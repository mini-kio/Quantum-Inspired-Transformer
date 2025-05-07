import torch
import numpy as np
import os
import json
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field


@dataclass
class HyperParameters:
    """중첩 모델에 대한 하이퍼파라미터 구성"""
    
    # 모델 구조 관련 파라미터
    d_model: int = 768
    nhead: int = 12
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1
    max_superposition_dim: int = 4
    activation: str = "gelu"
    
    # 중첩 상태 관련 파라미터
    superposition_degree: float = 0.7  # 중첩 정도(0~1)
    collapse_threshold: float = 0.5    # 상태 붕괴 임계값(0~1)
    interference_strength: float = 0.3  # 간섭 강도(0~1)
    
    # 학습 관련 파라미터
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 10000
    max_steps: int = 500000
    lr_scheduler: str = "inverse_sqrt"  # linear, cosine, inverse_sqrt
    
    # 손실 함수 가중치
    task_loss_weight: float = 0.6
    superposition_reg_weight: float = 0.2
    consistency_loss_weight: float = 0.2
    uncertainty_correction_weight: float = 0.2  # 새로 추가
    resource_penalty_weight: float = 0.1  # 새로 추가
    
    # 메타 학습 관련 파라미터
    use_meta_learning: bool = False
    meta_learning_rate: float = 1e-5
    meta_steps: int = 3
    
    # 효율성 관련 파라미터
    target_sparsity: float = 0.7
    parameter_sharing_ratio: float = 0.5
    computation_efficiency_target: float = 0.8
    
    # CollapseGate 관련 파라미터 (새로 추가)
    p_target: float = 0.5  # 목표 전환 확률
    alpha_init: float = 0.5  # 초기 soft/hard 붕괴 혼합 비율
    gate_type: str = "mlp"  # "mlp" 또는 "transformer"
    
    # 커리큘럼 학습 관련 파라미터 (새로 추가)
    use_curriculum: bool = True
    curriculum_epochs: List[int] = field(default_factory=lambda: [1, 3, 5])
    curriculum_difficulties: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    
    # 추론 관련 파라미터
    inference_mode: str = "adaptive"  # adaptive, efficient, accurate
    force_collapse_inference: bool = True
    reasoning_depth: int = 3
    
    # 기타 파라미터
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """하이퍼파라미터를 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'HyperParameters':
        """딕셔너리에서 하이퍼파라미터 객체 생성"""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class CurriculumScheduler:
    """
    난이도별 커리큘럼 학습 스케줄러
    
    에포크에 따라 데이터 난이도와 하이퍼파라미터를 동적으로 조정합니다.
    """
    
    def __init__(
        self,
        hparams: Union[Dict[str, Any], HyperParameters],
        difficulties: Optional[List[str]] = None,
        epoch_thresholds: Optional[List[int]] = None,
        total_epochs: int = 10
    ):
        """
        커리큘럼 스케줄러 초기화
        
        Args:
            hparams: 기본 하이퍼파라미터
            difficulties: 난이도 목록 ["easy", "medium", "hard"]
            epoch_thresholds: 난이도 변경 에포크 임계값
            total_epochs: 총 에포크 수
        """
        # 하이퍼파라미터 설정
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.hparams = hparams.to_dict()
            
        # 난이도 설정
        self.difficulties = difficulties or ["easy", "medium", "hard"]
        
        # 에포크 임계값 설정
        if epoch_thresholds is None:
            # 자동 임계값 계산: [1, 3, 5, ...]
            self.epoch_thresholds = [int(total_epochs * (i + 1) / (len(self.difficulties) + 1)) 
                                   for i in range(len(self.difficulties))]
        else:
            self.epoch_thresholds = epoch_thresholds
            
        # 난이도별 하이퍼파라미터 프리셋
        self.difficulty_presets = {
            "easy": {
                # 낮은 중첩, 높은 붕괴 - 간단한 문제에 적합
                "superposition_degree": 0.3,
                "collapse_threshold": 0.7,
                "interference_strength": 0.1,
                "p_target": 0.7,  # 자주 붕괴 (70%)
                "alpha_init": 0.3,  # hard 붕괴 비중 높음
                "task_loss_weight": 0.8,
                "superposition_reg_weight": 0.1,
                "consistency_loss_weight": 0.05,
                "uncertainty_correction_weight": 0.05,
                "resource_penalty_weight": 0.0  # 리소스 제약 없음
            },
            "medium": {
                # 중간 중첩, 중간 붕괴 - 보통 난이도에 적합
                "superposition_degree": 0.6,
                "collapse_threshold": 0.5,
                "interference_strength": 0.3,
                "p_target": 0.5,  # 균형있는 붕괴 (50%)
                "alpha_init": 0.5,  # 균형있는 soft/hard 혼합
                "task_loss_weight": 0.6,
                "superposition_reg_weight": 0.2,
                "consistency_loss_weight": 0.1,
                "uncertainty_correction_weight": 0.1,
                "resource_penalty_weight": 0.1  # 약간의 리소스 제약
            },
            "hard": {
                # 높은 중첩, 낮은 붕괴 - 복잡한 문제에 적합
                "superposition_degree": 0.9,
                "collapse_threshold": 0.3,
                "interference_strength": 0.5,
                "p_target": 0.3,  # 드물게 붕괴 (30%)
                "alpha_init": 0.7,  # soft 붕괴 비중 높음
                "task_loss_weight": 0.4,
                "superposition_reg_weight": 0.3,
                "consistency_loss_weight": 0.15,
                "uncertainty_correction_weight": 0.15,
                "resource_penalty_weight": 0.2  # 강한 리소스 제약
            }
        }
        
        # 현재 난이도 및 하이퍼파라미터 초기화
        self.current_difficulty = self.difficulties[0]
        self.current_hparams = self._merge_hparams(self.difficulty_presets[self.current_difficulty])
        
    def _merge_hparams(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """
        기본 하이퍼파라미터와 프리셋 병합
        
        Args:
            preset: 적용할 프리셋
            
        Returns:
            Dict[str, Any]: 병합된 하이퍼파라미터
        """
        merged = self.hparams.copy()
        
        # 프리셋 값으로 덮어쓰기
        for key, value in preset.items():
            if key in merged:
                merged[key] = value
                
        return merged
    
    def update(self, epoch: int) -> Dict[str, Any]:
        """
        에포크에 따른 난이도 및 하이퍼파라미터 업데이트
        
        Args:
            epoch: 현재 에포크
            
        Returns:
            Dict[str, Any]: 업데이트된 하이퍼파라미터
        """
        # 현재 에포크에 해당하는 난이도 결정
        new_difficulty = self.current_difficulty
        
        for i, threshold in enumerate(self.epoch_thresholds):
            if epoch >= threshold and i < len(self.difficulties):
                new_difficulty = self.difficulties[i]
        
        # 난이도가 변경된 경우 하이퍼파라미터 업데이트
        if new_difficulty != self.current_difficulty:
            self.current_difficulty = new_difficulty
            self.current_hparams = self._merge_hparams(self.difficulty_presets[new_difficulty])
            print(f"Curriculum Update: Switched to '{new_difficulty}' difficulty at epoch {epoch}")
        
        return self.current_hparams
    
    def get_difficulty_preset(self, difficulty: str) -> Dict[str, Any]:
        """
        특정 난이도의 하이퍼파라미터 프리셋 반환
        
        Args:
            difficulty: 난이도 이름
            
        Returns:
            Dict[str, Any]: 프리셋 하이퍼파라미터
        """
        if difficulty not in self.difficulty_presets:
            raise ValueError(f"Unknown difficulty: {difficulty}")
            
        return self.difficulty_presets[difficulty]
    
    def get_current_difficulty(self) -> str:
        """현재 난이도 반환"""
        return self.current_difficulty
    
    def get_current_hparams(self) -> Dict[str, Any]:
        """현재 하이퍼파라미터 반환"""
        return self.current_hparams
    
    def interpolate_hparams(self, epoch: float, smooth: bool = True) -> Dict[str, Any]:
        """
        에포크에 따라 난이도 간 부드러운 보간 수행
        
        Args:
            epoch: 현재 에포크 (소수점 허용)
            smooth: 부드러운 보간 수행 여부
            
        Returns:
            Dict[str, Any]: 보간된 하이퍼파라미터
        """
        # 현재 위치 결정
        current_idx = 0
        next_idx = 0
        interp_factor = 0.0
        
        for i, threshold in enumerate(self.epoch_thresholds):
            if epoch < threshold:
                if i == 0:
                    # 첫 번째 임계값 이전: 첫 번째 난이도만 사용
                    current_idx = 0
                    next_idx = 0
                    interp_factor = 0.0
                else:
                    # 두 난이도 사이
                    current_idx = i - 1
                    next_idx = i
                    prev_threshold = self.epoch_thresholds[i - 1] if i > 0 else 0
                    interp_factor = (epoch - prev_threshold) / (threshold - prev_threshold)
                break
        else:
            # 모든 임계값 이후: 마지막 난이도만 사용
            current_idx = len(self.difficulties) - 1
            next_idx = len(self.difficulties) - 1
            interp_factor = 1.0
        
        # 부드러운 보간 적용
        if smooth:
            # 0~1 사이를 코사인 커브로 보간 (더 부드러운 전환)
            interp_factor = 0.5 * (1 - math.cos(interp_factor * math.pi))
        
        # 현재/다음 난이도 가져오기
        current_difficulty = self.difficulties[current_idx]
        next_difficulty = self.difficulties[next_idx]
        
        # 하이퍼파라미터 보간
        if current_difficulty == next_difficulty:
            # 같은 난이도면 보간 불필요
            return self._merge_hparams(self.difficulty_presets[current_difficulty])
        else:
            # 두 난이도 사이 보간
            current_preset = self.difficulty_presets[current_difficulty]
            next_preset = self.difficulty_presets[next_difficulty]
            
            # 보간된 프리셋 생성
            interpolated_preset = {}
            for key in current_preset:
                if key in next_preset:
                    current_value = current_preset[key]
                    next_value = next_preset[key]
                    
                    # 숫자 값만 보간
                    if isinstance(current_value, (int, float)) and isinstance(next_value, (int, float)):
                        interpolated_preset[key] = current_value + interp_factor * (next_value - current_value)
                    else:
                        # 숫자가 아닌 값은 보간 비율에 따라 선택
                        interpolated_preset[key] = next_value if interp_factor > 0.5 else current_value
            
            return self._merge_hparams(interpolated_preset)
    
    def get_dataloaders_for_difficulty(self, 
                                      dataset,
                                      batch_size: int,
                                      difficulty: Optional[str] = None,
                                      num_workers: int = 4):
        """
        난이도에 따른 데이터 로더 생성
        
        Args:
            dataset: 원본 데이터셋
            batch_size: 배치 크기
            difficulty: 난이도 (없으면 현재 난이도)
            num_workers: 데이터 로더 워커 수
            
        Returns:
            torch.utils.data.DataLoader: 난이도에 맞는 데이터 로더
        """
        import torch
        from torch.utils.data import DataLoader, Subset, RandomSampler
        
        if difficulty is None:
            difficulty = self.current_difficulty
            
        # 데이터셋 길이
        dataset_size = len(dataset)
        
        if difficulty == "easy":
            # 쉬운 난이도: 데이터셋의 처음 30%만 사용 (단순한 패턴)
            subset_size = int(dataset_size * 0.3)
            indices = list(range(subset_size))
            subset = Subset(dataset, indices)
            
            # 작은 배치 크기 (더 자주 업데이트)
            actual_batch_size = batch_size // 2
            
            return DataLoader(
                subset,
                batch_size=max(1, actual_batch_size),
                shuffle=True,
                num_workers=num_workers
            )
            
        elif difficulty == "medium":
            # 중간 난이도: 데이터셋의 중간 60% 사용
            subset_size = int(dataset_size * 0.6)
            start_idx = int(dataset_size * 0.2)
            indices = list(range(start_idx, start_idx + subset_size))
            subset = Subset(dataset, indices)
            
            # 기본 배치 크기
            return DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
        elif difficulty == "hard":
            # 어려운 난이도: 전체 데이터셋 사용 + 가중치 샘플링
            # 데이터 크기에 비례한 샘플링 가중치 (큰 샘플에 더 높은 가중치)
            weights = self._compute_sample_weights(dataset)
            sampler = RandomSampler(dataset) if weights is None else torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=dataset_size,
                replacement=True
            )
            
            # 큰 배치 크기 (더 안정적인 학습)
            actual_batch_size = batch_size * 2
            
            return DataLoader(
                dataset,
                batch_size=actual_batch_size,
                sampler=sampler,
                num_workers=num_workers
            )
        else:
            # 기본: 전체 데이터셋, 일반 샘플링
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
    
    def _compute_sample_weights(self, dataset) -> Optional[torch.Tensor]:
        """
        데이터셋 샘플에 대한 가중치 계산
        
        Args:
            dataset: 데이터셋
            
        Returns:
            Optional[torch.Tensor]: 샘플 가중치
        """
        import torch
        
        try:
            # 샘플 복잡성 추정 (예: 시퀀스 길이, 토큰 수 등)
            weights = []
            
            # 데이터셋 구조에 따라 다른 방식 적용
            if hasattr(dataset, 'input_ids'):
                # Hugging Face 스타일 데이터셋
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if isinstance(sample, dict) and 'input_ids' in sample:
                        # 입력 ID 길이를 복잡성 지표로 사용
                        weight = len(sample['input_ids'])
                    else:
                        weight = 1.0
                    weights.append(weight)
            elif hasattr(dataset, 'data'):
                # PyTorch 내장 데이터셋
                # 예: MNIST, CIFAR 등의 이미지 데이터에서는 픽셀 분산 사용
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if isinstance(sample, tuple):
                        data = sample[0]
                        if isinstance(data, torch.Tensor):
                            # 이미지 데이터의 분산
                            weight = data.var().item() + 0.1
                        else:
                            weight = 1.0
                    else:
                        weight = 1.0
                    weights.append(weight)
            else:
                # 범용 접근: 기본 가중치 반환
                return None
            
            # 가중치 정규화 (0~1)
            weights = torch.tensor(weights, dtype=torch.float)
            weights = weights / weights.max()
            
            return weights
            
        except Exception as e:
            print(f"Warning: Failed to compute sample weights: {e}")
            return None


class LearnableCollapseScheduler(nn.Module):
    """
    학습 가능한 붕괴 스케줄러
    
    레이어 ID, 불확실성, 토큰 임베딩에 기반하여 붕괴 임계값을 예측합니다.
    메인 네트워크와 End-to-End로 함께 학습됩니다.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        max_superposition_dim: int = 4,
        scheduler_type: str = "mlp"  # "mlp" 또는 "transformer"
    ):
        """
        학습 가능한 붕괴 스케줄러 초기화
        
        Args:
            hidden_dim: 기본 히든 차원
            num_layers: 총 레이어 수
            max_superposition_dim: 최대 중첩 상태 차원
            scheduler_type: 스케줄러 아키텍처 유형
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_superposition_dim = max_superposition_dim
        self.scheduler_type = scheduler_type
        
        # 레이어 임베딩
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)
        
        # 스케줄러 아키텍처 설정
        if scheduler_type == "mlp":
            # MLP 기반 스케줄러
            self.scheduler_network = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # 히든 + 레이어 + 불확실성
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3)  # 임계값, p_target, alpha
            )
        else:
            # Transformer 기반 스케줄러
            self.input_projection = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_projection = nn.Linear(hidden_dim, 3)  # 임계값, p_target, alpha
            
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        # 레이어 임베딩 초기화
        nn.init.normal_(self.layer_embedding.weight, std=0.02)
        
        if self.scheduler_type == "mlp":
            # MLP 초기화
            for module in self.scheduler_network:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            # Transformer 초기화
            nn.init.normal_(self.input_projection.weight, std=0.02)
            nn.init.zeros_(self.input_projection.bias)
            
            nn.init.normal_(self.self_attn.in_proj_weight, std=0.02)
            nn.init.zeros_(self.self_attn.in_proj_bias)
            nn.init.normal_(self.self_attn.out_proj.weight, std=0.02)
            nn.init.zeros_(self.self_attn.out_proj.bias)
            
            for module in self.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            nn.init.normal_(self.output_projection.weight, std=0.02)
            nn.init.zeros_(self.output_projection.bias)
            
    def forward(
        self,
        token_embeddings: torch.Tensor,
        uncertainty: torch.Tensor,
        layer_id: int,
        base_threshold: float = 0.5,
        base_p_target: float = 0.5,
        base_alpha: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        학습 가능한 붕괴 스케줄러 순전파
        
        Args:
            token_embeddings: 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            uncertainty: 불확실성 추정값 [batch_size, seq_len, 1]
            layer_id: 현재 레이어 ID
            base_threshold: 기본 붕괴 임계값
            base_p_target: 기본 목표 전환 확률
            base_alpha: 기본 soft/hard 붕괴 혼합 비율
            
        Returns:
            Dict[str, torch.Tensor]: 스케줄러 예측값
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 레이어 임베딩 가져오기
        layer_emb = self.layer_embedding(torch.tensor(layer_id, device=token_embeddings.device))
        layer_emb = layer_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # 모든 정보 결합
        combined_input = torch.cat([token_embeddings, layer_emb, uncertainty], dim=-1)
        
        if self.scheduler_type == "mlp":
            # MLP 스케줄러로 예측
            raw_outputs = self.scheduler_network(combined_input)
        else:
            # Transformer 스케줄러로 예측
            x = self.input_projection(combined_input)
            attn_output, _ = self.self_attn(x, x, x)
            x = x + attn_output
            x = self.norm1(x)
            
            ffn_output = self.ffn(x)
            x = x + ffn_output
            x = self.norm2(x)
            
            raw_outputs = self.output_projection(x)
        
        # 출력 분리 및 활성화
        threshold = torch.sigmoid(raw_outputs[..., 0:1]) * 0.8 + 0.1  # 0.1~0.9 범위
        p_target = torch.sigmoid(raw_outputs[..., 1:2]) * 0.8 + 0.1  # 0.1~0.9 범위
        alpha = torch.sigmoid(raw_outputs[..., 2:3])  # 0~1 범위
        
        # 기본값에 보정치 추가
        final_threshold = base_threshold * torch.ones_like(threshold) * 0.7 + threshold * 0.3
        final_p_target = base_p_target * torch.ones_like(p_target) * 0.7 + p_target * 0.3
        final_alpha = base_alpha * torch.ones_like(alpha) * 0.7 + alpha * 0.3
        
        return {
            'threshold': final_threshold,
            'p_target': final_p_target,
            'alpha': final_alpha,
            'raw_outputs': raw_outputs
        }


class AdaptiveTuningScheduler:
    """
    중첩 및 붕괴 파라미터의 점진적 미세 조정 스케줄러
    """
    
    def __init__(
        self,
        hparams: Dict[str, Any],
        total_steps: int,
        steps_per_epoch: int,
        warmup_epochs: int = 1,
        schedule_type: str = 'linear'
    ):
        """
        적응형 미세 조정 스케줄러 초기화
        
        Args:
            hparams: 하이퍼파라미터
            total_steps: 총 학습 스텝 수
            steps_per_epoch: 에포크당 스텝 수
            warmup_epochs: 워밍업 에포크 수
            schedule_type: 스케줄 유형 ('linear', 'exponential', 'cosine')
        """
        self.hparams = hparams
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.schedule_type = schedule_type
        
        # 초기 및 최종 값 설정
        self.initial_values = {
            'superposition_degree': hparams.get('superposition_degree', 0.7) * 0.3,
            'collapse_threshold': hparams.get('collapse_threshold', 0.5) * 2.0,
            'interference_strength': hparams.get('interference_strength', 0.3) * 0.2,
            'task_loss_weight': hparams.get('task_loss_weight', 0.6) * 0.8,
            'superposition_reg_weight': hparams.get('superposition_reg_weight', 0.2) * 1.5,
            'consistency_loss_weight': hparams.get('consistency_loss_weight', 0.2) * 1.2,
            'p_target': hparams.get('p_target', 0.5) * 1.5,  # 초기에는 더 자주 붕괴
            'alpha_init': hparams.get('alpha_init', 0.5) * 0.5  # 초기에는 hard 붕괴 선호
        }
        
        self.final_values = {
            'superposition_degree': hparams.get('superposition_degree', 0.7),
            'collapse_threshold': hparams.get('collapse_threshold', 0.5),
            'interference_strength': hparams.get('interference_strength', 0.3),
            'task_loss_weight': hparams.get('task_loss_weight', 0.6),
            'superposition_reg_weight': hparams.get('superposition_reg_weight', 0.2),
            'consistency_loss_weight': hparams.get('consistency_loss_weight', 0.2),
            'p_target': hparams.get('p_target', 0.5),
            'alpha_init': hparams.get('alpha_init', 0.5)
        }
        
        # 현재 값 초기화
        self.current_values = self.initial_values.copy()
        
    def step(self, step: int) -> Dict[str, float]:
        """
        현재 스텝에 따른 파라미터 값 갱신
        
        Args:
            step: 현재 스텝
            
        Returns:
            Dict[str, float]: 현재 파라미터 값
        """
        # 워밍업 단계에서는 초기값 유지
        if step < self.warmup_steps:
            return self.current_values
        
        # 워밍업 이후 스케줄에 따라 값 조정
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
        if self.schedule_type == 'linear':
            # 선형 스케줄
            factor = progress
        elif self.schedule_type == 'exponential':
            # 지수 스케줄
            factor = 1.0 - math.exp(-5 * progress)
        elif self.schedule_type == 'cosine':
            # 코사인 스케줄
            factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        else:
            factor = progress
        
        # 각 파라미터 값 갱신
        for param_name in self.initial_values.keys():
            initial_value = self.initial_values[param_name]
            final_value = self.final_values[param_name]
            
            self.current_values[param_name] = initial_value + factor * (final_value - initial_value)
        
        return self.current_values
    
    def get_current_values(self) -> Dict[str, float]:
        """현재 파라미터 값 반환"""
        return self.current_values


class QuantumMetaScheduler:
    """
    태스크 및 데이터 특성에 따라 중첩 상태 파라미터를 자동 조정하는 메타 스케줄러
    """
    
    def __init__(
        self,
        hparams: Dict[str, Any],
        task_type: str = 'classification',
        adaptation_rate: float = 0.1,
        history_length: int = 10
    ):
        """
        양자 메타 스케줄러 초기화
        
        Args:
            hparams: 하이퍼파라미터
            task_type: 태스크 유형 ('classification', 'regression', 'generation')
            adaptation_rate: 적응 속도
            history_length: 히스토리 길이
        """
        self.hparams = hparams
        self.task_type = task_type
        self.adaptation_rate = adaptation_rate
        self.history_length = history_length
        
        # 태스크별 기본 파라미터 조정
        task_adjustments = {
            'classification': {
                'superposition_degree': 1.0,
                'collapse_threshold': 0.9,
                'interference_strength': 0.7,
                'p_target': 0.6,  # 분류는 조금 더 확정적
                'alpha_init': 0.4  # 분류는 hard 붕괴 선호
            },
            'regression': {
                'superposition_degree': 0.8,
                'collapse_threshold': 0.7,
                'interference_strength': 0.5,
                'p_target': 0.5,  # 회귀는 중간 정도
                'alpha_init': 0.5  # 회귀는 균형 선호
            },
            'generation': {
                'superposition_degree': 1.2,
                'collapse_threshold': 0.6,
                'interference_strength': 1.0,
                'p_target': 0.3,  # 생성은 중첩 상태 유지 선호
                'alpha_init': 0.7  # 생성은 soft 붕괴 선호
            }
        }
        
        # 현재 태스크 조정값 설정
        self.task_adjustment = task_adjustments.get(
            task_type, 
            {'superposition_degree': 1.0, 'collapse_threshold': 1.0, 'interference_strength': 1.0,
             'p_target': 0.5, 'alpha_init': 0.5}
        )
        
        # 성능 히스토리
        self.performance_history = []
        
        # 현재 파라미터 값
        self.current_values = {
            'superposition_degree': hparams.get('superposition_degree', 0.7) * self.task_adjustment['superposition_degree'],
            'collapse_threshold': hparams.get('collapse_threshold', 0.5) * self.task_adjustment['collapse_threshold'],
            'interference_strength': hparams.get('interference_strength', 0.3) * self.task_adjustment['interference_strength'],
            'p_target': hparams.get('p_target', 0.5) * self.task_adjustment['p_target'],
            'alpha_init': hparams.get('alpha_init', 0.5) * self.task_adjustment['alpha_init']
        }
        
        # 초기 기준 성능
        self.baseline_performance = None
        
    def update(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        성능 지표에 따라 파라미터 업데이트
        
        Args:
            performance_metrics: 성능 지표
            
        Returns:
            Dict[str, float]: 업데이트된 파라미터 값
        """
        # 성능 지표 처리
        if 'loss' in performance_metrics:
            current_performance = -performance_metrics['loss']  # 손실은 낮을수록 좋음
        elif 'accuracy' in performance_metrics:
            current_performance = performance_metrics['accuracy']
        elif 'score' in performance_metrics:
            current_performance = performance_metrics['score']
        else:
            # 기본 성능 지표가 없으면 현재 값 유지
            return self.current_values
        
        # 기준 성능 초기화
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            self.performance_history.append(current_performance)
            return self.current_values
        
        # 성능 히스토리 업데이트
        self.performance_history.append(current_performance)
        if len(self.performance_history) > self.history_length:
            self.performance_history.pop(0)
        
        # 최근 성능 추세 계산
        recent_trend = 0
        if len(self.performance_history) > 1:
            # 최근 성능 변화율
            recent_performances = self.performance_history[-3:]
            if len(recent_performances) > 1:
                recent_trend = (recent_performances[-1] - recent_performances[0]) / abs(recent_performances[0] + 1e-10)
        
        # 파라미터 조정 계수 계산
        adjustment_factor = self.adaptation_rate * recent_trend
        
        # 파라미터별 조정 로직
        adjustments = {
            # 성능 향상 시 중첩 정도 증가, 악화 시 감소
            'superposition_degree': adjustment_factor * 0.1,
            
            # 성능 향상 시 붕괴 임계값 감소, 악화 시 증가
            'collapse_threshold': -adjustment_factor * 0.05,
            
            # 성능 향상 시 간섭 강도 증가, 악화 시 감소
            'interference_strength': adjustment_factor * 0.1,
            
            # 성능 향상 시 p_target 감소 (더 적은 붕괴), 악화 시 증가
            'p_target': -adjustment_factor * 0.05,
            
            # 성능 향상 시 alpha 증가 (더 많은 soft 붕괴), 악화 시 감소
            'alpha_init': adjustment_factor * 0.05
        }
        
        # 파라미터 업데이트
        for param_name, adjustment in adjustments.items():
            base_value = self.hparams.get(param_name, self.current_values[param_name])
            task_adj = self.task_adjustment.get(param_name, 1.0)
            
            # 현재 값 조정
            self.current_values[param_name] = self.current_values[param_name] * (1 + adjustment)
            
            # 유효 범위 제한
            if param_name == 'superposition_degree':
                self.current_values[param_name] = min(max(self.current_values[param_name], 0.1), 0.95)
            elif param_name == 'collapse_threshold':
                self.current_values[param_name] = min(max(self.current_values[param_name], 0.2), 0.8)
            elif param_name == 'interference_strength':
                self.current_values[param_name] = min(max(self.current_values[param_name], 0.05), 0.5)
            elif param_name == 'p_target':
                self.current_values[param_name] = min(max(self.current_values[param_name], 0.1), 0.9)
            elif param_name == 'alpha_init':
                self.current_values[param_name] = min(max(self.current_values[param_name], 0.1), 0.9)
        
        return self.current_values


class HyperParameterOptimizer:
    """
    중첩 강도와 붕괴 임계값에 대한 자동 하이퍼파라미터 최적화
    """
    
    def __init__(
        self,
        model_builder: Callable[[Dict[str, Any]], nn.Module],
        evaluator: Callable[[nn.Module, Dict[str, Any]], float],
        base_config: Optional[HyperParameters] = None,
        search_space: Optional[Dict[str, Any]] = None,
        num_trials: int = 20,
        optimization_method: str = 'bayesian',
        output_dir: str = './hparam_results',
        device: str = 'cuda'
    ):
        """
        하이퍼파라미터 최적화기 초기화
        
        Args:
            model_builder: 하이퍼파라미터로 모델을 구축하는 함수
            evaluator: 모델을 평가하는 함수 (모델, 하이퍼파라미터를 받고 점수 반환)
            base_config: 기본 하이퍼파라미터 구성
            search_space: 탐색 공간 정의
            num_trials: 시도할 시험 횟수
            optimization_method: 최적화 방법 ('random', 'grid', 'bayesian')
            output_dir: 결과 저장 디렉토리
            device: 사용할 장치
        """
        self.model_builder = model_builder
        self.evaluator = evaluator
        self.base_config = base_config or HyperParameters()
        self.num_trials = num_trials
        self.optimization_method = optimization_method
        self.output_dir = output_dir
        self.device = device
        
        # 기본 검색 공간 정의
        default_search_space = {
            'superposition_degree': {'min': 0.3, 'max': 0.9, 'type': 'float'},
            'collapse_threshold': {'min': 0.2, 'max': 0.8, 'type': 'float'},
            'interference_strength': {'min': 0.1, 'max': 0.5, 'type': 'float'},
            'max_superposition_dim': {'values': [2, 3, 4, 6, 8], 'type': 'choice'},
            'learning_rate': {'min': 1e-5, 'max': 5e-4, 'type': 'float', 'log': True},
            'weight_decay': {'min': 0.001, 'max': 0.1, 'type': 'float', 'log': True},
            'task_loss_weight': {'min': 0.4, 'max': 0.8, 'type': 'float'},
            'superposition_reg_weight': {'min': 0.1, 'max': 0.4, 'type': 'float'},
            'consistency_loss_weight': {'min': 0.1, 'max': 0.4, 'type': 'float'},
            'uncertainty_correction_weight': {'min': 0.1, 'max': 0.3, 'type': 'float'},
            'resource_penalty_weight': {'min': 0.0, 'max': 0.3, 'type': 'float'},
            'target_sparsity': {'min': 0.5, 'max': 0.9, 'type': 'float'},
            'p_target': {'min': 0.2, 'max': 0.8, 'type': 'float'},
            'alpha_init': {'min': 0.2, 'max': 0.8, 'type': 'float'},
            'gate_type': {'values': ['mlp', 'transformer'], 'type': 'choice'}
        }
        
        self.search_space = search_space or default_search_space
        
        # 최적화 결과 저장
        self.trials = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # 베이지안 최적화를 위한 설정
        self.bayesian_state = None
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
    def sample_parameters(self, trial_id: int) -> Dict[str, Any]:
        """
        현재 검색 공간에서 하이퍼파라미터 샘플링
        
        Args:
            trial_id: 현재 시도 ID
            
        Returns:
            Dict[str, Any]: 샘플링된 하이퍼파라미터
        """
        if self.optimization_method == 'random':
            return self._random_sample()
        elif self.optimization_method == 'grid':
            return self._grid_sample(trial_id)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_sample(trial_id)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _random_sample(self) -> Dict[str, Any]:
        """무작위 샘플링"""
        sampled_params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                
                if param_config.get('log', False):
                    # 로그 스케일 샘플링
                    log_min = math.log(min_val)
                    log_max = math.log(max_val)
                    sampled_params[param_name] = math.exp(np.random.uniform(log_min, log_max))
                else:
                    # 선형 스케일 샘플링
                    sampled_params[param_name] = np.random.uniform(min_val, max_val)
                    
            elif param_type == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                sampled_params[param_name] = np.random.randint(min_val, max_val + 1)
                
            elif param_type == 'choice':
                values = param_config['values']
                sampled_params[param_name] = np.random.choice(values)
                
            elif param_type == 'bool':
                sampled_params[param_name] = bool(np.random.randint(0, 2))
        
        return sampled_params
    
    def _grid_sample(self, trial_id: int) -> Dict[str, Any]:
        """그리드 검색 샘플링"""
        
        # 그리드 포인트 생성
        grid_points = []
        grid_sizes = []
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                
                if param_config.get('log', False):
                    # 로그 스케일 그리드
                    log_min = math.log(min_val)
                    log_max = math.log(max_val)
                    num_points = param_config.get('num_points', 5)
                    points = [math.exp(log_min + i * (log_max - log_min) / (num_points - 1)) for i in range(num_points)]
                else:
                    # 선형 스케일 그리드
                    num_points = param_config.get('num_points', 5)
                    points = [min_val + i * (max_val - min_val) / (num_points - 1) for i in range(num_points)]
                    
            elif param_type == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                step = param_config.get('step', 1)
                points = list(range(min_val, max_val + 1, step))
                
            elif param_type == 'choice':
                points = param_config['values']
                
            elif param_type == 'bool':
                points = [False, True]
                
            grid_points.append(points)
            grid_sizes.append(len(points))
        
        # 총 그리드 크기 계산
        total_grid_size = np.prod(grid_sizes)
        
        # trial_id에 해당하는 인덱스 계산
        index = trial_id % total_grid_size
        
        # 인덱스를 각 파라미터의 인덱스로 변환
        param_indices = []
        for size in reversed(grid_sizes):
            param_indices.insert(0, index % size)
            index //= size
            
        # 최종 파라미터 구성
        sampled_params = {}
        for i, (param_name, _) in enumerate(self.search_space.items()):
            param_idx = param_indices[i]
            sampled_params[param_name] = grid_points[i][param_idx]
            
        return sampled_params
    
    def _bayesian_sample(self, trial_id: int) -> Dict[str, Any]:
        """베이지안 최적화 샘플링"""
        try:
            # 필요한 패키지 임포트 (설치되어 있어야 함)
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.stats import norm
        except ImportError:
            print("scikit-learn과 scipy가 필요합니다. pip install scikit-learn scipy")
            return self._random_sample()
        
        # 첫 번째 시도는 무작위 샘플링
        if trial_id == 0 or not self.trials:
            return self._random_sample()
        
        # 검색 공간 준비
        param_names = list(self.search_space.keys())
        param_bounds = []
        param_types = []
        
        for param_name in param_names:
            param_config = self.search_space[param_name]
            param_type = param_config['type']
            param_types.append(param_type)
            
            if param_type in ['float', 'int']:
                min_val = param_config['min']
                max_val = param_config['max']
                
                if param_type == 'float' and param_config.get('log', False):
                    # 로그 스케일 변환
                    min_val = math.log(min_val)
                    max_val = math.log(max_val)
                    
                param_bounds.append((min_val, max_val))
                
            elif param_type == 'choice':
                values = param_config['values']
                param_bounds.append((0, len(values) - 1))
                
            elif param_type == 'bool':
                param_bounds.append((0, 1))
        
        # 이전 시도에서 데이터 준비
        X = []
        y = []
        
        for trial in self.trials:
            # 파라미터 벡터 생성
            params_vector = []
            
            for i, param_name in enumerate(param_names):
                param_value = trial['params'].get(param_name)
                param_config = self.search_space[param_name]
                param_type = param_config['type']
                
                if param_type == 'float':
                    if param_config.get('log', False):
                        param_value = math.log(param_value)
                    params_vector.append(param_value)
                    
                elif param_type == 'int':
                    params_vector.append(param_value)
                    
                elif param_type == 'choice':
                    values = param_config['values']
                    param_idx = values.index(param_value)
                    params_vector.append(param_idx)
                    
                elif param_type == 'bool':
                    params_vector.append(int(param_value))
            
            X.append(params_vector)
            y.append(trial['score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # 가우시안 프로세스 회귀 모델 학습
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gp.fit(X, y)
        
        # 획득 함수: Expected Improvement
        def expected_improvement(x, gp, y_best):
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
            z = (mean - y_best) / (std + 1e-9)
            return (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
        
        # 현재 최고 점수
        y_best = max(y)
        
        # 획득 함수 최적화 (무작위 탐색으로 단순화)
        best_ei = -1
        best_params_vector = None
        
        for _ in range(100):  # 100회 무작위 시도
            params_vector = []
            
            for i, (param_name, param_config) in enumerate(self.search_space.items()):
                param_type = param_config['type']
                
                if param_type in ['float', 'int']:
                    min_val, max_val = param_bounds[i]
                    
                    if param_type == 'float':
                        param_value = np.random.uniform(min_val, max_val)
                    else:
                        param_value = np.random.randint(min_val, max_val + 1)
                        
                elif param_type == 'choice':
                    num_choices = len(param_config['values'])
                    param_value = np.random.randint(0, num_choices)
                    
                elif param_type == 'bool':
                    param_value = np.random.randint(0, 2)
                    
                params_vector.append(param_value)
            
            ei = expected_improvement(np.array(params_vector), gp, y_best)
            
            if ei > best_ei:
                best_ei = ei
                best_params_vector = params_vector
        
        # 파라미터 벡터를 원래 형식으로 변환
        sampled_params = {}
        
        for i, param_name in enumerate(param_names):
            param_config = self.search_space[param_name]
            param_type = param_config['type']
            param_value = best_params_vector[i]
            
            if param_type == 'float':
                if param_config.get('log', False):
                    param_value = math.exp(param_value)
                sampled_params[param_name] = param_value
                
            elif param_type == 'int':
                sampled_params[param_name] = int(param_value)
                
            elif param_type == 'choice':
                values = param_config['values']
                sampled_params[param_name] = values[int(param_value)]
                
            elif param_type == 'bool':
                sampled_params[param_name] = bool(int(param_value))
        
        return sampled_params
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 실행
        
        Returns:
            Dict[str, Any]: 최적화 결과
        """
        print(f"Starting hyperparameter optimization using {self.optimization_method} search...")
        print(f"Number of trials: {self.num_trials}")
        print(f"Search space: {json.dumps(self.search_space, indent=2)}")
        
        for trial_id in range(self.num_trials):
            print(f"\nTrial {trial_id + 1}/{self.num_trials}")
            
            # 하이퍼파라미터 샘플링
            sampled_params = self.sample_parameters(trial_id)
            
            # 기본 구성과 샘플링된 파라미터 병합
            config = self.base_config.to_dict()
            config.update(sampled_params)
            
            # 하이퍼파라미터 객체 생성
            hparams = HyperParameters.from_dict(config)
            
            # 모델 생성
            try:
                start_time = time.time()
                model = self.model_builder(hparams.to_dict())
                model.to(self.device)
                
                # 모델 평가
                score = self.evaluator(model, hparams.to_dict())
                
                # 소요 시간 계산
                elapsed_time = time.time() - start_time
                
                # 시도 결과 저장
                trial_result = {
                    'trial_id': trial_id,
                    'params': sampled_params,
                    'score': score,
                    'elapsed_time': elapsed_time
                }
                
                self.trials.append(trial_result)
                
                # 최고 점수 업데이트
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = hparams.to_dict()
                    self.best_trial = trial_result
                    
                    # 최고 구성 저장
                    self.save_best_config()
                
                print(f"Parameters: {json.dumps(sampled_params, indent=2)}")
                print(f"Score: {score:.6f}, Time: {elapsed_time:.2f}s")
                print(f"Best score so far: {self.best_score:.6f}")
                
            except Exception as e:
                print(f"Error in trial {trial_id}: {e}")
                
            # 진행 상황 저장
            self.save_progress()
        
        print("\nHyperparameter optimization completed!")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best parameters: {json.dumps(self.best_params, indent=2)}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trials': self.trials
        }
    
    def save_progress(self) -> None:
        """진행 상황 저장"""
        progress_path = os.path.join(self.output_dir, 'optimization_progress.json')
        
        with open(progress_path, 'w') as f:
            json.dump({
                'trials': self.trials,
                'best_score': self.best_score,
                'best_params': self.best_params
            }, f, indent=2)
    
    def save_best_config(self) -> None:
        """최고 구성 저장"""
        config_path = os.path.join(self.output_dir, 'best_config.json')
        
        with open(config_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
    
    def load_progress(self) -> None:
        """이전 진행 상황 로드"""
        progress_path = os.path.join(self.output_dir, 'optimization_progress.json')
        
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                
            self.trials = progress.get('trials', [])
            self.best_score = progress.get('best_score', float('-inf'))
            self.best_params = progress.get('best_params', None)
            
            print(f"Loaded {len(self.trials)} previous trials")
            print(f"Previous best score: {self.best_score:.6f}")
    
    def get_learning_rate_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        hparams: Dict[str, Any]
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        하이퍼파라미터에 따른 학습률 스케줄러 생성
        
        Args:
            optimizer: 최적화기
            hparams: 하이퍼파라미터
            
        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: 학습률 스케줄러
        """
        scheduler_type = hparams.get('lr_scheduler', 'inverse_sqrt')
        warmup_steps = hparams.get('warmup_steps', 10000)
        max_steps = hparams.get('max_steps', 500000)
        
        if scheduler_type == 'linear':
            # 선형 스케줄러
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
        elif scheduler_type == 'cosine':
            # 코사인 스케줄러
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=1e-6
            )
            
        elif scheduler_type == 'inverse_sqrt':
            # 역제곱근 스케줄러 (커스텀 함수로 구현)
            warmup_init_lr = hparams.get('learning_rate', 1e-4) * 0.01
            peak_lr = hparams.get('learning_rate', 1e-4)
            
            lambda_lr = lambda step: min(
                1.0, step / warmup_steps) * (1 / math.sqrt(max(step, warmup_steps) / warmup_steps)
            ) if step < max_steps else 0.0
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
            
        else:
            return None