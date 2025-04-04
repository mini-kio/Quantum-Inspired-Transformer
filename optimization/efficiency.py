import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComputationalEfficiencyFramework(nn.Module):
    """
    글로벌 계산 효율성 프레임워크
    
    희소 중첩 활성화와 공유 파라미터의 체계적 통합으로 계산 효율성 최적화
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, num_heads=8, sparsity_target=0.7):
        """
        계산 효율성 프레임워크 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            num_heads (int): 어텐션 헤드 수
            sparsity_target (float): 목표 희소성 (0~1, 높을수록 더 희소)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.num_heads = num_heads
        self.sparsity_target = sparsity_target
        
        # 계산량 추정기
        self.computation_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 희소성 컨트롤러
        self.sparsity_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_superposition_dim),
            nn.Sigmoid()
        )
        
        # 중첩 차원 선택기
        self.dimension_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_superposition_dim),
            nn.Softmax(dim=-1)
        )
        
        # 파라미터 공유 매핑
        self.parameter_sharing_map = nn.Parameter(
            torch.eye(max_superposition_dim)  # 초기값은 항등 매핑
        )
        
        # 계산 효율성 점수 함수
        self.efficiency_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def estimate_computation_load(self, token_embeddings, context=None):
        """
        입력에 따른 계산량 추정
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩
            context (torch.Tensor, optional): 문맥 임베딩
            
        Returns:
            torch.Tensor: 계산량 점수 (0~1, 높을수록 더 많은 계산 필요)
        """
        # 문맥이 제공되지 않은 경우 토큰 평균 사용
        if context is None:
            context = token_embeddings.mean(dim=1)
            
        # 토큰별 계산량 추정
        batch_size, seq_len, _ = token_embeddings.shape
        
        computation_scores = []
        for i in range(seq_len):
            score = self.computation_estimator(token_embeddings[:, i])
            computation_scores.append(score)
            
        # 계산 점수 스택
        computation_scores = torch.cat(computation_scores, dim=1)
        
        return computation_scores
    
    def generate_sparsity_mask(self, token_embeddings, computation_scores, context=None):
        """
        계산 효율성을 위한 희소성 마스크 생성
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩
            computation_scores (torch.Tensor): 계산량 점수
            context (torch.Tensor, optional): 문맥 임베딩
            
        Returns:
            torch.Tensor: 희소성 마스크 (0~1, 1은 활성화된 차원)
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 문맥이 제공되지 않은 경우 토큰 평균 사용
        if context is None:
            context = token_embeddings.mean(dim=1)
            
        # 차원별 희소성 점수 생성
        sparsity_scores = self.sparsity_controller(context).unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # 계산량에 따른 희소성 조정
        adjusted_sparsity = sparsity_scores * (1 - computation_scores.unsqueeze(-1))
        
        # 희소성 임계값 (sparsity_target에 따라 조정)
        threshold = torch.quantile(adjusted_sparsity, self.sparsity_target, dim=-1, keepdim=True)
        
        # 임계값 이상의 차원만 활성화하는 희소 마스크 생성
        sparse_mask = (adjusted_sparsity >= threshold).float()
        
        return sparse_mask
    
    def select_active_dimensions(self, context):
        """
        문맥에 따른 활성 중첩 차원 선택
        
        Args:
            context (torch.Tensor): 문맥 임베딩
            
        Returns:
            torch.Tensor: 차원별 활성화 가중치
        """
        return self.dimension_selector(context)
    
    def apply_parameter_sharing(self, superposition_state, active_dimensions):
        """
        파라미터 공유 매핑 적용
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태
            active_dimensions (torch.Tensor): 활성 차원 가중치
            
        Returns:
            torch.Tensor: 파라미터 공유가 적용된 중첩 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        
        # 활성 차원 가중치 확장
        active_dimensions = active_dimensions.unsqueeze(1).unsqueeze(-1)
        
        # 파라미터 공유 매핑 적용
        shared_state = torch.einsum('bsid,ij->bsjd', reshaped, self.parameter_sharing_map)
        
        # 활성 차원 가중치 적용
        weighted_state = shared_state * active_dimensions
        
        # 형태 복원
        return weighted_state.view(batch_size, seq_len, -1)
    
    def calculate_efficiency_score(self, original_state, optimized_state, context):
        """
        최적화 전후의 계산 효율성 점수 계산
        
        Args:
            original_state (torch.Tensor): 원본 상태
            optimized_state (torch.Tensor): 최적화된 상태
            context (torch.Tensor): 문맥 임베딩
            
        Returns:
            float: 계산 효율성 점수 (높을수록 더 효율적)
        """
        batch_size = context.shape[0]
        
        # 원본 및 최적화 상태의 평균
        original_mean = original_state.mean(dim=1)
        optimized_mean = optimized_state.mean(dim=1)
        
        # 각 배치에 대한 효율성 점수 계산
        efficiency_scores = []
        for b in range(batch_size):
            combined = torch.cat([original_mean[b], optimized_mean[b]]).unsqueeze(0)
            score = self.efficiency_scorer(combined)
            efficiency_scores.append(score)
            
        # 평균 효율성 점수
        avg_efficiency_score = torch.cat(efficiency_scores).mean().item()
        
        return avg_efficiency_score
    
    def forward(self, token_embeddings, superposition_state=None, context=None, apply_optimization=True):
        """
        계산 효율성 프레임워크 순전파
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩
            superposition_state (torch.Tensor, optional): 중첩 상태
            context (torch.Tensor, optional): 문맥 임베딩
            apply_optimization (bool): 최적화 적용 여부
            
        Returns:
            dict: 효율성 최적화 결과
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 문맥이 제공되지 않은 경우 토큰 평균 사용
        if context is None:
            context = token_embeddings.mean(dim=1)
            
        # 계산량 추정
        computation_scores = self.estimate_computation_load(token_embeddings, context)
        
        # 희소성 마스크 생성
        sparse_mask = self.generate_sparsity_mask(token_embeddings, computation_scores, context)
        
        # 활성 차원 선택
        active_dimensions = self.select_active_dimensions(context)
        
        if apply_optimization and superposition_state is not None:
            # 원본 상태 저장
            original_state = superposition_state.clone()
            
            # 희소 마스크 적용
            masked_state = superposition_state.view(
                batch_size, seq_len, self.max_superposition_dim, -1
            )
            masked_state = masked_state * sparse_mask.unsqueeze(-1)
            masked_state = masked_state.view(batch_size, seq_len, -1)
            
            # 파라미터 공유 적용
            optimized_state = self.apply_parameter_sharing(masked_state, active_dimensions)
            
            # 효율성 점수 계산
            efficiency_score = self.calculate_efficiency_score(original_state, optimized_state, context)
            
            return {
                'optimized_state': optimized_state,
                'sparse_mask': sparse_mask,
                'active_dimensions': active_dimensions,
                'computation_scores': computation_scores,
                'efficiency_score': efficiency_score
            }
        else:
            return {
                'sparse_mask': sparse_mask,
                'active_dimensions': active_dimensions,
                'computation_scores': computation_scores
            }


class DynamicSparsityEngine(nn.Module):
    """
    동적 희소성 엔진
    
    런타임에 계산 필요성에 따라 활성화를 희소화하여 효율성 극대화
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, num_layers=12, sparsity_levels=4):
        """
        동적 희소성 엔진 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            num_layers (int): 모델 레이어 수
            sparsity_levels (int): 희소성 수준 수
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.num_layers = num_layers
        self.sparsity_levels = sparsity_levels
        
        # 계산 효율성 프레임워크
        self.efficiency_framework = ComputationalEfficiencyFramework(
            hidden_dim=hidden_dim,
            max_superposition_dim=max_superposition_dim
        )
        
        # 레이어별 희소성 정책
        self.layer_sparsity_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * sparsity_levels),
            nn.Softmax(dim=-1)
        )
        
        # 희소 활성화를 위한 게이트
        self.sparsity_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(sparsity_levels)
        ])
        
        # 희소성 수준별 가중치
        self.sparsity_level_weights = nn.Parameter(
            torch.linspace(0.2, 0.9, sparsity_levels)  # 0.2에서 0.9까지 단계적 희소성
        )
        
    def compute_layer_sparsity(self, context):
        """
        레이어별 희소성 정책 계산
        
        Args:
            context (torch.Tensor): 문맥 임베딩
            
        Returns:
            torch.Tensor: 레이어 및 희소성 수준별 정책 가중치
        """
        batch_size = context.shape[0]
        
        # 희소성 정책 계산
        policy = self.layer_sparsity_policy(context)
        
        # 정책 재구성 [batch_size, num_layers, sparsity_levels]
        policy = policy.view(batch_size, self.num_layers, self.sparsity_levels)
        
        return policy
    
    def apply_sparse_activation(self, hidden_states, context, layer_idx):
        """
        희소 활성화 적용
        
        Args:
            hidden_states (torch.Tensor): 히든 상태 [batch_size, seq_len, hidden_dim]
            context (torch.Tensor): 문맥 임베딩 [batch_size, hidden_dim]
            layer_idx (int): 현재 레이어 인덱스
            
        Returns:
            torch.Tensor: 희소 활성화가 적용된 히든 상태
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 레이어별 희소성 정책 계산
        layer_policy = self.compute_layer_sparsity(context)
        
        # 현재 레이어의 희소성 수준 가중치
        current_policy = layer_policy[:, layer_idx]  # [batch_size, sparsity_levels]
        
        # 희소 활성화 적용
        sparse_states = []
        for i in range(self.sparsity_levels):
            # 현재 희소성 수준의 가중치
            level_weight = current_policy[:, i].unsqueeze(1).unsqueeze(2)
            
            # 희소성 게이트 적용
            gate = self.sparsity_gates[i](context).unsqueeze(1)
            
            # 희소성 마스크 생성
            sparsity_threshold = self.sparsity_level_weights[i]
            sparse_mask = (gate > sparsity_threshold).float()
            
            # 희소 활성화
            sparse_state = hidden_states * sparse_mask
            
            # 가중 희소 상태
            weighted_sparse_state = sparse_state * level_weight
            sparse_states.append(weighted_sparse_state)
            
        # 모든 희소성 수준의 상태 결합
        combined_sparse_state = torch.zeros_like(hidden_states)
        for state in sparse_states:
            combined_sparse_state += state
            
        return combined_sparse_state
    
    def forward(self, hidden_states, superposition_states=None, context=None, layer_indices=None):
        """
        동적 희소성 엔진 순전파
        
        Args:
            hidden_states (list or torch.Tensor): 레이어별 히든 상태 목록 또는 단일 상태
            superposition_states (list or torch.Tensor, optional): 레이어별 중첩 상태 목록 또는 단일 상태
            context (torch.Tensor, optional): 문맥 임베딩
            layer_indices (list, optional): 레이어 인덱스 목록
            
        Returns:
            dict: 희소 활성화 결과
        """
        # 단일 상태를 목록으로 변환
        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]
            
        if superposition_states is not None and not isinstance(superposition_states, list):
            superposition_states = [superposition_states]
            
        # 문맥이 제공되지 않은 경우 히든 상태의 평균 사용
        if context is None:
            context = hidden_states[0].mean(dim=1)
            
        # 레이어 인덱스가 제공되지 않은 경우 기본값 사용
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))
            
        # 희소 활성화 적용
        sparse_hidden_states = []
        for i, (hidden_state, layer_idx) in enumerate(zip(hidden_states, layer_indices)):
            sparse_state = self.apply_sparse_activation(hidden_state, context, layer_idx)
            sparse_hidden_states.append(sparse_state)
            
        # 중첩 상태 최적화
        optimized_superposition_states = []
        if superposition_states is not None:
            for i, superposition_state in enumerate(superposition_states):
                # 계산 효율성 프레임워크 적용
                efficiency_result = self.efficiency_framework(
                    hidden_states[i], superposition_state, context
                )
                
                optimized_superposition_states.append(efficiency_result['optimized_state'])
                
        result = {
            'sparse_hidden_states': sparse_hidden_states,
            'layer_policy': self.compute_layer_sparsity(context)
        }
        
        if superposition_states is not None:
            result['optimized_superposition_states'] = optimized_superposition_states
            
        return result


class SharedParameterOptimizer(nn.Module):
    """
    공유 파라미터 최적화기
    
    모델 전체에서 파라미터 공유를 통해 메모리 효율성 향상
    """
    
    def __init__(self, model, hidden_dim, sharing_groups=None):
        """
        공유 파라미터 최적화기 초기화
        
        Args:
            model (nn.Module): 최적화 대상 모델
            hidden_dim (int): 기본 히든 차원
            sharing_groups (dict, optional): 파라미터 공유 그룹 정의
        """
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        
        # 기본 공유 그룹 정의
        if sharing_groups is None:
            sharing_groups = {
                'attention': ['q_linear', 'k_linear', 'v_linear'],
                'feedforward': ['linear1', 'linear2'],
                'dual_state': ['to_superposition', 'from_superposition'],
                'collapse': ['mode_projections']
            }
        self.sharing_groups = sharing_groups
        
        # 공유 파라미터 매핑 행렬
        self.parameter_mapping = nn.ParameterDict()
        
        # 파라미터 공유 초기화
        self._initialize_parameter_sharing()
        
    def _initialize_parameter_sharing(self):
        """
        파라미터 공유 초기화
        """
        # 모델 파라미터 분석
        model_parameters = dict(self.model.named_parameters())
        
        # 각 공유 그룹에 대한 매핑 행렬 생성
        for group_name, param_patterns in self.sharing_groups.items():
            group_params = []
            
            # 그룹에 속하는 파라미터 찾기
            for pattern in param_patterns:
                for name, param in model_parameters.items():
                    if pattern in name:
                        group_params.append((name, param.shape))
                        
            if group_params:
                # 그룹 내 파라미터 크기의 최대값 찾기
                max_shape = max([p[1] for p in group_params], key=lambda x: torch.prod(torch.tensor(x)))
                
                # 매핑 행렬 생성
                self.parameter_mapping[group_name] = nn.Parameter(torch.eye(self.hidden_dim))
    
    def apply_parameter_sharing(self):
        """
        모델에 파라미터 공유 적용
        
        Returns:
            dict: 파라미터 공유 결과
        """
        sharing_results = {}
        model_parameters = dict(self.model.named_parameters())
        
        # 각 공유 그룹에 대해 파라미터 공유 적용
        for group_name, param_patterns in self.sharing_groups.items():
            group_results = []
            
            # 각 패턴에 대해 매핑 적용
            for pattern in param_patterns:
                for name, param in model_parameters.items():
                    if pattern in name and 'weight' in name:
                        # 원본 형태 저장
                        original_shape = param.shape
                        
                        # 파라미터 재구성
                        if len(original_shape) == 2:
                            # 선형 레이어
                            mapping_matrix = self.parameter_mapping[group_name]
                            
                            # 매핑 적용
                            param.data = torch.matmul(mapping_matrix, param.data)
                            
                            group_results.append(name)
                            
                        elif len(original_shape) == 4:
                            # 컨볼루션 레이어 또는 다차원 파라미터
                            mapping_matrix = self.parameter_mapping[group_name]
                            
                            # 파라미터 평탄화
                            param_flat = param.data.view(original_shape[0], -1)
                            
                            # 매핑 적용
                            mapped_param = torch.matmul(mapping_matrix[:original_shape[0], :original_shape[0]], param_flat)
                            
                            # 원래 형태로 복원
                            param.data = mapped_param.view(original_shape)
                            
                            group_results.append(name)
                            
            sharing_results[group_name] = group_results
                            
        return sharing_results
    
    def optimize_parameter_sharing(self, dataloader, loss_fn, num_batches=5):
        """
        파라미터 공유 최적화
        
        Args:
            dataloader (DataLoader): 데이터 로더
            loss_fn (callable): 손실 함수
            num_batches (int): 최적화할 배치 수
            
        Returns:
            dict: 최적화 결과
        """
        # 원본 성능 측정
        original_losses = []
        device = next(self.model.parameters()).device
        
        # 일부 배치로 원본 성능 측정
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 원본 모델 추론
            with torch.no_grad():
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    outputs = outputs.get('output', outputs.get('deterministic_state'))
                    
                loss = loss_fn(outputs, targets)
                original_losses.append(loss.item())
        
        # 평균 원본 손실
        avg_original_loss = sum(original_losses) / len(original_losses)
        
        # 매핑 행렬 최적화
        for group_name in self.parameter_mapping:
            # 그래디언트 초기화
            grad_accumulation = torch.zeros_like(self.parameter_mapping[group_name])
            
            # 일부 배치로 그래디언트 추정
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 현재 매핑으로 파라미터 공유 적용
                self.apply_parameter_sharing()
                
                # 모델 추론
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    outputs = outputs.get('output', outputs.get('deterministic_state'))
                    
                # 손실 계산
                loss = loss_fn(outputs, targets)
                
                # 그래디언트 계산
                grad = torch.autograd.grad(loss, self.parameter_mapping[group_name], retain_graph=True)[0]
                
                # 그래디언트 누적
                grad_accumulation += grad
                
            # 평균 그래디언트로 매핑 행렬 업데이트
            avg_grad = grad_accumulation / num_batches
            self.parameter_mapping[group_name].data -= 0.01 * avg_grad
            
            # 정규화 (직교성 유지)
            U, S, V = torch.svd(self.parameter_mapping[group_name].data)
            self.parameter_mapping[group_name].data = torch.matmul(U, V.t())
        
        # 최적화된 파라미터 공유 적용
        sharing_results = self.apply_parameter_sharing()
        
        # 최적화 후 성능 측정
        optimized_losses = []
        
        # 일부 배치로 최적화 후 성능 측정
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 최적화된 모델 추론
            with torch.no_grad():
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    outputs = outputs.get('output', outputs.get('deterministic_state'))
                    
                loss = loss_fn(outputs, targets)
                optimized_losses.append(loss.item())
        
        # 평균 최적화 후 손실
        avg_optimized_loss = sum(optimized_losses) / len(optimized_losses)
        
        return {
            'sharing_results': sharing_results,
            'original_loss': avg_original_loss,
            'optimized_loss': avg_optimized_loss,
            'improvement': avg_original_loss - avg_optimized_loss
        }
    
    def forward(self, apply_sharing=True):
        """
        공유 파라미터 최적화기 순전파
        
        Args:
            apply_sharing (bool): 파라미터 공유 적용 여부
            
        Returns:
            dict: 최적화 결과
        """
        if apply_sharing:
            # 파라미터 공유 적용
            sharing_results = self.apply_parameter_sharing()
            
            return {
                'sharing_results': sharing_results,
                'parameter_mapping': {k: v.data for k, v in self.parameter_mapping.items()}
            }
        else:
            return {
                'parameter_mapping': {k: v.data for k, v in self.parameter_mapping.items()}
            }
