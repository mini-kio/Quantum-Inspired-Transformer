import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CollapseGate(nn.Module):
    """
    토큰과 레이어별 중첩-확정 전환 비율을 학습하는 게이트 모듈
    
    입력 토큰 임베딩과 컨텍스트로부터 전환 확률 p를 결정하고,
    soft/hard collapse의 비율 α를 자동으로 조절합니다.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        max_superposition_dim: int = 4,
        layer_id: int = 0,
        num_layers: int = 12,
        gate_type: str = "mlp",  # "mlp" 또는 "transformer"
        alpha_learnable: bool = True
    ):
        """
        CollapseGate 초기화
        
        Args:
            hidden_dim: 기본 히든 차원
            max_superposition_dim: 최대 중첩 상태 차원
            layer_id: 현재 레이어 ID (0부터 시작)
            num_layers: 총 레이어 수
            gate_type: 게이트 아키텍처 유형 ("mlp" 또는 "transformer")
            alpha_learnable: soft/hard collapse 비율 α를 학습 가능하게 할지 여부
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.gate_type = gate_type
        
        # 레이어 ID 임베딩
        self.layer_embedding = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )
        
        # 레이어 ID 위치 인코딩
        layer_pos = torch.zeros(1, 1, hidden_dim)
        for i in range(0, hidden_dim, 2):
            denominator = 10000 ** (2 * i / hidden_dim)
            layer_pos[0, 0, i] = math.sin(layer_id / denominator)
            if i + 1 < hidden_dim:
                layer_pos[0, 0, i + 1] = math.cos(layer_id / denominator)
        self.register_buffer('layer_pos', layer_pos)
        
        # 게이트 아키텍처 설정
        if gate_type == "mlp":
            # MLP 게이트
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # 초소형 Transformer 게이트
            self.input_projection = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # 불확실성 추정기
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Soft/Hard Collapse 비율 α
        if alpha_learnable:
            # 학습 가능한 α 파라미터 (토큰별, 레이어별로 다름)
            self.alpha = nn.Parameter(torch.ones(1, 1, 1) * 0.5)  # 초기값 0.5
        else:
            # 고정된 α 값
            self.register_buffer('alpha', torch.ones(1, 1, 1) * 0.5)
            
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        # 레이어 임베딩 초기화 - 레이어 ID에 비례하는 값으로
        with torch.no_grad():
            self.layer_embedding.data.fill_(self.layer_id / self.num_layers)
            
        # 선형 레이어 초기화
        if self.gate_type == "mlp":
            for module in self.gate_network:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
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
                        
            nn.init.normal_(self.output_projection[0].weight, std=0.02)
            nn.init.zeros_(self.output_projection[0].bias)
            
    def estimate_uncertainty(self, superposition_state):
        """
        중첩 상태의 불확실성 추정
        
        Args:
            superposition_state: 중첩 상태 텐서
            
        Returns:
            불확실성 추정값
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 불확실성 추정
        return self.uncertainty_estimator(superposition_state)
    
    def compute_transition_probability(self, token_embeddings, context_embedding, uncertainty=None):
        """
        중첩-확정 전환 확률 p 계산
        
        Args:
            token_embeddings: 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            context_embedding: 컨텍스트 임베딩 [batch_size, hidden_dim]
            uncertainty: 불확실성 추정값 (선택사항)
            
        Returns:
            전환 확률 p [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 컨텍스트 임베딩 확장
        context_expanded = context_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # 레이어 위치 정보 확장
        layer_info = self.layer_embedding + self.layer_pos
        layer_info_expanded = layer_info.expand(batch_size, seq_len, -1)
        
        # 모든 정보 결합
        combined_input = torch.cat([token_embeddings, context_expanded, layer_info_expanded], dim=-1)
        
        if self.gate_type == "mlp":
            # MLP 게이트로 확률 계산
            transition_prob = self.gate_network(combined_input)
        else:
            # Transformer 게이트로 확률 계산
            x = self.input_projection(combined_input)
            attn_output, _ = self.self_attn(x, x, x)
            x = x + attn_output
            x = self.norm1(x)
            
            ffn_output = self.ffn(x)
            x = x + ffn_output
            x = self.norm2(x)
            
            transition_prob = self.output_projection(x)
        
        # 불확실성이 제공된 경우 확률 조정
        if uncertainty is not None:
            # 불확실성이 높을수록 확정 전환 확률 감소
            transition_prob = transition_prob * (1 - uncertainty * 0.5)
            
        return transition_prob
    
    def forward(self, 
               deterministic_state, 
               superposition_state, 
               context=None, 
               p_target=0.5,
               force_collapse=False):
        """
        CollapseGate 순전파
        
        Args:
            deterministic_state: 확정 상태 [batch_size, seq_len, hidden_dim]
            superposition_state: 중첩 상태 [batch_size, seq_len, hidden_dim * max_superposition_dim]
            context: 컨텍스트 임베딩 (없으면 토큰 평균)
            p_target: 목표 전환 확률
            force_collapse: 강제 붕괴 여부
            
        Returns:
            dict: 게이트 결과
        """
        batch_size, seq_len, _ = deterministic_state.shape
        
        # 컨텍스트가 제공되지 않은 경우 토큰 평균 사용
        if context is None:
            context = deterministic_state.mean(dim=1)
            
        # 불확실성 추정
        uncertainty = self.estimate_uncertainty(superposition_state)
        
        # 전환 확률 계산
        transition_prob = self.compute_transition_probability(
            deterministic_state, context, uncertainty
        )
        
        # 강제 붕괴 시 확률 1로 설정
        if force_collapse:
            transition_prob = torch.ones_like(transition_prob)
            
        # 전환 확률에 따른 마스크 생성 (확률적 이진 마스크)
        if self.training:
            # 학습 중: Gumbel-Softmax 릴랙세이션 사용
            temperature = 1.0
            logits = torch.log(transition_prob / (1 - transition_prob + 1e-10))
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            binary_mask = torch.sigmoid((logits + gumbel_noise) / temperature)
        else:
            # 추론 중: 확률적 이진 샘플링
            binary_mask = (torch.rand_like(transition_prob) < transition_prob).float()
            
        # 현재 alpha 값 (토큰 및 레이어별 조정)
        current_alpha = torch.sigmoid(self.alpha) * torch.ones_like(binary_mask)
        
        # 소프트 및 하드 붕괴 혼합
        collapsed_state = torch.zeros_like(deterministic_state)
        
        # 중첩 상태 재구성
        reshaped_superposition = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        
        # 소프트 붕괴: 중첩 상태의 가중 평균
        soft_collapsed = reshaped_superposition.mean(dim=2)
        
        # 하드 붕괴: 확정 상태로 직접 치환
        hard_collapsed = deterministic_state
        
        # alpha에 따른 소프트/하드 붕괴 혼합
        mixed_collapsed = current_alpha * soft_collapsed + (1 - current_alpha) * hard_collapsed
        
        # 최종 상태: binary_mask에 따라 원래 상태와 붕괴된 상태 혼합
        final_deterministic = (1 - binary_mask) * deterministic_state + binary_mask * mixed_collapsed
        
        # 중첩 상태도 부분적으로 조정
        # binary_mask가 1인 위치(붕괴 위치)에서는 중첩 차원 간 균일화
        binary_mask_expanded = binary_mask.unsqueeze(2).expand(-1, -1, self.max_superposition_dim, -1)
        
        # 균일화된 상태: 모든 중첩 차원이 동일한 값 (soft_collapsed)
        uniform_superposition = soft_collapsed.unsqueeze(2).expand(-1, -1, self.max_superposition_dim, -1)
        
        # 최종 중첩 상태: 마스크에 따라 원래 중첩 상태와 균일화된 상태 혼합
        final_superposition = (1 - binary_mask_expanded) * reshaped_superposition + binary_mask_expanded * uniform_superposition
        final_superposition = final_superposition.reshape(batch_size, seq_len, -1)
        
        # 리소스 효율성 계산: 평균 전환 확률과 목표 확률 간의 차이
        resource_efficiency = 1.0 - torch.abs(transition_prob.mean() - p_target)
        
        return {
            'deterministic_state': final_deterministic,
            'superposition_state': final_superposition,
            'transition_prob': transition_prob,
            'binary_mask': binary_mask,
            'alpha': current_alpha,
            'uncertainty': uncertainty,
            'resource_efficiency': resource_efficiency
        }


class StateCollapseFramework(nn.Module):
    """
    범용 상태 붕괴 프레임워크
    
    맥락, 태스크, 불확실성 수준에 따라 자동 조정되는 붕괴 메커니즘 제공
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, num_collapse_modes=3):
        """
        상태 붕괴 프레임워크 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            num_collapse_modes (int): 붕괴 모드 수 (점진적, 부분적, 완전)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.num_collapse_modes = num_collapse_modes
        
        # 맥락에 따른 붕괴 메커니즘 조정
        self.context_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_collapse_modes)
        )
        
        # 불확실성 추정기
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 최적 결정 시점 예측기 - 올바른 입력 차원 계산
        # uncertainty: 1차원 (uncertainty_estimator 출력), context: hidden_dim
        predictor_input_dim = 1 + hidden_dim
        self.decision_time_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 확률 분포 변화 감지기
        self.distribution_change_detector = nn.GRUCell(
            hidden_dim * max_superposition_dim, hidden_dim
        )
        
        # 각 붕괴 모드별 투영 레이어
        self.mode_projections = nn.ModuleList([
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim)
            for _ in range(num_collapse_modes)
        ])
        
        # 붕괴 후 상태 정규화
        self.collapsed_norm = nn.LayerNorm(hidden_dim)
        
        # CollapseGate 추가 - 모든 레이어와 공유
        self.collapse_gate = CollapseGate(
            hidden_dim=hidden_dim,
            max_superposition_dim=max_superposition_dim,
            layer_id=0,  # 기본값, 나중에 각 레이어에서 덮어씀
            num_layers=1  # 기본값, 나중에 설정
        )
        
    def estimate_uncertainty(self, superposition_state):
        """
        중첩 상태의 불확실성 추정
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 불확실성 추정값
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 중첩 상태의 분산 계산
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        variance = reshaped.var(dim=2).mean(dim=-1, keepdim=True)
        
        # 불확실성 추정
        uncertainty = self.uncertainty_estimator(superposition_state)
        
        # 분산과 추정된 불확실성 결합
        combined_uncertainty = 0.5 * (variance + uncertainty)
        
        return combined_uncertainty
    
    def detect_distribution_change(self, current_state, previous_state, hidden_state):
        """
        확률 분포의 급격한 변화 감지
        
        Args:
            current_state (torch.Tensor): 현재 중첩 상태
            previous_state (torch.Tensor): 이전 중첩 상태
            hidden_state (torch.Tensor): GRU 히든 상태
            
        Returns:
            tuple: (변화 감지 점수, 업데이트된 히든 상태)
        """
        batch_size, seq_len, _ = current_state.shape
        
        # 상태 변화 계산
        state_diff = current_state - previous_state
        state_diff_flat = state_diff.view(batch_size * seq_len, -1)
        
        # 히든 상태 평탄화
        hidden_flat = hidden_state.view(batch_size * seq_len, -1)
        
        # GRU를 사용하여 변화 추적
        new_hidden = self.distribution_change_detector(state_diff_flat, hidden_flat)
        
        # 변화 감지 점수 계산 (GRU 출력의 노름)
        change_score = torch.norm(new_hidden, dim=1, keepdim=True)
        change_score = torch.sigmoid(change_score)
        
        # 형태 복원
        change_score = change_score.view(batch_size, seq_len, 1)
        new_hidden = new_hidden.view(batch_size, seq_len, -1)
        
        return change_score, new_hidden
    
    def determine_optimal_decision_time(self, uncertainty, context):
        """
        전역 및 지역 불확실성을 고려한 최적 결정 시점 결정

        Args:
            uncertainty (torch.Tensor): 불확실성 추정값
            context (torch.Tensor): 맥락 정보

        Returns:
            torch.Tensor: 결정 시점 점수 (0에 가까울수록 즉시 붕괴)
        """
        batch_size, seq_len, _ = uncertainty.shape

        # 맥락 확장
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, seq_len, -1)

        # 불확실성과 맥락 결합
        combined = torch.cat([uncertainty, context], dim=-1)

        # 결정 시점 점수 계산
        decision_time = self.decision_time_predictor(combined)

        return decision_time
    
    def select_collapse_mode(self, context, uncertainty):
        """
        맥락과 불확실성에 따른 붕괴 모드 선택
        
        Args:
            context (torch.Tensor): 맥락 정보
            uncertainty (torch.Tensor): 불확실성 추정값
            
        Returns:
            torch.Tensor: 각 붕괴 모드의 가중치
        """
        batch_size, seq_len, _ = uncertainty.shape
        
        # 맥락 확장
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, seq_len, -1)
            
        # 붕괴 모드 점수 계산
        mode_scores = self.context_adapter(context)
        
        # 불확실성에 따른 점수 조정
        adjusted_scores = mode_scores * (1 - uncertainty) + uncertainty * torch.ones_like(mode_scores) / self.num_collapse_modes
        
        # 모드 가중치 계산
        mode_weights = F.softmax(adjusted_scores, dim=-1)
        
        return mode_weights
    
    def apply_collapse(self, superposition_state, mode_weights):
        """
        선택된 붕괴 모드에 따라 상태 붕괴 적용
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            mode_weights (torch.Tensor): 각 붕괴 모드의 가중치
            
        Returns:
            torch.Tensor: 붕괴된 확정 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 각 붕괴 모드별 투영
        mode_projections = []
        for i in range(self.num_collapse_modes):
            projection = self.mode_projections[i](superposition_state)
            mode_projections.append(projection.unsqueeze(-1))
            
        # 모든 모드 스택
        stacked_projections = torch.cat(mode_projections, dim=-1)
        
        # 가중치 적용 및 합산
        mode_weights = mode_weights.unsqueeze(-2)
        collapsed_state = torch.matmul(stacked_projections, mode_weights.transpose(-1, -2)).squeeze(-1)
        
        return self.collapsed_norm(collapsed_state)
    
    def forward(self, 
                superposition_state, 
                context, 
                previous_state=None, 
                hidden_state=None, 
                force_collapse=False, 
                layer_id=0, 
                num_layers=12,
                p_target=0.5):
        """
        상태 붕괴 프레임워크 순전파
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            context (torch.Tensor): 맥락 정보
            previous_state (torch.Tensor, optional): 이전 중첩 상태
            hidden_state (torch.Tensor, optional): GRU 히든 상태
            force_collapse (bool): 강제 붕괴 수행 여부
            layer_id (int): 현재 레이어 ID
            num_layers (int): 총 레이어 수
            p_target (float): 목표 전환 확률
            
        Returns:
            dict: 붕괴 결과 (확정 상태, 불확실성, 붕괴 여부 등)
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 초기 히든 상태 생성
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, seq_len, self.hidden_dim, device=superposition_state.device)
            
        # 초기 이전 상태 생성
        if previous_state is None:
            previous_state = superposition_state.clone()
            
        # 불확실성 추정
        uncertainty = self.estimate_uncertainty(superposition_state)
        
        # 분포 변화 감지
        change_score, new_hidden = self.detect_distribution_change(
            superposition_state, previous_state, hidden_state
        )
        
        # 결정 시점 결정
        decision_time = self.determine_optimal_decision_time(uncertainty, context)
        
        # 붕괴 조건: 강제 붕괴 또는 (변화 감지 + 낮은 결정 시점)
        collapse_condition = force_collapse | ((change_score > 0.7) & (decision_time < 0.3))
        
        # 붕괴 모드 선택
        mode_weights = self.select_collapse_mode(context, uncertainty)
        
        # 확정 상태 생성
        deterministic_state = torch.zeros(
            batch_size, seq_len, self.hidden_dim, 
            device=superposition_state.device
        )
        
        # 레이어별 CollapseGate 설정
        self.collapse_gate.layer_id = layer_id
        self.collapse_gate.num_layers = num_layers
        
        # CollapseGate 적용
        gate_result = self.collapse_gate(
            deterministic_state=deterministic_state,
            superposition_state=superposition_state,
            context=context,
            p_target=p_target,
            force_collapse=force_collapse
        )
        
        # 조건부 붕괴 적용
        collapsed_state = torch.zeros_like(superposition_state[:, :, :self.hidden_dim])
        
        for b in range(batch_size):
            for s in range(seq_len):
                if collapse_condition[b, s, 0]:
                    # 해당 위치의 상태 붕괴
                    collapsed_state[b, s] = self.apply_collapse(
                        superposition_state[b, s].unsqueeze(0).unsqueeze(0),
                        mode_weights[b, s].unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)
                else:
                    # CollapseGate 결과 사용
                    collapsed_state[b, s] = gate_result['deterministic_state'][b, s]
                    
        return {
            'collapsed_state': collapsed_state,
            'uncertainty': uncertainty,
            'change_score': change_score,
            'decision_time': decision_time,
            'collapse_condition': collapse_condition,
            'mode_weights': mode_weights,
            'hidden_state': new_hidden,
            'gate_result': gate_result,
            'transition_prob': gate_result['transition_prob'],
            'resource_efficiency': gate_result['resource_efficiency']
        }


class DynamicCollapseController(nn.Module):
    """
    점진적, 부분적, 완전 붕괴 모드 간 동적 전환 시스템
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4):
        """
        동적 붕괴 컨트롤러 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        
        # 태스크 복잡성 평가
        self.task_complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 모드 전환 컨트롤러 - 올바른 입력 차원 계산
        # context: hidden_dim, uncertainty: 1, task_complexity: 1
        mode_input_dim = hidden_dim + 2
        self.mode_transition_controller = nn.Sequential(
            nn.Linear(mode_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # 점진적, 부분적, 완전 모드
        )
        
        # 점진적 붕괴 비율 조절기
        self.gradual_rate_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 부분적 붕괴 마스크 생성기
        self.partial_mask_generator = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim * max_superposition_dim),
            nn.GELU(),
            nn.Linear(hidden_dim * max_superposition_dim, max_superposition_dim),
            nn.Sigmoid()
        )
        
        # CollapseGate 추가
        self.collapse_gate = CollapseGate(
            hidden_dim=hidden_dim,
            max_superposition_dim=max_superposition_dim,
            gate_type="transformer"  # 더 강력한 transformer 게이트 사용
        )
        
    def estimate_task_complexity(self, context):
        """
        맥락 기반 태스크 복잡성 추정
        
        Args:
            context (torch.Tensor): 맥락 정보
            
        Returns:
            torch.Tensor: 태스크 복잡성 점수 (0~1)
        """
        return self.task_complexity_estimator(context)
    
    def determine_collapse_mode(self, context, uncertainty, task_complexity):
        """
        맥락, 불확실성, 태스크 복잡성에 따른 붕괴 모드 결정
        
        Args:
            context (torch.Tensor): 맥락 정보
            uncertainty (torch.Tensor): 불확실성 점수
            task_complexity (torch.Tensor): 태스크 복잡성 점수
            
        Returns:
            torch.Tensor: 각 붕괴 모드의 가중치
        """
        batch_size = context.shape[0]
        
        # 입력 결합
        combined = torch.cat([
            context,
            uncertainty.reshape(batch_size, 1),
            task_complexity.reshape(batch_size, 1)
        ], dim=-1)
        
        # 모드 점수 계산
        mode_scores = self.mode_transition_controller(combined)
        
        # 모드 가중치 변환
        mode_weights = F.softmax(mode_scores, dim=-1)
        
        return mode_weights
    
    def gradual_collapse(self, superposition_state, deterministic_state, context):
        """
        점진적 붕괴 모드 적용
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            deterministic_state (torch.Tensor): 확정 상태 텐서
            context (torch.Tensor): 맥락 정보
            
        Returns:
            torch.Tensor: 점진적으로 붕괴된 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 맥락 확장
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, seq_len, -1)
            
        # 점진적 붕괴 비율 계산
        collapse_rate = self.gradual_rate_controller(context)
        
        # 중첩 상태에서 확정 상태 추출 (평균 취하기)
        batch_size, seq_len, _ = superposition_state.shape
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        superposition_mean = reshaped.mean(dim=2)  # [B, S, hidden_dim]

        # 점진적 붕괴 적용 (중첩과 확정의 가중 평균)
        gradually_collapsed = (1 - collapse_rate) * superposition_mean + collapse_rate * deterministic_state

        return gradually_collapsed
    
    def partial_collapse(self, superposition_state):
        """
        부분적 붕괴 모드 적용 (일부 중첩 차원만 붕괴)
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 부분적으로 붕괴된 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 붕괴 마스크 생성
        collapse_mask = self.partial_mask_generator(superposition_state)
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        
        # 붕괴 마스크 확장
        expanded_mask = collapse_mask.unsqueeze(-1).expand_as(reshaped)
        
        # 마스크된 평균 계산 (붕괴할 차원의 가중 평균)
        mask_sum = expanded_mask.sum(dim=2, keepdim=True).clamp(min=1e-10)
        masked_mean = (reshaped * expanded_mask).sum(dim=2, keepdim=True) / mask_sum
        
        # 부분 붕괴 적용
        partially_collapsed = reshaped * (1 - expanded_mask) + masked_mean * expanded_mask

        # 확정 상태로 변환 (평균 취하기)
        collapsed_deterministic = partially_collapsed.mean(dim=2)  # [B, S, hidden_dim]

        return collapsed_deterministic
    
    def full_collapse(self, superposition_state):
        """
        완전 붕괴 모드 적용
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 완전히 붕괴된 확정 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, -1
        )
        
        # 중첩 확률 계산 (진폭 제곱)
        amplitudes = torch.norm(reshaped, dim=-1)
        probabilities = F.softmax(amplitudes, dim=-1)
        
        # 확률에 따른 중첩 상태 가중 평균
        probabilities = probabilities.unsqueeze(-1)
        weighted_sum = (reshaped * probabilities).sum(dim=2)
        
        return weighted_sum
    
    def forward(
        self, 
        superposition_state, 
        deterministic_state, 
        context, 
        uncertainty=None,
        p_target=0.5):
        """
        동적 붕괴 컨트롤러 순전파
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            deterministic_state (torch.Tensor): 확정 상태 텐서
            context (torch.Tensor): 맥락 정보
            uncertainty (torch.Tensor, optional): 불확실성 점수
            p_target (float): 목표 전환 확률
            
        Returns:
            dict: 붕괴 결과 (붕괴된 상태, 모드 가중치 등)
        """
        batch_size = context.shape[0]
        
        # 불확실성이 제공되지 않은 경우 기본값 사용
        if uncertainty is None:
            uncertainty = torch.ones(batch_size, 1, device=context.device) * 0.5
            
        # 태스크 복잡성 추정
        task_complexity = self.estimate_task_complexity(context)
        
        # 붕괴 모드 결정
        mode_weights = self.determine_collapse_mode(context, uncertainty, task_complexity)
        
        # 각 붕괴 모드 적용
        gradual_state = self.gradual_collapse(superposition_state, deterministic_state, context)
        partial_state = self.partial_collapse(superposition_state)
        full_state = self.full_collapse(superposition_state)
        
        # 모드별 가중치 준비
        gradual_weight = mode_weights[:, 0].view(batch_size, 1, 1)
        partial_weight = mode_weights[:, 1].view(batch_size, 1, 1)
        full_weight = mode_weights[:, 2].view(batch_size, 1, 1)
        
        # 가중치에 따른 최종 붕괴 상태 계산
        collapsed_state = (
            gradual_weight * gradual_state +
            partial_weight * partial_state +
            full_weight * full_state
        )
        
        # CollapseGate 적용
        gate_result = self.collapse_gate(
            deterministic_state=deterministic_state,
            superposition_state=superposition_state,
            context=context,
            p_target=p_target
        )
        
        # CollapseGate의 deterministic_state와 weighted_collapsed 합치기
        final_collapsed_state = 0.5 * gate_result['deterministic_state'] + 0.5 * collapsed_state
        
        return {
            'collapsed_state': final_collapsed_state,
            'mode_weights': mode_weights,
            'task_complexity': task_complexity,
            'gradual_state': gradual_state,
            'partial_state': partial_state,
            'full_state': full_state,
            'gate_result': gate_result,
            'transition_prob': gate_result['transition_prob'],
            'resource_efficiency': gate_result['resource_efficiency']
        }