import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStateRepresentation(nn.Module):
    """
    통합 이중 상태 표현 시스템
    
    모든 처리 단계에서 일관된 이중 상태(중첩 상태와 확정 상태)를 관리하는 프레임워크
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, controller_dim=128):
        """
        이중 상태 표현 시스템 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            controller_dim (int): 글로벌 컨트롤러 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        
        # 글로벌 중첩 정도 컨트롤러
        self.global_controller = nn.Sequential(
            nn.Linear(hidden_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, 1),
            nn.Sigmoid()
        )
        
        # 중첩 상태와 확정 상태 간 변환 레이어
        self.to_superposition = nn.Linear(hidden_dim, hidden_dim * max_superposition_dim)
        self.from_superposition = nn.Linear(hidden_dim * max_superposition_dim, hidden_dim)
        
        # 상태 간 간섭 계산을 위한 파라미터
        self.interference_weights = nn.Parameter(
            torch.randn(max_superposition_dim, max_superposition_dim)
        )
        
        # 상태 정규화
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.superposition_norm = nn.LayerNorm(hidden_dim * max_superposition_dim)
        
        # NEW: Soft/Hard collapse mixer
        self.alpha_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # NEW: Collapse threshold predictor
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # NEW: Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def to_superposition_state(self, deterministic_state, context_embedding=None):
        """
        확정 상태에서 중첩 상태로 변환
        
        Args:
            deterministic_state (torch.Tensor): 확정 상태 텐서 [batch_size, seq_len, hidden_dim]
            context_embedding (torch.Tensor, optional): 컨텍스트 임베딩
            
        Returns:
            torch.Tensor: 중첩 상태 텐서 [batch_size, seq_len, hidden_dim * max_superposition_dim]
        """
        # 컨텍스트에 따른 중첩 정도 결정
        if context_embedding is not None:
            batch_size, seq_len, _ = deterministic_state.shape
            # 컨텍스트 임베딩 활용하여 글로벌 중첩 정도 계산
            superposition_degree = self.global_controller(context_embedding)
            superposition_degree = superposition_degree.view(batch_size, 1, 1)
        else:
            # 기본 중첩 정도 (중간값)
            superposition_degree = torch.tensor(0.5, device=deterministic_state.device)

        # 확정 상태를 중첩 상태로 변환
        superposition = self.to_superposition(deterministic_state)
        
        # 중첩 정도에 따른 가중치 적용
        weighted_superposition = superposition * superposition_degree
        
        return self.superposition_norm(weighted_superposition)

    def from_superposition_state(self, superposition_state, collapse_threshold=None, alpha=None, context=None):
        """
        중첩 상태에서 확정 상태로 변환 (상태 붕괴)
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            collapse_threshold (float, optional): 상태 붕괴 임계값
            alpha (float, optional): Soft/Hard collapse 혼합 비율
            context (torch.Tensor, optional): 컨텍스트 임베딩
            
        Returns:
            torch.Tensor: 확정 상태 텐서
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 중첩 상태 재구성
        superposition_reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, self.hidden_dim
        )
        
        # 중첩 상태 간 간섭 효과 계산
        interference = torch.einsum('bsih,ij->bsjh', superposition_reshaped, self.interference_weights)
        
        # 간섭을 포함한 중첩 상태 합산
        interference = interference.view(batch_size, seq_len, self.max_superposition_dim * self.hidden_dim)
        combined_state = superposition_state + 0.1 * interference
        
        # 컨텍스트 기반 알파값 예측
        if context is not None and alpha is None:
            alpha = self.alpha_controller(context).view(batch_size, 1, 1)
        elif alpha is None:
            alpha = torch.tensor(0.5, device=superposition_state.device)
            
        # 컨텍스트 기반 붕괴 임계값 예측
        if context is not None and collapse_threshold is None:
            collapse_threshold = self.threshold_predictor(context).view(batch_size, 1, 1)
        elif collapse_threshold is None:
            collapse_threshold = torch.tensor(0.5, device=superposition_state.device)
        
        # Soft Collapse: 중첩 상태의 직접 변환
        soft_collapsed = self.from_superposition(combined_state)
        
        # Hard Collapse: 확률적 샘플링
        # 중첩 상태에서 각 차원의 노름 계산
        norms = torch.norm(superposition_reshaped, dim=3, keepdim=True)
        probabilities = norms**2 / (norms**2).sum(dim=2, keepdim=True).clamp(min=1e-10)
        
        # 확률적 샘플링 또는 argmax
        if self.training:
            # 학습 중에는 Gumbel-Softmax를 사용한 샘플링
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probabilities) + 1e-10) + 1e-10)
            samples = F.softmax((torch.log(probabilities + 1e-10) + gumbel_noise) / 0.1, dim=2)
        else:
            # 추론 중에는 argmax 사용
            indices = probabilities.argmax(dim=2, keepdim=True)
            samples = torch.zeros_like(probabilities).scatter_(2, indices, 1.0)
            
        # 샘플링된 차원의 값 추출
        hard_collapsed = (superposition_reshaped * samples).sum(dim=2)
        
        # Soft와 Hard 붕괴 혼합
        collapsed_state = alpha * soft_collapsed + (1 - alpha) * hard_collapsed
        
        return self.state_norm(collapsed_state)

    def compute_interference(self, superposition_state):
        """
        중첩 상태 간 간섭 효과 계산
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 간섭 효과가 적용된 중첩 상태
        """
        batch_size, seq_len, _ = superposition_state.shape
        
        # 중첩 상태 재구성
        superposition_reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, self.hidden_dim
        )
        
        # 중첩 진폭의 제곱으로 확률 계산
        amplitudes = F.softmax(
            superposition_reshaped.norm(dim=-1, keepdim=True), 
            dim=2
        )
        
        # 간섭 효과 계산 (양자 영감 접근법)
        phase_shifts = torch.sin(torch.einsum('bsih,ij->bsjh', superposition_reshaped, self.interference_weights))
        interference = amplitudes * phase_shifts
        
        # 원래 중첩 상태에 간섭 효과 적용
        interfered_state = superposition_reshaped + 0.1 * interference
        
        return interfered_state.view(batch_size, seq_len, self.max_superposition_dim * self.hidden_dim)
    
    def estimate_uncertainty(self, superposition_state):
        """
        중첩 상태의 불확실성 추정
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 불확실성 추정값 [batch_size, seq_len, 1]
        """
        # 중첩 상태 재구성
        batch_size, seq_len, _ = superposition_state.shape
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, self.hidden_dim
        )
        
        # 엔트로피 계산을 위한 확률 분포
        norms = torch.norm(reshaped, dim=3)
        probs = norms**2 / (norms**2).sum(dim=2, keepdim=True).clamp(min=1e-10)
        
        # 엔트로피 계산 (불확실성 지표)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.max_superposition_dim, dtype=torch.float, 
                                            device=superposition_state.device))
        normalized_entropy = entropy / max_entropy
        
        # 모델 기반 불확실성 추정과 결합
        model_uncertainty = self.uncertainty_estimator(superposition_state)
        
        # 두 불확실성 지표 결합
        combined_uncertainty = 0.7 * normalized_entropy + 0.3 * model_uncertainty
        
        return combined_uncertainty

    def forward(self, input_state, is_superposition=False, collapse=False, context=None, 
                alpha=None, collapse_threshold=None, p_target=0.5):
        """
        이중 상태 표현 시스템 순전파
        
        Args:
            input_state (torch.Tensor): 입력 상태 텐서
            is_superposition (bool): 입력이 중첩 상태인지 여부
            collapse (bool): 상태 붕괴 수행 여부
            context (torch.Tensor, optional): 컨텍스트 임베딩
            alpha (float, optional): Soft/Hard collapse 혼합 비율
            collapse_threshold (float, optional): 상태 붕괴 임계값
            p_target (float): 목표 전환 확률
            
        Returns:
            torch.Tensor: 처리된 상태 텐서
        """
        if not is_superposition:
            # 확정 상태를 중첩 상태로 변환
            superposition_state = self.to_superposition_state(input_state, context)
            
            # 중첩 상태 간 간섭 효과 적용
            superposition_state = self.compute_interference(superposition_state)
            
            if collapse:
                # 필요한 경우 상태 붕괴 수행
                return self.from_superposition_state(
                    superposition_state, collapse_threshold, alpha, context
                )
            else:
                return superposition_state
        else:
            # 이미 중첩 상태인 경우
            if collapse:
                # 상태 붕괴 수행
                return self.from_superposition_state(
                    input_state, collapse_threshold, alpha, context
                )
            else:
                # 중첩 상태 간 간섭 효과 적용
                return self.compute_interference(input_state)
                
    def compute_transition_probability(self, deterministic_state, superposition_state, context=None):
        """
        현재 상태에 기반한 전환 확률 계산
        
        Args:
            deterministic_state (torch.Tensor): 확정 상태 텐서
            superposition_state (torch.Tensor): 중첩 상태 텐서
            context (torch.Tensor, optional): 컨텍스트 임베딩
            
        Returns:
            torch.Tensor: 전환 확률 [batch_size, seq_len, 1]
        """
        # 불확실성 추정
        uncertainty = self.estimate_uncertainty(superposition_state)
        
        # 컨텍스트 임베딩이 제공된 경우
        if context is not None:
            batch_size, seq_len, _ = deterministic_state.shape
            context_expanded = context.unsqueeze(1).expand(batch_size, seq_len, -1)
            
            # 컨텍스트와 불확실성 결합
            combined = torch.cat([deterministic_state, context_expanded, uncertainty], dim=-1)
            
            # 전환 확률 계산
            transition_prob = self.global_controller(combined)
        else:
            # 간단한 불확실성 기반 전환 확률
            transition_prob = 0.5 + 0.3 * uncertainty
            
        return transition_prob


class DualStateController(nn.Module):
    """
    이중 상태의 동적 중첩 정도를 조절하는 글로벌 컨트롤러
    """
    
    def __init__(self, hidden_dim, controller_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 글로벌 컨트롤러 네트워크
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, 3)  # 중첩 정도, 붕괴 임계값, 간섭 강도
        )
        
        # NEW: CollapseGate 파라미터 예측기
        self.collapse_gate_controller = nn.Sequential(
            nn.Linear(hidden_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, 2)  # p_target, alpha
        )
        
        # NEW: 리소스 효율성 컨트롤러
        self.efficiency_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context_embedding):
        """
        컨텍스트에 기반한 이중 상태 파라미터 계산
        
        Args:
            context_embedding (torch.Tensor): 컨텍스트 임베딩
            
        Returns:
            dict: 이중 상태 파라미터
        """
        outputs = self.controller(context_embedding)
        
        # 각 파라미터에 적절한 활성화 함수 적용
        superposition_degree = torch.sigmoid(outputs[:, 0]).unsqueeze(1)
        collapse_threshold = torch.sigmoid(outputs[:, 1]).unsqueeze(1)
        interference_strength = torch.sigmoid(outputs[:, 2]).unsqueeze(1)
        
        # CollapseGate 파라미터 예측
        gate_params = self.collapse_gate_controller(context_embedding)
        p_target = torch.sigmoid(gate_params[:, 0]).unsqueeze(1)  # 범위: 0~1
        alpha = torch.sigmoid(gate_params[:, 1]).unsqueeze(1)     # 범위: 0~1
        
        # 리소스 효율성 점수 계산
        efficiency_score = self.efficiency_controller(context_embedding)
        
        return {
            'superposition_degree': superposition_degree,
            'collapse_threshold': collapse_threshold,
            'interference_strength': interference_strength,
            'p_target': p_target,
            'alpha': alpha,
            'efficiency_score': efficiency_score
        }