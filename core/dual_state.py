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

    def from_superposition_state(self, superposition_state, collapse_threshold=None):
        """
        중첩 상태에서 확정 상태로 변환 (상태 붕괴)
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            collapse_threshold (float, optional): 상태 붕괴 임계값
            
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
        
        # 중첩 상태를 확정 상태로 변환
        deterministic_state = self.from_superposition(combined_state)
        
        return self.state_norm(deterministic_state)

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

    def forward(self, input_state, is_superposition=False, collapse=False, context=None):
        """
        이중 상태 표현 시스템 순전파
        
        Args:
            input_state (torch.Tensor): 입력 상태 텐서
            is_superposition (bool): 입력이 중첩 상태인지 여부
            collapse (bool): 상태 붕괴 수행 여부
            context (torch.Tensor, optional): 컨텍스트 임베딩
            
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
                return self.from_superposition_state(superposition_state)
            else:
                return superposition_state
        else:
            # 이미 중첩 상태인 경우
            if collapse:
                # 상태 붕괴 수행
                return self.from_superposition_state(input_state)
            else:
                # 중첩 상태 간 간섭 효과 적용
                return self.compute_interference(input_state)


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
        
    def forward(self, context_embedding):
        """
        컨텍스트에 기반한 이중 상태 파라미터 계산
        
        Args:
            context_embedding (torch.Tensor): 컨텍스트 임베딩
            
        Returns:
            tuple: (중첩 정도, 붕괴 임계값, 간섭 강도)
        """
        outputs = self.controller(context_embedding)
        
        # 각 파라미터에 적절한 활성화 함수 적용
        superposition_degree = torch.sigmoid(outputs[:, 0]).unsqueeze(1)
        collapse_threshold = torch.sigmoid(outputs[:, 1]).unsqueeze(1)
        interference_strength = torch.sigmoid(outputs[:, 2]).unsqueeze(1)
        
        return superposition_degree, collapse_threshold, interference_strength
