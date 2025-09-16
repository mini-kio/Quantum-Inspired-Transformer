import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    표준 피드포워드 네트워크
    """
    
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        """
        피드포워드 네트워크 초기화
        
        Args:
            d_model: 모델 차원
            dim_feedforward: 피드포워드 네트워크 내부 차원
            dropout: 드롭아웃 비율
            activation: 활성화 함수 ('relu', 'gelu')
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 활성화 함수 설정
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        """
        피드포워드 네트워크 순전파
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DualStateFeedForward(nn.Module):
    """
    이중 상태(확정/중첩)를 지원하는 피드포워드 네트워크
    """
    
    def __init__(self, d_model, dim_feedforward=2048, max_superposition_dim=4, dropout=0.1, activation='gelu'):
        """
        이중 상태 피드포워드 네트워크 초기화
        
        Args:
            d_model: 모델 차원
            dim_feedforward: 피드포워드 네트워크 내부 차원
            max_superposition_dim: 최대 중첩 상태 차원
            dropout: 드롭아웃 비율
            activation: 활성화 함수 ('relu', 'gelu')
        """
        super().__init__()
        self.d_model = d_model
        self.max_superposition_dim = max_superposition_dim
        
        # 확정 상태를 위한 표준 피드포워드 네트워크
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 중첩 상태를 위한 확장된 피드포워드 네트워크
        self.superposition_linear1 = nn.Linear(
            d_model * max_superposition_dim, 
            dim_feedforward * max_superposition_dim
        )
        self.superposition_linear2 = nn.Linear(
            dim_feedforward * max_superposition_dim, 
            d_model * max_superposition_dim
        )
        
        # 활성화 함수 설정
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 차원 간 상호작용을 모델링하기 위한 파라미터
        self.interaction_weights = nn.Parameter(
            torch.randn(max_superposition_dim, max_superposition_dim)
        )
        
    def forward_deterministic(self, x):
        """
        확정 상태를 위한 표준 피드포워드 네트워크 순전파
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
    def forward_superposition(self, x):
        """
        중첩 상태를 위한 확장된 피드포워드 네트워크 순전파
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 기본 피드포워드 연산
        output = self.superposition_linear2(
            self.dropout(
                self.activation(
                    self.superposition_linear1(x)
                )
            )
        )
        
        # 중첩 차원 간 상호작용 계산
        reshaped = output.view(batch_size, seq_len, self.max_superposition_dim, self.d_model)
        
        # 차원 간 상호작용 적용
        interaction = torch.einsum('bsid,ij->bsjd', reshaped, self.interaction_weights)
        
        # 원래 출력과 상호작용 결합
        interacted = reshaped + 0.1 * interaction
        
        # 원래 형태로 복원
        return interacted.view(batch_size, seq_len, -1)
    
    def forward(self, x, is_superposition=False):
        """
        이중 상태 피드포워드 네트워크 순전파
        
        Args:
            x: 입력 텐서
            is_superposition: 중첩 상태 여부
            
        Returns:
            출력 텐서
        """
        if is_superposition:
            return self.forward_superposition(x)
        else:
            return self.forward_deterministic(x)