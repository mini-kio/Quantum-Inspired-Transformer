import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    표준 사인-코사인 위치 인코딩
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        위치 인코딩 초기화
        
        Args:
            d_model: 모델 차원
            max_len: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 사인-코사인 인코딩
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, d_model] 형태로 변환
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록 (파라미터가 아님)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        입력에 위치 인코딩 적용
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        Returns:
            위치 인코딩이 적용된 텐서
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class QuantumPositionalEncoding(nn.Module):
    """
    중첩 상태를 위한 확장된 양자 영감 위치 인코딩
    """
    
    def __init__(self, d_model, max_superposition_dim=4, max_len=5000, dropout=0.1):
        """
        양자 위치 인코딩 초기화
        
        Args:
            d_model: 모델 차원
            max_superposition_dim: 최대 중첩 상태 차원
            max_len: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_superposition_dim = max_superposition_dim
        
        # 표준 위치 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 사인-코사인 인코딩
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, d_model] 형태로 변환
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록
        self.register_buffer('pe', pe)
        
        # 중첩 상태를 위한 위상 시프트 파라미터 (각 중첩 차원마다 다른 위상)
        self.phase_shifts = nn.Parameter(torch.randn(max_superposition_dim) * 0.1)
        
    def forward(self, x, is_superposition=False):
        """
        입력에 위치 인코딩 적용
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model] 또는 
               [batch_size, seq_len, d_model * max_superposition_dim]
            is_superposition: 중첩 상태 여부
            
        Returns:
            위치 인코딩이 적용된 텐서
        """
        if not is_superposition:
            # 확정 상태: 표준 위치 인코딩 적용
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
        else:
            # 중첩 상태: 각 중첩 차원에 다른 위상의 위치 인코딩 적용
            batch_size, seq_len, _ = x.shape
            
            # 중첩 상태 재구성
            reshaped = x.view(batch_size, seq_len, self.max_superposition_dim, self.d_model)
            
            # 각 중첩 차원에 위치 인코딩 적용
            for dim in range(self.max_superposition_dim):
                # 위상 시프트된 위치 인코딩 계산
                phase_shift = self.phase_shifts[dim]
                phase_pe = self.pe[:, :seq_len, :] * torch.cos(phase_shift) + self.pe[:, :seq_len, :].roll(shifts=1, dims=2) * torch.sin(phase_shift)
                
                # 해당 차원에 적용
                reshaped[:, :, dim, :] = reshaped[:, :, dim, :] + phase_pe
            
            # 원래 형태로 복원
            encoded = reshaped.view(batch_size, seq_len, -1)
            
            return self.dropout(encoded)