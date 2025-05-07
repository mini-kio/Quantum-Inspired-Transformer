import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuantumInspiredAttention(nn.Module):
    """
    양자 영감 어텐션 메커니즘
    
    표준 멀티헤드 어텐션을 확장하여 중첩 상태와 확정 상태 모두에 대해 작동하도록 개선
    """
    
    def __init__(self, d_model, nhead, max_superposition_dim=4, dropout=0.1):
        """
        양자 영감 어텐션 초기화
        
        Args:
            d_model (int): 모델 차원
            nhead (int): 어텐션 헤드 수
            max_superposition_dim (int): 최대 중첩 상태 차원
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_superposition_dim = max_superposition_dim
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        
        # 확정 상태를 위한 표준 선형 투영
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # 중첩 상태를 위한 확장된 선형 투영
        self.superposition_q_linear = nn.Linear(d_model * max_superposition_dim, d_model * max_superposition_dim)
        self.superposition_k_linear = nn.Linear(d_model * max_superposition_dim, d_model * max_superposition_dim)
        self.superposition_v_linear = nn.Linear(d_model * max_superposition_dim, d_model * max_superposition_dim)
        
        # 출력 투영
        self.out_linear = nn.Linear(d_model, d_model)
        self.superposition_out_linear = nn.Linear(d_model * max_superposition_dim, d_model * max_superposition_dim)
        
        # 양자 간섭 효과 모델링을 위한 파라미터
        self.interference_factors = nn.Parameter(torch.randn(nhead, max_superposition_dim, max_superposition_dim))
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 진폭 조절 파라미터
        self.amplitude_scaler = nn.Parameter(torch.ones(1))

        # Sparsity control - for resource efficiency
        self.sparsity_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Superposition sparsity control
        self.superposition_sparsity_gate = nn.Sequential(
            nn.Linear(d_model * max_superposition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_superposition_dim),
            nn.Sigmoid()
        )
        
    def _transform_for_superposition(self, x, batch_size, seq_len):
        """
        중첩 상태 텐서를 헤드별로 변환
        
        Args:
            x (torch.Tensor): 입력 중첩 상태 텐서
            batch_size (int): 배치 크기
            seq_len (int): 시퀀스 길이
            
        Returns:
            torch.Tensor: 헤드별로 변환된 중첩 상태 텐서
        """
        # [batch_size, seq_len, nhead, d_k, max_superposition_dim]로 재구성
        x = x.view(batch_size, seq_len, self.nhead, self.d_k, self.max_superposition_dim)
        
        # [batch_size, nhead, seq_len, d_k, max_superposition_dim]로 치환
        x = x.permute(0, 2, 1, 3, 4)
        
        return x
        
    def _apply_quantum_interference(self, q, k, v):
        """
        양자 간섭 효과 적용
        
        Args:
            q (torch.Tensor): 쿼리 텐서 [batch_size, nhead, seq_len, d_k, max_superposition_dim]
            k (torch.Tensor): 키 텐서 [batch_size, nhead, seq_len, d_k, max_superposition_dim]
            v (torch.Tensor): 값 텐서 [batch_size, nhead, seq_len, d_k, max_superposition_dim]
            
        Returns:
            tuple: 간섭 효과가 적용된 (쿼리, 키, 값) 텐서
        """
        batch_size, nhead, seq_len, d_k, max_dim = q.shape
        
        # 중첩 상태 간 간섭 계산
        for h in range(nhead):
            # 쿼리에 간섭 적용
            interference = torch.einsum('bsdi,ij->bsdj', q[:, h], self.interference_factors[h])
            q[:, h] = q[:, h] + 0.1 * self.amplitude_scaler * interference
            
            # 키에 간섭 적용
            interference = torch.einsum('bsdi,ij->bsdj', k[:, h], self.interference_factors[h])
            k[:, h] = k[:, h] + 0.1 * self.amplitude_scaler * interference
            
            # 값에 간섭 적용
            interference = torch.einsum('bsdi,ij->bsdj', v[:, h], self.interference_factors[h])
            v[:, h] = v[:, h] + 0.1 * self.amplitude_scaler * interference
            
        return q, k, v
    
    def _calculate_amplitude_weights(self, attention_scores):
        """
        어텐션 점수로부터 진폭 가중치 계산
        
        Args:
            attention_scores (torch.Tensor): 어텐션 점수
            
        Returns:
            torch.Tensor: 진폭 가중치
        """
        # BUG FIX 1: Removed the duplicate scaling by sqrt(d_k)
        # Only apply softmax, don't scale again
        weights = F.softmax(attention_scores, dim=-1)
        weights = self.dropout(weights)
        
        return weights
    
    def forward_deterministic(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """
        확정 상태에 대한 표준 멀티헤드 어텐션 적용
        
        Args:
            q (torch.Tensor): 쿼리 텐서
            k (torch.Tensor): 키 텐서
            v (torch.Tensor): 값 텐서
            attn_mask (torch.Tensor, optional): 어텐션 마스크
            key_padding_mask (torch.Tensor, optional): 키 패딩 마스크
            
        Returns:
            torch.Tensor: 어텐션 결과
        """
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]
        seq_len_v = v.shape[1]
        
        # BUG FIX 3: Add validation for key and value sequence lengths
        assert seq_len_k == seq_len_v, "Key and value sequence lengths must match"
        
        # 선형 변환 적용
        q = self.q_linear(q).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, seq_len_v, self.nhead, self.d_k).transpose(1, 2)
        
        # 어텐션 점수 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 마스크 적용
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
            
        # 어텐션 가중치 계산
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # 가중치 적용
        output = torch.matmul(weights, v)
        
        # 형상 변환
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        
        # 출력 투영
        output = self.out_linear(output)
        
        return output
    
    def forward_superposition(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """
        중첩 상태에 대한 확장된 멀티헤드 어텐션 적용
        
        Args:
            q (torch.Tensor): 중첩 상태 쿼리 텐서
            k (torch.Tensor): 중첩 상태 키 텐서
            v (torch.Tensor): 중첩 상태 값 텐서
            attn_mask (torch.Tensor, optional): 어텐션 마스크
            key_padding_mask (torch.Tensor, optional): 키 패딩 마스크
            
        Returns:
            torch.Tensor: 중첩 상태 어텐션 결과
        """
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]
        seq_len_v = v.shape[1]
        
        # BUG FIX 3: Add validation for key and value sequence lengths
        assert seq_len_k == seq_len_v, "Key and value sequence lengths must match"
        
        # 중첩 상태에 대한 선형 변환
        q = self.superposition_q_linear(q)
        k = self.superposition_k_linear(k)
        v = self.superposition_v_linear(v)
        
        # 중첩 상태 차원으로 변환
        q = self._transform_for_superposition(q, batch_size, seq_len_q)
        k = self._transform_for_superposition(k, batch_size, seq_len_k)
        v = self._transform_for_superposition(v, batch_size, seq_len_v)
        
        # 양자 간섭 효과 적용
        q, k, v = self._apply_quantum_interference(q, k, v)
        
        # 각 중첩 차원에 대해 어텐션 계산
        outputs = []
        for dim in range(self.max_superposition_dim):
            # 현재 중첩 차원의 쿼리, 키, 값 추출
            q_dim = q[..., dim]  # [batch_size, nhead, seq_len_q, d_k]
            k_dim = k[..., dim]  # [batch_size, nhead, seq_len_k, d_k]
            v_dim = v[..., dim]  # [batch_size, nhead, seq_len_v, d_k]
            
            # 어텐션 점수 계산
            scores = torch.matmul(q_dim, k_dim.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # 마스크 적용
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
                
            # 진폭 가중치 계산
            weights = self._calculate_amplitude_weights(scores)
            
            # 가중치 적용
            output_dim = torch.matmul(weights, v_dim)
            outputs.append(output_dim.unsqueeze(-1))
            
        # 모든 중첩 차원의 결과 결합
        output = torch.cat(outputs, dim=-1)  # [batch_size, nhead, seq_len_q, d_k, max_dim]
        
        # 형상 변환
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = output.view(batch_size, seq_len_q, -1)
        
        # 출력 투영
        output = self.superposition_out_linear(output)
        
        return output
    
    def apply_sparsity(self, state, target_sparsity=0.7):
        """
        Applies adaptive sparsity to attention weights for resource efficiency
        
        Args:
            state (torch.Tensor): Input state tensor [batch_size, seq_len, hidden_dim]
            target_sparsity (float): Target sparsity level (0-1)
            
        Returns:
            torch.Tensor: Sparsified state tensor
        """
        # Calculate sparsity mask
        sparsity_scores = self.sparsity_gate(state.mean(dim=1, keepdim=True))
        threshold = torch.quantile(sparsity_scores, target_sparsity)
        mask = (sparsity_scores >= threshold).float()
        
        # Apply mask
        return state * mask
    
    # BUG FIX 2: Add a separate method for superposition sparsity
    def apply_superposition_sparsity(self, state, target_sparsity=0.7):
        """
        Applies adaptive sparsity to superposition state
        
        Args:
            state (torch.Tensor): Input superposition state tensor [batch_size, seq_len, hidden_dim * max_superposition_dim]
            target_sparsity (float): Target sparsity level (0-1)
            
        Returns:
            torch.Tensor: Sparsified superposition state tensor
        """
        batch_size, seq_len, _ = state.shape
        
        # Calculate dimension-wise sparsity scores
        sparsity_scores = self.superposition_sparsity_gate(state.mean(dim=1, keepdim=True))
        
        # Apply target sparsity by creating a dimension mask
        threshold = torch.quantile(sparsity_scores, target_sparsity)
        dim_mask = (sparsity_scores >= threshold).float()
        
        # Reshape state for dimension-wise masking
        reshaped_state = state.view(batch_size, seq_len, self.max_superposition_dim, self.hidden_dim)
        
        # Apply mask along dimension axis
        masked_state = reshaped_state * dim_mask.unsqueeze(1).unsqueeze(-1)
        
        # Return to original shape
        return masked_state.view(batch_size, seq_len, -1)
    
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, is_superposition=False, 
                target_sparsity=None, apply_sparsity=False):
        """
        양자 영감 어텐션 순전파
        
        Args:
            q (torch.Tensor): 쿼리 텐서
            k (torch.Tensor): 키 텐서
            v (torch.Tensor): 값 텐서
            attn_mask (torch.Tensor, optional): 어텐션 마스크
            key_padding_mask (torch.Tensor, optional): 키 패딩 마스크
            is_superposition (bool): 중첩 상태 사용 여부
            target_sparsity (float, optional): Target sparsity level
            apply_sparsity (bool): Whether to apply sparsity
            
        Returns:
            torch.Tensor: 어텐션 결과
        """
        if is_superposition:
            output = self.forward_superposition(q, k, v, attn_mask, key_padding_mask)
            
            # BUG FIX 2: Use the correct sparsity method for superposition state
            if apply_sparsity and target_sparsity is not None:
                output = self.apply_superposition_sparsity(output, target_sparsity)
        else:
            output = self.forward_deterministic(q, k, v, attn_mask, key_padding_mask)
            
            # Apply regular sparsity for deterministic state
            if apply_sparsity and target_sparsity is not None:
                output = self.apply_sparsity(output, target_sparsity)
            
        return output