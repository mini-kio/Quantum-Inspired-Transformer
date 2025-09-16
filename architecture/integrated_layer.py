import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import QuantumInspiredAttention
from .feed import DualStateFeedForward
from core.collapse import CollapseGate


class IntegratedTransformerLayer(nn.Module):
    """
    확정 상태와 중첩 상태를 모두 처리할 수 있는 통합 트랜스포머 레이어
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        max_superposition_dim=4,
        activation='gelu',
        layer_id=0,
        num_layers=12,
        gate_type='mlp'
    ):
        """
        통합 트랜스포머 레이어 초기화
        
        Args:
            d_model: 모델 차원
            nhead: 어텐션 헤드 수
            dim_feedforward: 피드포워드 네트워크 내부 차원
            dropout: 드롭아웃 비율
            max_superposition_dim: 최대 중첩 상태 차원
            activation: 활성화 함수
            layer_id: 현재 레이어 ID
            num_layers: 총 레이어 수
            gate_type: CollapseGate 유형 ('mlp' 또는 'transformer')
        """
        super().__init__()
        self.d_model = d_model
        self.max_superposition_dim = max_superposition_dim
        self.layer_id = layer_id
        self.num_layers = num_layers
        
        # 양자 영감 어텐션
        self.attention = QuantumInspiredAttention(
            d_model=d_model,
            nhead=nhead,
            max_superposition_dim=max_superposition_dim,
            dropout=dropout
        )
        
        # 이중 상태 피드포워드 네트워크
        self.feed_forward = DualStateFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            max_superposition_dim=max_superposition_dim,
            dropout=dropout,
            activation=activation
        )
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 중첩 상태용 레이어 정규화
        self.superposition_norm1 = nn.LayerNorm(d_model * max_superposition_dim)
        self.superposition_norm2 = nn.LayerNorm(d_model * max_superposition_dim)
        
        # 중첩-확정 전환을 위한 CollapseGate
        self.collapse_gate = CollapseGate(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim,
            layer_id=layer_id,
            num_layers=num_layers,
            gate_type=gate_type
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward_deterministic(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None
    ):
        """
        확정 상태에 대한 트랜스포머 레이어 순전파
        
        Args:
            src: 입력 텐서 [batch_size, seq_len, d_model]
            src_mask: 어텐션 마스크
            src_key_padding_mask: 패딩 마스크
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        # 자기 어텐션
        attn_output = self.attention(
            q=src,
            k=src,
            v=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_superposition=False
        )
        
        # 첫 번째 잔여 연결 및 정규화
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(src, is_superposition=False)
        
        # 두 번째 잔여 연결 및 정규화
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src
    
    def forward_superposition(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None
    ):
        """
        중첩 상태에 대한 트랜스포머 레이어 순전파
        
        Args:
            src: 입력 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
            src_mask: 어텐션 마스크
            src_key_padding_mask: 패딩 마스크
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
        """
        # 자기 어텐션
        attn_output = self.attention(
            q=src,
            k=src,
            v=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_superposition=True
        )
        
        # 첫 번째 잔여 연결 및 정규화
        src = src + self.dropout(attn_output)
        src = self.superposition_norm1(src)
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(src, is_superposition=True)
        
        # 두 번째 잔여 연결 및 정규화
        src = src + self.dropout(ff_output)
        src = self.superposition_norm2(src)
        
        return src
    
    def forward(
        self,
        deterministic_state,
        superposition_state,
        mask=None,
        src_key_padding_mask=None,
        context=None,
        superposition_degree=None,
        collapse_threshold=None,
        interference_strength=None,
        p_target=0.5
    ):
        """
        통합 트랜스포머 레이어 순전파
        
        Args:
            deterministic_state: 확정 상태 텐서
            superposition_state: 중첩 상태 텐서
            mask: 어텐션 마스크
            src_key_padding_mask: 패딩 마스크
            context: 컨텍스트 임베딩
            superposition_degree: 중첩 정도 (0~1)
            collapse_threshold: 상태 붕괴 임계값 (0~1)
            interference_strength: 간섭 강도 (0~1)
            p_target: 목표 전환 확률 (0~1)
            
        Returns:
            dict: 레이어 결과
        """
        # 확정 상태 처리
        deterministic_output = self.forward_deterministic(
            src=deterministic_state,
            src_mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 중첩 상태 처리
        superposition_output = self.forward_superposition(
            src=superposition_state,
            src_mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # CollapseGate 적용
        gate_result = self.collapse_gate(
            deterministic_state=deterministic_output,
            superposition_state=superposition_output,
            context=context,
            p_target=p_target
        )
        
        return {
            'deterministic_state': gate_result['deterministic_state'],
            'superposition_state': gate_result['superposition_state'],
            'uncertainty': gate_result['uncertainty'],
            'transition_prob': gate_result['transition_prob'],
            'resource_efficiency': gate_result['resource_efficiency'],
            'binary_mask': gate_result['binary_mask'],
            'alpha': gate_result['alpha']
        }


class IntegratedDecoderLayer(nn.Module):
    """
    확정 상태와 중첩 상태를 모두 처리할 수 있는 통합 디코더 레이어
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        max_superposition_dim=4,
        activation='gelu',
        layer_id=0,
        num_layers=12,
        gate_type='mlp'
    ):
        """
        통합 디코더 레이어 초기화
        
        Args:
            d_model: 모델 차원
            nhead: 어텐션 헤드 수
            dim_feedforward: 피드포워드 네트워크 내부 차원
            dropout: 드롭아웃 비율
            max_superposition_dim: 최대 중첩 상태 차원
            activation: 활성화 함수
            layer_id: 현재 레이어 ID
            num_layers: 총 레이어 수
            gate_type: CollapseGate 유형 ('mlp' 또는 'transformer')
        """
        super().__init__()
        self.d_model = d_model
        self.max_superposition_dim = max_superposition_dim
        self.layer_id = layer_id
        self.num_layers = num_layers
        
        # 자기 어텐션
        self.self_attn = QuantumInspiredAttention(
            d_model=d_model,
            nhead=nhead,
            max_superposition_dim=max_superposition_dim,
            dropout=dropout
        )
        
        # 크로스 어텐션
        self.cross_attn = QuantumInspiredAttention(
            d_model=d_model,
            nhead=nhead,
            max_superposition_dim=max_superposition_dim,
            dropout=dropout
        )
        
        # 이중 상태 피드포워드 네트워크
        self.feed_forward = DualStateFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            max_superposition_dim=max_superposition_dim,
            dropout=dropout,
            activation=activation
        )
        
        # 레이어 정규화 (확정 상태용)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 레이어 정규화 (중첩 상태용)
        self.superposition_norm1 = nn.LayerNorm(d_model * max_superposition_dim)
        self.superposition_norm2 = nn.LayerNorm(d_model * max_superposition_dim)
        self.superposition_norm3 = nn.LayerNorm(d_model * max_superposition_dim)
        
        # 중첩-확정 전환을 위한 CollapseGate
        self.collapse_gate = CollapseGate(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim,
            layer_id=layer_id,
            num_layers=num_layers,
            gate_type=gate_type
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward_deterministic(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        """
        확정 상태에 대한 디코더 레이어 순전파
        
        Args:
            tgt: 타겟 텐서 [batch_size, seq_len, d_model]
            memory: 메모리 텐서 [batch_size, src_len, d_model]
            tgt_mask: 타겟 어텐션 마스크
            memory_mask: 메모리 어텐션 마스크
            tgt_key_padding_mask: 타겟 패딩 마스크
            memory_key_padding_mask: 메모리 패딩 마스크
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        # 자기 어텐션
        self_attn_output = self.self_attn(
            q=tgt,
            k=tgt,
            v=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_superposition=False
        )
        
        # 첫 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        
        # 크로스 어텐션
        cross_attn_output = self.cross_attn(
            q=tgt,
            k=memory,
            v=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_superposition=False
        )
        
        # 두 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(tgt, is_superposition=False)
        
        # 세 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt
    
    def forward_superposition(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        """
        중첩 상태에 대한 디코더 레이어 순전파
        
        Args:
            tgt: 타겟 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
            memory: 메모리 텐서 [batch_size, src_len, d_model * max_superposition_dim]
            tgt_mask: 타겟 어텐션 마스크
            memory_mask: 메모리 어텐션 마스크
            tgt_key_padding_mask: 타겟 패딩 마스크
            memory_key_padding_mask: 메모리 패딩 마스크
            
        Returns:
            출력 텐서 [batch_size, seq_len, d_model * max_superposition_dim]
        """
        # 자기 어텐션
        self_attn_output = self.self_attn(
            q=tgt,
            k=tgt,
            v=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_superposition=True
        )
        
        # 첫 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.superposition_norm1(tgt)
        
        # 크로스 어텐션
        cross_attn_output = self.cross_attn(
            q=tgt,
            k=memory,
            v=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_superposition=True
        )
        
        # 두 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.superposition_norm2(tgt)
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(tgt, is_superposition=True)
        
        # 세 번째 잔여 연결 및 정규화
        tgt = tgt + self.dropout(ff_output)
        tgt = self.superposition_norm3(tgt)
        
        return tgt
    
    def forward(
        self,
        deterministic_state,
        superposition_state,
        memory,
        memory_superposition=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        context=None,
        superposition_degree=None,
        collapse_threshold=None,
        interference_strength=None,
        p_target=0.5
    ):
        """
        통합 디코더 레이어 순전파
        
        Args:
            deterministic_state: 확정 상태 텐서
            superposition_state: 중첩 상태 텐서
            memory: 메모리 확정 상태 텐서
            memory_superposition: 메모리 중첩 상태 텐서
            tgt_mask: 타겟 어텐션 마스크
            memory_mask: 메모리 어텐션 마스크
            tgt_key_padding_mask: 타겟 패딩 마스크
            memory_key_padding_mask: 메모리 패딩 마스크
            context: 컨텍스트 임베딩
            superposition_degree: 중첩 정도 (0~1)
            collapse_threshold: 상태 붕괴 임계값 (0~1)
            interference_strength: 간섭 강도 (0~1)
            p_target: 목표 전환 확률 (0~1)
            
        Returns:
            dict: 레이어 결과
        """
        # 확정 상태 처리
        deterministic_output = self.forward_deterministic(
            tgt=deterministic_state,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 중첩 상태 처리
        if memory_superposition is None:
            # 메모리 중첩 상태가 없는 경우, 확정 상태에서 변환
            batch_size, src_len, _ = memory.shape
            # 중첩 상태 생성 로직은 실제 구현에서 상태 변환 모듈을 사용해야 함
            # 여기서는 간단히 복제하여 확장
            memory_superposition = memory.repeat(1, 1, self.max_superposition_dim)
        
        superposition_output = self.forward_superposition(
            tgt=superposition_state,
            memory=memory_superposition,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # CollapseGate 적용
        gate_result = self.collapse_gate(
            deterministic_state=deterministic_output,
            superposition_state=superposition_output,
            context=context,
            p_target=p_target
        )
        
        return {
            'deterministic_state': gate_result['deterministic_state'],
            'superposition_state': gate_result['superposition_state'],
            'uncertainty': gate_result['uncertainty'],
            'transition_prob': gate_result['transition_prob'],
            'resource_efficiency': gate_result['resource_efficiency'],
            'binary_mask': gate_result['binary_mask'],
            'alpha': gate_result['alpha']
        }