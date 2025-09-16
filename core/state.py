import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalStateManager(nn.Module):
    """
    글로벌 상태 관리 엔진
    
    모델 전체에서 중첩 상태의 일관성을 유지하고, 계층적 상태 전파 메커니즘을 제공
    """
    
    def __init__(self, hidden_dim, num_layers, max_superposition_dim=4):
        """
        글로벌 상태 관리 엔진 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            num_layers (int): 모델 레이어 수
            max_superposition_dim (int): 최대 중첩 상태 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_superposition_dim = max_superposition_dim
        
        # 레이어 간 상태 전파를 위한 게이트
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * max_superposition_dim * 2, hidden_dim * max_superposition_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers - 1)
        ])
        
        # 전역 상태 메모리
        self.global_memory = nn.Parameter(torch.zeros(1, 1, hidden_dim * max_superposition_dim))
        
        # 상태 일관성 유지를 위한 모듈
        self.consistency_projection = nn.Linear(
            hidden_dim * max_superposition_dim, 
            hidden_dim * max_superposition_dim
        )
        
        # 상태 정규화
        self.norm = nn.LayerNorm(hidden_dim * max_superposition_dim)
        
    def propagate_states(self, layer_states):
        """
        레이어 간 상태 전파
        
        Args:
            layer_states (list): 각 레이어의 상태 리스트
            
        Returns:
            list: 전파된 상태 리스트
        """
        propagated_states = [layer_states[0]]
        
        for i in range(self.num_layers - 1):
            # 현재 레이어와 다음 레이어의 상태 병합
            current_state = propagated_states[-1]
            next_state = layer_states[i + 1]
            
            # 게이트 계산
            gate_input = torch.cat([
                current_state.mean(dim=1),
                next_state.mean(dim=1)
            ], dim=-1)
            gate = self.layer_gates[i](gate_input).unsqueeze(1)
            
            # 게이트를 사용하여 상태 전파
            # 차원 불일치 처리: 두 상태의 마지막 차원을 맞춘다
            current_dim = current_state.shape[-1]
            next_dim = next_state.shape[-1]

            if current_dim == next_dim:
                # 차원이 같으면 기존 방식 사용
                propagated_next = gate * current_state + (1 - gate) * next_state
            else:
                # 차원이 다르면 더 큰 차원으로 통일
                target_dim = max(current_dim, next_dim)

                if current_dim < target_dim:
                    # current_state를 확장 (복제하여 확장)
                    expand_factor = target_dim // current_dim
                    current_expanded = current_state.repeat(1, 1, expand_factor)
                    if target_dim % current_dim != 0:
                        # 나머지가 있으면 패딩
                        remaining = target_dim - current_expanded.shape[-1]
                        padding = current_state[:, :, :remaining]
                        current_expanded = torch.cat([current_expanded, padding], dim=-1)
                    current_for_calc = current_expanded
                else:
                    current_for_calc = current_state

                if next_dim < target_dim:
                    # next_state를 확장 (복제하여 확장)
                    expand_factor = target_dim // next_dim
                    next_expanded = next_state.repeat(1, 1, expand_factor)
                    if target_dim % next_dim != 0:
                        # 나머지가 있으면 패딩
                        remaining = target_dim - next_expanded.shape[-1]
                        padding = next_state[:, :, :remaining]
                        next_expanded = torch.cat([next_expanded, padding], dim=-1)
                    next_for_calc = next_expanded
                else:
                    next_for_calc = next_state

                propagated_next = gate * current_for_calc + (1 - gate) * next_for_calc

            propagated_states.append(propagated_next)
            
        return propagated_states
    
    
    def exchange_information(self, states, exchange_rate=0.1):
        """
        토큰 간, 레이어 간 상태 정보 교환
        
        Args:
            states (list): 각 레이어의 상태 리스트
            exchange_rate (float): 정보 교환 비율
            
        Returns:
            list: 정보 교환 후 상태 리스트
        """
        exchanged_states = []
        
        for state in states:
            batch_size, seq_len, _ = state.shape
            
            # 토큰 간 정보 교환 (주변 토큰과의 정보 공유)
            token_exchange = F.avg_pool1d(
                state.transpose(1, 2),
                kernel_size=3,
                stride=1,
                padding=1
            ).transpose(1, 2)
            
            # 원본 상태와 교환된 정보 결합
            exchanged = (1 - exchange_rate) * state + exchange_rate * token_exchange
            exchanged_states.append(exchanged)
            
        return exchanged_states

    def ensure_consistency(self, superposition_states):
        """Model-wide consistency for superposition states.

        Args:
            superposition_states (list[Tensor]): list of [B, S, H*K] states

        Returns:
            list[Tensor]: consistent states, each [B, S, H*K]
        """
        batch_size, seq_len = superposition_states[0].shape[:2]

        consistent_states = []
        for state in superposition_states:
            reshaped = state.view(batch_size, seq_len, self.max_superposition_dim, self.hidden_dim)

            expected_size = self.hidden_dim * self.max_superposition_dim
            gm_flat = self.global_memory.view(-1)
            if gm_flat.numel() != expected_size:
                new_mem = torch.zeros(expected_size, device=self.global_memory.device, dtype=self.global_memory.dtype)
                self.global_memory.data = new_mem.view(1, 1, -1)
                gm_flat = self.global_memory.view(-1)

            global_memory_reshaped = gm_flat.view(self.max_superposition_dim, self.hidden_dim)
            global_memory_expanded = global_memory_reshaped.unsqueeze(0).expand(
                batch_size * seq_len, self.max_superposition_dim, self.hidden_dim
            )

            memory_attention = torch.bmm(
                reshaped.view(batch_size * seq_len, self.max_superposition_dim, self.hidden_dim),
                global_memory_expanded.transpose(-2, -1)
            ).view(batch_size, seq_len, self.max_superposition_dim, self.max_superposition_dim)

            memory_attention = torch.diagonal(memory_attention, dim1=-2, dim2=-1)
            consistency_weights = F.softmax(memory_attention, dim=-1).unsqueeze(-1)

            gated = reshaped * consistency_weights
            gated = gated.view(batch_size, seq_len, -1)

            consistent_state = self.consistency_projection(gated)
            consistent_state = self.norm(consistent_state)
            consistent_states.append(consistent_state)

        return consistent_states

    def update_global_memory(self, states):
        """
        전역 상태 메모리 업데이트
        
        Args:
            states (list): 각 레이어의 상태 리스트
        """
        # 모든 레이어의 상태 평균 계산
        avg_state = torch.stack([state.mean(dim=1).mean(dim=0) for state in states]).mean(dim=0)
        
        # 글로벌 메모리 업데이트 (지수 이동 평균)
        self.global_memory.data = 0.9 * self.global_memory.data + 0.1 * avg_state.unsqueeze(0).unsqueeze(0)
    
    def forward(self, layer_states, update_memory=True):
        """
        글로벌 상태 관리 엔진 순전파
        
        Args:
            layer_states (list): 각 레이어의 상태 리스트
            update_memory (bool): 글로벌 메모리 업데이트 여부
            
        Returns:
            list: 관리된 상태 리스트
        """
        # 레이어 간 상태 전파
        propagated_states = self.propagate_states(layer_states)
        
        # 중첩 상태의 일관성 보장
        consistent_states = self.ensure_consistency(propagated_states)
        
        # 토큰 간, 레이어 간 상태 정보 교환
        exchanged_states = self.exchange_information(consistent_states)
        
        # 필요한 경우 글로벌 메모리 업데이트
        if update_memory:
            self.update_global_memory(exchanged_states)
            
        return exchanged_states


class HierarchicalStateProtocol(nn.Module):
    """
    단순화된 계층적 상태 처리 메커니즘
    """

    def __init__(self, hidden_dim, num_levels=3, max_superposition_dim=4):
        """
        계층적 상태 프로토콜 초기화

        Args:
            hidden_dim (int): 기본 히든 차원
            num_levels (int): 계층 수준 수 (단순화를 위해 고정)
            max_superposition_dim (int): 최대 중첩 상태 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim

        # 모든 레벨에서 동일한 차원을 사용하여 차원 불일치 문제 해결
        self.state_dim = hidden_dim * max_superposition_dim

        # 멀티레벨 어텐션 메커니즘
        self.multi_level_attention = nn.MultiheadAttention(
            embed_dim=self.state_dim,
            num_heads=min(8, self.state_dim // 64),  # 적절한 헤드 수
            dropout=0.1,
            batch_first=True
        )

        # 상태 변환 및 정규화
        self.norm1 = nn.LayerNorm(self.state_dim)
        self.norm2 = nn.LayerNorm(self.state_dim)

        # 피드포워드 네트워크
        self.ff = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.Dropout(0.1)
        )

        # 계층별 게이팅
        self.level_gate = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Sigmoid()
        )

    def forward(self, input_state):
        """
        단순화된 계층적 상태 처리

        Args:
            input_state (torch.Tensor): 입력 상태 텐서 [B, S, H*K]

        Returns:
            torch.Tensor: 처리된 상태 텐서 [B, S, H*K]
        """
        # 입력 상태 정규화
        normalized_input = self.norm1(input_state)

        # 멀티레벨 셀프 어텐션
        attended_state, _ = self.multi_level_attention(
            normalized_input, normalized_input, normalized_input
        )

        # 잔차 연결
        state_after_attention = input_state + attended_state

        # 정규화
        normalized_state = self.norm2(state_after_attention)

        # 피드포워드 처리
        ff_output = self.ff(normalized_state)

        # 게이팅 메커니즘 적용
        gate = self.level_gate(normalized_state)
        gated_output = gate * ff_output + (1 - gate) * normalized_state

        # 최종 잔차 연결
        final_state = state_after_attention + gated_output

        return final_state
