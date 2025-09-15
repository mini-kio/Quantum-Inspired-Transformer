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
    
    def ensure_consistency(self, superposition_states):
        """
        모델 전체에서 중첩 상태의 일관성 보장
        
        Args:
            superposition_states (list): 중첩 상태 리스트
            
        Returns:
            list: 일관성이 유지된 중첩 상태 리스트
        """
        batch_size, seq_len = superposition_states[0].shape[:2]
        
        # 글로벌 상태 메모리 확장
        global_memory = self.global_memory.expand(batch_size, seq_len, -1)
        
        consistent_states = []
        for state in superposition_states:
            # state: [B, S, H * K]  (K = max_superposition_dim)
            # 글로벌 메모리를 참조하여 각 중첩 차원(K)별 가중치 산출
            reshaped = state.view(batch_size, seq_len, self.max_superposition_dim, self.hidden_dim)
            # [B*S, K, H] @ [B*S, H, 1] -> [B*S, K, 1]
            # global_memory를 올바른 형태로 처리
            # global_memory의 실제 크기 확인하고 적절히 reshape
            expected_size = self.hidden_dim * self.max_superposition_dim
            if global_memory.numel() == expected_size:
                # 올바른 크기인 경우
                global_memory_flat = global_memory.view(-1)  # [H*K]
            else:
                # 크기가 다른 경우 초기화된 상태로 되돌림
                global_memory_flat = torch.zeros(expected_size, device=global_memory.device, dtype=global_memory.dtype)
                # global_memory 재초기화
                self.global_memory.data = global_memory_flat.view(1, 1, -1)

            # [H*K] -> [K, H]로 reshape
            global_memory_reshaped = global_memory_flat.view(self.max_superposition_dim, self.hidden_dim)
            global_memory_expanded = global_memory_reshaped.unsqueeze(0).expand(batch_size * seq_len, self.max_superposition_dim, self.hidden_dim)

            # attention 계산 - [B*S, K, H] @ [B*S, H, K] -> [B*S, K, K]
            memory_attention = torch.bmm(
                reshaped.view(batch_size * seq_len, self.max_superposition_dim, self.hidden_dim),
                global_memory_expanded.transpose(-2, -1)  # [B*S, H, K]
            ).view(batch_size, seq_len, self.max_superposition_dim, self.max_superposition_dim)

            # 대각합을 취해서 [B, S, K] 차원으로 축소
            memory_attention = torch.diagonal(memory_attention, dim1=-2, dim2=-1)
            # 차원 K에 대한 소프트맥스 가중치
            consistency_weights = F.softmax(memory_attention, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

            # K 차원별 게이팅 적용 후 결합
            gated = reshaped * consistency_weights  # [B, S, K, H]
            gated = gated.view(batch_size, seq_len, -1)  # [B, S, H*K]

            # 일관성 프로젝션 및 정규화
            consistent_state = self.consistency_projection(gated)
            consistent_state = self.norm(consistent_state)

            consistent_states.append(consistent_state)
        
        return consistent_states
    
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

    def ensure_consistency_fixed(self, superposition_states):
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
        consistent_states = self.ensure_consistency_fixed(propagated_states)
        
        # 토큰 간, 레이어 간 상태 정보 교환
        exchanged_states = self.exchange_information(consistent_states)
        
        # 필요한 경우 글로벌 메모리 업데이트
        if update_memory:
            self.update_global_memory(exchanged_states)
            
        return exchanged_states


class HierarchicalStateProtocol(nn.Module):
    """
    계층적 상태 전파 메커니즘으로 다양한 추상화 수준 연결
    """
    
    def __init__(self, hidden_dim, num_levels=3, max_superposition_dim=4):
        """
        계층적 상태 프로토콜 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            num_levels (int): 계층 수준 수
            max_superposition_dim (int): 최대 중첩 상태 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.max_superposition_dim = max_superposition_dim
        
        # 각 계층 수준의 표현 차원
        # 최상위 차원은 실제 입력 차원(hidden_dim * max_superposition_dim)과 일치해야 함
        # 이후 각 단계에서 2배씩 축소되는 구조를 기준으로 차원을 정의
        top_dim = hidden_dim * max_superposition_dim
        self.level_dims = [top_dim // (2 ** i) for i in range(num_levels)]
        
        # 상향 전파 프로젝션 (하위 -> 상위)
        self.up_projections = nn.ModuleList([
            nn.Linear(self.level_dims[i], self.level_dims[i-1])
            for i in range(1, num_levels)
        ])
        
        # 하향 전파 프로젝션 (상위 -> 하위)
        self.down_projections = nn.ModuleList([
            nn.Linear(self.level_dims[i], self.level_dims[i+1])
            for i in range(num_levels - 1)
        ])
        
        # 레벨별 적응형 게이트
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.level_dims[i], 1),
                nn.Sigmoid()
            )
            for i in range(num_levels)
        ])

        # align previous accumulated state to the current combine dimension (level_dims[i])
        # index 0 is identity (same dim), i>=1 maps from level_dims[0] -> level_dims[i]
        self.up_prev_aligns = nn.ModuleList([
            nn.Identity() if i == 0 else nn.Linear(self.level_dims[0], self.level_dims[i])
            for i in range(num_levels)
        ])

        # 출력 정렬용 레벨별 프로젝션: 각 레벨 출력을 동일 목표 차원으로 매핑
        target_total_dim = hidden_dim * max_superposition_dim
        base_dim = target_total_dim // num_levels
        remainder = target_total_dim - base_dim * num_levels
        self.level_out_dims = [base_dim] * num_levels
        self.level_out_dims[-1] += remainder  # 합이 정확히 target_total_dim이 되도록 조정
        self.level_align = nn.ModuleList([
            nn.Linear(self.level_dims[i], self.level_out_dims[i])
            for i in range(num_levels)
        ])
        
    def propagate_up(self, level_states):
        """
        하위에서 상위 수준으로 상태 전파
        
        Args:
            level_states (list): 각 수준의 상태 리스트
            
        Returns:
            list: 상향 전파된 상태 리스트
        """
        propagated_states = [level_states[0]]
        
        for i in range(self.num_levels - 1):
            # 하위 수준 상태를 상위 수준으로 투영
            up_propagated = self.up_projections[i](level_states[i + 1])
            
            # 게이트 계산: 해당 레벨 상태를 기준으로 일관된 차원 사용
            gate = self.level_gates[i](level_states[i])

            # align accumulated state to current combine dim
            prev_aligned = self.up_prev_aligns[i](propagated_states[-1])
            
            # 게이트를 사용하여 상태 결합 (동일 차원)
            combined = gate * prev_aligned + (1 - gate) * up_propagated
            propagated_states.append(combined)
            
        return propagated_states
    
    def propagate_down(self, level_states):
        """
        상위에서 하위 수준으로 상태 전파
        
        Args:
            level_states (list): 각 수준의 상태 리스트
            
        Returns:
            list: 하향 전파된 상태 리스트
        """
        propagated_states = [level_states[0]]
        
        for i in range(self.num_levels - 1):
            # 상위 수준 상태를 하위 수준으로 투영
            down_propagated = self.down_projections[i](propagated_states[-1])
            
            # 게이트 계산
            gate = self.level_gates[i + 1](level_states[i + 1])
            
            # 게이트를 사용하여 상태 결합
            combined = gate * level_states[i + 1] + (1 - gate) * down_propagated
            propagated_states.append(combined)
            
        return propagated_states
    
    def forward(self, input_state):
        """
        계층적 상태 프로토콜 순전파
        
        Args:
            input_state (torch.Tensor): 입력 상태 텐서
            
        Returns:
            torch.Tensor: 계층적 처리된 상태 텐서
        """
        batch_size, seq_len, _ = input_state.shape
        
        # 초기 레벨 상태 생성
        level_states = []
        current_state = input_state
        
        for i in range(self.num_levels):
            if i == 0:
                # 첫 번째 수준은 입력 상태 그대로 사용
                level_states.append(current_state)
            else:
                # 다음 수준은 이전 수준의 압축된 상태
                compressed = current_state.view(
                    batch_size, seq_len, 2, -1
                ).mean(dim=2)
                level_states.append(compressed)
                current_state = compressed
                
        # 상향 전파
        up_states = self.propagate_up(level_states)
        
        # 하향 전파
        down_states = self.propagate_down(up_states)
        
        # 모든 레벨의 정보를 통합 (특징 차원 정렬 후 연결)
        aligned = []
        for i, state in enumerate(down_states):
            # [B, S, level_dim] -> [B, S, level_out_dim]
            aligned.append(self.level_align[i](state))
        integrated_state = torch.cat(aligned, dim=-1)  # [B, S, H * K]

        return integrated_state
