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
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers - 1)
        ])
        
        # 전역 상태 메모리
        self.global_memory = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
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
            propagated_next = gate * current_state + (1 - gate) * next_state
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
            # 글로벌 메모리를 참조하여 일관성 유지
            memory_attention = torch.bmm(
                state.view(batch_size * seq_len, -1, self.hidden_dim),
                global_memory.view(batch_size * seq_len, self.hidden_dim, 1)
            ).view(batch_size, seq_len, -1)
            
            # 일관성 가중치 계산
            consistency_weights = F.softmax(memory_attention, dim=-1)
            
            # 일관성 프로젝션 적용
            consistent_state = self.consistency_projection(state)
            consistent_state = consistent_state * consistency_weights
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
        self.level_dims = [hidden_dim * 2**(num_levels - i - 1) for i in range(num_levels)]
        
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
            
            # 게이트 계산
            gate = self.level_gates[i](propagated_states[-1])
            
            # 게이트를 사용하여 상태 결합
            combined = gate * propagated_states[-1] + (1 - gate) * up_propagated
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
        
        # 모든 레벨의 정보를 통합
        integrated_state = torch.cat([
            F.interpolate(
                state.transpose(1, 2), 
                size=self.hidden_dim * self.max_superposition_dim // self.num_levels
            ).transpose(1, 2)
            for state in down_states
        ], dim=-1)
        
        return integrated_state
