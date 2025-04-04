import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        
        # 최적 결정 시점 예측기
        self.decision_time_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
    
    def forward(self, superposition_state, context, previous_state=None, hidden_state=None, force_collapse=False):
        """
        상태 붕괴 프레임워크 순전파
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            context (torch.Tensor): 맥락 정보
            previous_state (torch.Tensor, optional): 이전 중첩 상태
            hidden_state (torch.Tensor, optional): GRU 히든 상태
            force_collapse (bool): 강제 붕괴 수행 여부
            
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
                    
        return {
            'collapsed_state': collapsed_state,
            'uncertainty': uncertainty,
            'change_score': change_score,
            'decision_time': decision_time,
            'collapse_condition': collapse_condition,
            'mode_weights': mode_weights,
            'hidden_state': new_hidden
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
        
        # 모드 전환 컨트롤러
        self.mode_transition_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
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
        
        # 확장된 확정 상태 생성
        expanded_deterministic = torch.zeros_like(superposition_state)
        expanded_deterministic[:, :, :self.hidden_dim] = deterministic_state
        
        # 점진적 붕괴 적용 (중첩과 확정의 가중 평균)
        gradually_collapsed = (1 - collapse_rate) * superposition_state + collapse_rate * expanded_deterministic
        
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
        
        # 형태 복원
        partially_collapsed = partially_collapsed.view(batch_size, seq_len, -1)
        
        return partially_collapsed
    
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
    
    def forward(self, superposition_state, deterministic_state, context, uncertainty=None):
        """
        동적 붕괴 컨트롤러 순전파
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            deterministic_state (torch.Tensor): 확정 상태 텐서
            context (torch.Tensor): 맥락 정보
            uncertainty (torch.Tensor, optional): 불확실성 점수
            
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
        
        return {
            'collapsed_state': collapsed_state,
            'mode_weights': mode_weights,
            'task_complexity': task_complexity,
            'gradual_state': gradual_state,
            'partial_state': partial_state,
            'full_state': full_state
        }
