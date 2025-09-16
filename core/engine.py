import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InferenceEngine(nn.Module):
    """
    통합 추론 엔진
    
    내부 중첩 상태와 외부 명시적 추론을 결합한 하이브리드 추론 시스템
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, max_reasoning_steps=5):
        """
        통합 추론 엔진 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            max_reasoning_steps (int): 최대 추론 단계 수
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.max_reasoning_steps = max_reasoning_steps
        
        # 추론 모드 선택기
        self.reasoning_mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # 암시적/명시적 모드
            nn.Softmax(dim=-1)
        )
        
        # 추론 깊이 컨트롤러
        self.reasoning_depth_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_reasoning_steps),
            nn.Softmax(dim=-1)
        )
        
        # 명시적 추론 단계 모듈
        self.explicit_reasoning_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(max_reasoning_steps)
        ])
        
        # 암시적 추론 모듈 (중첩 상태에서 작동)
        self.implicit_reasoning = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * max_superposition_dim),
            nn.LayerNorm(hidden_dim * max_superposition_dim)
        )
        
        # 추론 결과 결합기
        self.reasoning_combiner = nn.Sequential(
            nn.Linear(hidden_dim * (max_reasoning_steps + 2), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 중간 추론 상태 메모리
        self.reasoning_memory = nn.GRUCell(hidden_dim, hidden_dim)
        
    def select_reasoning_mode(self, context):
        """
        맥락 기반 추론 모드 선택 (암시적/명시적)
        
        Args:
            context (torch.Tensor): 맥락 정보
            
        Returns:
            torch.Tensor: 추론 모드 가중치 [암시적, 명시적]
        """
        return self.reasoning_mode_selector(context)
    
    def determine_reasoning_depth(self, context):
        """
        복잡성에 따른 추론 깊이 결정
        
        Args:
            context (torch.Tensor): 맥락 정보
            
        Returns:
            torch.Tensor: 각 추론 단계의 가중치
        """
        return self.reasoning_depth_controller(context)
    
    def explicit_reasoning(self, state, context, depth_weights=None):
        """
        명시적 추론 수행 (단계별 추론)
        
        Args:
            state (torch.Tensor): 초기 상태 텐서
            context (torch.Tensor): 맥락 정보
            depth_weights (torch.Tensor, optional): 각 단계의 가중치
            
        Returns:
            tuple: (최종 추론 상태, 중간 추론 상태 목록)
        """
        batch_size = state.shape[0]
        
        # 초기 메모리 상태
        memory = torch.zeros_like(state)
        
        # 중간 추론 상태 저장
        intermediate_states = [state]
        
        # 맥락 확장
        if context.dim() == 2:
            context_expanded = context.unsqueeze(1).expand(-1, state.shape[1], -1)
        else:
            context_expanded = context
            
        # 추론 깊이가 지정되지 않은 경우 균등 분포 사용
        if depth_weights is None:
            depth_weights = torch.ones(batch_size, self.max_reasoning_steps, device=state.device)
            depth_weights = depth_weights / self.max_reasoning_steps
            
        # 단계별 추론 수행
        for i in range(self.max_reasoning_steps):
            # 현재 상태와 맥락 결합
            combined = torch.cat([intermediate_states[-1], context_expanded], dim=-1)
            
            # 추론 단계 적용
            next_state = self.explicit_reasoning_steps[i](combined)
            
            # 메모리 업데이트
            memory_flat = memory.view(-1, self.hidden_dim)
            state_flat = next_state.view(-1, self.hidden_dim)
            updated_memory = self.reasoning_memory(state_flat, memory_flat)
            memory = updated_memory.view(state.shape)
            
            # 메모리를 참조하여 상태 업데이트
            gated_state = next_state + 0.1 * memory
            intermediate_states.append(gated_state)
            
        # 깊이 가중치 준비
        depth_weights = depth_weights.unsqueeze(1).unsqueeze(2)
        
        # 가중 합계 계산 (배치별 다른 깊이 적용)
        weighted_sum = torch.zeros_like(state)
        for b in range(batch_size):
            for i in range(self.max_reasoning_steps):
                weighted_sum[b] += depth_weights[b, 0, i] * intermediate_states[i+1][b]
                
        return weighted_sum, intermediate_states
    
    def implicit_reasoning(self, superposition_state):
        """
        암시적 추론 수행 (중첩 상태 기반)
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 암시적 추론 결과
        """
        # 중첩 상태에 대한 직접 추론
        return self.implicit_reasoning(superposition_state)
    
    def forward(self, deterministic_state, superposition_state, context):
        """
        통합 추론 엔진 순전파
        
        Args:
            deterministic_state (torch.Tensor): 확정 상태 텐서
            superposition_state (torch.Tensor): 중첩 상태 텐서
            context (torch.Tensor): 맥락 정보
            
        Returns:
            dict: 추론 결과
        """
        batch_size = deterministic_state.shape[0]
        
        # 추론 모드 선택
        mode_weights = self.select_reasoning_mode(context)
        implicit_weight = mode_weights[:, 0].view(batch_size, 1, 1)
        explicit_weight = mode_weights[:, 1].view(batch_size, 1, 1)
        
        # 추론 깊이 결정
        depth_weights = self.determine_reasoning_depth(context)
        
        # 명시적 추론 수행
        explicit_result, intermediate_states = self.explicit_reasoning(
            deterministic_state, context, depth_weights
        )
        
        # 암시적 추론 수행
        implicit_result = self.implicit_reasoning(superposition_state)
        
        # 암시적 추론 결과 변환 (중첩 상태에서 확정 상태로)
        implicit_deterministic = implicit_result.view(
            batch_size, -1, self.max_superposition_dim, self.hidden_dim
        ).mean(dim=2)
        
        # 추론 모드에 따른 가중 평균
        combined_result = (
            implicit_weight * implicit_deterministic +
            explicit_weight * explicit_result
        )
        
        return {
            'result': combined_result,
            'mode_weights': mode_weights,
            'depth_weights': depth_weights,
            'explicit_result': explicit_result,
            'implicit_result': implicit_result,
            'intermediate_states': intermediate_states
        }


class ReasoningDepthAdapter(nn.Module):
    """
    복잡성에 따른 추론 깊이 자동 조절 기능
    """
    
    def __init__(self, hidden_dim, min_depth=1, max_depth=8):
        """
        추론 깊이 어댑터 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            min_depth (int): 최소 추론 깊이
            max_depth (int): 최대 추론 깊이
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # 복잡성 추정기
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 깊이 맵핑
        self.depth_mapping = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_depth - min_depth + 1),
            nn.Softmax(dim=-1)
        )
        
    def estimate_complexity(self, context):
        """
        맥락 기반 복잡성 추정
        
        Args:
            context (torch.Tensor): 맥락 정보
            
        Returns:
            torch.Tensor: 복잡성 점수 (0~1)
        """
        return self.complexity_estimator(context)
    
    def forward(self, context):
        """
        추론 깊이 어댑터 순전파
        
        Args:
            context (torch.Tensor): 맥락 정보
            
        Returns:
            tuple: (추론 깊이, 복잡성 점수)
        """
        # 복잡성 추정
        complexity = self.estimate_complexity(context)
        
        # 깊이 분포 계산
        depth_distribution = self.depth_mapping(context)
        
        # 기대 깊이 계산
        expected_depth = 0
        for i in range(self.max_depth - self.min_depth + 1):
            expected_depth += (self.min_depth + i) * depth_distribution[:, i]
            
        return expected_depth, complexity


class MultiHypothesisTracker(nn.Module):
    """
    다중 가설 추적 및 결합을 위한 글로벌 메모리 시스템
    """
    
    def __init__(self, hidden_dim, max_hypotheses=4):
        """
        다중 가설 트래커 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_hypotheses (int): 최대 가설 수
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hypotheses = max_hypotheses
        
        # 가설 스코어링
        self.hypothesis_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 가설 업데이트 게이트
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 가설 결합기
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * max_hypotheses, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 글로벌 메모리 (가설 저장)
        self.register_buffer(
            'hypotheses',
            torch.zeros(1, max_hypotheses, hidden_dim)
        )
        self.register_buffer(
            'hypothesis_scores',
            torch.zeros(1, max_hypotheses)
        )
        
    def update_hypotheses(self, new_hypothesis, force_update=False):
        """
        새 가설 업데이트
        
        Args:
            new_hypothesis (torch.Tensor): 새 가설 상태
            force_update (bool): 강제 업데이트 여부
            
        Returns:
            tuple: (업데이트된 가설 목록, 업데이트된 점수 목록)
        """
        batch_size = new_hypothesis.shape[0]
        device = new_hypothesis.device
        
        # 배치별 처리
        updated_hypotheses = []
        updated_scores = []
        
        for b in range(batch_size):
            # 현재 인스턴스의 가설 및 점수
            current_hyp = self.hypotheses[0].clone()
            current_scores = self.hypothesis_scores[0].clone()
            
            # 새 가설 점수
            new_score = self.hypothesis_scorer(new_hypothesis[b]).item()
            
            # 가장 낮은 점수의 가설 찾기
            min_idx = torch.argmin(current_scores).item()
            
            # 업데이트 조건: 강제 업데이트 또는 새 점수가 더 높음
            if force_update or new_score > current_scores[min_idx]:
                # 현재 가설과 새 가설 비교
                similarities = torch.cosine_similarity(
                    new_hypothesis[b].unsqueeze(0),
                    current_hyp,
                    dim=1
                )
                
                # 가장 유사한 가설 찾기
                max_sim_idx = torch.argmax(similarities).item()
                max_sim = similarities[max_sim_idx].item()
                
                if max_sim > 0.8:  # 높은 유사도: 기존 가설 업데이트
                    # 게이트 계산
                    gate_input = torch.cat([
                        new_hypothesis[b],
                        current_hyp[max_sim_idx]
                    ]).unsqueeze(0)
                    gate = self.update_gate(gate_input).item()
                    
                    # 가설 업데이트
                    current_hyp[max_sim_idx] = (
                        gate * new_hypothesis[b] +
                        (1 - gate) * current_hyp[max_sim_idx]
                    )
                    
                    # 점수 업데이트
                    current_scores[max_sim_idx] = max(new_score, current_scores[max_sim_idx])
                else:  # 낮은 유사도: 가장 낮은 점수의 가설 교체
                    current_hyp[min_idx] = new_hypothesis[b]
                    current_scores[min_idx] = new_score
            
            # 결과 저장
            updated_hypotheses.append(current_hyp.unsqueeze(0))
            updated_scores.append(current_scores.unsqueeze(0))
            
        # 결과 결합
        updated_hypotheses = torch.cat(updated_hypotheses, dim=0)
        updated_scores = torch.cat(updated_scores, dim=0)
        
        return updated_hypotheses, updated_scores
    
    def combine_hypotheses(self, context=None):
        """
        가설 결합하여 최종 상태 생성
        
        Args:
            context (torch.Tensor, optional): 맥락 정보
            
        Returns:
            torch.Tensor: 결합된 가설 상태
        """
        batch_size = 1 if context is None else context.shape[0]
        device = self.hypotheses.device
        
        # 정규화된 가설 점수
        norm_scores = F.softmax(self.hypothesis_scores, dim=1)
        
        # 가설 가중 합계
        combined = torch.zeros(batch_size, self.hidden_dim, device=device)
        for b in range(batch_size):
            batch_hyp = self.hypotheses[0] if batch_size == 1 else self.hypotheses[b]
            batch_scores = norm_scores[0] if batch_size == 1 else norm_scores[b]
            
            # 점수에 따른 가중 평균
            weighted_sum = (batch_hyp * batch_scores.unsqueeze(1)).sum(dim=0)
            combined[b] = weighted_sum
            
        # 컨텍스트가 있는 경우 추가 가공
        if context is not None:
            # 모든 가설 연결
            all_hypotheses = self.hypotheses.expand(batch_size, -1, -1)
            flat_hypotheses = all_hypotheses.reshape(batch_size, -1)
            
            # 가설 결합기 적용
            combined = self.combiner(flat_hypotheses)
            
        return combined
    
    def forward(self, new_hypothesis=None, context=None, force_update=False):
        """
        다중 가설 트래커 순전파
        
        Args:
            new_hypothesis (torch.Tensor, optional): 새 가설 상태
            context (torch.Tensor, optional): 맥락 정보
            force_update (bool): 강제 업데이트 여부
            
        Returns:
            dict: 가설 트래킹 결과
        """
        if new_hypothesis is not None:
            # 가설 업데이트
            updated_hypotheses, updated_scores = self.update_hypotheses(
                new_hypothesis, force_update
            )
            
            # 클래스 변수 업데이트
            self.hypotheses = updated_hypotheses
            self.hypothesis_scores = updated_scores
            
        # 가설 결합
        combined = self.combine_hypotheses(context)
        
        return {
            'combined_hypothesis': combined,
            'hypotheses': self.hypotheses,
            'hypothesis_scores': self.hypothesis_scores
        }
