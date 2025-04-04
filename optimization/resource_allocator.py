import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResourceAllocator(nn.Module):
    """
    자동 리소스 할당 시스템
    
    토큰, 문맥, 태스크 복잡성에 따라 계산 자원을 동적으로 분배
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, num_heads=8):
        """
        리소스 할당 시스템 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            num_heads (int): 어텐션 헤드 수
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.num_heads = num_heads
        
        # 토큰 복잡성 추정기
        self.token_complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 문맥 복잡성 추정기
        self.context_complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 태스크 복잡성 추정기
        self.task_complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 중첩 차원 할당기
        self.superposition_dim_allocator = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_superposition_dim),
            nn.Softmax(dim=-1)
        )
        
        # 계산 심도 할당기 (레이어별 집중도)
        self.computation_depth_allocator = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=-1)
        )
        
        # 리소스 효율성 컨트롤러
        self.efficiency_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def estimate_token_complexity(self, token_embeddings):
        """
        각 토큰의 복잡성 추정
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: 토큰 복잡성 점수 [batch_size, seq_len, 1]
        """
        return self.token_complexity_estimator(token_embeddings)
        
    def estimate_context_complexity(self, context_embedding):
        """
        전체 문맥의 복잡성 추정
        
        Args:
            context_embedding (torch.Tensor): 문맥 임베딩 [batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 문맥 복잡성 점수 [batch_size, 1]
        """
        return self.context_complexity_estimator(context_embedding)
        
    def estimate_task_complexity(self, context_embedding):
        """
        태스크의 복잡성 추정
        
        Args:
            context_embedding (torch.Tensor): 문맥 임베딩 [batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 태스크 복잡성 점수 [batch_size, 1]
        """
        return self.task_complexity_estimator(context_embedding)
        
    def allocate_superposition_dims(self, token_embeddings, token_complexity, context_complexity, task_complexity):
        """
        각 토큰에 중첩 차원 할당
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            token_complexity (torch.Tensor): 토큰 복잡성 점수 [batch_size, seq_len, 1]
            context_complexity (torch.Tensor): 문맥 복잡성 점수 [batch_size, 1]
            task_complexity (torch.Tensor): 태스크 복잡성 점수 [batch_size, 1]
            
        Returns:
            torch.Tensor: 중첩 차원 할당 가중치 [batch_size, seq_len, max_superposition_dim]
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 문맥 및 태스크 복잡성 확장
        context_complexity = context_complexity.expand(batch_size, seq_len, 1)
        task_complexity = task_complexity.expand(batch_size, seq_len, 1)
        
        # 모든 복잡성 정보 결합
        combined_features = torch.cat([
            token_embeddings,
            token_complexity,
            context_complexity,
            task_complexity
        ], dim=-1)
        
        # 중첩 차원 할당
        dim_weights = self.superposition_dim_allocator(combined_features)
        
        return dim_weights
        
    def allocate_computation_depth(self, token_embeddings, token_complexity, context_complexity, task_complexity):
        """
        계산 심도 할당 (어텐션 헤드별 가중치)
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            token_complexity (torch.Tensor): 토큰 복잡성 점수 [batch_size, seq_len, 1]
            context_complexity (torch.Tensor): 문맥 복잡성 점수 [batch_size, 1]
            task_complexity (torch.Tensor): 태스크 복잡성 점수 [batch_size, 1]
            
        Returns:
            torch.Tensor: 계산 심도 할당 가중치 [batch_size, seq_len, num_heads]
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 문맥 및 태스크 복잡성 확장
        context_complexity = context_complexity.expand(batch_size, seq_len, 1)
        task_complexity = task_complexity.expand(batch_size, seq_len, 1)
        
        # 모든 복잡성 정보 결합
        combined_features = torch.cat([
            token_embeddings,
            token_complexity,
            context_complexity,
            task_complexity
        ], dim=-1)
        
        # 계산 심도 할당
        depth_weights = self.computation_depth_allocator(combined_features)
        
        return depth_weights
        
    def optimize_resource_efficiency(self, token_embeddings):
        """
        리소스 효율성 최적화 비율 계산
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: 리소스 효율성 비율 [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 시퀀스 내 모든 토큰에 대해 효율성 비율 계산
        efficiency_ratios = []
        for i in range(seq_len):
            ratio = self.efficiency_controller(token_embeddings[:, i])
            efficiency_ratios.append(ratio)
            
        # 효율성 비율 스택
        efficiency_ratios = torch.stack(efficiency_ratios, dim=1)
        
        return efficiency_ratios
        
    def create_computation_mask(self, superposition_weights, efficiency_ratio, threshold=0.2):
        """
        계산 마스크 생성 (효율성을 위한 희소 활성화)
        
        Args:
            superposition_weights (torch.Tensor): 중첩 차원 할당 가중치
            efficiency_ratio (torch.Tensor): 리소스 효율성 비율
            threshold (float): 마스크 임계값
            
        Returns:
            torch.Tensor: 계산 마스크
        """
        # 가중치가 높은 차원만 활성화하는 희소 마스크 생성
        sparse_mask = (superposition_weights > threshold).float()
        
        # 효율성 비율에 따른 마스크 조정
        adjusted_mask = sparse_mask * efficiency_ratio
        
        return adjusted_mask
    
    def forward(self, token_embeddings, context_embedding=None):
        """
        자동 리소스 할당 시스템 순전파
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩 [batch_size, seq_len, hidden_dim]
            context_embedding (torch.Tensor, optional): 문맥 임베딩 [batch_size, hidden_dim]
            
        Returns:
            dict: 리소스 할당 결과
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 문맥 임베딩이 제공되지 않은 경우, 토큰 임베딩의 평균 사용
        if context_embedding is None:
            context_embedding = token_embeddings.mean(dim=1)
            
        # 복잡성 추정
        token_complexity = self.estimate_token_complexity(token_embeddings)
        context_complexity = self.estimate_context_complexity(context_embedding)
        task_complexity = self.estimate_task_complexity(context_embedding)
        
        # 중첩 차원 및 계산 심도 할당
        superposition_weights = self.allocate_superposition_dims(
            token_embeddings, token_complexity, context_complexity, task_complexity
        )
        computation_depth = self.allocate_computation_depth(
            token_embeddings, token_complexity, context_complexity, task_complexity
        )
        
        # 리소스 효율성 최적화
        efficiency_ratio = self.optimize_resource_efficiency(token_embeddings)
        
        # 계산 마스크 생성
        computation_mask = self.create_computation_mask(superposition_weights, efficiency_ratio)
        
        return {
            'token_complexity': token_complexity,
            'context_complexity': context_complexity,
            'task_complexity': task_complexity,
            'superposition_weights': superposition_weights,
            'computation_depth': computation_depth,
            'efficiency_ratio': efficiency_ratio,
            'computation_mask': computation_mask
        }


class DynamicComputeEngine(nn.Module):
    """
    동적 계산 엔진
    
    리소스 할당 시스템의 지시에 따라 계산 리소스를 효율적으로 분배
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4, num_heads=8):
        """
        동적 계산 엔진 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            num_heads (int): 어텐션 헤드 수
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        self.num_heads = num_heads
        
        # 리소스 할당 시스템
        self.resource_allocator = ResourceAllocator(
            hidden_dim=hidden_dim,
            max_superposition_dim=max_superposition_dim,
            num_heads=num_heads
        )
        
        # 동적 계산 행렬
        self.computation_matrix = nn.Parameter(
            torch.randn(num_heads, max_superposition_dim, hidden_dim, hidden_dim)
        )
        
        # 희소 어텐션 모듈
        self.sparse_attention = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(max_superposition_dim)
        ])
        
        # 게이팅 메커니즘
        self.gate_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def apply_dynamic_computation(self, token_embeddings, superposition_weights, computation_mask):
        """
        동적 계산 적용
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩
            superposition_weights (torch.Tensor): 중첩 차원 할당 가중치
            computation_mask (torch.Tensor): 계산 마스크
            
        Returns:
            torch.Tensor: 처리된 임베딩
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 결과 저장 텐서 초기화
        processed_embeddings = torch.zeros_like(token_embeddings)
        
        # 각 중첩 차원에 대해 처리
        for dim in range(self.max_superposition_dim):
            # 현재 차원의 가중치 및 마스크
            dim_weight = superposition_weights[:, :, dim].unsqueeze(-1)
            dim_mask = computation_mask[:, :, dim].unsqueeze(-1)
            
            # 희소 어텐션 적용
            attended = self.sparse_attention[dim](token_embeddings)
            
            # 가중치 및 마스크 적용
            processed = attended * dim_weight * dim_mask
            
            # 결과에 추가
            processed_embeddings += processed
            
        return processed_embeddings
    
    def forward(self, token_embeddings, context_embedding=None):
        """
        동적 계산 엔진 순전파
        
        Args:
            token_embeddings (torch.Tensor): 토큰 임베딩
            context_embedding (torch.Tensor, optional): 문맥 임베딩
            
        Returns:
            dict: 동적 계산 결과
        """
        # 리소스 할당
        allocation_result = self.resource_allocator(token_embeddings, context_embedding)
        
        # 동적 계산 적용
        processed_embeddings = self.apply_dynamic_computation(
            token_embeddings,
            allocation_result['superposition_weights'],
            allocation_result['computation_mask']
        )
        
        # 입력과 처리된 임베딩 결합
        combined = torch.cat([token_embeddings, processed_embeddings], dim=-1)
        
        # 게이트 계산
        gate = self.gate_controller(combined)
        
        # 게이트 적용한 최종 출력
        output = gate * processed_embeddings + (1 - gate) * token_embeddings
        
        return {
            'output': output,
            'allocation_result': allocation_result,
            'gate': gate
        }
