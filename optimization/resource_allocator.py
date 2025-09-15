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


    
