import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniversalLoss(nn.Module):
    """
    다양한 태스크 유형에 일관되게 적용 가능한 통합 손실 함수
    """
    
    def __init__(self, hidden_dim, task_types=None, alpha=0.5, beta=0.3, gamma=0.2):
        """
        통합 손실 함수 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            task_types (list, optional): 태스크 유형 목록
            alpha (float): 태스크 손실 가중치
            beta (float): 중첩 상태 정규화 가중치
            gamma (float): 일관성 손실 가중치
        """
        super().__init__()
        
        # 기본 태스크 유형 설정
        if task_types is None:
            task_types = ['classification', 'regression', 'generation', 'autoencoding']
        self.task_types = task_types
        
        # 가중치 설정
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 태스크별 헤드
        self.task_heads = nn.ModuleDict({
            'classification': nn.Linear(hidden_dim, 1),  # 분류용 (로짓)
            'regression': nn.Linear(hidden_dim, 1),      # 회귀용
            'generation': nn.Linear(hidden_dim, hidden_dim),  # 생성용
            'autoencoding': nn.Linear(hidden_dim, hidden_dim)  # 자동 인코딩용
        })
        
        # 태스크 가중치 예측기
        self.task_weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, len(task_types)),
            nn.Softmax(dim=-1)
        )
        
        # 중첩 상태 정규화 모듈
        self.superposition_regularizer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 일관성 손실 모듈
        self.consistency_loss_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def compute_task_loss(self, output, target, task_type, superposition_state=None):
        """
        태스크별 손실 계산
        
        Args:
            output (torch.Tensor): 모델 출력
            target (torch.Tensor): 목표 값
            task_type (str): 태스크 유형
            superposition_state (torch.Tensor, optional): 중첩 상태
            
        Returns:
            torch.Tensor: 계산된 태스크 손실
        """
        if task_type == 'classification':
            # 분류 태스크용 크로스 엔트로피 손실
            task_output = self.task_heads['classification'](output)
            loss = F.cross_entropy(task_output, target)
            
        elif task_type == 'regression':
            # 회귀 태스크용 MSE 손실
            task_output = self.task_heads['regression'](output)
            loss = F.mse_loss(task_output, target)
            
        elif task_type == 'generation':
            # 생성 태스크용 손실
            task_output = self.task_heads['generation'](output)
            
            # 다음 토큰 예측 (쉬프트된 타겟)
            shifted_target = target[:, 1:, :]
            shifted_output = task_output[:, :-1, :]
            
            loss = F.mse_loss(shifted_output, shifted_target)
            
        elif task_type == 'autoencoding':
            # 자동 인코딩 태스크용 손실
            task_output = self.task_heads['autoencoding'](output)
            loss = F.mse_loss(task_output, target)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return loss
    
    def compute_superposition_regularization(self, superposition_state, deterministic_state):
        """
        중첩 상태 정규화 손실 계산
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태
            deterministic_state (torch.Tensor): 확정 상태
            
        Returns:
            torch.Tensor: 중첩 상태 정규화 손실
        """
        batch_size, seq_len, _ = deterministic_state.shape
        
        # 중첩 상태 재구성
        superposition_mean = superposition_state.view(batch_size, seq_len, -1, deterministic_state.shape[2]).mean(dim=2)
        
        # 중첩과 확정 상태 결합
        combined = torch.cat([superposition_mean, deterministic_state], dim=-1)
        
        # 정규화 계수 계산
        regularization_factor = self.superposition_regularizer(combined)
        
        # 중첩 상태의 분산에 기반한 정규화
        superposition_variance = superposition_state.view(batch_size, seq_len, -1, deterministic_state.shape[2]).var(dim=2)
        
        # 정규화 손실: 너무 높거나 낮은 분산 페널티
        regularization_loss = ((superposition_variance - 0.5).abs() * regularization_factor).mean()
        
        return regularization_loss
    
    def compute_consistency_loss(self, deterministic_state, deterministic_states, collapsed_state=None):
        """
        일관성 손실 계산
        
        Args:
            deterministic_state (torch.Tensor): 현재 확정 상태
            deterministic_states (list): 이전 확정 상태 목록
            collapsed_state (torch.Tensor, optional): 붕괴된 상태
            
        Returns:
            torch.Tensor: 일관성 손실
        """
        if not deterministic_states:
            return torch.tensor(0.0, device=deterministic_state.device)
            
        # 이전 상태와의 일관성 손실
        consistency_losses = []
        
        for prev_state in deterministic_states:
            # 상태 결합
            combined = torch.cat([deterministic_state, prev_state], dim=-1)
            
            # 일관성 손실 추정
            inconsistency = self.consistency_loss_estimator(combined)
            
            # 일관성 손실 추가
            consistency_losses.append(inconsistency.mean())
            
        # 붕괴된 상태가 제공된 경우, 추가 일관성 계산
        if collapsed_state is not None:
            combined = torch.cat([deterministic_state, collapsed_state], dim=-1)
            collapse_inconsistency = self.consistency_loss_estimator(combined)
            consistency_losses.append(collapse_inconsistency.mean() * 2.0)  # 붕괴 일관성에 더 높은 가중치
            
        # 모든 일관성 손실의 평균
        avg_consistency_loss = torch.stack(consistency_losses).mean()
        
        return avg_consistency_loss
    
    def predict_task_weights(self, context_embedding):
        """
        문맥에 따른 태스크 가중치 예측
        
        Args:
            context_embedding (torch.Tensor): 문맥 임베딩
            
        Returns:
            torch.Tensor: 태스크별 가중치
        """
        return self.task_weight_predictor(context_embedding)
    
    def forward(self, outputs, targets, task_type=None, context_embedding=None, **kwargs):
        """
        통합 손실 함수 순전파
        
        Args:
            outputs (dict): 모델 출력 (확정 상태, 중첩 상태 등)
            targets (torch.Tensor): 목표 값
            task_type (str, optional): 태스크 유형
            context_embedding (torch.Tensor, optional): 문맥 임베딩
            **kwargs: 추가 인자
            
        Returns:
            dict: 손실 계산 결과
        """
        # 필요한 상태 추출
        deterministic_state = outputs.get('output', outputs.get('deterministic_state'))
        superposition_state = outputs.get('superposition_state')
        deterministic_states = outputs.get('deterministic_states', [])
        collapsed_state = outputs.get('collapsed_state')
        
        # 태스크 가중치 예측
        if context_embedding is not None:
            task_weights = self.predict_task_weights(context_embedding)
        else:
            # 기본 가중치는 균등 분포
            task_weights = torch.ones(len(self.task_types), device=deterministic_state.device)
            task_weights = task_weights / len(self.task_types)
            
        # 태스크 손실 계산
        task_losses = {}
        
        if task_type is not None:
            # 단일 태스크 유형이 지정된 경우
            task_losses[task_type] = self.compute_task_loss(
                deterministic_state, targets, task_type, superposition_state
            )
            total_task_loss = task_losses[task_type]
        else:
            # 모든 태스크 유형에 대해 가중 손실 계산
            for i, t_type in enumerate(self.task_types):
                task_losses[t_type] = self.compute_task_loss(
                    deterministic_state, targets, t_type, superposition_state
                )
                
                if i == 0:
                    total_task_loss = task_weights[i] * task_losses[t_type]
                else:
                    total_task_loss += task_weights[i] * task_losses[t_type]
        
        # 중첩 상태 정규화 손실 계산
        if superposition_state is not None:
            superposition_reg_loss = self.compute_superposition_regularization(
                superposition_state, deterministic_state
            )
        else:
            superposition_reg_loss = torch.tensor(0.0, device=deterministic_state.device)
            
        # 일관성 손실 계산
        consistency_loss = self.compute_consistency_loss(
            deterministic_state, deterministic_states, collapsed_state
        )
        
        # 총 손실 계산
        total_loss = (
            self.alpha * total_task_loss +
            self.beta * superposition_reg_loss +
            self.gamma * consistency_loss
        )
        
        return {
            'loss': total_loss,
            'task_loss': total_task_loss,
            'superposition_reg_loss': superposition_reg_loss,
            'consistency_loss': consistency_loss,
            'task_losses': task_losses,
            'task_weights': task_weights
        }


class MetaLearningOptimizer(nn.Module):
    """
    메타-학습을 통한 태스크 독립적 최적 중첩 패턴 발견
    """
    
    def __init__(self, model, hidden_dim, learning_rate=1e-4, meta_steps=3):
        """
        메타-학습 최적화기 초기화
        
        Args:
            model (nn.Module): 최적화 대상 모델
            hidden_dim (int): 기본 히든 차원
            learning_rate (float): 메타-학습률
            meta_steps (int): 메타 최적화 단계 수
        """
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.meta_steps = meta_steps
        
        # 메타-학습 컨트롤러
        self.meta_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 메타-파라미터 생성기
        self.meta_parameter_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 최적 중첩 패턴 메모리
        self.optimal_patterns = nn.Parameter(
            torch.randn(10, hidden_dim)  # 10개의 기본 패턴
        )
        
        # 패턴 가중치 생성기
        self.pattern_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, 10),
            nn.Softmax(dim=-1)
        )
        
    def generate_meta_parameters(self, context_embedding):
        """
        문맥에 따른 메타-파라미터 생성
        
        Args:
            context_embedding (torch.Tensor): 문맥 임베딩
            
        Returns:
            torch.Tensor: 메타-파라미터
        """
        return self.meta_parameter_generator(context_embedding)
    
    def select_optimal_pattern(self, context_embedding):
        """
        문맥에 따른 최적 중첩 패턴 선택
        
        Args:
            context_embedding (torch.Tensor): 문맥 임베딩
            
        Returns:
            torch.Tensor: 선택된 중첩 패턴
        """
        # 패턴 가중치 계산
        pattern_weights = self.pattern_weight_generator(context_embedding)
        
        # 가중 패턴 계산
        batch_size = context_embedding.shape[0]
        weighted_pattern = torch.zeros(batch_size, self.hidden_dim, device=context_embedding.device)
        
        for i in range(10):  # 10개의 기본 패턴
            pattern = self.optimal_patterns[i].unsqueeze(0).expand(batch_size, -1)
            weighted_pattern += pattern_weights[:, i].unsqueeze(1) * pattern
            
        return weighted_pattern
    
    def meta_optimization_step(self, inputs, targets, context_embedding=None, loss_fn=None):
        """
        메타-최적화 단계 수행
        
        Args:
            inputs (torch.Tensor): 입력 데이터
            targets (torch.Tensor): 목표 값
            context_embedding (torch.Tensor, optional): 문맥 임베딩
            loss_fn (callable, optional): 손실 함수
            
        Returns:
            dict: 메타-최적화 결과
        """
        # 기본 손실 함수
        if loss_fn is None:
            loss_fn = F.mse_loss
            
        # 문맥 임베딩이 없는 경우 생성
        if context_embedding is None:
            context_embedding = inputs.mean(dim=1)
            
        # 메타-파라미터 생성
        meta_params = self.generate_meta_parameters(context_embedding)
        
        # 최적 중첩 패턴 선택
        optimal_pattern = self.select_optimal_pattern(context_embedding)
        
        # 메타-최적화 결과 저장
        meta_results = {
            'meta_losses': [],
            'meta_grads': [],
            'intermediate_outputs': []
        }
        
        # 메타-학습 단계 수행
        for i in range(self.meta_steps):
            # 현재 메타-파라미터와 문맥 임베딩 결합
            combined_context = torch.cat([context_embedding, meta_params], dim=-1)
            
            # 메타-컨트롤러 적용
            meta_update = self.meta_controller(combined_context)
            
            # 메타-파라미터 업데이트
            meta_params = meta_params + self.learning_rate * meta_update
            
            # 임시 모델 생성 (원본 모델의 복사본)
            temp_model = type(self.model)(**vars(self.model))
            temp_model.load_state_dict(self.model.state_dict())
            
            # 임시 모델에 중첩 패턴 및 메타-파라미터 적용
            for name, param in temp_model.named_parameters():
                if 'dual_state' in name or 'superposition' in name:
                    # 중첩 관련 파라미터에 영향
                    param.data = param.data + 0.01 * torch.einsum('bh,...->b...', optimal_pattern, param.data)
            
            # 임시 모델로 추론
            temp_outputs = temp_model(inputs, context=meta_params)
            
            # 손실 계산
            if isinstance(temp_outputs, dict):
                output = temp_outputs.get('output', temp_outputs.get('deterministic_state'))
            else:
                output = temp_outputs
                
            meta_loss = loss_fn(output, targets)
            
            # 메타-그래디언트 계산
            meta_grad = torch.autograd.grad(meta_loss, meta_params, retain_graph=True)[0]
            
            # 결과 저장
            meta_results['meta_losses'].append(meta_loss.item())
            meta_results['meta_grads'].append(meta_grad.norm().item())
            meta_results['intermediate_outputs'].append(output)
            
            # 메타-파라미터 업데이트
            meta_params = meta_params - self.learning_rate * meta_grad
            
        return {
            'meta_results': meta_results,
            'optimal_pattern': optimal_pattern,
            'final_meta_params': meta_params
        }
    
    def forward(self, inputs, targets, context_embedding=None, loss_fn=None, apply_optimization=True):
        """
        메타-학습 최적화기 순전파
        
        Args:
            inputs (torch.Tensor): 입력 데이터
            targets (torch.Tensor): 목표 값
            context_embedding (torch.Tensor, optional): 문맥 임베딩
            loss_fn (callable, optional): 손실 함수
            apply_optimization (bool): 최적화 결과 적용 여부
            
        Returns:
            dict: 최적화 결과
        """
        # 메타-최적화 수행
        meta_optimization = self.meta_optimization_step(
            inputs, targets, context_embedding, loss_fn
        )
        
        if apply_optimization:
            # 최적화 결과를 실제 모델에 적용
            optimal_pattern = meta_optimization['optimal_pattern']
            meta_params = meta_optimization['final_meta_params']
            
            # 모델의 중첩 관련 파라미터 업데이트
            for name, param in self.model.named_parameters():
                if 'dual_state' in name or 'superposition' in name:
                    param.data = param.data + 0.001 * torch.einsum('bh,...->b...', optimal_pattern, param.data)
                    
            # 최적화된 모델로 추론
            optimized_outputs = self.model(inputs, context=meta_params)
            
            return {
                'meta_optimization': meta_optimization,
                'optimized_outputs': optimized_outputs
            }
        else:
            return {
                'meta_optimization': meta_optimization
            }
