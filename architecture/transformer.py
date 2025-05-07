import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, List, Dict, Tuple, Union, Any

from ..core.dual_state import DualStateRepresentation, DualStateController
from ..core.state_management import GlobalStateManager, HierarchicalStateProtocol
from ..core.collapse import StateCollapseFramework, DynamicCollapseController, CollapseGate
from ..core.inference_engine import InferenceEngine, ReasoningDepthAdapter, MultiHypothesisTracker
from .attention import QuantumInspiredAttention
from .position_encoding import PositionalEncoding, QuantumPositionalEncoding
from .feed_forward import FeedForward, DualStateFeedForward
from .integrated_layer import IntegratedTransformerLayer, IntegratedDecoderLayer


class QuantumInspiredTransformerEncoder(nn.Module):
    """
    양자 영감 트랜스포머 인코더
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_superposition_dim: int = 4,
        activation: str = "gelu",
        gate_type: str = "mlp"  # "mlp" 또는 "transformer"
    ):
        """
        양자 영감 트랜스포머 인코더 초기화
        
        Args:
            d_model (int): 모델 차원
            nhead (int): 어텐션 헤드 수
            num_layers (int): 인코더 레이어 수
            dim_feedforward (int): 피드포워드 네트워크 차원
            dropout (float): 드롭아웃 비율
            max_superposition_dim (int): 최대 중첩 상태 차원
            activation (str): 활성화 함수 유형
            gate_type (str): CollapseGate 유형
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_superposition_dim = max_superposition_dim
        self.gate_type = gate_type
        
        # 이중 상태 표현 시스템
        self.dual_state_system = DualStateRepresentation(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 이중 상태 컨트롤러
        self.state_controller = DualStateController(hidden_dim=d_model)
        
        # 위치 인코딩
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.quantum_position_encoding = QuantumPositionalEncoding(
            d_model=d_model, 
            max_superposition_dim=max_superposition_dim, 
            dropout=dropout
        )
        
        # 인코더 레이어
        self.layers = nn.ModuleList([
            IntegratedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_superposition_dim=max_superposition_dim,
                activation=activation,
                layer_id=i,  # 레이어 ID 추가
                num_layers=num_layers,  # 총 레이어 수 추가
                gate_type=gate_type  # CollapseGate 유형 추가
            ) for i in range(num_layers)
        ])
        
        # 글로벌 상태 관리 엔진
        self.state_manager = GlobalStateManager(
            hidden_dim=d_model,
            num_layers=num_layers,
            max_superposition_dim=max_superposition_dim
        )
        
        # 계층적 상태 프로토콜
        self.hierarchical_protocol = HierarchicalStateProtocol(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 상태 붕괴 프레임워크
        self.collapse_framework = StateCollapseFramework(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 동적 붕괴 컨트롤러
        self.collapse_controller = DynamicCollapseController(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 통합 추론 엔진
        self.inference_engine = InferenceEngine(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 다중 가설 트래커
        self.hypothesis_tracker = MultiHypothesisTracker(
            hidden_dim=d_model
        )
        
        # 정규화 레이어
        self.norm = nn.LayerNorm(d_model)
        
        # 초기화
        self._reset_parameters()
        
    def _reset_parameters(self):
        """
        모델 파라미터 초기화
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_all_states: bool = False,
        force_collapse: bool = False,
        p_target: float = 0.5,  # 목표 전환 확률 추가
        superposition_degree: Optional[float] = None,
        collapse_threshold: Optional[float] = None,
        interference_strength: Optional[float] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        양자 영감 트랜스포머 인코더 순전파
        
        Args:
            src (torch.Tensor): 소스 시퀀스 [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): 어텐션 마스크
            src_key_padding_mask (torch.Tensor, optional): 소스 패딩 마스크
            context (torch.Tensor, optional): 컨텍스트 정보
            return_all_states (bool): 모든 상태 반환 여부
            force_collapse (bool): 강제 상태 붕괴 여부
            p_target (float): 목표 전환 확률
            superposition_degree (float, optional): 중첩 정도 (0~1)
            collapse_threshold (float, optional): 붕괴 임계값 (0~1)
            interference_strength (float, optional): 간섭 강도 (0~1)
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 인코딩 결과
        """
        batch_size, seq_len, _ = src.shape
        
        # 위치 인코딩 적용
        src = self.position_encoding(src)
        
        # 컨텍스트가 없는 경우 초기 컨텍스트 생성
        if context is None:
            context = src.mean(dim=1)  # [batch_size, d_model]
            
        # 이중 상태 컨트롤러로 중첩 파라미터 계산
        if any(param is None for param in [superposition_degree, collapse_threshold, interference_strength]):
            controller_params = self.state_controller(context)
            if superposition_degree is None:
                superposition_degree = controller_params['superposition_degree']
            if collapse_threshold is None:
                collapse_threshold = controller_params['collapse_threshold']
            if interference_strength is None:
                interference_strength = controller_params['interference_strength']
        
        # 초기 상태를 이중 상태로 변환
        superposition_state = self.dual_state_system.to_superposition_state(src, context)
        # 양자 위치 인코딩 적용
        superposition_state = self.quantum_position_encoding(superposition_state, is_superposition=True)
        deterministic_state = src
        
        # 레이어별 상태 저장
        layer_superposition_states = []
        layer_deterministic_states = []
        
        # 이전 상태 및 히든 상태 초기화
        previous_state = superposition_state.clone()
        hidden_state = torch.zeros(batch_size, seq_len, self.d_model, device=src.device)
        
        # 불확실성, 전환 확률, 리소스 효율성 추적
        uncertainties = []
        transition_probs = []
        resource_efficiencies = []
        
        # 순차적으로 레이어 처리
        for i, layer in enumerate(self.layers):
            # 레이어 결과 계산
            layer_result = layer(
                deterministic_state=deterministic_state,
                superposition_state=superposition_state,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                context=context,
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength,
                p_target=p_target  # 목표 전환 확률 전달
            )
            
            # 레이어 결과 업데이트
            deterministic_state = layer_result['deterministic_state']
            superposition_state = layer_result['superposition_state']
            
            # 불확실성 정보 저장
            if 'uncertainty' in layer_result:
                uncertainties.append(layer_result['uncertainty'])
            
            # 전환 확률 정보 저장
            if 'transition_prob' in layer_result:
                transition_probs.append(layer_result['transition_prob'])
                
            # 리소스 효율성 정보 저장
            if 'resource_efficiency' in layer_result:
                resource_efficiencies.append(layer_result.get('resource_efficiency', torch.tensor(0.0)))
            
            # 레이어 상태 저장
            layer_deterministic_states.append(deterministic_state)
            layer_superposition_states.append(superposition_state)
            
            # 필요한 경우 중간 상태 붕괴
            if (i + 1) % 4 == 0 and i < self.num_layers - 1:
                # 상태 붕괴 프레임워크 적용
                collapse_result = self.collapse_framework(
                    superposition_state=superposition_state,
                    context=context,
                    previous_state=previous_state,
                    hidden_state=hidden_state,
                    force_collapse=False,  # 중간 레이어는 조건부 붕괴
                    layer_id=i,            # 레이어 ID 전달
                    num_layers=self.num_layers,  # 총 레이어 수 전달
                    p_target=p_target      # 목표 전환 확률 전달
                )
                
                # 붕괴 조건이 충족된 위치만 업데이트
                collapse_mask = collapse_result['collapse_condition']
                for b in range(batch_size):
                    for s in range(seq_len):
                        if collapse_mask[b, s, 0]:
                            deterministic_state[b, s] = collapse_result['collapsed_state'][b, s]
                
                # 상태 및 히든 상태 업데이트
                previous_state = superposition_state.clone()
                hidden_state = collapse_result['hidden_state']
                
                # 붕괴 이후 새 중첩 상태 생성
                superposition_state = self.dual_state_system.to_superposition_state(
                    deterministic_state, context
                )
        
        # 글로벌 상태 관리 적용
        managed_superposition_states = self.state_manager(layer_superposition_states)
        
        # 계층적 상태 프로토콜 적용 (마지막 상태에만)
        hierarchical_state = self.hierarchical_protocol(managed_superposition_states[-1])
        
        # 최종 상태 붕괴 (강제 또는 마지막 레이어)
        if force_collapse:
            # 동적 붕괴 컨트롤러 적용
            collapse_result = self.collapse_controller(
                superposition_state=hierarchical_state,
                deterministic_state=layer_deterministic_states[-1],
                context=context,
                p_target=p_target  # 목표 전환 확률 전달
            )
            final_deterministic = collapse_result['collapsed_state']
        else:
            # 표준 붕괴 프레임워크 적용
            collapse_result = self.collapse_framework(
                superposition_state=hierarchical_state,
                context=context,
                previous_state=previous_state,
                hidden_state=hidden_state,
                force_collapse=True,  # 마지막 레이어는 강제 붕괴
                layer_id=self.num_layers-1,  # 레이어 ID 전달
                num_layers=self.num_layers,  # 총 레이어 수 전달
                p_target=p_target  # 목표 전환 확률 전달
            )
            final_deterministic = collapse_result['collapsed_state']
        
        # 정규화 적용
        output = self.norm(final_deterministic)
        
        # 통합 추론 엔진 적용
        inference_result = self.inference_engine(
            deterministic_state=output,
            superposition_state=hierarchical_state,
            context=context
        )
        
        # 다중 가설 트래킹 업데이트
        hypothesis_result = self.hypothesis_tracker(
            new_hypothesis=inference_result['result'].mean(dim=1),
            context=context
        )
        
        if return_all_states:
            # 평균 불확실성, 전환 확률, 리소스 효율성 계산
            avg_uncertainty = torch.cat(uncertainties).mean() if uncertainties else torch.tensor(0.0)
            avg_transition_prob = torch.cat(transition_probs).mean() if transition_probs else torch.tensor(0.0)
            avg_resource_efficiency = torch.tensor(
                sum([eff.item() for eff in resource_efficiencies]) / len(resource_efficiencies)
                if resource_efficiencies else 0.0
            )
            
            return {
                'output': output,
                'inference_result': inference_result['result'],
                'superposition_state': hierarchical_state,
                'deterministic_states': layer_deterministic_states,
                'superposition_states': managed_superposition_states,
                'collapse_result': collapse_result,
                'hypothesis': hypothesis_result['combined_hypothesis'],
                'uncertainty': avg_uncertainty,  # 평균 불확실성 추가
                'transition_prob': avg_transition_prob,  # 평균 전환 확률 추가
                'resource_efficiency': avg_resource_efficiency,  # 평균 리소스 효율성 추가
                'gate_result': collapse_result.get('gate_result', {})  # CollapseGate 결과 추가
            }
        else:
            return output


class QuantumInspiredTransformerDecoder(nn.Module):
    """
    양자 영감 트랜스포머 디코더
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_superposition_dim: int = 4,
        activation: str = "gelu",
        gate_type: str = "mlp"  # "mlp" 또는 "transformer"
    ):
        """
        양자 영감 트랜스포머 디코더 초기화
        
        Args:
            d_model (int): 모델 차원
            nhead (int): 어텐션 헤드 수
            num_layers (int): 디코더 레이어 수
            dim_feedforward (int): 피드포워드 네트워크 차원
            dropout (float): 드롭아웃 비율
            max_superposition_dim (int): 최대 중첩 상태 차원
            activation (str): 활성화 함수 유형
            gate_type (str): CollapseGate 유형
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_superposition_dim = max_superposition_dim
        self.gate_type = gate_type
        
        # 이중 상태 표현 시스템
        self.dual_state_system = DualStateRepresentation(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 이중 상태 컨트롤러
        self.state_controller = DualStateController(hidden_dim=d_model)
        
        # 위치 인코딩
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.quantum_position_encoding = QuantumPositionalEncoding(
            d_model=d_model, 
            max_superposition_dim=max_superposition_dim, 
            dropout=dropout
        )
        
        # 디코더 레이어
        self.layers = nn.ModuleList([
            IntegratedDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_superposition_dim=max_superposition_dim,
                activation=activation,
                layer_id=i,  # 레이어 ID 추가
                num_layers=num_layers,  # 총 레이어 수 추가
                gate_type=gate_type  # CollapseGate 유형 추가
            ) for i in range(num_layers)
        ])
        
        # 글로벌 상태 관리 엔진
        self.state_manager = GlobalStateManager(
            hidden_dim=d_model,
            num_layers=num_layers,
            max_superposition_dim=max_superposition_dim
        )
        
        # 계층적 상태 프로토콜
        self.hierarchical_protocol = HierarchicalStateProtocol(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 상태 붕괴 프레임워크
        self.collapse_framework = StateCollapseFramework(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 동적 붕괴 컨트롤러
        self.collapse_controller = DynamicCollapseController(
            hidden_dim=d_model,
            max_superposition_dim=max_superposition_dim
        )
        
        # 추론 깊이 어댑터
        self.reasoning_depth_adapter = ReasoningDepthAdapter(hidden_dim=d_model)
        
        # 정규화 레이어
        self.norm = nn.LayerNorm(d_model)
        
        # 초기화
        self._reset_parameters()
        
    def _reset_parameters(self):
        """
        모델 파라미터 초기화
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        memory_superposition: Optional[torch.Tensor] = None,
        return_all_states: bool = False,
        force_collapse: bool = False,
        p_target: float = 0.5,  # 목표 전환 확률 추가
        superposition_degree: Optional[float] = None,
        collapse_threshold: Optional[float] = None,
        interference_strength: Optional[float] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        양자 영감 트랜스포머 디코더 순전파
        
        Args:
            tgt (torch.Tensor): 타겟 시퀀스 [batch_size, seq_len, d_model]
            memory (torch.Tensor): 인코더 메모리 [batch_size, src_len, d_model]
            tgt_mask (torch.Tensor, optional): 타겟 어텐션 마스크
            memory_mask (torch.Tensor, optional): 메모리 어텐션 마스크
            tgt_key_padding_mask (torch.Tensor, optional): 타겟 패딩 마스크
            memory_key_padding_mask (torch.Tensor, optional): 메모리 패딩 마스크
            context (torch.Tensor, optional): 컨텍스트 정보
            memory_superposition (torch.Tensor, optional): 인코더 중첩 상태
            return_all_states (bool): 모든 상태 반환 여부
            force_collapse (bool): 강제 상태 붕괴 여부
            p_target (float): 목표 전환 확률
            superposition_degree (float, optional): 중첩 정도 (0~1)
            collapse_threshold (float, optional): 붕괴 임계값 (0~1)
            interference_strength (float, optional): 간섭 강도 (0~1)
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 디코딩 결과
        """
        batch_size, seq_len, _ = tgt.shape
        
        # 위치 인코딩 적용
        tgt = self.position_encoding(tgt)
        
        # 컨텍스트가 없는 경우 메모리와 타겟에서 생성
        if context is None:
            memory_context = memory.mean(dim=1)  # [batch_size, d_model]
            tgt_context = tgt.mean(dim=1)  # [batch_size, d_model]
            context = (memory_context + tgt_context) / 2
            
        # 이중 상태 컨트롤러로 중첩 파라미터 계산
        if any(param is None for param in [superposition_degree, collapse_threshold, interference_strength]):
            controller_params = self.state_controller(context)
            if superposition_degree is None:
                superposition_degree = controller_params['superposition_degree']
            if collapse_threshold is None:
                collapse_threshold = controller_params['collapse_threshold']
            if interference_strength is None:
                interference_strength = controller_params['interference_strength']
        
        # 초기 상태를 이중 상태로 변환
        superposition_state = self.dual_state_system.to_superposition_state(tgt, context)
        # 양자 위치 인코딩 적용
        superposition_state = self.quantum_position_encoding(superposition_state, is_superposition=True)
        deterministic_state = tgt
        
        # 레이어별 상태 저장
        layer_superposition_states = []
        layer_deterministic_states = []
        
        # 이전 상태 및 히든 상태 초기화
        previous_state = superposition_state.clone()
        hidden_state = torch.zeros(batch_size, seq_len, self.d_model, device=tgt.device)
        
        # 레이어별 불확실성, 전환 확률, 리소스 효율성 저장
        uncertainties = []
        transition_probs = []
        resource_efficiencies = []
        
        # 순차적으로 레이어 처리
        for i, layer in enumerate(self.layers):
            # 레이어 결과 계산
            layer_result = layer(
                deterministic_state=deterministic_state,
                superposition_state=superposition_state,
                memory=memory,
                memory_superposition=memory_superposition,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                context=context,
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength,
                p_target=p_target  # 목표 전환 확률 전달
            )
            
            # 레이어 결과 업데이트
            deterministic_state = layer_result['deterministic_state']
            superposition_state = layer_result['superposition_state']
            
            # 불확실성 정보 저장
            if 'uncertainty' in layer_result:
                uncertainties.append(layer_result['uncertainty'])
            
            # 전환 확률 정보 저장
            if 'transition_prob' in layer_result:
                transition_probs.append(layer_result['transition_prob'])
                
            # 리소스 효율성 정보 저장
            if 'resource_efficiency' in layer_result:
                resource_efficiencies.append(layer_result.get('resource_efficiency', torch.tensor(0.0)))
            
            # 레이어 상태 저장
            layer_deterministic_states.append(deterministic_state)
            layer_superposition_states.append(superposition_state)
            
            # 필요한 경우 중간 상태 붕괴
            if (i + 1) % 4 == 0 and i < self.num_layers - 1:
                # 상태 붕괴 프레임워크 적용
                collapse_result = self.collapse_framework(
                    superposition_state=superposition_state,
                    context=context,
                    previous_state=previous_state,
                    hidden_state=hidden_state,
                    force_collapse=False,  # 중간 레이어는 조건부 붕괴
                    layer_id=i,            # 레이어 ID 전달
                    num_layers=self.num_layers,  # 총 레이어 수 전달
                    p_target=p_target      # 목표 전환 확률 전달
                )
                
                # 붕괴 조건이 충족된 위치만 업데이트
                collapse_mask = collapse_result['collapse_condition']
                for b in range(batch_size):
                    for s in range(seq_len):
                        if collapse_mask[b, s, 0]:
                            deterministic_state[b, s] = collapse_result['collapsed_state'][b, s]
                
                # 상태 및 히든 상태 업데이트
                previous_state = superposition_state.clone()
                hidden_state = collapse_result['hidden_state']
                
                # 붕괴 이후 새 중첩 상태 생성
                superposition_state = self.dual_state_system.to_superposition_state(
                    deterministic_state, context
                )
        
        # 글로벌 상태 관리 적용
        managed_superposition_states = self.state_manager(layer_superposition_states)
        
        # 계층적 상태 프로토콜 적용 (마지막 상태에만)
        hierarchical_state = self.hierarchical_protocol(managed_superposition_states[-1])
        
        # 최종 상태 붕괴 (강제 또는 마지막 레이어)
        if force_collapse:
            # 동적 붕괴 컨트롤러 적용
            collapse_result = self.collapse_controller(
                superposition_state=hierarchical_state,
                deterministic_state=layer_deterministic_states[-1],
                context=context,
                p_target=p_target  # 목표 전환 확률 전달
            )
            final_deterministic = collapse_result['collapsed_state']
        else:
            # 표준 붕괴 프레임워크 적용
            collapse_result = self.collapse_framework(
                superposition_state=hierarchical_state,
                context=context,
                previous_state=previous_state,
                hidden_state=hidden_state,
                force_collapse=True,  # 마지막 레이어는 강제 붕괴
                layer_id=self.num_layers-1,  # 레이어 ID 전달
                num_layers=self.num_layers,  # 총 레이어 수 전달
                p_target=p_target  # 목표 전환 확률 전달
            )
            final_deterministic = collapse_result['collapsed_state']
        
        # 정규화 적용
        output = self.norm(final_deterministic)
        
        # 추론 깊이 계산
        reasoning_depth, complexity = self.reasoning_depth_adapter(context)
        
        if return_all_states:
            # 평균 불확실성, 전환 확률, 리소스 효율성 계산
            avg_uncertainty = torch.cat(uncertainties).mean() if uncertainties else torch.tensor(0.0)
            avg_transition_prob = torch.cat(transition_probs).mean() if transition_probs else torch.tensor(0.0)
            avg_resource_efficiency = torch.tensor(
                sum([eff.item() for eff in resource_efficiencies]) / len(resource_efficiencies)
                if resource_efficiencies else 0.0
            )
            
            return {
                'output': output,
                'superposition_state': hierarchical_state,
                'deterministic_states': layer_deterministic_states,
                'superposition_states': managed_superposition_states,
                'collapse_result': collapse_result,
                'reasoning_depth': reasoning_depth,
                'complexity': complexity,
                'uncertainty': avg_uncertainty,  # 평균 불확실성 추가
                'transition_prob': avg_transition_prob,  # 평균 전환 확률 추가
                'resource_efficiency': avg_resource_efficiency,  # 평균 리소스 효율성 추가
                'gate_result': collapse_result.get('gate_result', {})  # CollapseGate 결과 추가
            }
        else:
            return output


class QuantumInspiredTransformer(nn.Module):
    """
    최적화된 하이브리드 양자 영감 트랜스포머 통합 모델
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_superposition_dim: int = 4,
        activation: str = "gelu",
        gate_type: str = "mlp",  # CollapseGate 유형 추가
        custom_encoder: Optional[nn.Module] = None,
        custom_decoder: Optional[nn.Module] = None,
        vocab_size: Optional[int] = None,
        pad_token_id: int = 0
    ):
        """
        양자 영감 트랜스포머 모델 초기화
        
        Args:
            d_model (int): 모델 차원
            nhead (int): 어텐션 헤드 수
            num_encoder_layers (int): 인코더 레이어 수
            num_decoder_layers (int): 디코더 레이어 수
            dim_feedforward (int): 피드포워드 네트워크 차원
            dropout (float): 드롭아웃 비율
            max_superposition_dim (int): 최대 중첩 상태 차원
            activation (str): 활성화 함수 유형
            gate_type (str): CollapseGate 유형
            custom_encoder (nn.Module, optional): 사용자 정의 인코더
            custom_decoder (nn.Module, optional): 사용자 정의 디코더
            vocab_size (int, optional): 어휘 크기 (임베딩 레이어 사용 시 필요)
            pad_token_id (int): 패딩 토큰 ID
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_superposition_dim = max_superposition_dim
        self.gate_type = gate_type
        self.pad_token_id = pad_token_id
        
        # 입력 임베딩 레이어 (vocab_size가 제공된 경우)
        self.has_embeddings = vocab_size is not None
        if self.has_embeddings:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
            self.embedding_scale = math.sqrt(d_model)
        
        # 인코더 생성
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = QuantumInspiredTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_superposition_dim=max_superposition_dim,
                activation=activation,
                gate_type=gate_type
            )
            
        # 디코더 생성
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            self.decoder = QuantumInspiredTransformerDecoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_superposition_dim=max_superposition_dim,
                activation=activation,
                gate_type=gate_type
            )
        
        # 출력 선형 레이어 (vocab_size가 제공된 경우)
        if self.has_embeddings:
            self.output_projection = nn.Linear(d_model, vocab_size)
            # 가중치 공유 (임베딩과 출력 프로젝션)
            self.output_projection.weight = self.embedding.weight
            
        # 초기화
        self._reset_parameters()
        
    def _reset_parameters(self):
        """
        모델 파라미터 초기화
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _prepare_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor]
    ]:
        """
        마스크 준비
        
        Args:
            src: 소스 시퀀스
            tgt: 타겟 시퀀스
            src_key_padding_mask: 소스 패딩 마스크
            tgt_key_padding_mask: 타겟 패딩 마스크
            
        Returns:
            Tuple: (src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask)
        """
        # 소스 패딩 마스크가 없는 경우 생성
        if src_key_padding_mask is None and self.has_embeddings:
            src_key_padding_mask = (src == self.pad_token_id)
        
        # 타겟 패딩 마스크가 없는 경우 생성
        if tgt_key_padding_mask is None and self.has_embeddings:
            tgt_key_padding_mask = (tgt == self.pad_token_id)
        
        # 소스/메모리 마스크 (일반적으로 None)
        src_mask = None
        memory_mask = None
        
        # 타겟 마스크 (casual attention)
        tgt_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        return src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        후속 마스크 생성 (디코더 self-attention에서 사용)
        
        Args:
            sz: 시퀀스 길이
            
        Returns:
            torch.Tensor: 후속 마스크
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _prepare_inputs(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력 준비
        
        Args:
            src: 소스 시퀀스
            tgt: 타겟 시퀀스
            
        Returns:
            Tuple: (src_emb, tgt_emb)
        """
        # 임베딩이 있는 경우 적용
        if self.has_embeddings:
            if src.dim() == 2:  # [batch_size, seq_len]
                src_emb = self.embedding(src) * self.embedding_scale
            else:  # [batch_size, seq_len, d_model]
                src_emb = src
                
            if tgt.dim() == 2:  # [batch_size, seq_len]
                tgt_emb = self.embedding(tgt) * self.embedding_scale
            else:  # [batch_size, seq_len, d_model]
                tgt_emb = tgt
        else:
            src_emb = src
            tgt_emb = tgt
            
        return src_emb, tgt_emb
                
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_all_states: bool = False,
        force_collapse: bool = False,
        p_target: float = 0.5,  # 목표 전환 확률 추가
        superposition_degree: Optional[float] = None,
        collapse_threshold: Optional[float] = None,
        interference_strength: Optional[float] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        양자 영감 트랜스포머 순전파
        
        Args:
            src (torch.Tensor): 소스 시퀀스 [batch_size, src_len, d_model] 또는 [batch_size, src_len]
            tgt (torch.Tensor): 타겟 시퀀스 [batch_size, tgt_len, d_model] 또는 [batch_size, tgt_len]
            src_mask (torch.Tensor, optional): 소스 어텐션 마스크
            tgt_mask (torch.Tensor, optional): 타겟 어텐션 마스크
            memory_mask (torch.Tensor, optional): 메모리 어텐션 마스크
            src_key_padding_mask (torch.Tensor, optional): 소스 패딩 마스크
            tgt_key_padding_mask (torch.Tensor, optional): 타겟 패딩 마스크
            memory_key_padding_mask (torch.Tensor, optional): 메모리 패딩 마스크
            context (torch.Tensor, optional): 컨텍스트 정보
            return_all_states (bool): 모든 상태 반환 여부
            force_collapse (bool): 강제 상태 붕괴 여부
            p_target (float): 목표 전환 확률
            superposition_degree (float, optional): 중첩 정도 (0~1)
            collapse_threshold (float, optional): 붕괴 임계값 (0~1)
            interference_strength (float, optional): 간섭 강도 (0~1)
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 트랜스포머 출력
        """
        # 입력 임베딩 적용
        src_emb, tgt_emb = self._prepare_inputs(src, tgt)
        
        # 마스크 준비
        if any(mask is None for mask in [src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask]):
            prepared_masks = self._prepare_masks(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
            src_mask = prepared_masks[0] if src_mask is None else src_mask
            tgt_mask = prepared_masks[1] if tgt_mask is None else tgt_mask
            memory_mask = prepared_masks[2] if memory_mask is None else memory_mask
            src_key_padding_mask = prepared_masks[3] if src_key_padding_mask is None else src_key_padding_mask
            tgt_key_padding_mask = prepared_masks[4] if tgt_key_padding_mask is None else tgt_key_padding_mask
        
        # memory_key_padding_mask가 없으면 src_key_padding_mask 사용
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        
        # 컨텍스트 생성
        if context is None:
            src_context = src_emb.mean(dim=1)  # [batch_size, d_model]
            tgt_context = tgt_emb.mean(dim=1)  # [batch_size, d_model]
            context = (src_context + tgt_context) / 2
            
        # 인코더 순전파
        if return_all_states:
            encoder_output = self.encoder(
                src=src_emb,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                context=context,
                return_all_states=True,
                force_collapse=False,
                p_target=p_target,  # 목표 전환 확률 전달
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength
            )
            memory = encoder_output['output']
            memory_superposition = encoder_output['superposition_state']
        else:
            memory = self.encoder(
                src=src_emb,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                context=context,
                p_target=p_target,  # 목표 전환 확률 전달
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength
            )
            memory_superposition = None
            
        # 디코더 순전파
        if return_all_states:
            decoder_output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                context=context,
                memory_superposition=memory_superposition,
                return_all_states=True,
                force_collapse=force_collapse,
                p_target=p_target,  # 목표 전환 확률 전달
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength
            )
            output = decoder_output['output']
        else:
            output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                context=context,
                memory_superposition=memory_superposition,
                p_target=p_target,  # 목표 전환 확률 전달
                superposition_degree=superposition_degree,
                collapse_threshold=collapse_threshold,
                interference_strength=interference_strength
            )
        
        # 출력 프로젝션 적용 (vocab_size가 있는 경우)
        if self.has_embeddings:
            if return_all_states:
                decoder_output['logits'] = self.output_projection(output)
                decoder_output['output'] = output  # 원본 출력도 유지
                output = decoder_output
            else:
                output = self.output_projection(output)
            
        if return_all_states and not isinstance(output, dict):
            # 평균 불확실성, 전환 확률, 리소스 효율성 계산
            encoder_uncertainty = encoder_output.get('uncertainty', torch.tensor(0.0))
            decoder_uncertainty = decoder_output.get('uncertainty', torch.tensor(0.0))
            avg_uncertainty = (encoder_uncertainty + decoder_uncertainty) / 2
            
            encoder_transition_prob = encoder_output.get('transition_prob', torch.tensor(0.0))
            decoder_transition_prob = decoder_output.get('transition_prob', torch.tensor(0.0))
            avg_transition_prob = (encoder_transition_prob + decoder_transition_prob) / 2
            
            encoder_resource_efficiency = encoder_output.get('resource_efficiency', torch.tensor(0.0))
            decoder_resource_efficiency = decoder_output.get('resource_efficiency', torch.tensor(0.0))
            avg_resource_efficiency = (encoder_resource_efficiency + decoder_resource_efficiency) / 2
            
            return {
                'output': output,
                'logits': self.output_projection(output) if self.has_embeddings else None,
                'encoder_output': encoder_output,
                'decoder_output': decoder_output,
                'uncertainty': avg_uncertainty,  # 평균 불확실성 추가
                'transition_prob': avg_transition_prob,  # 평균 전환 확률 추가
                'resource_efficiency': avg_resource_efficiency  # 평균 리소스 효율성 추가
            }
        else:
            return output