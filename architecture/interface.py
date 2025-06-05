import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Union, Any, Tuple


class QuantumInspiredInterface(nn.Module):
    """
    범용 인터페이스 계층
    
    기존 트랜스포머 모델과의 호환성을 위한 표준화된 인터페이스
    """
    
    def __init__(
        self,
        model: nn.Module,
        d_model: int = 768,
        max_superposition_dim: int = 4,
        mode: str = 'default',
        compatibility_mode: bool = False
    ):
        """
        범용 인터페이스 계층 초기화
        
        Args:
            model (nn.Module): 양자 영감 트랜스포머 모델
            d_model (int): 모델 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
            mode (str): 작동 모드 ('default', 'efficient', 'accurate')
            compatibility_mode (bool): 기존 트랜스포머와의 호환성 모드
        """
        super().__init__()
        self.model = model
        self.d_model = d_model
        self.max_superposition_dim = max_superposition_dim
        self.mode = mode
        self.compatibility_mode = compatibility_mode
        
        # 입력 변환기
        self.input_adapter = nn.ModuleDict({
            'default': nn.Identity(),
            'huggingface': nn.Linear(d_model, d_model),
            'pytorch': nn.Linear(d_model, d_model),
            'tensorflow': nn.Linear(d_model, d_model)
        })
        
        # 출력 변환기
        self.output_adapter = nn.ModuleDict({
            'default': nn.Identity(),
            'huggingface': nn.Linear(d_model, d_model),
            'pytorch': nn.Linear(d_model, d_model),
            'tensorflow': nn.Linear(d_model, d_model)
        })
        
        # 상태 변환기 (중첩 상태 <-> 확정 상태)
        self.state_adapter = DualStateAdapter(d_model, max_superposition_dim)
        
        # 모드 컨트롤러
        self.mode_controller = ModeController()
        
        # 호환성 계층
        if compatibility_mode:
            self.compatibility_layer = CompatibilityLayer(d_model)
        
    def adapt_inputs(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        backend: str = 'default'
    ) -> Dict[str, torch.Tensor]:
        """
        외부 입력을 내부 형식으로 변환
        
        Args:
            inputs: 외부 입력 (텐서 또는 딕셔너리)
            backend: 백엔드 유형 ('default', 'huggingface', 'pytorch', 'tensorflow')
            
        Returns:
            Dict[str, torch.Tensor]: 변환된 내부 입력
        """
        # 백엔드별 입력 형식 처리
        if backend not in self.input_adapter:
            backend = 'default'
            
        if isinstance(inputs, dict):
            # 딕셔너리 입력 처리
            adapted_inputs = {}
            
            # 주요 키 변환
            key_mapping = {
                'input_ids': 'src',
                'attention_mask': 'src_key_padding_mask',
                'decoder_input_ids': 'tgt',
                'decoder_attention_mask': 'tgt_key_padding_mask',
                'encoder_outputs': 'memory',
                'past_key_values': 'cache',
                'inputs_embeds': 'src_emb',
                'decoder_inputs_embeds': 'tgt_emb'
            }
            
            # 키 매핑에 따른 변환
            for key, value in inputs.items():
                if key in key_mapping:
                    adapted_key = key_mapping[key]
                    
                    # 텐서 변환
                    if isinstance(value, torch.Tensor):
                        adapted_inputs[adapted_key] = self.input_adapter[backend](value)
                    else:
                        adapted_inputs[adapted_key] = value
                else:
                    # 매핑되지 않은 키는 그대로 유지
                    adapted_inputs[key] = value
                    
            return adapted_inputs
            
        elif isinstance(inputs, torch.Tensor):
            # 텐서 입력 처리
            adapted_tensor = self.input_adapter[backend](inputs)
            return {'src': adapted_tensor}
            
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
    def adapt_outputs(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        backend: str = 'default'
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        내부 출력을 외부 형식으로 변환
        
        Args:
            outputs: 내부 출력 (텐서 또는 딕셔너리)
            backend: 백엔드 유형 ('default', 'huggingface', 'pytorch', 'tensorflow')
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 변환된 외부 출력
        """
        # 백엔드별 출력 형식 처리
        if backend not in self.output_adapter:
            backend = 'default'
            
        if isinstance(outputs, dict):
            # 딕셔너리 출력 처리
            adapted_outputs = {}
            
            # 주요 키 변환
            key_mapping = {
                'output': 'logits',
                'deterministic_state': 'hidden_states',
                'superposition_state': 'quantum_states',
                'collapsed_state': 'attentions',
                'inference_result': 'cross_attentions',
                'hypothesis': 'past_key_values'
            }
            
            # 키 매핑에 따른 변환
            for key, value in outputs.items():
                if key in key_mapping:
                    adapted_key = key_mapping[key]
                    
                    # 텐서 변환
                    if isinstance(value, torch.Tensor):
                        adapted_outputs[adapted_key] = self.output_adapter[backend](value)
                    elif isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value):
                        adapted_outputs[adapted_key] = [self.output_adapter[backend](v) for v in value]
                    else:
                        adapted_outputs[adapted_key] = value
                else:
                    # 매핑되지 않은 키는 그대로 유지
                    adapted_outputs[key] = value
                    
            # 호환성 모드에서 필요한 키 추가
            if self.compatibility_mode:
                if backend == 'huggingface' and 'logits' in adapted_outputs:
                    # Hugging Face 형식에 필요한 추가 키
                    if 'hidden_states' not in adapted_outputs:
                        adapted_outputs['hidden_states'] = None
                    if 'attentions' not in adapted_outputs:
                        adapted_outputs['attentions'] = None
                
            return adapted_outputs
            
        elif isinstance(outputs, torch.Tensor):
            # 텐서 출력 처리
            return self.output_adapter[backend](outputs)
            
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")
    
    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        backend: str = 'default',
        mode: Optional[str] = None,
        return_all_states: bool = False,
        force_collapse: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        범용 인터페이스 계층 순전파
        
        Args:
            inputs: 외부 입력 (텐서 또는 딕셔너리)
            backend: 백엔드 유형 ('default', 'huggingface', 'pytorch', 'tensorflow')
            mode: 작동 모드 ('default', 'efficient', 'accurate')
            return_all_states: 모든 상태 반환 여부
            force_collapse: 강제 상태 붕괴 여부
            **kwargs: 추가 인자
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 변환된 외부 출력
        """
        # 작동 모드 설정
        current_mode = mode if mode is not None else self.mode
        
        # 입력 변환
        adapted_inputs = self.adapt_inputs(inputs, backend)
        
        # 모드 설정에 따른 추가 설정
        mode_settings = self.mode_controller.get_mode_settings(current_mode)
        
        # 모델 추론
        if self.compatibility_mode:
            # 호환성 모드에서는 호환성 계층을 통해 실행
            outputs = self.compatibility_layer(
                self.model,
                adapted_inputs,
                return_all_states=return_all_states or mode_settings.get('return_all_states', False),
                force_collapse=force_collapse or mode_settings.get('force_collapse', False),
                **kwargs
            )
        else:
            # 직접 모델 실행
            model_kwargs = {
                'return_all_states': return_all_states or mode_settings.get('return_all_states', False),
                'force_collapse': force_collapse or mode_settings.get('force_collapse', False)
            }
            
            # 추가 인자 병합
            model_kwargs.update(kwargs)
            
            # 중첩 상태가 필요한 경우 상태 어댑터 적용
            if mode_settings.get('use_superposition', True):
                adapted_inputs = self.state_adapter.prepare_superposition(adapted_inputs)
            
            # 모델 실행
            outputs = self.model(**adapted_inputs, **model_kwargs)
        
        # 출력 변환
        adapted_outputs = self.adapt_outputs(outputs, backend)
        
        return adapted_outputs


class DualStateAdapter(nn.Module):
    """
    중첩 상태와 확정 상태 간 변환을 위한 어댑터
    """
    
    def __init__(self, hidden_dim, max_superposition_dim=4):
        """
        이중 상태 어댑터 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
            max_superposition_dim (int): 최대 중첩 상태 차원
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superposition_dim = max_superposition_dim
        
        # 확정 상태 -> 중첩 상태 변환
        self.to_superposition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * max_superposition_dim)
        )
        
        # 중첩 상태 -> 확정 상태 변환
        self.from_superposition = nn.Sequential(
            nn.Linear(hidden_dim * max_superposition_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def prepare_superposition(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        입력에 중첩 상태 준비
        
        Args:
            inputs (dict): 입력 텐서 딕셔너리
            
        Returns:
            dict: 중첩 상태가 추가된 입력
        """
        # 중첩 상태가 이미 있는지 확인
        if 'src_superposition' in inputs or 'tgt_superposition' in inputs:
            return inputs
            
        # 딥 카피로 입력 복사
        adapted_inputs = {k: v for k, v in inputs.items()}
        
        # 소스 임베딩이 있는 경우 중첩 상태 생성
        if 'src_emb' in inputs:
            adapted_inputs['src_superposition'] = self.to_superposition(inputs['src_emb'])
            
        # 타겟 임베딩이 있는 경우 중첩 상태 생성
        if 'tgt_emb' in inputs:
            adapted_inputs['tgt_superposition'] = self.to_superposition(inputs['tgt_emb'])
            
        return adapted_inputs
    
    def collapse_superposition(self, superposition_state: torch.Tensor) -> torch.Tensor:
        """
        중첩 상태 붕괴
        
        Args:
            superposition_state (torch.Tensor): 중첩 상태 텐서
            
        Returns:
            torch.Tensor: 확정 상태 텐서
        """
        return self.from_superposition(superposition_state)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        collapse_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        이중 상태 어댑터 순전파
        
        Args:
            inputs (dict): 입력 텐서 딕셔너리
            collapse_outputs (bool): 출력 상태 붕괴 여부
            
        Returns:
            dict: 상태 변환된 텐서 딕셔너리
        """
        # 입력에 중첩 상태 준비
        adapted_inputs = self.prepare_superposition(inputs)
        
        # 출력 상태 붕괴가 필요하면 준비
        if collapse_outputs and 'superposition_state' in adapted_inputs:
            adapted_inputs['deterministic_state'] = self.collapse_superposition(
                adapted_inputs['superposition_state']
            )
            
        return adapted_inputs


class ModeController:
    """
    인터페이스 작동 모드 컨트롤러
    """
    
    def __init__(self):
        """
        모드 컨트롤러 초기화
        """
        # 기본 모드 설정
        self.mode_settings = {
            'default': {
                'use_superposition': True,
                'return_all_states': False,
                'force_collapse': False,
                'efficiency_level': 0.5
            },
            'efficient': {
                'use_superposition': True,
                'return_all_states': False,
                'force_collapse': True,
                'efficiency_level': 0.8
            },
            'accurate': {
                'use_superposition': True,
                'return_all_states': True,
                'force_collapse': False,
                'efficiency_level': 0.2
            },
            'compatibility': {
                'use_superposition': False,
                'return_all_states': False,
                'force_collapse': True,
                'efficiency_level': 0.5
            }
        }
        
    def get_mode_settings(self, mode: str) -> Dict[str, Any]:
        """
        작동 모드 설정 가져오기
        
        Args:
            mode (str): 작동 모드
            
        Returns:
            Dict[str, Any]: 모드 설정
        """
        if mode not in self.mode_settings:
            mode = 'default'
            
        return self.mode_settings[mode]
    
    def register_mode(self, mode: str, settings: Dict[str, Any]) -> None:
        """
        새 작동 모드 등록
        
        Args:
            mode (str): 작동 모드 이름
            settings (Dict[str, Any]): 모드 설정
        """
        self.mode_settings[mode] = settings
        
    def update_mode(self, mode: str, updates: Dict[str, Any]) -> None:
        """
        기존 작동 모드 업데이트
        
        Args:
            mode (str): 작동 모드 이름
            updates (Dict[str, Any]): 업데이트할 설정
        """
        if mode in self.mode_settings:
            self.mode_settings[mode].update(updates)


class CompatibilityLayer:
    """
    기존 트랜스포머 모델과의 호환성을 위한 계층
    """
    
    def __init__(self, hidden_dim):
        """
        호환성 계층 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
        """
        self.hidden_dim = hidden_dim
        
        # 백엔드별 호환성 어댑터
        self.backend_adapters = {
            'huggingface': HuggingFaceAdapter(hidden_dim),
            'pytorch': PyTorchAdapter(hidden_dim),
            'tensorflow': TensorFlowAdapter(hidden_dim)
        }
        
    def __call__(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        backend: str = 'default',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        호환성 계층 호출
        
        Args:
            model (nn.Module): 모델
            inputs (Dict[str, torch.Tensor]): 입력 텐서 딕셔너리
            backend (str): 백엔드 유형
            **kwargs: 추가 인자
            
        Returns:
            Dict[str, torch.Tensor]: 호환성 조정된 출력
        """
        if backend in self.backend_adapters:
            # 백엔드별 어댑터 사용
            return self.backend_adapters[backend](model, inputs, **kwargs)
        else:
            # 기본 직접 실행
            return model(**inputs, **kwargs)


class HuggingFaceAdapter:
    """
    Hugging Face 모델과의 호환성 어댑터
    """
    
    def __init__(self, hidden_dim):
        """
        Hugging Face 어댑터 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
        """
        self.hidden_dim = hidden_dim
        
    def __call__(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Hugging Face 어댑터 호출
        
        Args:
            model (nn.Module): 모델
            inputs (Dict[str, torch.Tensor]): 입력 텐서 딕셔너리
            **kwargs: 추가 인자
            
        Returns:
            Dict[str, torch.Tensor]: 호환성 조정된 출력
        """
        # Hugging Face 형식으로 입력 변환
        hf_inputs = {}
        
        # 주요 키 변환
        key_mapping = {
            'src': 'input_ids',
            'src_key_padding_mask': 'attention_mask',
            'tgt': 'decoder_input_ids',
            'tgt_key_padding_mask': 'decoder_attention_mask',
            'memory': 'encoder_outputs',
            'cache': 'past_key_values',
            'src_emb': 'inputs_embeds',
            'tgt_emb': 'decoder_inputs_embeds'
        }
        
        # 키 매핑에 따른 변환
        for key, value in inputs.items():
            if key in key_mapping:
                hf_inputs[key_mapping[key]] = value
            else:
                hf_inputs[key] = value
                
        # 모델 실행
        outputs = model(**hf_inputs, **kwargs)
        
        # 출력 변환
        adapted_outputs = {}
        
        # 출력 키 변환
        if isinstance(outputs, dict):
            output_key_mapping = {
                'logits': 'output',
                'hidden_states': 'deterministic_state',
                'last_hidden_state': 'output',
                'attentions': 'collapsed_state',
                'past_key_values': 'cache'
            }
            
            # 키 매핑에 따른 변환
            for key, value in outputs.items():
                if key in output_key_mapping:
                    adapted_outputs[output_key_mapping[key]] = value
                else:
                    adapted_outputs[key] = value
        else:
            # 텐서 출력은 'output'으로 매핑
            adapted_outputs['output'] = outputs
            
        return adapted_outputs


class PyTorchAdapter:
    """
    PyTorch 트랜스포머 모델과의 호환성 어댑터
    """
    
    def __init__(self, hidden_dim):
        """
        PyTorch 어댑터 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
        """
        self.hidden_dim = hidden_dim
        
    def __call__(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        PyTorch 어댑터 호출
        
        Args:
            model (nn.Module): 모델
            inputs (Dict[str, torch.Tensor]): 입력 텐서 딕셔너리
            **kwargs: 추가 인자
            
        Returns:
            Dict[str, torch.Tensor]: 호환성 조정된 출력
        """
        # PyTorch 트랜스포머 형식으로 입력 변환
        pt_inputs = {}
        
        # 주요 키 변환
        key_mapping = {
            'src': 'src',
            'tgt': 'tgt',
            'src_key_padding_mask': 'src_key_padding_mask',
            'tgt_key_padding_mask': 'tgt_key_padding_mask',
            'memory': 'memory',
            'src_mask': 'src_mask',
            'tgt_mask': 'tgt_mask',
            'memory_mask': 'memory_mask'
        }
        
        # 키 매핑에 따른 변환
        for key, value in inputs.items():
            if key in key_mapping:
                pt_inputs[key_mapping[key]] = value
                
        # 모델 실행
        outputs = model(**pt_inputs, **kwargs)
        
        # 출력 변환
        if isinstance(outputs, dict):
            return outputs
        elif isinstance(outputs, torch.Tensor):
            return {'output': outputs}
        else:
            return {'output': outputs[0], 'deterministic_state': outputs[1]}


class TensorFlowAdapter:
    """
    TensorFlow 모델과의 호환성 어댑터 (PyTorch 환경에서 TF 스타일 인터페이스 에뮬레이션)
    """
    
    def __init__(self, hidden_dim):
        """
        TensorFlow 어댑터 초기화
        
        Args:
            hidden_dim (int): 기본 히든 차원
        """
        self.hidden_dim = hidden_dim
        
    def __call__(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        TensorFlow 어댑터 호출
        
        Args:
            model (nn.Module): 모델
            inputs (Dict[str, torch.Tensor]): 입력 텐서 딕셔너리
            **kwargs: 추가 인자
            
        Returns:
            Dict[str, torch.Tensor]: 호환성 조정된 출력
        """
        # TensorFlow 스타일의 인터페이스로 변환
        tf_inputs = {}
        
        # 주요 키 변환
        key_mapping = {
            'src': 'inputs',
            'src_key_padding_mask': 'attention_mask',
            'tgt': 'decoder_inputs',
            'tgt_key_padding_mask': 'decoder_attention_mask',
            'memory': 'encoder_outputs',
            'cache': 'cache'
        }
        
        # 키 매핑에 따른 변환
        for key, value in inputs.items():
            if key in key_mapping:
                tf_inputs[key_mapping[key]] = value
            else:
                tf_inputs[key] = value
                
        # 모델 실행
        outputs = model(**tf_inputs, **kwargs)
        
        # 출력 변환
        adapted_outputs = {}
        
        # 출력 키 변환
        if isinstance(outputs, dict):
            output_key_mapping = {
                'logits': 'output',
                'hidden_states': 'deterministic_state',
                'final_state': 'output',
                'attention_weights': 'collapsed_state',
                'cache': 'cache'
            }
            
            # 키 매핑에 따른 변환
            for key, value in outputs.items():
                if key in output_key_mapping:
                    adapted_outputs[output_key_mapping[key]] = value
                else:
                    adapted_outputs[key] = value
        else:
            # 텐서 출력은 'output'으로 매핑
            adapted_outputs['output'] = outputs
            
        return adapted_outputs


class ScalableModelInterface:
    """
    다양한 크기의 모델에 적용 가능한 확장성 인터페이스
    """
    
    def __init__(self, config=None):
        """
        확장성 인터페이스 초기화
        
        Args:
            config (Dict, optional): 모델 구성
        """
        self.config = config or {}
        
    def build_model(
        self,
        size: str = 'base',
        custom_config: Dict = None,
        pretrained: bool = False,
        checkpoint_path: str = None
    ) -> nn.Module:
        """
        모델 크기에 따른 양자 영감 트랜스포머 생성
        
        Args:
            size (str): 모델 크기 ('small', 'base', 'large', 'xl')
            custom_config (Dict, optional): 사용자 정의 구성
            pretrained (bool): 사전 훈련된 모델 로드 여부
            checkpoint_path (str, optional): 체크포인트 경로
            
        Returns:
            nn.Module: 구성된 모델
        """
        # 모델 크기별 기본 구성
        size_configs = {
            'small': {
                'd_model': 384,
                'nhead': 6,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'dim_feedforward': 1536,
                'max_superposition_dim': 3,
            },
            'base': {
                'd_model': 768,
                'nhead': 12,
                'num_encoder_layers': 12,
                'num_decoder_layers': 12,
                'dim_feedforward': 3072,
                'max_superposition_dim': 4,
            },
            'large': {
                'd_model': 1024,
                'nhead': 16,
                'num_encoder_layers': 24,
                'num_decoder_layers': 24,
                'dim_feedforward': 4096,
                'max_superposition_dim': 6,
            },
            'xl': {
                'd_model': 2048,
                'nhead': 32,
                'num_encoder_layers': 36,
                'num_decoder_layers': 36,
                'dim_feedforward': 8192,
                'max_superposition_dim': 8,
            }
        }
        
        # 기본 크기 구성 선택
        if size not in size_configs:
            size = 'base'
            
        model_config = size_configs[size].copy()
        
        # 사용자 정의 구성이 있으면 병합
        if custom_config:
            model_config.update(custom_config)

        # Only keep parameters accepted by QuantumInspiredTransformer
        allowed_keys = {
            'd_model',
            'nhead',
            'num_encoder_layers',
            'num_decoder_layers',
            'dim_feedforward',
            'dropout',
            'max_superposition_dim',
            'activation',
            'gate_type',
            'custom_encoder',
            'custom_decoder',
            'vocab_size',
            'pad_token_id',
        }
        model_config = {k: v for k, v in model_config.items() if k in allowed_keys}
            
        # 공통 기본 옵션 설정
        model_config.setdefault('dropout', 0.1)
        model_config.setdefault('activation', 'gelu')
        
        # 필요한 모듈 임포트
        from architecture.transformer import QuantumInspiredTransformer
        
        # 모델 인스턴스 생성
        model = QuantumInspiredTransformer(**model_config)
        
        # 사전 훈련된 모델 로드
        if pretrained and checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
        return model
    
    def create_interface(
        self,
        model: nn.Module,
        backend: str = 'default',
        mode: str = 'default',
        compatibility_mode: bool = False
    ) -> QuantumInspiredInterface:
        """
        모델 인터페이스 생성
        
        Args:
            model (nn.Module): 양자 영감 트랜스포머 모델
            backend (str): 백엔드 유형
            mode (str): 작동 모드
            compatibility_mode (bool): 호환성 모드 여부
            
        Returns:
            QuantumInspiredInterface: 모델 인터페이스
        """
        # 모델 구성 추출
        if hasattr(model, 'd_model'):
            d_model = model.d_model
        else:
            d_model = 768  # 기본값
            
        if hasattr(model, 'max_superposition_dim'):
            max_superposition_dim = model.max_superposition_dim
        else:
            max_superposition_dim = 4  # 기본값
            
        # 인터페이스 생성
        interface = QuantumInspiredInterface(
            model=model,
            d_model=d_model,
            max_superposition_dim=max_superposition_dim,
            mode=mode,
            compatibility_mode=compatibility_mode
        )
        
        return interface
