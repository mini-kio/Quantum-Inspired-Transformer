import torch
import torch.nn as nn
import math
import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple


class ScalingLaws:
    """
    모델 크기와 중첩 복잡성에 따른 효율적 스케일링 법칙
    """
    
    @staticmethod
    def compute_optimal_dimensions(
        model_size: int,
        task_complexity: float = 0.5,
        efficiency_target: float = 0.7
    ) -> Dict[str, int]:
        """
        모델 크기에 따른 최적 차원 계산
        
        Args:
            model_size (int): 모델 크기 (백만 파라미터 단위)
            task_complexity (float): 태스크 복잡성 (0~1)
            efficiency_target (float): 목표 효율성 (0~1)
            
        Returns:
            Dict[str, int]: 최적 모델 차원
        """
        # 기본 스케일링 법칙: 차원 ~ sqrt(모델 크기)
        base_dimension = int(16 * math.sqrt(model_size))
        
        # 복잡성에 따른 조정
        complexity_factor = 0.8 + 0.4 * task_complexity
        
        # 효율성 목표에 따른 조정
        efficiency_factor = 1.0 + 0.5 * (1.0 - efficiency_target)
        
        # 최종 차원 계산
        d_model = int(base_dimension * complexity_factor * efficiency_factor)
        d_model = max(d_model, 128)  # 최소 차원
        
        # 차원을 64의 배수로 정렬
        d_model = ((d_model + 63) // 64) * 64
        
        # 기타 차원 계산
        nhead = max(d_model // 64, 4)  # 헤드 수
        dim_feedforward = d_model * 4  # 피드포워드 네트워크 차원
        
        # 중첩 차원 계산 (태스크 복잡성에 따라)
        max_superposition_dim = max(int(2 + 6 * task_complexity), 2)
        
        # 레이어 수 계산
        num_layers = max(int(math.log(model_size) * 2 * complexity_factor), 2)
        
        return {
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'max_superposition_dim': max_superposition_dim,
            'num_encoder_layers': num_layers,
            'num_decoder_layers': num_layers
        }
    
    @staticmethod
    def compute_computational_cost(
        config: Dict[str, Any],
        seq_length: int = 512
    ) -> Dict[str, float]:
        """
        모델 구성에 따른 계산 비용 추정
        
        Args:
            config (Dict[str, Any]): 모델 구성
            seq_length (int): 시퀀스 길이
            
        Returns:
            Dict[str, float]: 계산 비용 (FLOPS, 메모리 등)
        """
        d_model = config['d_model']
        nhead = config['nhead']
        dim_feedforward = config['dim_feedforward']
        num_encoder_layers = config['num_encoder_layers']
        num_decoder_layers = config['num_decoder_layers']
        max_superposition_dim = config.get('max_superposition_dim', 4)
        
        # 기본 트랜스포머 비용
        base_attention_flops = 4 * seq_length * seq_length * d_model
        base_ffn_flops = 2 * seq_length * d_model * dim_feedforward
        
        # 양자 영감 오버헤드
        quantum_attention_overhead = seq_length * d_model * max_superposition_dim
        quantum_ffn_overhead = seq_length * dim_feedforward * max_superposition_dim
        quantum_state_management = seq_length * d_model * d_model * max_superposition_dim
        
        # 총 FLOPS 계산
        encoder_flops = num_encoder_layers * (
            base_attention_flops + base_ffn_flops +
            quantum_attention_overhead + quantum_ffn_overhead + quantum_state_management
        )
        
        decoder_flops = num_decoder_layers * (
            2 * base_attention_flops + base_ffn_flops +
            2 * quantum_attention_overhead + quantum_ffn_overhead + quantum_state_management
        )
        
        total_flops = encoder_flops + decoder_flops
        
        # 메모리 사용량 추정 (바이트 단위)
        param_bytes = 4  # 부동소수점 (32비트)
        
        encoder_params = num_encoder_layers * (
            4 * d_model * d_model +  # 어텐션 행렬
            2 * dim_feedforward * d_model +  # FFN 행렬
            max_superposition_dim * d_model * d_model  # 양자 상태 표현
        )
        
        decoder_params = num_decoder_layers * (
            6 * d_model * d_model +  # 어텐션 행렬 (자기 + 크로스)
            2 * dim_feedforward * d_model +  # FFN 행렬
            max_superposition_dim * d_model * d_model  # 양자 상태 표현
        )
        
        total_params = encoder_params + decoder_params
        memory_usage = total_params * param_bytes
        
        # 활성화 메모리 추정
        activation_memory = seq_length * (
            d_model * (num_encoder_layers + num_decoder_layers) * (1 + max_superposition_dim)
        ) * param_bytes
        
        return {
            'total_flops': total_flops,
            'total_params': total_params,
            'memory_params': memory_usage,
            'memory_activations': activation_memory,
            'total_memory': memory_usage + activation_memory
        }
    
    @staticmethod
    def find_optimal_configuration(
        target_size: Union[str, int],
        target_hardware: str = 'gpu',
        seq_length: int = 512,
        task_complexity: float = 0.5
    ) -> Dict[str, Any]:
        """
        목표 크기와 하드웨어에 최적화된 모델 구성 탐색
        
        Args:
            target_size (Union[str, int]): 목표 모델 크기 ('small', 'medium', 'large', 'xl') 또는 백만 파라미터
            target_hardware (str): 대상 하드웨어 ('gpu', 'cpu', 'tpu')
            seq_length (int): 시퀀스 길이
            task_complexity (float): 태스크 복잡성 (0~1)
            
        Returns:
            Dict[str, Any]: 최적 모델 구성
        """
        # 크기 문자열을 파라미터 수로 변환
        if isinstance(target_size, str):
            size_mapping = {
                'tiny': 10,  # 10M 파라미터
                'small': 50,  # 50M 파라미터
                'medium': 100,  # 100M 파라미터
                'base': 250,  # 250M 파라미터
                'large': 750,  # 750M 파라미터
                'xl': 1500,  # 1.5B 파라미터
                'xxl': 3000   # 3B 파라미터
            }
            
            if target_size not in size_mapping:
                target_size = 'base'
                
            model_size = size_mapping[target_size]
        else:
            model_size = target_size
        
        # 하드웨어별 효율성 목표
        hardware_efficiency = {
            'gpu': 0.75,
            'cpu': 0.9,  # CPU는 메모리 효율성 중시
            'tpu': 0.6   # TPU는 병렬성 중시
        }
        
        if target_hardware not in hardware_efficiency:
            target_hardware = 'gpu'
            
        efficiency_target = hardware_efficiency[target_hardware]
        
        # 기본 차원 계산
        dimensions = ScalingLaws.compute_optimal_dimensions(
            model_size=model_size,
            task_complexity=task_complexity,
            efficiency_target=efficiency_target
        )
        
        # 하드웨어별 최적화
        if target_hardware == 'gpu':
            # GPU를 위한 최적화: 병렬 처리에 유리한 설정
            dimensions['nhead'] = max(dimensions['nhead'], 8)  # 최소 8개 헤드
            dimensions['d_model'] = ((dimensions['d_model'] + 127) // 128) * 128  # 128의 배수로
            
        elif target_hardware == 'cpu':
            # CPU를 위한 최적화: 메모리 효율성 중시
            dimensions['max_superposition_dim'] = min(dimensions['max_superposition_dim'], 4)  # 제한된 중첩 차원
            dimensions['dim_feedforward'] = int(dimensions['dim_feedforward'] * 0.75)  # 작은 FFN
            
        elif target_hardware == 'tpu':
            # TPU를 위한 최적화: 행렬 연산 효율성 중시
            dimensions['d_model'] = ((dimensions['d_model'] + 127) // 128) * 128  # 128의 배수로
            
        # 계산 비용 추정
        cost = ScalingLaws.compute_computational_cost(dimensions, seq_length)
        
        # 결과 구성
        config = {
            **dimensions,
            'model_size_m_params': model_size,
            'target_hardware': target_hardware,
            'computational_cost': cost,
            'task_complexity': task_complexity,
            'recommended_batch_size': ScalingLaws.recommend_batch_size(
                dimensions, target_hardware, seq_length
            )
        }
        
        return config
    
    @staticmethod
    def recommend_batch_size(
        dimensions: Dict[str, int],
        target_hardware: str,
        seq_length: int
    ) -> int:
        """
        최적 배치 크기 추천
        
        Args:
            dimensions (Dict[str, int]): 모델 차원
            target_hardware (str): 대상 하드웨어
            seq_length (int): 시퀀스 길이
            
        Returns:
            int: 추천 배치 크기
        """
        d_model = dimensions['d_model']
        max_superposition_dim = dimensions.get('max_superposition_dim', 4)
        
        # 메모리 요구사항 추정
        memory_per_sample = seq_length * d_model * (1 + max_superposition_dim) * 4  # 바이트
        
        # 하드웨어별 메모리 예산
        hardware_memory = {
            'gpu': 12 * 1024 * 1024 * 1024,  # 12GB
            'cpu': 32 * 1024 * 1024 * 1024,  # 32GB
            'tpu': 16 * 1024 * 1024 * 1024   # 16GB
        }
        
        if target_hardware not in hardware_memory:
            target_hardware = 'gpu'
            
        available_memory = hardware_memory[target_hardware]
        
        # 모델 크기를 고려한 메모리 조정
        model_memory = ScalingLaws.compute_computational_cost(dimensions, seq_length)['memory_params']
        
        # 활성화에 사용 가능한 메모리
        activation_memory = available_memory * 0.7 - model_memory  # 70%만 활성화에 사용
        
        # 배치 크기 계산
        max_batch_size = int(activation_memory / memory_per_sample)
        
        # 하드웨어별 최소 배치 크기
        min_batch_sizes = {'gpu': 4, 'cpu': 1, 'tpu': 8}
        min_batch_size = min_batch_sizes.get(target_hardware, 1)
        
        # 최종 배치 크기 (2의 거듭제곱으로 정렬)
        batch_size = max(min_batch_size, 1 << (max_batch_size.bit_length() - 1))
        
        return min(batch_size, max_batch_size)
 


class AdaptiveConfigurationFramework:
    """
    다양한 하드웨어 환경에 적응하는 자동 구성 메커니즘
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        적응형 구성 프레임워크 초기화
        
        Args:
            config_dir (str, optional): 구성 파일 디렉토리
        """
        self.config_dir = config_dir or "./configs"
        
        # 구성 캐시
        self.cached_configs = {}
        
        # 하드웨어 프로필 (기본값)
        self.hardware_profiles = {
            'default': {
                'memory': 16 * 1024 * 1024 * 1024,  # 16GB
                'compute_capability': 7.5,  # CUDA compute capability
                'processor_count': 8,
                'cache_size': 16 * 1024 * 1024  # 16MB
            },
            'cpu': {
                'memory': 32 * 1024 * 1024 * 1024,  # 32GB
                'compute_capability': 0.0,
                'processor_count': 16,
                'cache_size': 32 * 1024 * 1024  # 32MB
            },
            'tpu': {
                'memory': 16 * 1024 * 1024 * 1024,  # 16GB
                'compute_capability': 0.0,
                'processor_count': 8,
                'cache_size': 16 * 1024 * 1024  # 16MB
            }
        }
        
        # 하드웨어 프로필 로드
        self._load_hardware_profiles()
        
    def _load_hardware_profiles(self) -> None:
        """
        하드웨어 프로필 파일 로드
        """
        if self.config_dir and os.path.exists(self.config_dir):
            profile_path = os.path.join(self.config_dir, "hardware_profiles.json")
            
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        profiles = json.load(f)
                        
                    # 프로필 병합
                    self.hardware_profiles.update(profiles)
                except Exception as e:
                    print(f"Error loading hardware profiles: {e}")
    
    def detect_hardware(self) -> Dict[str, Any]:
        """
        현재 하드웨어 환경 감지
        
        Returns:
            Dict[str, Any]: 하드웨어 사양
        """
        hardware_info = {}
        
        # 메모리 정보
        try:
            if torch.cuda.is_available():
                # GPU 메모리
                hardware_info['memory'] = torch.cuda.get_device_properties(0).total_memory
                hardware_info['compute_capability'] = (
                    torch.cuda.get_device_properties(0).major +
                    torch.cuda.get_device_properties(0).minor / 10
                )
                hardware_info['device_type'] = 'gpu'
                hardware_info['device_count'] = torch.cuda.device_count()
            else:
                # CPU 메모리 (근사값)
                import psutil
                hardware_info['memory'] = psutil.virtual_memory().total
                hardware_info['device_type'] = 'cpu'
                hardware_info['processor_count'] = os.cpu_count() or 1
        except:
            # 탐지 실패 시 기본값 사용
            hardware_info['memory'] = 8 * 1024 * 1024 * 1024  # 8GB
            hardware_info['device_type'] = 'cpu'
            hardware_info['processor_count'] = 4
            
        return hardware_info
    
    def generate_adaptive_config(
        self,
        target_size: str = 'base',
        task_complexity: float = 0.5,
        hardware_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        현재 하드웨어에 적응하는 구성 생성
        
        Args:
            target_size (str): 목표 모델 크기
            task_complexity (float): 태스크 복잡성
            hardware_override (Dict[str, Any], optional): 하드웨어 정보 재정의
            
        Returns:
            Dict[str, Any]: 적응형 모델 구성
        """
        # 하드웨어 정보 가져오기
        hardware_info = hardware_override or self.detect_hardware()
        device_type = hardware_info.get('device_type', 'cpu')
        
        # 캐시 키 생성
        cache_key = f"{target_size}_{task_complexity}_{device_type}"
        
        # 캐시된 구성이 있으면 반환
        if cache_key in self.cached_configs:
            return self.cached_configs[cache_key]
        
        # 하드웨어 프로필 선택
        profile = self.hardware_profiles.get(device_type, self.hardware_profiles['default'])
        
        # 메모리 조정
        if 'memory' in hardware_info:
            profile['memory'] = hardware_info['memory']
            
        # 프로세서 수 조정
        if 'processor_count' in hardware_info:
            profile['processor_count'] = hardware_info['processor_count']
            
        # 계산 능력 조정
        if 'compute_capability' in hardware_info:
            profile['compute_capability'] = hardware_info['compute_capability']
        
        # 최적 구성 탐색
        optimal_config = ScalingLaws.find_optimal_configuration(
            target_size=target_size,
            target_hardware=device_type,
            task_complexity=task_complexity
        )
        
        # 하드웨어 제약에 따른 조정
        
        # 메모리 제약
        max_model_params = int(profile['memory'] * 0.4 / 4)  # 가용 메모리의 40%, 4바이트 파라미터
        
        if optimal_config['computational_cost']['total_params'] > max_model_params:
            # 구성 축소
            scale_factor = 0.9 * max_model_params / optimal_config['computational_cost']['total_params']
            
            # 차원 축소
            optimal_config['d_model'] = int(optimal_config['d_model'] * math.sqrt(scale_factor))
            optimal_config['d_model'] = ((optimal_config['d_model'] + 63) // 64) * 64  # 64의 배수로
            
            optimal_config['dim_feedforward'] = optimal_config['d_model'] * 4
            
            # 중첩 차원 조정
            optimal_config['max_superposition_dim'] = max(
                2, 
                min(optimal_config['max_superposition_dim'], 
                    int(optimal_config['max_superposition_dim'] * scale_factor)
                )
            )
            
            # 계산 비용 재추정
            optimal_config['computational_cost'] = ScalingLaws.compute_computational_cost(
                optimal_config, seq_length=512
            )
        
        # 계산 능력 제약
        if device_type == 'gpu' and profile['compute_capability'] < 7.0:
            # 오래된 GPU에 최적화
            optimal_config['max_superposition_dim'] = min(optimal_config['max_superposition_dim'], 3)
            
        # 프로세서 수에 따른 병렬화 조정
        processor_count = profile.get('processor_count', 1)
        optimal_config['recommended_parallelism'] = {
            'model_parallel': processor_count >= 4,
            'superposition_parallel': processor_count >= 2 and optimal_config['max_superposition_dim'] > 2,
            'hybrid': processor_count >= 8
        }
        
        # 구성 캐시
        self.cached_configs[cache_key] = optimal_config
        
        return optimal_config
    
    def save_config(self, config: Dict[str, Any], name: str) -> str:
        """
        구성 저장
        
        Args:
            config (Dict[str, Any]): 모델 구성
            name (str): 구성 이름
            
        Returns:
            str: 구성 파일 경로
        """
        if self.config_dir:
            os.makedirs(self.config_dir, exist_ok=True)
            
            config_path = os.path.join(self.config_dir, f"{name}.json")
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return config_path
        
        return ""
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        구성 로드
        
        Args:
            name (str): 구성 이름
            
        Returns:
            Dict[str, Any]: 모델 구성
        """
        if self.config_dir:
            config_path = os.path.join(self.config_dir, f"{name}.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
                    
        return None
