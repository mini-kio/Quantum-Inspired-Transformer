#!/usr/bin/env python3
"""
Quantum-Inspired Transformer 테스트 코드
Shape 오류 및 기본 동작 확인을 위한 종합 테스트
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
import os
import traceback
from typing import Dict, Any, Tuple

# 프로젝트 모듈 import
from architecture.transformer import QuantumInspiredTransformer, QuantumInspiredTransformerEncoder
from architecture.attention import QuantumInspiredAttention
from architecture.position_encoding import PositionalEncoding, QuantumPositionalEncoding
from architecture.feed_forward import FeedForward, DualStateFeedForward
from architecture.integrated_layer import IntegratedTransformerLayer
from core.dual_state import DualStateRepresentation, DualStateController
from core.state_management import GlobalStateManager
from core.collapse import StateCollapseFramework
from optimization.resource_allocator import ResourceAllocator
from optimization.learning import UniversalLoss
from training.hyperparameters import HyperParameters


class QuantumTransformerTester:
    """Quantum-Inspired Transformer 종합 테스트 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"테스트 디바이스: {self.device}")
        
        # 기본 테스트 설정
        self.test_configs = {
            "tiny": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "max_superposition_dim": 2,
                "vocab_size": 1000,
                "max_seq_len": 32
            },
            "small": {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "dim_feedforward": 256,
                "max_superposition_dim": 4,
                "vocab_size": 5000,
                "max_seq_len": 64
            },
            "base": {
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048,
                "max_superposition_dim": 4,
                "vocab_size": 10000,
                "max_seq_len": 128
            }
        }
        
        self.results = {}
    
    def test_dual_state_representation(self, config_name: str = "tiny"):
        """이중 상태 표현 시스템 테스트"""
        print(f"\n=== 이중 상태 표현 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 이중 상태 시스템 초기화
            dual_state = DualStateRepresentation(
                hidden_dim=config["d_model"],
                max_superposition_dim=config["max_superposition_dim"]
            ).to(self.device)
            
            # 테스트 데이터 생성
            batch_size = 4
            seq_len = config["max_seq_len"]
            deterministic_state = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"입력 확정 상태 shape: {deterministic_state.shape}")
            
            # 확정 상태 → 중첩 상태 변환 테스트
            superposition_state = dual_state.to_superposition_state(deterministic_state)
            expected_superposition_shape = (
                batch_size, seq_len, 
                config["d_model"] * config["max_superposition_dim"]
            )
            print(f"중첩 상태 shape: {superposition_state.shape}")
            print(f"예상 shape: {expected_superposition_shape}")
            
            assert superposition_state.shape == expected_superposition_shape, \
                f"중첩 상태 shape 불일치: {superposition_state.shape} != {expected_superposition_shape}"
            
            # 중첩 상태 → 확정 상태 변환 테스트
            recovered_state = dual_state.from_superposition_state(superposition_state)
            print(f"복원된 확정 상태 shape: {recovered_state.shape}")
            
            assert recovered_state.shape == deterministic_state.shape, \
                f"복원된 상태 shape 불일치: {recovered_state.shape} != {deterministic_state.shape}"
            
            print("✅ 이중 상태 표현 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 이중 상태 표현 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_quantum_attention(self, config_name: str = "tiny"):
        """양자 영감 어텐션 테스트"""
        print(f"\n=== 양자 영감 어텐션 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 양자 어텐션 초기화
            attention = QuantumInspiredAttention(
                d_model=config["d_model"],
                nhead=config["nhead"],
                max_superposition_dim=config["max_superposition_dim"],
                dropout=0.1
            ).to(self.device)
            
            # 테스트 데이터 생성
            batch_size = 4
            seq_len = config["max_seq_len"]
            
            # 확정 상태 테스트
            deterministic_input = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"확정 상태 입력 shape: {deterministic_input.shape}")
            
            deterministic_output = attention.forward_deterministic(deterministic_input)
            print(f"확정 상태 출력 shape: {deterministic_output.shape}")
            
            assert deterministic_output.shape == deterministic_input.shape, \
                f"확정 상태 출력 shape 불일치: {deterministic_output.shape} != {deterministic_input.shape}"
            
            # 중첩 상태 테스트
            superposition_input = torch.randn(
                batch_size, seq_len, 
                config["d_model"] * config["max_superposition_dim"]
            ).to(self.device)
            
            print(f"중첩 상태 입력 shape: {superposition_input.shape}")
            
            superposition_output = attention.forward_superposition(superposition_input)
            print(f"중첩 상태 출력 shape: {superposition_output.shape}")
            
            assert superposition_output.shape == superposition_input.shape, \
                f"중첩 상태 출력 shape 불일치: {superposition_output.shape} != {superposition_input.shape}"
            
            print("✅ 양자 영감 어텐션 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 양자 영감 어텐션 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_position_encoding(self, config_name: str = "tiny"):
        """위치 인코딩 테스트"""
        print(f"\n=== 위치 인코딩 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 표준 위치 인코딩
            pos_encoding = PositionalEncoding(
                d_model=config["d_model"],
                dropout=0.1
            ).to(self.device)
            
            # 양자 위치 인코딩
            quantum_pos_encoding = QuantumPositionalEncoding(
                d_model=config["d_model"],
                max_superposition_dim=config["max_superposition_dim"],
                dropout=0.1
            ).to(self.device)
            
            # 테스트 데이터
            batch_size = 4
            seq_len = config["max_seq_len"]
            
            # 표준 위치 인코딩 테스트
            input_tensor = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"입력 텐서 shape: {input_tensor.shape}")
            
            pos_encoded = pos_encoding(input_tensor)
            print(f"위치 인코딩 출력 shape: {pos_encoded.shape}")
            
            assert pos_encoded.shape == input_tensor.shape, \
                f"위치 인코딩 출력 shape 불일치: {pos_encoded.shape} != {input_tensor.shape}"
            
            # 양자 위치 인코딩 테스트
            quantum_pos_encoded = quantum_pos_encoding(input_tensor)
            print(f"양자 위치 인코딩 출력 shape: {quantum_pos_encoded.shape}")
            
            expected_quantum_shape = (
                batch_size, seq_len,
                config["d_model"] * config["max_superposition_dim"]
            )
            assert quantum_pos_encoded.shape == expected_quantum_shape, \
                f"양자 위치 인코딩 출력 shape 불일치: {quantum_pos_encoded.shape} != {expected_quantum_shape}"
            
            print("✅ 위치 인코딩 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 위치 인코딩 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_feed_forward(self, config_name: str = "tiny"):
        """피드포워드 네트워크 테스트"""
        print(f"\n=== 피드포워드 네트워크 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 표준 피드포워드
            ff = FeedForward(
                d_model=config["d_model"],
                dim_feedforward=config["dim_feedforward"],
                dropout=0.1
            ).to(self.device)
            
            # 이중 상태 피드포워드
            dual_ff = DualStateFeedForward(
                d_model=config["d_model"],
                dim_feedforward=config["dim_feedforward"],
                max_superposition_dim=config["max_superposition_dim"],
                dropout=0.1
            ).to(self.device)
            
            # 테스트 데이터
            batch_size = 4
            seq_len = config["max_seq_len"]
            
            # 표준 피드포워드 테스트
            input_tensor = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"입력 텐서 shape: {input_tensor.shape}")
            
            ff_output = ff(input_tensor)
            print(f"피드포워드 출력 shape: {ff_output.shape}")
            
            assert ff_output.shape == input_tensor.shape, \
                f"피드포워드 출력 shape 불일치: {ff_output.shape} != {input_tensor.shape}"
            
            # 이중 상태 피드포워드 테스트
            dual_output = dual_ff(input_tensor)
            print(f"이중 상태 피드포워드 출력 shape: {dual_output.shape}")
            
            # 이중 상태 출력은 원본 크기와 같아야 함 (내부적으로 변환 후 다시 원래 크기로)
            assert dual_output.shape == input_tensor.shape, \
                f"이중 상태 피드포워드 출력 shape 불일치: {dual_output.shape} != {input_tensor.shape}"
            
            print("✅ 피드포워드 네트워크 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 피드포워드 네트워크 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_integrated_layer(self, config_name: str = "tiny"):
        """통합 트랜스포머 레이어 테스트"""
        print(f"\n=== 통합 트랜스포머 레이어 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 통합 레이어 초기화
            layer = IntegratedTransformerLayer(
                d_model=config["d_model"],
                nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                dropout=0.1,
                max_superposition_dim=config["max_superposition_dim"],
                layer_id=0,
                num_layers=config["num_layers"]
            ).to(self.device)
            
            # 테스트 데이터
            batch_size = 4
            seq_len = config["max_seq_len"]
            input_tensor = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"입력 텐서 shape: {input_tensor.shape}")
            
            # 레이어 통과
            output = layer(input_tensor)
            print(f"레이어 출력 shape: {output.shape}")
            
            assert output.shape == input_tensor.shape, \
                f"레이어 출력 shape 불일치: {output.shape} != {input_tensor.shape}"
            
            print("✅ 통합 트랜스포머 레이어 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 통합 트랜스포머 레이어 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_transformer_encoder(self, config_name: str = "tiny"):
        """트랜스포머 인코더 테스트"""
        print(f"\n=== 트랜스포머 인코더 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 인코더 초기화
            encoder = QuantumInspiredTransformerEncoder(
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=0.1,
                max_superposition_dim=config["max_superposition_dim"]
            ).to(self.device)
            
            # 테스트 데이터
            batch_size = 4
            seq_len = config["max_seq_len"]
            input_tensor = torch.randn(
                batch_size, seq_len, config["d_model"]
            ).to(self.device)
            
            print(f"입력 텐서 shape: {input_tensor.shape}")
            
            # 인코더 통과
            output = encoder(input_tensor)
            print(f"인코더 출력 shape: {output.shape}")
            
            assert output.shape == input_tensor.shape, \
                f"인코더 출력 shape 불일치: {output.shape} != {input_tensor.shape}"
            
            print("✅ 트랜스포머 인코더 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 트랜스포머 인코더 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_full_transformer(self, config_name: str = "tiny"):
        """전체 트랜스포머 모델 테스트"""
        print(f"\n=== 전체 트랜스포머 모델 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            # 하이퍼파라미터 설정
            hyperparams = HyperParameters(
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["dim_feedforward"],
                vocab_size=config["vocab_size"],
                max_seq_len=config["max_seq_len"],
                max_superposition_dim=config["max_superposition_dim"],
                dropout=0.1,
                learning_rate=1e-4
            )
            
            # 전체 모델 초기화
            model = QuantumInspiredTransformer(hyperparams).to(self.device)
            
            # 테스트 데이터 (토큰 ID)
            batch_size = 4
            seq_len = config["max_seq_len"]
            input_ids = torch.randint(
                0, config["vocab_size"], 
                (batch_size, seq_len)
            ).to(self.device)
            
            print(f"입력 토큰 shape: {input_ids.shape}")
            
            # 모델 순전파
            with torch.no_grad():
                output = model(input_ids)
                print(f"모델 출력 shape: {output.shape}")
                
                expected_output_shape = (batch_size, seq_len, config["vocab_size"])
                assert output.shape == expected_output_shape, \
                    f"모델 출력 shape 불일치: {output.shape} != {expected_output_shape}"
            
            # 그래디언트 계산 테스트
            model.train()
            output = model(input_ids)
            loss = output.sum()  # 간단한 손실 함수
            loss.backward()
            
            # 파라미터에 그래디언트가 계산되었는지 확인
            has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert has_grad, "그래디언트가 계산되지 않았습니다"
            
            print("✅ 전체 트랜스포머 모델 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 전체 트랜스포머 모델 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def test_memory_usage(self, config_name: str = "small"):
        """메모리 사용량 테스트"""
        print(f"\n=== 메모리 사용량 테스트 ({config_name}) ===")
        config = self.test_configs[config_name]
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                initial_memory = torch.cuda.memory_allocated()
                print(f"초기 메모리 사용량: {initial_memory / 1024**2:.2f} MB")
            
            # 하이퍼파라미터 설정
            hyperparams = HyperParameters(
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["dim_feedforward"],
                vocab_size=config["vocab_size"],
                max_seq_len=config["max_seq_len"],
                max_superposition_dim=config["max_superposition_dim"],
                dropout=0.1,
                learning_rate=1e-4
            )
            
            # 모델 초기화
            model = QuantumInspiredTransformer(hyperparams).to(self.device)
            
            if torch.cuda.is_available():
                model_memory = torch.cuda.memory_allocated() - initial_memory
                print(f"모델 메모리 사용량: {model_memory / 1024**2:.2f} MB")
            
            # 추론 테스트
            model.eval()
            batch_size = 8
            seq_len = config["max_seq_len"]
            input_ids = torch.randint(
                0, config["vocab_size"], 
                (batch_size, seq_len)
            ).to(self.device)
            
            with torch.no_grad():
                output = model(input_ids)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                print(f"최대 메모리 사용량: {peak_memory / 1024**2:.2f} MB")
                
                # 메모리 정리
                torch.cuda.empty_cache()
            
            print("✅ 메모리 사용량 테스트 통과")
            return True
            
        except Exception as e:
            print(f"❌ 메모리 사용량 테스트 실패: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("=" * 60)
        print("Quantum-Inspired Transformer 종합 테스트 시작")
        print("=" * 60)
        
        test_methods = [
            "test_dual_state_representation",
            "test_quantum_attention", 
            "test_position_encoding",
            "test_feed_forward",
            "test_integrated_layer",
            "test_transformer_encoder",
            "test_full_transformer",
            "test_memory_usage"
        ]
        
        configs_to_test = ["tiny", "small"]
        passed_tests = 0
        total_tests = 0
        
        for config_name in configs_to_test:
            print(f"\n{'='*20} {config_name.upper()} 설정 테스트 {'='*20}")
            
            for test_method in test_methods:
                total_tests += 1
                try:
                    method = getattr(self, test_method)
                    if method(config_name):
                        passed_tests += 1
                        self.results[f"{test_method}_{config_name}"] = "PASS"
                    else:
                        self.results[f"{test_method}_{config_name}"] = "FAIL"
                except Exception as e:
                    print(f"❌ {test_method} ({config_name}) 실행 중 오류: {e}")
                    self.results[f"{test_method}_{config_name}"] = "ERROR"
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        print(f"총 테스트: {total_tests}")
        print(f"통과: {passed_tests}")
        print(f"실패: {total_tests - passed_tests}")
        print(f"성공률: {passed_tests/total_tests*100:.1f}%")
        
        print("\n상세 결과:")
        for test_name, result in self.results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"{status} {test_name}: {result}")
        
        return passed_tests == total_tests


def main():
    """메인 테스트 실행 함수"""
    tester = QuantumTransformerTester()
    
    try:
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 모든 테스트가 성공적으로 통과했습니다!")
            return 0
        else:
            print("\n⚠️ 일부 테스트에서 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        print(f"\n💥 테스트 실행 중 치명적 오류 발생: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
