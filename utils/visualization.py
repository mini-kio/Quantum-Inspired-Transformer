import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math


class QuantumStateVisualizer:
    """
    양자 상태 시각화 도구
    
    중첩 상태, 확정 상태, 상태 붕괴 과정 등을 시각화하기 위한 유틸리티
    """
    
    def __init__(
        self,
        max_superposition_dim: int = 4,
        device: str = "cpu",
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "viridis"
    ):
        """
        양자 상태 시각화기 초기화
        
        Args:
            max_superposition_dim: 최대 중첩 차원
            device: 시각화에 사용할 장치
            figsize: 기본 그림 크기
            cmap: 기본 컬러맵
        """
        self.max_superposition_dim = max_superposition_dim
        self.device = device
        self.figsize = figsize
        self.cmap = cmap
        
        # 커스텀 컬러맵 정의
        self.quantum_cmap = LinearSegmentedColormap.from_list(
            "quantum",
            [(0, "#000033"), (0.25, "#0066CC"), (0.5, "#00CCFF"), 
             (0.75, "#CCFFFF"), (1, "#FFFFFF")]
        )
        
        # 상태 간섭 컬러맵
        self.interference_cmap = LinearSegmentedColormap.from_list(
            "interference",
            [(0, "#000000"), (0.3, "#CC0000"), (0.6, "#FFCC00"), (1, "#FFFFFF")]
        )
        
        # 시각화 스타일 설정
        plt.style.use("seaborn-darkgrid")
        
    def visualize_superposition(
        self,
        superposition_state: torch.Tensor,
        title: str = "Superposition State Visualization",
        max_tokens: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        중첩 상태 시각화
        
        Args:
            superposition_state: 중첩 상태 텐서 [batch_size, seq_len, dim]
            title: 그림 제목
            max_tokens: 표시할 최대 토큰 수
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # CPU로 이동
        if superposition_state.device != torch.device("cpu"):
            superposition_state = superposition_state.cpu()
        
        # 텐서 형태 확인
        batch_size, seq_len, dim = superposition_state.shape
        
        # 중첩 차원 계산
        d_model = dim // self.max_superposition_dim
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, d_model
        )
        
        # 표시할 토큰 수 제한
        seq_len = min(seq_len, max_tokens)
        
        # 각 중첩 차원의 노름 계산
        norms = torch.norm(reshaped[:, :seq_len], dim=-1)  # [batch_size, seq_len, max_superposition_dim]
        
        # 그림 생성
        fig, axs = plt.subplots(
            2, 2, figsize=self.figsize, 
            gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [3, 1]}
        )
        
        # 각 배치의 첫 번째 샘플만 표시
        sample_idx = 0
        
        # 1. 중첩 상태 히트맵
        ax = axs[0, 0]
        im = ax.imshow(
            norms[sample_idx].numpy(), 
            aspect='auto', 
            cmap=self.quantum_cmap
        )
        ax.set_title("Superposition State Amplitudes", fontsize=12)
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 2. 중첩 차원 분포 (평균)
        ax = axs[0, 1]
        mean_norms = norms[sample_idx].mean(dim=0).numpy()
        ax.barh(
            range(self.max_superposition_dim),
            mean_norms,
            color=[plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim) 
                  for i in range(self.max_superposition_dim)]
        )
        ax.set_title("Average Dimension Amplitude", fontsize=12)
        ax.set_xlabel("Amplitude", fontsize=10)
        ax.set_ylabel("Dimension", fontsize=10)
        ax.set_yticks(range(self.max_superposition_dim))
        ax.tick_params(axis='both', labelsize=8)
        
        # 3. 토큰별 분포 (stacked)
        ax = axs[1, 0]
        token_indices = np.arange(seq_len)
        bottom = np.zeros(seq_len)
        for i in range(self.max_superposition_dim):
            ax.bar(
                token_indices, 
                norms[sample_idx, :, i].numpy(), 
                bottom=bottom,
                label=f"Dim {i}",
                alpha=0.7,
                color=plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim)
            )
            bottom += norms[sample_idx, :, i].numpy()
        ax.set_title("Token-wise Superposition Distribution", fontsize=12)
        ax.set_xlabel("Token Position", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xticks(token_indices)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        
        # 4. 전체 확률 분포
        ax = axs[1, 1]
        # 확률 분포 계산 (노름 제곱)
        probs = (norms[sample_idx] ** 2).sum(dim=1).numpy()
        probs = probs / probs.sum()
        ax.pie(
            probs, 
            labels=[f"T{i}" for i in range(seq_len)],
            autopct='%1.1f%%',
            colors=[plt.cm.get_cmap(self.cmap)(i/seq_len) for i in range(seq_len)],
            textprops={'fontsize': 8}
        )
        ax.set_title("Token Probability Distribution", fontsize=12)
        
        # 전체 제목 설정
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_collapse_process(
        self,
        superposition_state: torch.Tensor,
        collapsed_state: torch.Tensor,
        collapse_weights: Optional[torch.Tensor] = None,
        title: str = "Quantum State Collapse Visualization",
        max_tokens: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        상태 붕괴 과정 시각화
        
        Args:
            superposition_state: 중첩 상태 텐서
            collapsed_state: 붕괴된 확정 상태 텐서
            collapse_weights: 붕괴 가중치
            title: 그림 제목
            max_tokens: 표시할 최대 토큰 수
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # CPU로 이동
        if superposition_state.device != torch.device("cpu"):
            superposition_state = superposition_state.cpu()
        if collapsed_state.device != torch.device("cpu"):
            collapsed_state = collapsed_state.cpu()
        if collapse_weights is not None and collapse_weights.device != torch.device("cpu"):
            collapse_weights = collapse_weights.cpu()
        
        # 텐서 형태 확인
        batch_size, seq_len, dim = superposition_state.shape
        
        # 중첩 차원 계산
        d_model = dim // self.max_superposition_dim
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, d_model
        )
        
        # 표시할 토큰 수 제한
        seq_len = min(seq_len, max_tokens)
        
        # 각 중첩 차원의 노름 계산
        norms = torch.norm(reshaped[:, :seq_len], dim=-1)  # [batch_size, seq_len, max_superposition_dim]
        
        # 확정 상태의 노름 계산
        deterministic_norms = torch.norm(
            collapsed_state[:, :seq_len], dim=-1
        ).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 그림 생성
        fig, axs = plt.subplots(
            2, 2, figsize=self.figsize, 
            gridspec_kw={'height_ratios': [1, 1]}
        )
        
        # 각 배치의 첫 번째 샘플만 표시
        sample_idx = 0
        
        # 1. 중첩 상태 (Before)
        ax = axs[0, 0]
        im = ax.imshow(
            norms[sample_idx].numpy(), 
            aspect='auto', 
            cmap=self.quantum_cmap
        )
        ax.set_title("Before Collapse: Superposition State", fontsize=12)
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 2. 붕괴 가중치 (Collapse Weights)
        ax = axs[0, 1]
        
        if collapse_weights is not None:
            # 붕괴 가중치가 제공된 경우
            collapse_weights_np = collapse_weights[:, :seq_len].numpy()
            im = ax.imshow(
                collapse_weights_np[sample_idx], 
                aspect='auto', 
                cmap='Reds'
            )
            ax.set_title("Collapse Weights", fontsize=12)
        else:
            # 가중치 추정 (중첩 차원의 확률 분포)
            probs = (norms[sample_idx] ** 2)
            probs = probs / probs.sum(dim=1, keepdim=True)
            im = ax.imshow(
                probs.numpy(), 
                aspect='auto', 
                cmap='Reds'
            )
            ax.set_title("Estimated Collapse Weights (Probabilities)", fontsize=12)
            
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 3. 상태 붕괴 과정 시각화
        ax = axs[1, 0]
        
        # 붕괴 전후 비교를 위한 데이터 준비
        token_indices = np.arange(seq_len)
        width = 0.35
        
        # 각 중첩 차원의 평균 노름
        mean_norms = norms[sample_idx].mean(dim=0).numpy()
        
        # 확정 상태의 평균 노름
        mean_deterministic = deterministic_norms[sample_idx].mean(dim=0).numpy()
        
        # 평균 노름으로 정규화된 각 차원의 노름
        normalized_norms = norms[sample_idx] / (norms[sample_idx].sum(dim=1, keepdim=True) + 1e-10)
        
        # 각 중첩 차원 표시
        bottom = np.zeros(seq_len)
        for i in range(self.max_superposition_dim):
            ax.bar(
                token_indices - width/2, 
                normalized_norms[:, i].numpy(), 
                width=width,
                bottom=bottom,
                label=f"Dim {i}" if i == 0 else None,
                alpha=0.7,
                color=plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim)
            )
            bottom += normalized_norms[:, i].numpy()
            
        # 붕괴 후 확정 상태 표시
        ax.bar(
            token_indices + width/2, 
            deterministic_norms[sample_idx].numpy().flatten() / deterministic_norms[sample_idx].sum().numpy(), 
            width=width,
            label="Collapsed",
            color="red",
            alpha=0.7
        )
        
        ax.set_title("Superposition vs Collapsed State", fontsize=12)
        ax.set_xlabel("Token Position", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.set_xticks(token_indices)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        
        # 4. 상태 에너지 비교
        ax = axs[1, 1]
        
        # 상태 에너지 계산 (노름 제곱)
        superposition_energy = (norms[sample_idx] ** 2).sum(dim=1).numpy()
        collapsed_energy = (deterministic_norms[sample_idx] ** 2).numpy().flatten()
        
        # 정규화
        superposition_energy = superposition_energy / superposition_energy.max()
        collapsed_energy = collapsed_energy / collapsed_energy.max()
        
        # 바 플롯
        barwidth = 0.35
        energy_idx = np.arange(seq_len)
        
        ax.bar(
            energy_idx - barwidth/2, 
            superposition_energy, 
            barwidth, 
            label="Superposition",
            color="blue",
            alpha=0.7
        )
        ax.bar(
            energy_idx + barwidth/2, 
            collapsed_energy, 
            barwidth, 
            label="Collapsed",
            color="red",
            alpha=0.7
        )
        
        # 에너지 보존 계산
        energy_conservation = np.sum(collapsed_energy) / (np.sum(superposition_energy) + 1e-10)
        ax.text(
            0.5, 0.95, 
            f"Energy Conservation: {energy_conservation:.2f}",
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        ax.set_title("State Energy Comparison", fontsize=12)
        ax.set_xlabel("Token Position", fontsize=10)
        ax.set_ylabel("Normalized Energy", fontsize=10)
        ax.set_xticks(energy_idx)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        
        # 전체 제목 설정
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_interference(
        self,
        superposition_state: torch.Tensor,
        interfered_state: torch.Tensor,
        interference_matrix: Optional[torch.Tensor] = None,
        title: str = "Quantum Interference Visualization",
        max_tokens: int = 4,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        양자 간섭 효과 시각화
        
        Args:
            superposition_state: 원본 중첩 상태 텐서
            interfered_state: 간섭 효과가 적용된 중첩 상태 텐서
            interference_matrix: 간섭 행렬 (없으면 자동 추정)
            title: 그림 제목
            max_tokens: 표시할 최대 토큰 수
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # CPU로 이동
        if superposition_state.device != torch.device("cpu"):
            superposition_state = superposition_state.cpu()
        if interfered_state.device != torch.device("cpu"):
            interfered_state = interfered_state.cpu()
        if interference_matrix is not None and interference_matrix.device != torch.device("cpu"):
            interference_matrix = interference_matrix.cpu()
        
        # 텐서 형태 확인
        batch_size, seq_len, dim = superposition_state.shape
        
        # 중첩 차원 계산
        d_model = dim // self.max_superposition_dim
        
        # 중첩 상태 재구성
        original_reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, d_model
        )
        interfered_reshaped = interfered_state.view(
            batch_size, seq_len, self.max_superposition_dim, d_model
        )
        
        # 표시할 토큰 수 제한
        seq_len = min(seq_len, max_tokens)
        
        # 각 중첩 차원의 노름 계산
        original_norms = torch.norm(original_reshaped[:, :seq_len], dim=-1)
        interfered_norms = torch.norm(interfered_reshaped[:, :seq_len], dim=-1)
        
        # 간섭 효과 계산
        interference_effect = interfered_norms - original_norms
        
        # 그림 생성
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        
        # 각 배치의 첫 번째 샘플만 표시
        sample_idx = 0
        
        # 1. 원본 중첩 상태
        ax = axs[0, 0]
        im = ax.imshow(
            original_norms[sample_idx].numpy(), 
            aspect='auto', 
            cmap=self.quantum_cmap
        )
        ax.set_title("Original Superposition State", fontsize=12)
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 2. 간섭 효과 적용 후
        ax = axs[0, 1]
        im = ax.imshow(
            interfered_norms[sample_idx].numpy(), 
            aspect='auto', 
            cmap=self.quantum_cmap
        )
        ax.set_title("After Interference", fontsize=12)
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 3. 간섭 행렬
        ax = axs[1, 0]
        
        if interference_matrix is not None:
            # 간섭 행렬이 제공된 경우
            interference_matrix_np = interference_matrix.numpy()
            im = ax.imshow(
                interference_matrix_np, 
                aspect='auto', 
                cmap=self.interference_cmap
            )
            ax.set_title("Interference Matrix", fontsize=12)
        else:
            # 간섭 행렬 추정
            # 간단한 상관 행렬 계산
            corr_matrix = np.corrcoef(
                original_norms[sample_idx].numpy().T
            )
            im = ax.imshow(
                corr_matrix, 
                aspect='auto', 
                cmap=self.interference_cmap
            )
            ax.set_title("Estimated Interference Pattern", fontsize=12)
            
        ax.set_xlabel("Dimension", fontsize=10)
        ax.set_ylabel("Dimension", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(self.max_superposition_dim))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 4. 간섭 효과 강도
        ax = axs[1, 1]
        im = ax.imshow(
            interference_effect[sample_idx].numpy(), 
            aspect='auto', 
            cmap="coolwarm"
        )
        ax.set_title("Interference Effect Magnitude", fontsize=12)
        ax.set_xlabel("Superposition Dimension", fontsize=10)
        ax.set_ylabel("Token Position", fontsize=10)
        ax.set_xticks(range(self.max_superposition_dim))
        ax.set_yticks(range(seq_len))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax)
        
        # 전체 제목 설정
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        훈련 지표 시각화
        
        Args:
            metrics: 훈련 지표 딕셔너리
            title: 그림 제목
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # 그림 생성
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 손실 및 메트릭
        ax = axs[0, 0]
        
        # 손실 관련 메트릭 찾기
        loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower() and len(v) > 0}
        
        for metric_name, values in loss_metrics.items():
            # None 값 처리
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            ax.plot(
                valid_indices, 
                valid_values, 
                label=metric_name,
                alpha=0.7
            )
            
        ax.set_title("Loss Metrics", fontsize=12)
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Loss Value", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. 양자 파라미터
        ax = axs[0, 1]
        
        # 양자 관련 파라미터 찾기
        quantum_params = {k: v for k, v in metrics.items() 
                        if any(p in k.lower() for p in 
                              ['superposition', 'collapse', 'interference']) 
                        and len(v) > 0}
        
        for param_name, values in quantum_params.items():
            # None 값 처리
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            ax.plot(
                valid_indices, 
                valid_values, 
                label=param_name,
                alpha=0.7
            )
            
        ax.set_title("Quantum Parameters", fontsize=12)
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Parameter Value", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 3. 학습률 및 효율성
        ax = axs[1, 0]
        
        # 학습률 관련 메트릭 찾기
        lr_metrics = {k: v for k, v in metrics.items() 
                    if ('learning_rate' in k.lower() or 'lr' == k.lower() or 'efficiency' in k.lower()) 
                    and len(v) > 0}
        
        for metric_name, values in lr_metrics.items():
            # None 값 처리
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            ax.plot(
                valid_indices, 
                valid_values, 
                label=metric_name,
                alpha=0.7
            )
            
        ax.set_title("Learning Rate & Efficiency", fontsize=12)
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. 정확도 및 기타 메트릭
        ax = axs[1, 1]
        
        # 정확도 관련 메트릭 찾기
        accuracy_metrics = {k: v for k, v in metrics.items() 
                          if ('accuracy' in k.lower() or 'acc' in k.lower() 
                             or 'f1' in k.lower() or 'bleu' in k.lower() or 'rouge' in k.lower()) 
                          and len(v) > 0}
        
        for metric_name, values in accuracy_metrics.items():
            # None 값 처리
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            ax.plot(
                valid_indices, 
                valid_values, 
                label=metric_name,
                alpha=0.7
            )
            
        ax.set_title("Accuracy Metrics", fontsize=12)
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 전체 제목 설정
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_3d_superposition(
        self,
        superposition_state: torch.Tensor,
        token_idx: int = 0,
        title: str = "3D Visualization of Superposition State",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        중첩 상태의 3D 시각화
        
        Args:
            superposition_state: 중첩 상태 텐서
            token_idx: 시각화할 토큰 인덱스
            title: 그림 제목
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # CPU로 이동
        if superposition_state.device != torch.device("cpu"):
            superposition_state = superposition_state.cpu()
        
        # 텐서 형태 확인
        batch_size, seq_len, dim = superposition_state.shape
        
        # 중첩 차원 계산
        d_model = dim // self.max_superposition_dim
        
        # 중첩 상태 재구성
        reshaped = superposition_state.view(
            batch_size, seq_len, self.max_superposition_dim, d_model
        )
        
        # 각 배치의 첫 번째 샘플만 표시
        sample_idx = 0
        
        # 선택한 토큰의 중첩 상태
        token_state = reshaped[sample_idx, token_idx]  # [max_superposition_dim, d_model]
        
        # 차원 감소를 위한 PCA 적용
        try:
            from sklearn.decomposition import PCA
            
            # 최소 3개 이상의 차원 필요
            if token_state.shape[1] < 3:
                # 차원 확장
                token_state_expanded = torch.cat([
                    token_state, 
                    torch.zeros(self.max_superposition_dim, 3 - token_state.shape[1])
                ], dim=1)
                token_state_np = token_state_expanded.numpy()
            else:
                # PCA 적용
                pca = PCA(n_components=3)
                token_state_np = pca.fit_transform(token_state.numpy())
        except ImportError:
            # scikit-learn 없으면 간단한 차원 감소 적용
            if token_state.shape[1] < 3:
                # 차원 확장
                token_state_np = torch.cat([
                    token_state, 
                    torch.zeros(self.max_superposition_dim, 3 - token_state.shape[1])
                ], dim=1).numpy()
            else:
                # 처음 3개 차원만 사용
                token_state_np = token_state[:, :3].numpy()
        
        # 그림 생성
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 각 차원의 노름 계산
        norms = np.linalg.norm(token_state.numpy(), axis=1)
        
        # 점 크기 계산
        point_sizes = 100 * (norms / norms.max()) ** 2
        
        # 컬러맵에서 색상 가져오기
        colors = [plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim) 
                for i in range(self.max_superposition_dim)]
        
        # 각 중첩 차원을 3D 공간에 표시
        for i in range(self.max_superposition_dim):
            ax.scatter(
                token_state_np[i, 0],
                token_state_np[i, 1],
                token_state_np[i, 2],
                s=point_sizes[i],
                color=colors[i],
                alpha=0.7,
                label=f"Dim {i} (norm: {norms[i]:.2f})"
            )
            
            # 원점에서 해당 차원으로 화살표 표시
            ax.quiver(
                0, 0, 0,
                token_state_np[i, 0],
                token_state_np[i, 1],
                token_state_np[i, 2],
                color=colors[i],
                alpha=0.5
            )
        
        # 축 설정
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        
        # 원점 표시
        ax.scatter(0, 0, 0, color='black', s=50, marker='x')
        
        # 제목 설정
        ax.set_title(f"{title} (Token {token_idx})", fontsize=12)
        
        # 범례 설정
        ax.legend(fontsize=8, loc='upper right')
        
        # 그리드 설정
        ax.grid(True, alpha=0.3)
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_quantum_state_evolution(
        self,
        superposition_states: List[torch.Tensor],
        steps: Optional[List[int]] = None,
        token_idx: int = 0,
        title: str = "Quantum State Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        양자 상태 진화 시각화
        
        Args:
            superposition_states: 시간에 따른 중첩 상태 텐서 목록
            steps: 각 상태의 스텝 (없으면 자동 생성)
            token_idx: 시각화할 토큰 인덱스
            title: 그림 제목
            save_path: 저장 경로
            
        Returns:
            plt.Figure: 시각화 그림
        """
        # 스텝 자동 생성
        if steps is None:
            steps = list(range(len(superposition_states)))
            
        # 텐서 형태 확인
        batch_size, seq_len, dim = superposition_states[0].shape
        
        # 중첩 차원 계산
        d_model = dim // self.max_superposition_dim
        
        # 각 배치의 첫 번째 샘플만 표시
        sample_idx = 0
        
        # 그림 생성
        fig, axs = plt.subplots(
            2, 2, figsize=self.figsize, 
            gridspec_kw={'height_ratios': [1, 1.5]}
        )
        
        # 1. 상태 노름 진화
        ax = axs[0, 0]
        
        # 각 상태의 중첩 차원별 노름 추출
        state_norms = []
        
        for state_tensor in superposition_states:
            # CPU로 이동
            if state_tensor.device != torch.device("cpu"):
                state_tensor = state_tensor.cpu()
                
            # 중첩 상태 재구성
            reshaped = state_tensor.view(
                batch_size, seq_len, self.max_superposition_dim, d_model
            )
            
            # 선택한 토큰의 각 중첩 차원의 노름 계산
            token_norms = torch.norm(reshaped[sample_idx, token_idx], dim=1).numpy()
            state_norms.append(token_norms)
            
        # 중첩 차원별 노름 진화 그래프
        state_norms = np.array(state_norms)  # [num_steps, max_superposition_dim]
        
        for i in range(self.max_superposition_dim):
            ax.plot(
                steps,
                state_norms[:, i],
                label=f"Dim {i}",
                color=plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim),
                alpha=0.7
            )
            
        ax.set_title(f"Dimension Norms Evolution (Token {token_idx})", fontsize=12)
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. 상태 확률 진화
        ax = axs[0, 1]
        
        # 각 상태의 중첩 차원별 확률 계산
        state_probs = []
        
        for norm_vector in state_norms:
            # 노름 제곱으로 확률 계산
            prob_vector = norm_vector ** 2
            # 정규화
            prob_vector = prob_vector / prob_vector.sum()
            state_probs.append(prob_vector)
            
        state_probs = np.array(state_probs)  # [num_steps, max_superposition_dim]
        
        # 각 스텝의 확률 분포 스택 그래프
        bottom = np.zeros(len(steps))
        
        for i in range(self.max_superposition_dim):
            ax.fill_between(
                steps,
                bottom,
                bottom + state_probs[:, i],
                label=f"Dim {i}" if i == 0 else None,
                color=plt.cm.get_cmap(self.cmap)(i/self.max_superposition_dim),
                alpha=0.7
            )
            bottom += state_probs[:, i]
            
        ax.set_title(f"Dimension Probability Evolution (Token {token_idx})", fontsize=12)
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. 상태 진화 히트맵
        ax = axs[1, 0]
        
        # 히트맵용 데이터 준비
        max_display_steps = min(len(steps), 20)  # 최대 20개 스텝 표시
        step_indices = np.linspace(0, len(steps) - 1, max_display_steps, dtype=int)
        
        # 선택된 스텝의 확률 추출
        selected_probs = state_probs[step_indices]
        
        # 히트맵 표시
        im = ax.imshow(
            selected_probs.T,  # 전치하여 차원을 행으로, 스텝을 열로 표시
            aspect='auto',
            cmap=self.quantum_cmap,
            extent=[min(steps), max(steps), -0.5, self.max_superposition_dim - 0.5]
        )
        
        ax.set_title(f"State Evolution Heatmap (Token {token_idx})", fontsize=12)
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Superposition Dimension", fontsize=10)
        ax.set_yticks(range(self.max_superposition_dim))
        ax.tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=ax, label="Probability")
        
        # 4. 엔트로피 진화
        ax = axs[1, 1]
        
        # 각 상태의 확률 분포 엔트로피 계산
        entropies = []
        
        for prob_vector in state_probs:
            # 섀넌 엔트로피 계산
            epsilon = 1e-10  # 로그 0 방지
            entropy = -np.sum(prob_vector * np.log2(prob_vector + epsilon))
            entropies.append(entropy)
            
        # 최대 가능 엔트로피 (균등 분포)
        max_entropy = np.log2(self.max_superposition_dim)
        
        # 정규화된 엔트로피
        normalized_entropies = np.array(entropies) / max_entropy
        
        # 엔트로피 진화 그래프
        ax.plot(
            steps,
            entropies,
            label="Entropy",
            color="blue",
            alpha=0.7
        )
        
        # 최대 엔트로피 기준선
        ax.axhline(
            y=max_entropy,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Max Entropy ({max_entropy:.2f})"
        )
        
        # 정규화된 엔트로피 (두 번째 y축)
        ax2 = ax.twinx()
        ax2.plot(
            steps,
            normalized_entropies,
            label="Normalized",
            color="green",
            alpha=0.7
        )
        ax2.set_ylabel("Normalized Entropy", fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.set_ylim([0, 1.05])
        
        # 첫 번째 y축 설정
        ax.set_title(f"Quantum State Entropy Evolution (Token {token_idx})", fontsize=12)
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Entropy (bits)", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # 범례 결합
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        
        # 전체 제목 설정
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def to_html_string(self, fig: plt.Figure) -> str:
        """
        matplotlib 그림을 HTML 문자열로 변환
        
        Args:
            fig: matplotlib 그림
            
        Returns:
            str: HTML img 태그 문자열
        """
        # 이미지를 바이트 스트림으로 저장
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # 바이트를 base64로 인코딩
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # HTML img 태그 생성
        html_string = f'<img src="data:image/png;base64,{img_str}" alt="Visualization"/>'
        
        return html_string
    
    def close_all(self):
        """모든 그림 닫기"""
        plt.close('all')
