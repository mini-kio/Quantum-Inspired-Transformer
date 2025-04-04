import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os
import json
import time
import math
import logging
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Iterator

# 프로젝트의 다른 모듈 임포트
from ..architecture.transformer import QuantumInspiredTransformer
from ..core.dual_state import DualStateController
from ..optimization.learning import UniversalLoss, MetaLearningOptimizer
from ..optimization.efficiency import ComputationalEfficiencyFramework
from ..training.hyperparameters import HyperParameters, AdaptiveTuningScheduler, QuantumMetaScheduler


class QuantumTransformerTrainer:
    """
    양자 영감 트랜스포머 모델 훈련 파이프라인
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        hparams: Optional[Dict[str, Any]] = None,
        output_dir: str = './outputs',
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,
        fp16: bool = False
    ):
        """
        훈련 파이프라인 초기화
        
        Args:
            model: 양자 영감 트랜스포머 모델
            train_dataloader: 훈련 데이터 로더
            val_dataloader: 검증 데이터 로더
            hparams: 하이퍼파라미터
            output_dir: 출력 디렉토리
            device: 사용할 장치
            distributed: 분산 훈련 여부
            fp16: 16비트 부동소수점 사용 여부
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.hparams = hparams or {}
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed
        self.fp16 = fp16
        
        # 하이퍼파라미터 설정
        if not isinstance(self.hparams, dict):
            self.hparams = self.hparams.to_dict() if hasattr(self.hparams, 'to_dict') else vars(self.hparams)
        
        # 기본 하이퍼파라미터
        self.hparams.setdefault('learning_rate', 1e-4)
        self.hparams.setdefault('weight_decay', 0.01)
        self.hparams.setdefault('beta1', 0.9)
        self.hparams.setdefault('beta2', 0.999)
        self.hparams.setdefault('warmup_steps', 10000)
        self.hparams.setdefault('max_steps', 500000)
        self.hparams.setdefault('save_steps', 10000)
        self.hparams.setdefault('log_steps', 100)
        self.hparams.setdefault('eval_steps', 5000)
        self.hparams.setdefault('gradient_accumulation_steps', 1)
        self.hparams.setdefault('max_grad_norm', 1.0)
        
        # 훈련 상태 초기화
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 모델을 장치로 이동
        self.model.to(self.device)
        
        # 분산 훈련 설정
        if self.distributed:
            self.setup_distributed()
        
        # 최적화기 및 스케줄러 설정
        self.setup_optimizer()
        
        # 손실 함수 설정
        self.setup_loss_function()
        
        # 효율성 프레임워크 설정
        self.setup_efficiency_framework()
        
        # 16비트 부동소수점 설정
        if self.fp16:
            self.setup_fp16()
        
        # 적응형 스케줄러 설정
        self.setup_adaptive_schedulers()
        
        # 체크포인트 복원
        self.restore_checkpoint()
        
    def setup_logging(self) -> None:
        """로깅 설정"""
        log_file = os.path.join(self.output_dir, 'training.log')
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
        
    def setup_distributed(self) -> None:
        """분산 훈련 설정"""
        if not dist.is_initialized():
            self.logger.warning("Distributed training requested but not initialized. Running in non-distributed mode.")
            self.distributed = False
            return
        
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if 'cuda' in str(self.device) else None
        )
        
        self.logger.info(f"Distributed training enabled. Rank: {self.local_rank}/{self.world_size}")
        
    def setup_optimizer(self) -> None:
        """최적화기 및 스케줄러 설정"""
        # 옵티마이저 그룹 설정
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # 옵티마이저 초기화
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams['learning_rate'],
            betas=(self.hparams['beta1'], self.hparams['beta2'])
        )
        
        # 스케줄러 설정
        warmup_steps = self.hparams['warmup_steps']
        max_steps = self.hparams['max_steps']
        
        # 역제곱근 스케줄러
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def setup_loss_function(self) -> None:
        """손실 함수 설정"""
        # UniversalLoss 초기화
        self.loss_fn = UniversalLoss(
            hidden_dim=self.model.d_model if hasattr(self.model, 'd_model') else 768,
            task_types=['classification', 'regression', 'generation'],
            alpha=self.hparams.get('task_loss_weight', 0.6),
            beta=self.hparams.get('superposition_reg_weight', 0.2),
            gamma=self.hparams.get('consistency_loss_weight', 0.2)
        )
        
        # 필요한 경우 메타 학습 최적화기 설정
        if self.hparams.get('use_meta_learning', False):
            self.meta_optimizer = MetaLearningOptimizer(
                model=self.model,
                hidden_dim=self.model.d_model if hasattr(self.model, 'd_model') else 768,
                learning_rate=self.hparams.get('meta_learning_rate', 1e-5),
                meta_steps=self.hparams.get('meta_steps', 3)
            )
            
    def setup_efficiency_framework(self) -> None:
        """효율성 프레임워크 설정"""
        self.efficiency_framework = ComputationalEfficiencyFramework(
            hidden_dim=self.model.d_model if hasattr(self.model, 'd_model') else 768,
            max_superposition_dim=self.model.max_superposition_dim if hasattr(self.model, 'max_superposition_dim') else 4,
            sparsity_target=self.hparams.get('target_sparsity', 0.7)
        )
        
    def setup_fp16(self) -> None:
        """16비트 부동소수점 설정"""
        try:
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
            self.autocast = autocast
            self.logger.info("FP16 training enabled")
        except ImportError:
            self.logger.warning("PyTorch version does not support AMP. Running in FP32 mode.")
            self.fp16 = False
            
    def setup_adaptive_schedulers(self) -> None:
        """적응형 스케줄러 설정"""
        # 적응형 튜닝 스케줄러
        steps_per_epoch = len(self.train_dataloader)
        self.tuning_scheduler = AdaptiveTuningScheduler(
            hparams=self.hparams,
            total_steps=self.hparams.get('max_steps', 500000),
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=self.hparams.get('warmup_epochs', 1),
            schedule_type=self.hparams.get('schedule_type', 'linear')
        )
        
        # 양자 메타 스케줄러
        self.meta_scheduler = QuantumMetaScheduler(
            hparams=self.hparams,
            task_type=self.hparams.get('task_type', 'classification'),
            adaptation_rate=self.hparams.get('adaptation_rate', 0.1)
        )
        
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        체크포인트 저장
        
        Args:
            is_best: 현재 모델이 최고 성능인지 여부
        """
        if self.distributed and self.local_rank != 0:
            return
        
        # 기본 체크포인트 저장
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pt')
        best_checkpoint_path = os.path.join(self.output_dir, 'best_checkpoint.pt')
        
        # 저장할 상태 준비
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'hparams': self.hparams
        }
        
        # 체크포인트 저장
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at step {self.step} to {checkpoint_path}")
        
        # 최고 모델 저장
        if is_best:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)
            self.logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
            
    def restore_checkpoint(self) -> None:
        """이전 체크포인트 복원"""
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint found. Starting from scratch.")
            return
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 상태 복원
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # 옵티마이저 및 스케줄러 상태 복원
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # 훈련 상태 복원
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        
        self.logger.info(f"Restored checkpoint from step {self.step}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        단일 훈련 스텝 수행
        
        Args:
            batch: 데이터 배치
            
        Returns:
            Dict[str, float]: 훈련 측정 지표
        """
        self.model.train()
        
        # 입력 및 타겟 추출
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device) if 'labels' in batch else None
        
        # 적응형 파라미터 적용
        adaptive_params = self.tuning_scheduler.step(self.step)
        inputs['superposition_degree'] = adaptive_params.get('superposition_degree')
        inputs['collapse_threshold'] = adaptive_params.get('collapse_threshold')
        inputs['interference_strength'] = adaptive_params.get('interference_strength')
        
        # 효율성 최적화 적용
        if self.step % self.hparams.get('efficiency_steps', 10) == 0:
            # 컨텍스트 추출
            context = inputs.get('input_embeddings', inputs.get('input_ids')).mean(dim=1)
            efficiency_result = self.efficiency_framework(context)
            inputs['computation_mask'] = efficiency_result.get('computation_mask')
        
        # 메타 학습 사용 여부
        use_meta = self.hparams.get('use_meta_learning', False) and self.step % self.hparams.get('meta_steps', 100) == 0
        
        # FP16 사용 여부에 따른 순전파
        if self.fp16:
            with self.autocast():
                if use_meta:
                    # 메타 학습 최적화기 적용
                    meta_result = self.meta_optimizer(inputs, labels)
                    outputs = meta_result['optimized_outputs']
                else:
                    # 일반 순전파
                    outputs = self.model(**inputs, return_all_states=True)
                
                # 손실 계산
                loss_inputs = {
                    'outputs': outputs,
                    'targets': labels,
                    'task_type': self.hparams.get('task_type', 'classification'),
                    'context_embedding': outputs.get('context', inputs.get('input_embeddings', inputs.get('input_ids')).mean(dim=1))
                }
                loss_result = self.loss_fn(**loss_inputs)
                loss = loss_result['loss']
                
            # 그래디언트 스케일링 및 역전파
            self.scaler.scale(loss).backward()
            
            if (self.step + 1) % self.hparams.get('gradient_accumulation_steps', 1) == 0:
                # 그래디언트 클리핑
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.hparams.get('max_grad_norm', 1.0)
                )
                
                # 파라미터 업데이트
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
        else:
            # FP32 모드에서의 순전파
            if use_meta:
                # 메타 학습 최적화기 적용
                meta_result = self.meta_optimizer(inputs, labels)
                outputs = meta_result['optimized_outputs']
            else:
                # 일반 순전파
                outputs = self.model(**inputs, return_all_states=True)
            
            # 손실 계산
            loss_inputs = {
                'outputs': outputs,
                'targets': labels,
                'task_type': self.hparams.get('task_type', 'classification'),
                'context_embedding': outputs.get('context', inputs.get('input_embeddings', inputs.get('input_ids')).mean(dim=1))
            }
            loss_result = self.loss_fn(**loss_inputs)
            loss = loss_result['loss']
            
            # 역전파
            loss.backward()
            
            if (self.step + 1) % self.hparams.get('gradient_accumulation_steps', 1) == 0:
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.hparams.get('max_grad_norm', 1.0)
                )
                
                # 파라미터 업데이트
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
        
        # 메트릭 추출
        metrics = {
            'loss': loss.item(),
            'task_loss': loss_result['task_loss'].item(),
            'superposition_reg_loss': loss_result['superposition_reg_loss'].item(),
            'consistency_loss': loss_result['consistency_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.hparams['learning_rate']
        }
        
        # 메타 스케줄러 업데이트
        if self.step % self.hparams.get('meta_update_steps', 100) == 0:
            updated_params = self.meta_scheduler.update(metrics)
            metrics.update({f"meta_{k}": v for k, v in updated_params.items()})
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        모델 평가
        
        Returns:
            Dict[str, float]: 평가 측정 지표
        """
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # 추가 메트릭 추적
        metrics = {}
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            # 입력 및 타겟 추출
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device) if 'labels' in batch else None
            
            # 순전파
            outputs = self.model(**inputs, return_all_states=True, force_collapse=True)
            
            # 손실 계산
            loss_inputs = {
                'outputs': outputs,
                'targets': labels,
                'task_type': self.hparams.get('task_type', 'classification'),
                'context_embedding': outputs.get('context', inputs.get('input_embeddings', inputs.get('input_ids')).mean(dim=1))
            }
            loss_result = self.loss_fn(**loss_inputs)
            loss = loss_result['loss']
            
            batch_size = labels.size(0) if labels is not None else inputs[list(inputs.keys())[0]].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 추가 메트릭 계산 (태스크별)
            task_type = self.hparams.get('task_type', 'classification')
            
            if task_type == 'classification':
                # 분류 태스크 메트릭 계산 (정확도)
                if 'logits' in outputs:
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=-1)
                    accuracy = (preds == labels).float().mean().item()
                    metrics.setdefault('accuracy', 0.0)
                    metrics['accuracy'] += accuracy * batch_size
                    
            elif task_type == 'regression':
                # 회귀 태스크 메트릭 계산 (MSE, MAE)
                if 'prediction' in outputs:
                    prediction = outputs['prediction']
                    mse = ((prediction - labels) ** 2).mean().item()
                    mae = (prediction - labels).abs().mean().item()
                    metrics.setdefault('mse', 0.0)
                    metrics.setdefault('mae', 0.0)
                    metrics['mse'] += mse * batch_size
                    metrics['mae'] += mae * batch_size
        
        # 평균 손실 및 메트릭 계산
        avg_loss = total_loss / total_samples
        
        eval_metrics = {
            'val_loss': avg_loss
        }
        
        # 추가 메트릭 평균 계산
        for metric_name, metric_value in metrics.items():
            eval_metrics[f"val_{metric_name}"] = metric_value / total_samples
        
        # 최고 성능 업데이트
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            is_best = True
        else:
            is_best = False
            
        # 정확도가 있으면 추가 최고 성능 지표로 사용
        if 'val_accuracy' in eval_metrics and eval_metrics['val_accuracy'] > self.best_val_metric:
            self.best_val_metric = eval_metrics['val_accuracy']
            is_best = True
        
        # 최고 모델 저장
        if is_best:
            self.save_checkpoint(is_best=True)
        
        return eval_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        모델 훈련 실행
        
        Returns:
            Dict[str, Any]: 훈련 결과
        """
        max_steps = self.hparams.get('max_steps', 500000)
        save_steps = self.hparams.get('save_steps', 10000)
        log_steps = self.hparams.get('log_steps', 100)
        eval_steps = self.hparams.get('eval_steps', 5000)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Max steps: {max_steps}")
        self.logger.info(f"Save steps: {save_steps}")
        self.logger.info(f"Log steps: {log_steps}")
        self.logger.info(f"Eval steps: {eval_steps}")
        
        # 훈련 시작 시간
        start_time = time.time()
        
        # 훈련 지표 추적
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'superposition_degree': [],
            'collapse_threshold': [],
            'interference_strength': []
        }
        
        # 메인 훈련 루프
        while self.step < max_steps:
            self.epoch += 1
            self.logger.info(f"Starting epoch {self.epoch}")
            
            # 에포크 내 배치 반복
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}"):
                # 훈련 스텝 수행
                metrics = self.train_step(batch)
                
                # 지표 기록
                for key, value in metrics.items():
                    if key not in metrics_history:
                        metrics_history[key] = []
                    metrics_history[key].append(value)
                
                # 로깅
                if self.step % log_steps == 0:
                    lr = metrics['learning_rate']
                    loss = metrics['loss']
                    elapsed_time = time.time() - start_time
                    
                    log_message = (
                        f"Step {self.step}/{max_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"LR: {lr:.6f} | "
                        f"Time: {elapsed_time:.2f}s"
                    )
                    
                    # 적응형 파라미터 로깅
                    adaptive_params = self.tuning_scheduler.get_current_values()
                    
                    for param_name, param_value in adaptive_params.items():
                        log_message += f" | {param_name}: {param_value:.4f}"
                        metrics_history.setdefault(param_name, []).append(param_value)
                    
                    self.logger.info(log_message)
                
                # 평가
                if self.step % eval_steps == 0:
                    self.logger.info(f"Evaluating at step {self.step}...")
                    eval_metrics = self.evaluate()
                    
                    # 평가 결과 로깅
                    self.logger.info(f"Validation results: {json.dumps(eval_metrics, indent=2)}")
                    
                    # 평가 지표 기록
                    for key, value in eval_metrics.items():
                        if key not in metrics_history:
                            metrics_history[key] = []
                        # 평가는 매 스텝마다 하지 않으므로 중간 스텝은 None으로 채움
                        while len(metrics_history[key]) < len(metrics_history['train_loss']):
                            metrics_history[key].append(None)
                        metrics_history[key].append(value)
                
                # 체크포인트 저장
                if self.step % save_steps == 0:
                    self.save_checkpoint()
                
                # 지표 저장
                if self.step % (save_steps // 5) == 0:
                    self.save_metrics(metrics_history)
                    
                # 스텝 증가
                self.step += 1
                
                # 최대 스텝 도달 시 종료
                if self.step >= max_steps:
                    break
                    
            # 에포크 종료 후 평가
            self.logger.info(f"Epoch {self.epoch} completed. Evaluating...")
            eval_metrics = self.evaluate()
            self.logger.info(f"Epoch {self.epoch} validation results: {json.dumps(eval_metrics, indent=2)}")
        
        # 훈련 완료 체크포인트 저장
        self.save_checkpoint()
        
        # 최종 지표 저장
        self.save_metrics(metrics_history)
        
        # 훈련 시간 계산
        total_time = time.time() - start_time
        
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation metric: {self.best_val_metric:.4f}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'total_steps': self.step,
            'epochs': self.epoch,
            'total_time': total_time
        }
    
    def save_metrics(self, metrics_history: Dict[str, List]) -> None:
        """
        훈련 지표 저장
        
        Args:
            metrics_history: 지표 히스토리
        """
        if self.distributed and self.local_rank != 0:
            return
            
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)


class UncertaintyDatasetManager:
    """
    불확실성 기반 데이터셋 관리
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        uncertainty_threshold: float = 0.7,
        superposition_dim: int = 4,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        불확실성 데이터셋 관리자 초기화
        
        Args:
            model: 양자 영감 트랜스포머 모델
            dataset: 원본 데이터셋
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
            uncertainty_threshold: 불확실성 임계값
            superposition_dim: 중첩 차원
            device: 사용할 장치
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uncertainty_threshold = uncertainty_threshold
        self.superposition_dim = superposition_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델을 장치로 이동
        self.model.to(self.device)
        
        # 불확실성 점수
        self.uncertainty_scores = None
        
        # 불확실성 기반 샘플링 가중치
        self.sampling_weights = None
        
    @torch.no_grad()
    def compute_uncertainty_scores(self) -> np.ndarray:
        """
        데이터셋의 각 샘플에 대한 불확실성 점수 계산
        
        Returns:
            np.ndarray: 불확실성 점수
        """
        self.model.eval()
        
        # 데이터 로더 생성
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        # 불확실성 점수 저장소
        all_uncertainties = []
        
        for batch in tqdm(dataloader, desc="Computing uncertainty scores"):
            # 입력 추출
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            
            # 중첩 상태 반환을 위한 설정
            inputs['return_all_states'] = True
            
            # 순전파
            outputs = self.model(**inputs)
            
            # 불확실성 추정
            if 'superposition_state' in outputs:
                # 중첩 상태에서 불확실성 계산
                superposition_state = outputs['superposition_state']
                batch_size = superposition_state.size(0)
                
                # 중첩 상태 재구성
                reshaped = superposition_state.view(
                    batch_size, -1, self.superposition_dim, self.model.d_model
                )
                
                # 중첩 상태의 분산 계산
                variance = reshaped.var(dim=2).mean(dim=-1)
                uncertainties = variance.mean(dim=1).cpu().numpy()
            else:
                # 중첩 상태가 없는 경우 기본값 사용
                batch_size = list(inputs.values())[0].size(0)
                uncertainties = np.ones(batch_size) * 0.5
            
            all_uncertainties.append(uncertainties)
        
        # 모든 불확실성 점수 결합
        self.uncertainty_scores = np.concatenate(all_uncertainties)
        
        return self.uncertainty_scores
    
    def create_sampling_weights(self, alpha: float = 1.0) -> np.ndarray:
        """
        불확실성 점수에 기반한 샘플링 가중치 생성
        
        Args:
            alpha: 불확실성 영향력 계수 (0: 균등 샘플링, 1: 불확실성 기반 샘플링)
            
        Returns:
            np.ndarray: 샘플링 가중치
        """
        if self.uncertainty_scores is None:
            self.compute_uncertainty_scores()
        
        # 불확실성 점수 정규화
        normalized_scores = self.uncertainty_scores / self.uncertainty_scores.sum()
        
        # 균등 가중치
        uniform_weights = np.ones_like(normalized_scores) / len(normalized_scores)
        
        # 불확실성 기반 가중치 계산
        self.sampling_weights = (1 - alpha) * uniform_weights + alpha * normalized_scores
        self.sampling_weights /= self.sampling_weights.sum()
        
        return self.sampling_weights
    
    def create_stratified_datasets(self, num_levels: int = 3) -> List[torch.utils.data.Dataset]:
        """
        불확실성 수준에 따라 데이터셋 층화
        
        Args:
            num_levels: 불확실성 수준 수
            
        Returns:
            List[torch.utils.data.Dataset]: 층화된 데이터셋 목록
        """
        if self.uncertainty_scores is None:
            self.compute_uncertainty_scores()
        
        # 불확실성 점수 정렬 및 인덱스 추출
        sorted_indices = np.argsort(self.uncertainty_scores)
        
        # 각 레벨당 샘플 수
        samples_per_level = len(sorted_indices) // num_levels
        
        # 층화 데이터셋 생성
        stratified_datasets = []
        
        for i in range(num_levels):
            # 현재 레벨의 시작 및 끝 인덱스
            start_idx = i * samples_per_level
            end_idx = (i + 1) * samples_per_level if i < num_levels - 1 else len(sorted_indices)
            
            # 현재 레벨의 인덱스
            level_indices = sorted_indices[start_idx:end_idx]
            
            # 부분 데이터셋 생성
            level_dataset = torch.utils.data.Subset(self.dataset, level_indices)
            stratified_datasets.append(level_dataset)
        
        return stratified_datasets
    
    def create_weighted_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """
        불확실성 기반 가중치 샘플러 생성
        
        Returns:
            torch.utils.data.WeightedRandomSampler: 가중치 샘플러
        """
        if self.sampling_weights is None:
            self.create_sampling_weights()
        
        # 가중치 샘플러 생성
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.sampling_weights,
            num_samples=len(self.sampling_weights),
            replacement=True
        )
        
        return sampler
    
    def create_uncertainty_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        use_weighted_sampling: bool = True,
        alpha: float = 1.0
    ) -> torch.utils.data.DataLoader:
        """
        불확실성 기반 데이터 로더 생성
        
        Args:
            batch_size: 배치 크기
            shuffle: 셔플 여부
            use_weighted_sampling: 가중치 샘플링 사용 여부
            alpha: 불확실성 영향력 계수
            
        Returns:
            torch.utils.data.DataLoader: 데이터 로더
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if use_weighted_sampling:
            # 가중치 샘플러 생성
            if self.sampling_weights is None:
                self.create_sampling_weights(alpha=alpha)
                
            sampler = self.create_weighted_sampler()
            
            # 데이터 로더 생성
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.num_workers
            )
        else:
            # 일반 데이터 로더 생성
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )
            
        return dataloader
    
    def create_curriculum_dataloaders(
        self,
        num_stages: int = 3,
        batch_size: Optional[int] = None
    ) -> List[torch.utils.data.DataLoader]:
        """
        난이도에 따른 커리큘럼 데이터 로더 생성
        
        Args:
            num_stages: 커리큘럼 단계 수
            batch_size: 배치 크기
            
        Returns:
            List[torch.utils.data.DataLoader]: 커리큘럼 데이터 로더 목록
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 층화 데이터셋 생성
        stratified_datasets = self.create_stratified_datasets(num_levels=num_stages)
        
        # 커리큘럼 데이터 로더 생성
        curriculum_dataloaders = []
        
        for dataset in stratified_datasets:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            curriculum_dataloaders.append(dataloader)
            
        return curriculum_dataloaders
    
    def get_high_uncertainty_samples(
        self,
        threshold: Optional[float] = None,
        max_samples: int = 100
    ) -> List[int]:
        """
        높은 불확실성을 가진 샘플 인덱스 반환
        
        Args:
            threshold: 불확실성 임계값
            max_samples: 최대 샘플 수
            
        Returns:
            List[int]: 높은 불확실성 샘플 인덱스
        """
        if self.uncertainty_scores is None:
            self.compute_uncertainty_scores()
        
        if threshold is None:
            threshold = self.uncertainty_threshold
        
        # 불확실성 임계값 이상의 샘플 선택
        high_uncertainty_indices = np.where(self.uncertainty_scores >= threshold)[0]
        
        # 상위 max_samples개 선택
        if len(high_uncertainty_indices) > max_samples:
            sorted_indices = np.argsort(self.uncertainty_scores[high_uncertainty_indices])[::-1]
            high_uncertainty_indices = high_uncertainty_indices[sorted_indices[:max_samples]]
        
        return high_uncertainty_indices.tolist()
