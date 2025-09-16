import torch
import torch.nn as nn
import argparse
import os
import json
import logging
import sys

# Ensure local modules can be imported when running as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Any, Optional, List

# Import the necessary modules from our project
from architecture.transformer import QuantumInspiredTransformer
from core.dual_state import DualStateController
from optimization.resource import ResourceAllocator
from optimization.learning import UniversalLoss, MetaLearningOptimizer
from optimization.efficiency import ComputationalEfficiencyFramework
from architecture.interface import QuantumInspiredInterface, ScalableModelInterface
from architecture.scaling import ScalingLaws, AdaptiveConfigurationFramework
from training.params import HyperParameters, HyperParameterOptimizer
from training.pipeline import QuantumTransformerTrainer, UncertaintyDatasetManager
from utils.visualization import QuantumStateVisualizer


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train a Quantum-Inspired Transformer model")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "small", "base", "large", "xl"],
                        help="Model size configuration")
    parser.add_argument("--max_superposition_dim", type=int, default=4,
                        help="Maximum superposition dimension")
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "regression", "generation"],
                        help="Task type for training")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Maximum gradient norm for clipping")
    
    # Quantum-specific parameters
    parser.add_argument("--superposition_degree", type=float, default=0.7,
                        help="Initial degree of superposition (0-1)")
    parser.add_argument("--collapse_threshold", type=float, default=0.5,
                        help="Threshold for state collapse (0-1)")
    parser.add_argument("--interference_strength", type=float, default=0.3,
                        help="Strength of interference effects (0-1)")
    
    # NEW: CollapseGate parameters
    parser.add_argument("--p_target", type=float, default=0.5,
                       help="Target transition probability for CollapseGate")
    parser.add_argument("--alpha_init", type=float, default=0.5,
                       help="Initial soft/hard collapse mixing ratio")
    parser.add_argument("--gate_type", type=str, default="mlp", choices=["mlp", "transformer"],
                       help="CollapseGate architecture type")
    
    # NEW: Curriculum learning parameters
    parser.add_argument("--use_curriculum", type=bool, default=True,
                       help="Whether to use curriculum learning")
    parser.add_argument("--curriculum_epochs", type=str, default="1,3,5",
                       help="Comma-separated list of epoch thresholds for curriculum stages")
    parser.add_argument("--curriculum_difficulties", type=str, default="easy,medium,hard",
                       help="Comma-separated list of curriculum difficulty levels")
    
    # Efficiency parameters
    parser.add_argument("--target_sparsity", type=float, default=0.7,
                        help="Target sparsity for efficient computation (0-1)")
    parser.add_argument("--use_meta_learning", type=bool, default=False,
                        help="Whether to use meta-learning optimization")
    
    # NEW: Resource efficiency parameters
    parser.add_argument("--resource_penalty_weight", type=float, default=0.1,
                       help="Weight for resource penalty in loss function")
    parser.add_argument("--uncertainty_correction_weight", type=float, default=0.2,
                       help="Weight for uncertainty correction in loss function")
                       
    # NEW: Learnable collapse scheduler
    parser.add_argument("--use_learnable_collapse", type=bool, default=True,
                       help="Whether to use learnable collapse scheduler")
    
    # I/O parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log training metrics every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Save model checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate model every N steps")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


def setup_logger(output_dir: str) -> logging.Logger:
    """Set up logger for training"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_dataset(data_dir: str, task_type: str) -> tuple:
    """
    Load dataset based on task type.
    This is a placeholder - implement your actual data loading logic.
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    try:
        from datasets import load_dataset
        
        if task_type == "classification":
            # For demonstration, load GLUE SST-2 for sentiment classification
            dataset = load_dataset("glue", "sst2")
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]
        elif task_type == "regression":
            # For demonstration, load GLUE STS-B for regression
            dataset = load_dataset("glue", "stsb")
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]
        elif task_type == "generation":
            # For demonstration, load a subset of CNN/DailyMail for summarization
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            train_dataset = dataset["train"].select(range(1000))  # Limit for demo
            val_dataset = dataset["validation"].select(range(100))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
    except ImportError:
        # Create dummy datasets if datasets library is not available
        from torch.utils.data import TensorDataset
        
        # Create random data
        x_train = torch.randn(1000, 10)
        y_train = torch.randint(0, 2, (1000,)) if task_type == "classification" else torch.randn(1000)
        
        x_val = torch.randn(100, 10)
        y_val = torch.randint(0, 2, (100,)) if task_type == "classification" else torch.randn(100)
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        
        print("Using dummy datasets (install 'datasets' library for real data)")
            
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset, val_dataset, batch_size: int, num_workers: int = 4
) -> tuple:
    """Create dataloaders for training and validation sets"""
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader


def create_curriculum_dataloaders(
    train_dataset, val_dataset, batch_size: int, num_workers: int = 4, model=None, device=None
) -> dict:
    """
    Create curriculum dataloaders for different difficulty levels
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        batch_size: Batch size
        num_workers: Number of workers
        model: Model to compute uncertainty (optional)
        device: Device to run model on
        
    Returns:
        dict: Dictionary of curriculum dataloaders
    """
    # If model provided, use it to estimate uncertainty
    if model is not None:
        # Create uncertainty dataset manager
        uncertainty_manager = UncertaintyDatasetManager(
            model=model,
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # Create curriculum datasets based on uncertainty
        curriculum_datasets = uncertainty_manager.create_curriculum_datasets()
        
        # Create dataloaders for each difficulty level
        curriculum_dataloaders = {}
        for difficulty, dataset in curriculum_datasets.items():
            # Adjust batch size based on difficulty
            current_batch_size = batch_size
            if difficulty == "easy":
                current_batch_size = batch_size // 2  # Smaller batch for easy examples
            elif difficulty == "hard":
                current_batch_size = batch_size * 2  # Larger batch for hard examples
                
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=max(1, current_batch_size),
                shuffle=True,
                num_workers=num_workers
            )
            curriculum_dataloaders[difficulty] = dataloader
            
        # Add validation dataloader
        curriculum_dataloaders["validation"] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return curriculum_dataloaders
    else:
        # Simple curriculum based on dataset size
        dataset_size = len(train_dataset)
        
        # Easy: first 30% of data
        easy_size = int(dataset_size * 0.3)
        easy_indices = list(range(easy_size))
        easy_dataset = torch.utils.data.Subset(train_dataset, easy_indices)
        
        # Medium: middle 40% of data
        medium_start = easy_size
        medium_end = int(dataset_size * 0.7)
        medium_indices = list(range(medium_start, medium_end))
        medium_dataset = torch.utils.data.Subset(train_dataset, medium_indices)
        
        # Hard: remaining 30% of data
        hard_start = medium_end
        hard_indices = list(range(hard_start, dataset_size))
        hard_dataset = torch.utils.data.Subset(train_dataset, hard_indices)
        
        # Create dataloaders
        curriculum_dataloaders = {
            "easy": torch.utils.data.DataLoader(
                easy_dataset,
                batch_size=max(1, batch_size // 2),
                shuffle=True,
                num_workers=num_workers
            ),
            "medium": torch.utils.data.DataLoader(
                medium_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            ),
            "hard": torch.utils.data.DataLoader(
                hard_dataset,
                batch_size=batch_size * 2,
                shuffle=True,
                num_workers=num_workers
            ),
            "validation": torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        }
        
        return curriculum_dataloaders


def build_model(args) -> nn.Module:
    """Build a quantum-inspired transformer model based on arguments"""
    # Create a scalable model interface
    model_interface = ScalableModelInterface()
    
    # Get adaptive configuration for current hardware
    adaptive_framework = AdaptiveConfigurationFramework()
    config = adaptive_framework.generate_adaptive_config(
        target_size=args.model_size,
        task_complexity=0.5  # Medium complexity by default
    )
    
    # Override with command-line arguments
    config.update({
        "max_superposition_dim": args.max_superposition_dim,
        "superposition_degree": args.superposition_degree,
        "collapse_threshold": args.collapse_threshold,
        "interference_strength": args.interference_strength,
        "target_sparsity": args.target_sparsity,
        "p_target": args.p_target,
        "alpha_init": args.alpha_init,
        "gate_type": args.gate_type
    })
    
    # Build the model
    model = model_interface.build_model(
        size=args.model_size,
        custom_config=config,
        pretrained=False
    )
    
    return model


def main():
    """Main function to train a Quantum-Inspired Transformer model"""
    args = parse_args()
    
    # Process curriculum arguments
    if isinstance(args.curriculum_epochs, str):
        args.curriculum_epochs = [int(x) for x in args.curriculum_epochs.split(",")]
        
    if isinstance(args.curriculum_difficulties, str):
        args.curriculum_difficulties = args.curriculum_difficulties.split(",")
    
    # Set up logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, val_dataset = load_dataset(args.data_dir, args.task_type)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, val_dataset, args.batch_size
    )
    
    # Build model
    logger.info(f"Building {args.model_size} Quantum-Inspired Transformer model...")
    model = build_model(args)
    
    # Create curriculum dataloaders if enabled
    if args.use_curriculum:
        logger.info("Creating curriculum dataloaders...")
        curriculum_dataloaders = create_curriculum_dataloaders(
            train_dataset, val_dataset, args.batch_size, 
            model=model, device=args.device
        )
        logger.info(f"Created curriculum dataloaders with difficulties: {list(curriculum_dataloaders.keys())}")
    
    # Set up hyperparameters
    hparams = HyperParameters(
        # Model structure
        d_model=model.d_model if hasattr(model, 'd_model') else 768,
        nhead=model.nhead if hasattr(model, 'nhead') else 12,
        num_encoder_layers=model.num_encoder_layers if hasattr(model, 'num_encoder_layers') else 12,
        num_decoder_layers=model.num_decoder_layers if hasattr(model, 'num_decoder_layers') else 12,
        max_superposition_dim=args.max_superposition_dim,
        
        # Quantum parameters
        superposition_degree=args.superposition_degree,
        collapse_threshold=args.collapse_threshold,
        interference_strength=args.interference_strength,
        
        # NEW: CollapseGate parameters
        p_target=args.p_target,
        alpha_init=args.alpha_init,
        gate_type=args.gate_type,
        
        # NEW: Loss function weights
        task_loss_weight=0.6,
        superposition_reg_weight=0.2,
        consistency_loss_weight=0.2,
        uncertainty_correction_weight=args.uncertainty_correction_weight,
        resource_penalty_weight=args.resource_penalty_weight,
        
        # NEW: Curriculum learning parameters
        use_curriculum=args.use_curriculum,
        curriculum_epochs=args.curriculum_epochs,
        curriculum_difficulties=args.curriculum_difficulties,
        
        
        # Training parameters
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=int(0.1 * len(train_dataloader) * args.epochs),
        max_steps=len(train_dataloader) * args.epochs,
        
        # Efficiency parameters
        target_sparsity=args.target_sparsity,
        use_meta_learning=args.use_meta_learning,
        
        # Other parameters
        fp16=args.fp16,
        gradient_checkpointing=False
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = QuantumTransformerTrainer(
        model=model,
        train_dataloader=train_dataloader if not args.use_curriculum else None,
        val_dataloader=val_dataloader,
        hparams=hparams.to_dict(),
        output_dir=args.output_dir,
        device=args.device,
        distributed=args.distributed,
        fp16=args.fp16
    )
    
    # Start training
    logger.info("Starting training...")
    results = trainer.train()
    
    # Log final results
    logger.info(f"Training completed with results: {json.dumps(results, indent=2)}")
    
    # Save the final model
    model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Visualize model states if requested
    if hasattr(args, "visualize") and args.visualize:
        logger.info("Creating visualizations...")
        visualizer = QuantumStateVisualizer(
            max_superposition_dim=args.max_superposition_dim,
            device=args.device
        )
        
        # Create sample input
        sample_input = torch.randn(1, 10, model.d_model).to(args.device)
        
        # Get model states
        with torch.no_grad():
            outputs = model(sample_input, return_all_states=True)
            
        # Visualize superposition state
        if "superposition_state" in outputs:
            fig = visualizer.visualize_superposition(
                outputs["superposition_state"].cpu(),
                title="Superposition State Visualization",
                max_tokens=8,
                save_path=os.path.join(args.output_dir, "superposition_state.png")
            )
            
        # Visualize collapse process
        if "superposition_state" in outputs and "output" in outputs:
            fig = visualizer.visualize_collapse_process(
                outputs["superposition_state"].cpu(),
                outputs["output"].cpu(),
                title="Quantum State Collapse Visualization",
                max_tokens=8,
                save_path=os.path.join(args.output_dir, "collapse_process.png")
            )
            
        logger.info(f"Visualizations saved to {args.output_dir}")
    
    return results


# Example dataset configurations
DATASET_EXAMPLES = """
Example Datasets:

1. Classification Task (Sentiment Analysis):
   - Hugging Face Dataset: "glue/sst2"
   - Command: python example.py --task_type classification --data_dir ./data --model_size small

2. Regression Task (Semantic Similarity):
   - Hugging Face Dataset: "glue/stsb" 
   - Command: python example.py --task_type regression --data_dir ./data --model_size base

3. Generation Task (Summarization):
   - Hugging Face Dataset: "cnn_dailymail/3.0.0"
   - Command: python example.py --task_type generation --data_dir ./data --model_size large --max_superposition_dim 6

4. Custom Math Dataset with Curriculum Learning:
   - Create dataset with varying difficulty levels
   - Command: python example.py --task_type classification --use_curriculum True --curriculum_epochs 1,3,5 --curriculum_difficulties easy,medium,hard
   
5. Resource-Efficient Configuration:
   - Command: python example.py --target_sparsity 0.8 --p_target 0.7 --resource_penalty_weight 0.3
"""


if __name__ == "__main__":
    print(DATASET_EXAMPLES)
    main()