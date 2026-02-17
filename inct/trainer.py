"""
Training utilities for Instant NGP models.

Provides a Trainer class with support for:
- AdamW optimizer
- Learning rate scheduling
- Checkpointing
- Logging (console and optional wandb)
- Early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from tqdm import tqdm
import time
import json

from .model import InstantNGPModel
from .utils import psnr, mse


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-15
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # 'cosine', 'step', 'none'
    lr_warmup_steps: int = 100
    lr_min: float = 1e-5
    step_lr_gamma: float = 0.1
    step_lr_milestones: List[int] = field(default_factory=lambda: [500, 750])
    
    # Training loop
    num_epochs: int = 100
    log_interval: int = 10
    eval_interval: int = 50
    checkpoint_interval: int = 100
    
    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 1e-6
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints_inct"
    save_best: bool = True
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "inct"
    wandb_run_name: Optional[str] = None
    
    # Loss
    loss_fn: str = "mse"  # 'mse', 'l1', 'huber'


class Trainer:
    """
    Trainer for Instant NGP models.
    
    Handles the training loop, optimization, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: InstantNGPModel,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: InstantNGPModel to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
        
        # Setup loss function
        if config.loss_fn == "mse":
            self.loss_fn = F.mse_loss
        elif config.loss_fn == "l1":
            self.loss_fn = F.l1_loss
        elif config.loss_fn == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")
        
        # Setup learning rate scheduler
        self.scheduler = None
        self._setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        
        # Early stopping
        self.early_stopping_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        config = self.config
        
        if config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.lr_min,
            )
        elif config.lr_scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.step_lr_milestones,
                gamma=config.step_lr_gamma,
            )
        elif config.lr_scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    'model': self.model.config,
                    'training': vars(self.config),
                },
            )
        except ImportError:
            print("Warning: wandb not installed. Disabling wandb logging.")
            self.config.use_wandb = False
    
    def _warmup_lr(self, step: int):
        """Apply learning rate warmup."""
        if step < self.config.lr_warmup_steps:
            warmup_factor = (step + 1) / self.config.lr_warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with 'coords' and 'values' tensors
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move to device
        coords = batch['coords'].to(self.device)
        values = batch['values'].to(self.device)
        
        # Forward pass
        pred = self.model(coords)
        
        # Compute loss
        loss = self.loss_fn(pred, values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader: DataLoader, n_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation
            n_batches: Number of batches to evaluate (None = all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if n_batches is not None and i >= n_batches:
                    break
                
                coords = batch['coords'].to(self.device)
                values = batch['values'].to(self.device)
                
                pred = self.model(coords)
                
                loss = self.loss_fn(pred, values)
                batch_mse = F.mse_loss(pred, values)
                
                batch_size = coords.shape[0]
                total_loss += loss.item() * batch_size
                total_mse += batch_mse.item() * batch_size
                n_samples += batch_size
        
        avg_loss = total_loss / max(n_samples, 1)
        avg_mse = total_mse / max(n_samples, 1)
        avg_psnr = psnr(avg_mse)
        
        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'psnr': avg_psnr,
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Args:
            train_dataloader: DataLoader for training
            val_dataloader: Optional DataLoader for validation
            verbose: Print progress
            
        Returns:
            Dictionary with training history
        """
        config = self.config
        start_time = time.time()
        
        if verbose:
            print(f"Starting training for {config.num_epochs} epochs")
            print(f"Device: {self.device}")
            print(self.model.get_model_info())
        
        for epoch in range(config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            n_batches = 0
            
            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", 
                       disable=not verbose)
            
            for batch in pbar:
                # Learning rate warmup
                self._warmup_lr(self.global_step)
                
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1
                
                self.global_step += 1
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss:.6f}',
                    'lr': f'{current_lr:.2e}',
                })
                
                # Logging
                if self.global_step % config.log_interval == 0:
                    self.train_losses.append(loss)
                    self.learning_rates.append(current_lr)
                    
                    if self.config.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': loss,
                            'train/lr': current_lr,
                            'train/step': self.global_step,
                        })
            
            # End of epoch
            avg_train_loss = epoch_loss / max(n_batches, 1)
            
            # Learning rate scheduling (after warmup)
            if self.scheduler is not None and self.global_step >= config.lr_warmup_steps:
                self.scheduler.step()
            
            # Validation
            if val_dataloader is not None and (epoch + 1) % config.eval_interval == 0:
                eval_metrics = self.evaluate(val_dataloader, n_batches=10)
                self.eval_losses.append(eval_metrics['loss'])
                
                if verbose:
                    print(f"\n  Validation - Loss: {eval_metrics['loss']:.6f}, "
                          f"PSNR: {eval_metrics['psnr']:.2f} dB")
                
                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss': eval_metrics['loss'],
                        'val/psnr': eval_metrics['psnr'],
                        'epoch': epoch,
                    })
                
                # Check for best model
                if config.save_best and eval_metrics['loss'] < self.best_loss:
                    self.best_loss = eval_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    if verbose:
                        print(f"  ✅ New best model saved!")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if config.early_stopping_patience > 0:
                    if self.early_stopping_counter >= config.early_stopping_patience:
                        if verbose:
                            print(f"\n⚠️ Early stopping after {epoch + 1} epochs")
                        break
            
            # Checkpointing
            if (epoch + 1) % config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Training summary
        total_time = time.time() - start_time
        summary = {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'total_time': total_time,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training completed in {total_time:.1f}s")
            print(f"Total steps: {self.global_step}")
            print(f"Best validation loss: {self.best_loss:.6f}")
            print(f"Best PSNR: {psnr(self.best_loss):.2f} dB")
        
        # Save summary
        with open(self.checkpoint_dir / 'training_summary.json', 'w') as f:
            json.dump({
                'total_epochs': summary['total_epochs'],
                'total_steps': summary['total_steps'],
                'total_time': summary['total_time'],
                'best_loss': summary['best_loss'],
            }, f, indent=2)
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
        
        return summary
    
    def save_checkpoint(self, filename: str):
        """Save a training checkpoint."""
        checkpoint = {
            'model_config': self.model.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses[-100:],  # Keep last 100
            'eval_losses': self.eval_losses,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load state
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval_losses = checkpoint.get('eval_losses', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")
