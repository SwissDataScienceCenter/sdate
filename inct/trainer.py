"""
Training utilities for Instant NGP models.

Provides a Trainer class with support for:
- AdamW optimizer
- Learning rate scheduling
- Checkpointing
- Logging (console and optional wandb)
- Early stopping
- Chunked DCT loss for hybrid compression optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from tqdm import tqdm
import time
import json
import math

from .model import InstantNGPModel
from .utils import psnr, mse

# Try to import torch_dct for DCT operations
try:
    from torch_dct import dct_2d, idct_2d
    _HAS_TORCH_DCT = True
except ImportError:
    _HAS_TORCH_DCT = False


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
    
    # Chunked DCT Loss Configuration
    use_chunked_dct_loss: bool = False  # Enable hybrid DCT loss for chunks
    dct_block_size: int = 8  # DCT block size (must match encoder)
    dct_quality: int = 80  # Quality parameter for adaptive quantization
    soft_quantize_temp: float = 1.0  # Temperature for soft quantization (lower = harder)
    rate_weight: float = 0.01  # Weight for rate proxy (L1 regularization on DCT coeffs)
    distortion_weight: float = 1.0  # Weight for distortion term
    adaptive_quantization: bool = True  # Use adaptive quantization matrix


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
        
        # Setup chunked DCT loss components
        if config.use_chunked_dct_loss:
            if not _HAS_TORCH_DCT:
                raise ImportError(
                    "torch_dct is required for chunked DCT loss. "
                    "Install with: pip install torch-dct"
                )
            self._setup_dct_components()
    
    def _setup_dct_components(self):
        """Setup DCT-related components for chunked loss."""
        config = self.config
        bs = config.dct_block_size
        
        # Generate a frequency-distance based quantization pattern for ANY block size
        # This replaces the hardcoded 8x8 JPEG matrix with a principled approach:
        # - DC component (0,0) has lowest quantization (preserve most)
        # - Higher frequencies have progressively higher quantization
        # - Based on L2 distance from DC in frequency space
        self._base_pattern = self._generate_frequency_pattern(bs).to(self.device)
        
        # Precompute static quantization matrix if not adaptive
        if not config.adaptive_quantization:
            quality = config.dct_quality
            if quality < 50:
                scale = 5000 / max(quality, 1)
            else:
                scale = 200 - 2 * quality
            self._static_q_matrix = torch.clamp(
                self._base_pattern * (scale / 100), min=1.0
            ).to(self.device)
        else:
            self._static_q_matrix = None
    
    def _generate_frequency_pattern(self, block_size: int) -> torch.Tensor:
        """
        Generate a quantization pattern based on frequency distance from DC.
        
        This creates a principled quantization matrix for ANY block size,
        not just 8x8. The pattern reflects the perceptual importance of
        DCT frequencies: low frequencies (near DC) are more important
        than high frequencies.
        
        The pattern follows: q(u,v) ∝ 1 + α * sqrt(u² + v²)
        where (u,v) is the frequency index and α controls the slope.
        
        Args:
            block_size: Size of the DCT block (can be any positive integer)
        
        Returns:
            Normalized quantization pattern of shape (block_size, block_size)
        """
        # Create frequency indices
        u = torch.arange(block_size, dtype=torch.float32)
        v = torch.arange(block_size, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        
        # Compute L2 distance from DC (normalized by block size)
        freq_distance = torch.sqrt(uu**2 + vv**2) / (block_size * math.sqrt(2))
        
        # Generate pattern: DC component = 1, increases with frequency distance
        # The slope parameter (4.0) is chosen to roughly match JPEG behavior at 8x8
        # but generalizes smoothly to larger block sizes
        alpha = 4.0
        pattern = 1.0 + alpha * freq_distance
        
        # Normalize to [0, 1] range
        pattern = pattern / pattern.max()
        
        return pattern
    
    def _compute_adaptive_quantization_matrix(
        self, dct_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive quantization matrix scaled to actual DCT coefficient magnitudes.
        
        For adaptive mode, the quantization matrix is derived from the actual
        statistics of the DCT coefficients, making it optimal for any block size.
        
        Args:
            dct_coeffs: Shape (N, 1, block_size, block_size) where N is number of blocks
        
        Returns:
            q_matrix: Adaptive quantization matrix of shape (1, 1, block_size, block_size)
        """
        # Compute per-frequency statistics across all blocks
        freq_std = dct_coeffs.std(dim=0, keepdim=True)  # (1, 1, bs, bs)
        
        # Scale by frequency statistics
        adaptive_scale = freq_std + 1e-6
        
        # Apply quality-based scaling
        quality = self.config.dct_quality
        if quality < 50:
            quality_scale = 5000 / max(quality, 1)
        else:
            quality_scale = 200 - 2 * quality
        
        # Combine: base pattern × frequency scale × quality scale
        q_matrix = self._base_pattern.unsqueeze(0).unsqueeze(0) * adaptive_scale * (quality_scale / 100)
        q_matrix = torch.clamp(q_matrix, min=1.0)
        
        return q_matrix
    
    def _soft_quantize(
        self, x: torch.Tensor, q_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft quantization that is differentiable.
        
        Uses a soft rounding operation: x + tanh((x - round(x)) / temp) * 0.5
        This approximates hard rounding while maintaining gradients.
        
        Args:
            x: Tensor to quantize
            q_matrix: Quantization matrix
        
        Returns:
            Soft-quantized tensor
        """
        temp = self.config.soft_quantize_temp
        
        # Divide by quantization matrix
        scaled = x / q_matrix
        
        # Soft rounding using tanh approximation
        # As temp → 0, this approaches hard rounding
        rounded = torch.round(scaled)
        diff = scaled - rounded
        soft_rounded = rounded + torch.tanh(diff / temp) * 0.5
        
        # Multiply back by quantization matrix (dequantize)
        return soft_rounded * q_matrix
    
    def _reshape_chunk_to_blocks(
        self, 
        residual: torch.Tensor,
        chunk_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int]]:
        """
        Reshape a chunk of residuals into DCT blocks.
        
        For a chunk of shape (H, W, N) or flattened voxels, reshape into
        (num_blocks, 1, block_size, block_size) for batch DCT processing.
        
        This matches the approach in encode_residuals_batched from Section 9.
        
        Args:
            residual: Residual tensor, either (H, W, N) or flattened (n_voxels,)
            chunk_shape: Shape of the chunk (H, W, N)
        
        Returns:
            blocks: Tensor of shape (num_blocks, 1, bs, bs)
            padded_shape: (H_pad, W_pad, N)
            n_blocks: (nb_h, nb_w)
        """
        bs = self.config.dct_block_size
        H, W, N = chunk_shape
        
        # Pad to make H and W divisible by block_size
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        
        # Reshape to (H, W, N) if flattened
        if residual.dim() == 1:
            residual = residual.reshape(H, W, N)
        
        # Pad residuals - use 'constant' mode (zero padding) to avoid reflect padding constraints
        # Reflect padding requires padding < input_dim, which may not hold for small chunks
        if pad_h > 0 or pad_w > 0:
            # Permute to (N, H, W) for padding, then permute back
            residual = residual.permute(2, 0, 1)  # (N, H, W)
            residual = F.pad(
                residual,  # (N, H, W)
                (0, pad_w, 0, pad_h),  # pad W and H dimensions
                mode='constant',  # Use zero padding to handle small chunks
                value=0
            )
            residual = residual.permute(1, 2, 0)  # (H_pad, W_pad, N)
        
        H_pad, W_pad = residual.shape[:2]
        nb_h = H_pad // bs
        nb_w = W_pad // bs
        
        # Process each projection slice separately
        # Reshape: (H_pad, W_pad, N) -> (nb_h, bs, nb_w, bs, N)
        blocks_all = []
        for proj_idx in range(N):
            proj_residual = residual[:, :, proj_idx]  # (H_pad, W_pad)
            
            # Reshape into blocks: (nb_h, bs, nb_w, bs) -> (nb_h * nb_w, 1, bs, bs)
            blocks = (
                proj_residual
                .reshape(nb_h, bs, nb_w, bs)
                .permute(0, 2, 1, 3)  # (nb_h, nb_w, bs, bs)
                .reshape(-1, 1, bs, bs)  # (nb_h * nb_w, 1, bs, bs)
            )
            blocks_all.append(blocks)
        
        # Concatenate all projection blocks: (nb_h * nb_w * N, 1, bs, bs)
        blocks = torch.cat(blocks_all, dim=0)
        
        return blocks, (H_pad, W_pad, N), (nb_h, nb_w)
    
    def _blocks_to_residual(
        self,
        blocks: torch.Tensor,
        padded_shape: Tuple[int, int, int],
        n_blocks: Tuple[int, int],
        original_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Reconstruct residual from DCT blocks.
        
        Args:
            blocks: Tensor of shape (num_blocks, 1, bs, bs)
            padded_shape: (H_pad, W_pad, N)
            n_blocks: (nb_h, nb_w)
            original_shape: (H, W, N)
        
        Returns:
            residual: Tensor of shape (H, W, N)
        """
        bs = self.config.dct_block_size
        H_pad, W_pad, N = padded_shape
        H, W, _ = original_shape
        nb_h, nb_w = n_blocks
        
        blocks_per_proj = nb_h * nb_w
        
        # Reconstruct each projection
        proj_residuals = []
        for proj_idx in range(N):
            start_idx = proj_idx * blocks_per_proj
            end_idx = start_idx + blocks_per_proj
            proj_blocks = blocks[start_idx:end_idx]  # (nb_h * nb_w, 1, bs, bs)
            
            # Reshape: (nb_h * nb_w, 1, bs, bs) -> (nb_h, nb_w, bs, bs)
            proj_blocks = proj_blocks.squeeze(1).reshape(nb_h, nb_w, bs, bs)
            
            # Reassemble: (nb_h, nb_w, bs, bs) -> (H_pad, W_pad)
            proj_residual = (
                proj_blocks
                .permute(0, 2, 1, 3)  # (nb_h, bs, nb_w, bs)
                .reshape(H_pad, W_pad)
            )
            
            # Remove padding
            proj_residuals.append(proj_residual[:H, :W])
        
        # Stack projections: (H, W, N)
        return torch.stack(proj_residuals, dim=2)
    
    def _chunked_dct_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: torch.Tensor,
        chunk_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the chunked DCT loss for hybrid compression optimization.
        
        Pipeline:
        1. INR predicts x_hat
        2. Compute residual r = x - x_hat
        3. Split residual into k×k slice blocks
        4. DCT → coefficients c
        5. Soft quantize c
        6. Rate proxy: L1 regularization on c
        7. Distortion: MSE on reconstructed x_hat + IDCT(c_hat)
        
        Args:
            pred: Model predictions, shape (n_voxels, 1)
            target: Ground truth values, shape (n_voxels, 1)
            coords: Coordinates, shape (n_voxels, 3)
            chunk_shape: Shape of the chunk (H, W, N)
        
        Returns:
            loss: Combined loss value
            metrics: Dictionary with individual loss components
        """
        # 1. Residual computation
        residual = (target - pred).squeeze(-1)  # (n_voxels,)
        
        # 2. Reshape residual into DCT blocks
        # This matches encode_residuals_batched structure
        blocks, padded_shape, n_blocks = self._reshape_chunk_to_blocks(
            residual, chunk_shape
        )
        
        # 3. Compute 2D DCT on all blocks
        dct_coeffs = dct_2d(blocks)  # (num_blocks, 1, bs, bs)
        
        # 4. Compute quantization matrix
        if self.config.adaptive_quantization:
            q_matrix = self._compute_adaptive_quantization_matrix(dct_coeffs)
        else:
            q_matrix = self._static_q_matrix.unsqueeze(0).unsqueeze(0)
        
        # 5. Soft quantize coefficients
        soft_quantized = self._soft_quantize(dct_coeffs, q_matrix)
        
        # 6. Rate proxy: L1 norm on quantized coefficients
        # Lower L1 = fewer bits needed for entropy coding
        rate_loss = torch.abs(soft_quantized / q_matrix).mean()
        
        # 7. Distortion term: reconstruct and measure error
        # IDCT to get reconstructed residual
        reconstructed_residual_blocks = idct_2d(soft_quantized)
        
        # Reshape back to residual
        reconstructed_residual = self._blocks_to_residual(
            reconstructed_residual_blocks,
            padded_shape,
            n_blocks,
            chunk_shape,
        )
        
        # Final reconstruction: x_hat + reconstructed_residual
        # Need to reshape pred to (H, W, N)
        pred_reshaped = pred.squeeze(-1).reshape(chunk_shape)
        final_reconstruction = pred_reshaped + reconstructed_residual
        
        # Distortion: MSE between final reconstruction and target
        target_reshaped = target.squeeze(-1).reshape(chunk_shape)
        distortion_loss = F.mse_loss(final_reconstruction, target_reshaped)
        
        # Combined loss
        total_loss = (
            self.config.distortion_weight * distortion_loss +
            self.config.rate_weight * rate_loss
        )
        
        # Metrics for logging
        metrics = {
            'distortion_loss': distortion_loss.item(),
            'rate_loss': rate_loss.item(),
            'dct_coeff_l1': torch.abs(dct_coeffs).mean().item(),
            'quantized_coeff_l1': torch.abs(soft_quantized / q_matrix).mean().item(),
        }
        
        return total_loss, metrics
    
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
            batch: Dictionary with 'coords' and 'values' tensors.
                   - Standard mode: coords (n_voxels, 3), values (n_voxels, 1)
                   - Chunked mode: coords (H, W, N, 3), values (H, W, N, 1), chunk_shape (3,)
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move to device
        coords = batch['coords'].to(self.device)
        values = batch['values'].to(self.device)
        
        # Check if we have structured chunk data (4D tensors)
        is_structured_chunk = coords.dim() == 4 and 'chunk_shape' in batch
        
        if is_structured_chunk:
            # Extract chunk shape from the data itself
            H, W, N = coords.shape[:3]
            chunk_shape = (H, W, N)
            
            # Flatten coords and values for forward pass
            coords_flat = coords.reshape(-1, 3)
            values_flat = values.reshape(-1, 1)
        else:
            coords_flat = coords
            values_flat = values
            chunk_shape = None
        
        # Forward pass
        pred = self.model(coords_flat)
        
        # Compute loss
        if self.config.use_chunked_dct_loss and chunk_shape is not None:
            # Use chunked DCT loss
            loss, metrics = self._chunked_dct_loss(pred, values_flat, coords_flat, chunk_shape)
            
            # Store metrics for logging
            self._last_dct_metrics = metrics
        else:
            # Standard loss
            loss = self.loss_fn(pred, values_flat)
            self._last_dct_metrics = None
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_chunked(
        self, 
        batch: Dict[str, torch.Tensor],
        chunk_shape: Tuple[int, int, int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform a single training step with chunked DCT loss.
        
        This is an explicit API for when chunk_shape is not in the batch dict.
        
        Args:
            batch: Dictionary with 'coords' and 'values' tensors
            chunk_shape: Shape of the chunk (H, W, N)
            
        Returns:
            loss: Loss value
            metrics: Dictionary with loss components
        """
        if not self.config.use_chunked_dct_loss:
            raise ValueError(
                "use_chunked_dct_loss must be True to use train_step_chunked"
            )
        
        self.model.train()
        
        # Move to device
        coords = batch['coords'].to(self.device)
        values = batch['values'].to(self.device)
        
        # Forward pass
        pred = self.model(coords)
        
        # Compute chunked DCT loss
        loss, metrics = self._chunked_dct_loss(pred, values, coords, chunk_shape)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), metrics
    
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
                postfix = {
                    'loss': f'{loss:.6f}',
                    'lr': f'{current_lr:.2e}',
                }
                
                # Add DCT metrics to progress bar if available
                if hasattr(self, '_last_dct_metrics') and self._last_dct_metrics is not None:
                    postfix['dist'] = f"{self._last_dct_metrics['distortion_loss']:.4f}"
                    postfix['rate'] = f"{self._last_dct_metrics['rate_loss']:.4f}"
                
                pbar.set_postfix(postfix)
                
                # Logging
                if self.global_step % config.log_interval == 0:
                    self.train_losses.append(loss)
                    self.learning_rates.append(current_lr)
                    
                    if self.config.use_wandb:
                        import wandb
                        log_dict = {
                            'train/loss': loss,
                            'train/lr': current_lr,
                            'train/step': self.global_step,
                        }
                        
                        # Log DCT metrics if available
                        if hasattr(self, '_last_dct_metrics') and self._last_dct_metrics is not None:
                            for key, value in self._last_dct_metrics.items():
                                log_dict[f'train/{key}'] = value
                        
                        wandb.log(log_dict)
            
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
