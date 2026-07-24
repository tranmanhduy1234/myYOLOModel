"""
training_logger.py
==================
Comprehensive training logger for NMSFreeDetector.

This module provides detailed tracking of:
- Gradients (histograms, RMS norms, total norm)
- Weights (statistics, histograms, update ratios)
- Loss components (total, o2m, o2o, iou, cls, dfl)
- Learning rate and schedule
- EMA statistics
- System metrics (GPU memory, timing)

Example usage:
    from src.utils.training_logger import TrainingLogger

    logger = TrainingLogger(writer, log_interval=10)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            global_step = epoch * len(dataloader) + step

            # Forward and backward
            loss.backward()

            # Log BEFORE optimizer step
            logger.log_gradients(model, global_step)

            # Optimizer step
            optimizer.step()

            # Log AFTER optimizer step
            logger.log_weights(model, global_step)
            logger.log_learning_rate(optimizer, global_step, epoch)
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Comprehensive training logger for NMSFreeDetector.
    Tracks gradients, weights, activations, BN stats, and system metrics.
    """

    def __init__(self, writer: SummaryWriter, log_interval: int = 10,
                 histogram_interval: int = 100):
        """
        Args:
            writer: TensorBoard SummaryWriter instance
            log_interval: Steps between logging scalar metrics
            histogram_interval: Steps between logging histograms (more expensive)
        """
        self.writer = writer
        self.log_interval = log_interval
        self.histogram_interval = histogram_interval
        self.step = 0
        self.epoch = 0

        # Buffers for moving averages
        self.loss_buffer = []
        self.grad_norm_buffer = []

    # ========================================================================
    # 1. GRADIENT LOGGING
    # ========================================================================

    def log_gradients(self, model: nn.Module, step: int) -> float:
        """
        Log gradient statistics for all parameters.

        Args:
            model: The model to track
            step: Current training step

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        grad_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.numel() > 0:
                # Histogram (log every N steps to reduce overhead)
                if step % self.histogram_interval == 0:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, step)

                # RMS Norm - logged at specified interval
                if step % self.log_interval == 0:
                    w = param.data
                    rms = param.grad.norm().item() / math.sqrt(w.numel())
                    self.writer.add_scalar(f"Gradients_RMS/{name}", rms, step)

                # Per-parameter norm for total calculation
                param_norm = param.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2

        # Total gradient norm
        total_norm = total_norm ** 0.5

        if step % self.log_interval == 0:
            self.writer.add_scalar("Gradients/total_norm", total_norm, step)

            # Moving average of gradient norm
            self.grad_norm_buffer.append(total_norm)
            if len(self.grad_norm_buffer) > 100:
                self.grad_norm_buffer.pop(0)
            avg_grad_norm = sum(self.grad_norm_buffer) / len(self.grad_norm_buffer)
            self.writer.add_scalar("Gradients/avg_norm", avg_grad_norm, step)

        return total_norm

    # ========================================================================
    # 2. WEIGHT & BIAS LOGGING
    # ========================================================================

    def log_weights(self, model: nn.Module, step: int):
        """
        Log weight statistics for all parameters.

        Args:
            model: The model to track
            step: Current training step
        """
        if step % self.log_interval != 0:
            return

        for name, param in model.named_parameters():
            w = param.data

            # Histogram (log less frequently)
            if step % self.histogram_interval == 0:
                self.writer.add_histogram(f"Weights/{name}", param, step)

            # Statistics
            self.writer.add_scalar(f"Weights_Stats/{name}/std", w.std().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/mean", w.mean().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/rms",
                                 w.norm().item() / math.sqrt(w.numel()), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/max", w.max().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/min", w.min().item(), step)

    def log_weight_updates(self, model: nn.Module, prev_params: list,
                          lr: float, step: int):
        """
        Log weight update ratios (delta_w / w).

        Args:
            model: Current model
            prev_params: List of parameter values before optimizer step
            lr: Current learning rate
            step: Training step
        """
        if step % self.log_interval != 0:
            return

        idx = 0
        for name, param in model.named_parameters():
            if param.grad is not None and idx < len(prev_params):
                prev_w = prev_params[idx]
                update = param.data - prev_w

                # Update ratio
                ratio = (update.abs() / (prev_w.abs() + 1e-8)).mean().item()
                self.writer.add_scalar(f"Update_Ratio/{name}", ratio, step)

                # Update magnitude
                update_mag = update.norm().item()
                self.writer.add_scalar(f"Update_Magnitude/{name}", update_mag, step)

                idx += 1

    # ========================================================================
    # 3. LEARNING RATE TRACKING
    # ========================================================================

    def log_learning_rate(self, optimizer, step, epoch=None):
        """
        Log learning rate for each parameter group.

        Args:
            optimizer: PyTorch optimizer
            step: Training step
            epoch: Current epoch (optional)
        """
        if step % self.log_interval != 0:
            return

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f"Learning_Rate/group_{i}", lr, step)

            # Weight decay
            if 'weight_decay' in param_group:
                wd = param_group['weight_decay']
                self.writer.add_scalar(f"Weight_Decay/group_{i}", wd, step)

        # Log epoch if provided
        if epoch is not None:
            self.writer.add_scalar("Training/epoch", epoch, step)

    # ========================================================================
    # 4. LOSS COMPONENTS TRACKING
    # ========================================================================

    def log_losses(self, items: dict, step: int, phase: str = "train"):
        """
        Log all loss components.

        Args:
            items: Dictionary containing loss components
            step: Training step
            phase: "train" or "val"
        """
        # Total loss
        self.writer.add_scalar(f"{phase}/loss_total", items["loss"], step)

        # Branch losses
        self.writer.add_scalar(f"{phase}/loss_o2m", items["loss_o2m"], step)
        self.writer.add_scalar(f"{phase}/loss_o2o", items["loss_o2o"], step)

        # Component losses - O2M
        self.writer.add_scalar(f"{phase}/o2m_iou", items["o2m/iou"], step)
        self.writer.add_scalar(f"{phase}/o2m_cls", items["o2m/cls"], step)
        self.writer.add_scalar(f"{phase}/o2m_dfl", items["o2m/dfl"], step)

        # Component losses - O2O
        self.writer.add_scalar(f"{phase}/o2o_iou", items["o2o/iou"], step)
        self.writer.add_scalar(f"{phase}/o2o_cls", items["o2o/cls"], step)
        self.writer.add_scalar(f"{phase}/o2o_dfl", items["o2o/dfl"], step)

        # Positive anchors
        self.writer.add_scalar(f"{phase}/o2m_n_pos", items["o2m/n_pos"], step)
        self.writer.add_scalar(f"{phase}/o2o_n_pos", items["o2o/n_pos"], step)

    def log_loss_ratios(self, items: dict, step: int, phase: str = "train"):
        """Log loss component ratios."""
        total = items["loss"] + 1e-8

        # Branch ratios
        self.writer.add_scalar(f"{phase}/ratio_o2m", items["loss_o2m"] / total, step)
        self.writer.add_scalar(f"{phase}/ratio_o2o", items["loss_o2o"] / total, step)

        # Component ratios within branches
        o2m_total = items["loss_o2m"] + 1e-8
        self.writer.add_scalar(f"{phase}/o2m_ratio_iou", items["o2m/iou"] / o2m_total, step)
        self.writer.add_scalar(f"{phase}/o2m_ratio_cls", items["o2m/cls"] / o2m_total, step)
        self.writer.add_scalar(f"{phase}/o2m_ratio_dfl", items["o2m/dfl"] / o2m_total, step)


# ========================================================================
# HELPER FUNCTIONS FOR SYSTEM TRACKING
# ========================================================================

def logGPU_memory(writer, step):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        writer.add_scalar("System/GPU_memory_allocated_GB", allocated, step)
        writer.add_scalar("System/GPU_memory_reserved_GB", reserved, step)
        writer.add_scalar("System/GPU_max_memory_allocated_GB", max_allocated, step)
        writer.add_scalar("System/GPU_memory_utilization",
                         allocated / (reserved + 1e-8), step)


class TimeTracker:
    """Track timing statistics."""
    def __init__(self, writer):
        self.writer = writer
        self.batch_times = []

    def log_batch_time(self, batch_time, step):
        self.batch_times.append(batch_time)
        if len(self.batch_times) > 100:
            self.batch_times.pop(0)
        avg_time = sum(self.batch_times) / len(self.batch_times)
        self.writer.add_scalar("Time/batch_time_ms", batch_time * 1000, step)
        self.writer.add_scalar("Time/avg_batch_time_ms", avg_time * 1000, step)

    def log_throughput(self, batch_size, batch_time, step):
        throughput = batch_size / batch_time
        self.writer.add_scalar("Time/throughput_samples_per_sec", throughput, step)


# ========================================================================
# EMA STATISTICS LOGGING
# ========================================================================

def logEMA_stats(writer, ema, step):
    """Log EMA statistics."""
    if ema is not None:
        # Current decay value
        current_decay = ema._current_decay()
        writer.add_scalar("EMA/current_decay", current_decay, step)
        writer.add_scalar("EMA/updates", ema.updates, step)

        # Warmup progress
        warmup_progress = min(ema.updates / max(1, ema.warmup_updates), 1.0)
        writer.add_scalar("EMA/warmup_progress", warmup_progress, step)


def logEMA_param_stats(writer, ema_model, step, prefix="EMA"):
    """Log statistics of EMA model parameters."""
    if ema_model is None:
        return

    total_norm = 0.0
    param_count = 0

    for name, param in ema_model.named_parameters():
        if param.dtype.is_floating_point:
            param_norm = param.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1

    total_norm = total_norm ** 0.5
    writer.add_scalar(f"{prefix}/param_norm", total_norm, step)
    writer.add_scalar(f"{prefix}/param_count", param_count, step)


# ========================================================================
# BATCH NORM STATISTICS LOGGING
# ========================================================================

def logBatchNorm_stats(writer, model, step):
    """Log BatchNorm running statistics."""
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            safe_name = name.replace('.', '/')

            # Running mean and var
            writer.add_scalar(f"BN/{safe_name}/running_mean",
                              module.running_mean.mean().item(), step)
            writer.add_scalar(f"BN/{safe_name}/running_var",
                              module.running_var.mean().item(), step)

            # Learned parameters
            if module.weight is not None:
                writer.add_scalar(f"BN/{safe_name}/gamma_mean",
                                  module.weight.mean().item(), step)
                writer.add_scalar(f"BN/{safe_name}/gamma_std",
                                  module.weight.std().item(), step)
            if module.bias is not None:
                writer.add_scalar(f"BN/{safe_name}/beta_mean",
                                  module.bias.mean().item(), step)
                writer.add_scalar(f"BN/{safe_name}/beta_std",
                                  module.bias.std().item(), step)


# ========================================================================
# ACTIVATION TRACKING
# ========================================================================

class ActivationTracker:
    """Track activation statistics during forward pass."""

    def __init__(self, writer):
        self.writer = writer
        self.hooks = []
        self.step = 0

    def register_hooks(self, model):
        """Register forward hooks to track activations."""
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._log_activation(name, output, self.step)
            return hook

        # Register hooks on key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.SiLU)):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def _log_activation(self, name, tensor, step):
        """Log activation statistics."""
        if tensor.numel() == 0:
            return

        # Compute statistics
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        max_val = tensor.max().item()
        min_val = tensor.min().item()

        # Log to tensorboard
        safe_name = name.replace('.', '/')
        self.writer.add_scalar(f"Activations/{safe_name}/mean", mean_val, step)
        self.writer.add_scalar(f"Activations/{safe_name}/std", std_val, step)
        self.writer.add_scalar(f"Activations/{safe_name}/max", max_val, step)
        self.writer.add_scalar(f"Activations/{safe_name}/min", min_val, step)

    def set_step(self, step):
        self.step = step

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ========================================================================
# WEIGHT UPDATE RATIO TRACKING
# ========================================================================

def logWeight_update_ratio(writer, model, prev_params, lr, step,
                           log_interval: int = 10):
    """
    Log weight update ratios (delta_w / w).

    Args:
        writer: TensorBoard writer
        model: Current model
        prev_params: List of parameter values before optimizer step
        lr: Current learning rate
        step: Training step
        log_interval: Logging frequency
    """
    if step % log_interval != 0:
        return

    idx = 0
    for name, param in model.named_parameters():
        if param.grad is not None and idx < len(prev_params):
            prev_w = prev_params[idx]
            update = param.data - prev_w

            # Update ratio
            ratio = (update.abs() / (prev_w.abs() + 1e-8)).mean().item()
            writer.add_scalar(f"Update_Ratio/{name}", ratio, step)

            # Update magnitude
            update_mag = update.norm().item()
            writer.add_scalar(f"Update_Magnitude/{name}", update_mag, step)

            idx += 1
