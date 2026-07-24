# Training Hyperparameters and Logging Guide

## Overview

This document outlines all hyperparameters and metrics to track during training for deep understanding of model behavior.

---

## 1. GRADIENT TRACKING (Per Layer)

### 1.1 Histogram Logging
```python
def logGradient_histogram(writer, model, step):
    """Log gradient histogram for each parameter."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, step)
```

**What to track:**
- Gradient histogram per layer (identify vanishing/exploding gradients)
- Gradient mean per layer
- Gradient standard deviation per layer
- Gradient min/max values per layer

### 1.2 RMS Norm (Root Mean Square Norm)
```python
def logGradient_rms(writer, model, step):
    """Log gradient RMS norm for each parameter."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            w = param.data
            rms = param.grad.norm().item() / math.sqrt(w.numel())
            writer.add_scalar(f"Gradients_RMS/{name}", rms, step)
```

**Why track RMS norm:**
- Indicates gradient scale relative to parameter size
- Useful for detecting gradient explosion
- Helps adjust learning rate per layer

### 1.3 Gradient Norm Distribution
```python
def logGradient_norm_stats(writer, model, step):
    """Log gradient norm statistics."""
    total_norm = 0.0
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms.append((name, param_norm))
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar("Gradients/total_norm", total_norm, step)
    return grad_norms
```

---

## 2. WEIGHT & BIAS TRACKING (Per Layer)

### 2.1 Weight Histogram
```python
def logWeight_histogram(writer, model, step):
    """Log weight histogram for each parameter."""
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, step)
```

### 2.2 Weight Statistics
```python
def logWeight_stats(writer, model, step):
    """Log weight statistics for each parameter."""
    for name, param in model.named_parameters():
        w = param.data
        writer.add_scalar(f"Weights_Stats/{name}/std", w.std().item(), step)
        writer.add_scalar(f"Weights_Stats/{name}/mean", w.mean().item(), step)
        writer.add_scalar(f"Weights_Stats/{name}/rms", 
                         w.norm().item() / math.sqrt(w.numel()), step)
        writer.add_scalar(f"Weights_Stats/{name}/max", w.max().item(), step)
        writer.add_scalar(f"Weights_Stats/{name}/min", w.min().item(), step)
```

**Why track weight statistics:**
- Detect weight saturation (in activations like sigmoid/tanh)
- Monitor weight initialization effectiveness
- Track weight decay effects

### 2.3 Weight Update Tracking
```python
def logWeight_update_ratio(writer, model, prev_params, lr, step):
    """Log weight update ratio: delta_w / w."""
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
```

**Why track update ratios:**
- Detect learning rate that's too high (unstable training)
- Detect learning rate that's too low (very small updates)
- Monitor parameter convergence

---

## 3. LEARNING RATE TRACKING

### 3.1 Learning Rate per Epoch/Step
```python
def logLearning_rate(writer, optimizer, step, epoch=None):
    """Log learning rate for each parameter group."""
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        writer.add_scalar(f"Learning_Rate/group_{i}", lr, step)
        
        # Weight decay
        if 'weight_decay' in param_group:
            wd = param_group['weight_decay']
            writer.add_scalar(f"Weight_Decay/group_{i}", wd, step)
            
    # Log epoch if provided
    if epoch is not None:
        writer.add_scalar("Training/epoch", epoch, step)
```

### 3.2 Learning Rate Schedule Visualization
```python
def logLR_schedule(writer, scheduler, steps, step):
    """Log learning rate schedule progression."""
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar("LR_Schedule/current", current_lr, step)
    
    # Track warmup progress if applicable
    if hasattr(scheduler, 'warmup_steps'):
        warmup_progress = min(step / scheduler.warmup_steps, 1.0)
        writer.add_scalar("LR_Schedule/warmup_progress", warmup_progress, step)
```

---

## 4. LOSS COMPONENTS TRACKING

### 4.1 Total and Component Losses (Already in tb_logger.py)
```python
# Already implemented in src/utils/tb_logger.py
def log_loss_items(writer, items, step, phase="train"):
    # Total loss + loss per branch
    writer.add_scalars(f"{phase}/loss_total", {
        "total": items["loss"],
        "o2m": items["loss_o2m"],
        "o2o": items["loss_o2o"],
    }, step)
    # ... etc
```

### 4.2 Loss Ratio Tracking
```python
def logLoss_ratios(writer, items, step, phase="train"):
    """Track ratios between loss components."""
    total = items["loss"] + 1e-8
    writer.add_scalars(f"{phase}/loss_ratios", {
        "o2m_ratio": items["loss_o2m"] / total,
        "o2o_ratio": items["loss_o2o"] / total,
    }, step)
    
    # Track individual component contributions
    writer.add_scalars(f"{phase}/o2m_component_ratios", {
        "iou": items["o2m/iou"] / (items["loss_o2m"] + 1e-8),
        "cls": items["o2m/cls"] / (items["loss_o2m"] + 1e-8),
        "dfl": items["o2m/dfl"] / (items["loss_o2m"] + 1e-8),
    }, step)
```

### 4.3 Loss Smoothing (Moving Average)
```python
class LossSmoother:
    """Track smoothed loss for better visualization."""
    def __init__(self, window=100):
        self.window = window
        self.buffer = []
        
    def update(self, loss):
        self.buffer.append(loss)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return sum(self.buffer) / len(self.buffer)
    
    def log_smoothed(self, writer, loss, step, tag="loss/smoothed"):
        smoothed = self.update(loss)
        writer.add_scalar(tag, smoothed, step)
        return smoothed
```

---

## 5. EMA (EXPONENTIAL MOVING AVERAGE) TRACKING

### 5.1 EMA Decay Tracking
```python
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
```

### 5.2 EMA Parameter Statistics
```python
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
```

---

## 6. ACTIVATION TRACKING

### 6.1 Activation Statistics (Forward Hooks)
```python
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
```

### 6.2 Activation Histograms
```python
def logActivation_histograms(writer, activations_dict, step):
    """Log activation histograms for specific layers."""
    for name, activation in activations_dict.items():
        if isinstance(activation, torch.Tensor):
            safe_name = name.replace('.', '/')
            writer.add_histogram(f"Activations_hist/{safe_name}", activation, step)
```

---

## 7. BATCH NORMALIZATION TRACKING

### 7.1 BN Statistics Tracking
```python
def logBatchNorm_stats(writer, model, step):
    """Log BatchNorm running statistics."""
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            safe_name = name.replace('.', '/')
            
            # Running mean and var
            writer.add_scalar(f"BN/{safe_name}/running_mean", module.running_mean.mean().item(), step)
            writer.add_scalar(f"BN/{safe_name}/running_var", module.running_var.mean().item(), step)
            
            # Learned parameters
            if module.weight is not None:
                writer.add_scalar(f"BN/{safe_name}/gamma_mean", module.weight.mean().item(), step)
                writer.add_scalar(f"BN/{safe_name}/gamma_std", module.weight.std().item(), step)
            if module.bias is not None:
                writer.add_scalar(f"BN/{safe_name}/beta_mean", module.bias.mean().item(), step)
                writer.add_scalar(f"BN/{safe_name}/beta_std", module.bias.std().item(), step)
```

### 7.2 BN Activation Distribution
```python
def logBatchNorm_distribution(writer, model, step):
    """Log BN input/output distribution statistics."""
    # This requires forward hooks, similar to activation tracking
    pass
```

---

## 8. SYSTEM RESOURCE TRACKING

### 8.1 GPU Memory Tracking
```python
import torch

def logGPU_memory(writer, step):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        # Current GPU memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        writer.add_scalar("System/GPU_memory_allocated_GB", allocated, step)
        
        # Maximum memory allocated
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        writer.add_scalar("System/GPU_max_memory_allocated_GB", max_allocated, step)
        
        # Current memory reserved
        reserved = torch.cuda.memory_reserved() / 1024**3
        writer.add_scalar("System/GPU_memory_reserved_GB", reserved, step)
        
        # Memory summary
        writer.add_scalar("System/GPU_memory_utilization", 
                         allocated / (reserved + 1e-8), step)
```

### 8.2 Timing Statistics
```python
import time

class TimeTracker:
    """Track timing statistics."""
    def __init__(self, writer):
        self.writer = writer
        self.batch_times = []
        self.data_load_times = []
        self.forward_times = []
        self.backward_times = []
        
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
```

---

## 9. COMPLETE LOGGING IMPLEMENTATION

### 9.1 Complete Logger Class
```python
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    Comprehensive training logger for NMSFreeDetector.
    Tracks gradients, weights, activations, BN stats, and system metrics.
    """
    
    def __init__(self, writer: SummaryWriter, log_interval: int = 10):
        self.writer = writer
        self.log_interval = log_interval
        self.step = 0
        self.epoch = 0
        
        # Buffers for moving averages
        self.loss_buffer = []
        self.grad_norm_buffer = []
        
    # ========================================================================
    # 1. GRADIENT LOGGING
    # ========================================================================
    
    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient statistics for all parameters."""
        total_norm = 0.0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.numel() > 0:
                # Histogram (log every N steps to reduce overhead)
                if step % (self.log_interval * 10) == 0:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
                
                # RMS Norm
                w = param.data
                rms = param.grad.norm().item() / math.sqrt(w.numel())
                self.writer.add_scalar(f"Gradients_RMS/{name}", rms, step)
                
                # Per-parameter norm for total calculation
                param_norm = param.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2
        
        # Total gradient norm
        total_norm = total_norm ** 0.5
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
        """Log weight statistics for all parameters."""
        for name, param in model.named_parameters():
            w = param.data
            
            # Histogram (log less frequently)
            if step % (self.log_interval * 10) == 0:
                self.writer.add_histogram(f"Weights/{name}", param, step)
            
            # Statistics
            self.writer.add_scalar(f"Weights_Stats/{name}/std", w.std().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/mean", w.mean().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/rms", 
                                 w.norm().item() / math.sqrt(w.numel()), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/max", w.max().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/min", w.min().item(), step)
    
    def log_weight_updates(self, model: nn.Module, prev_params: list, lr: float, step: int):
        """Log weight update ratios (delta_w / w)."""
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
        """Log learning rate for each parameter group."""
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
        """Log all loss components."""
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
        self.writer.add_scalar(f"phase}/o2o_cls", items["o2o/cls"], step)
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
```

---

## 10. SYSTEM METRICS TRACKING

### 10.1 GPU Memory
```python
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
```

### 10.2 Timing
```python
import time

class TimeTracker:
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
```

---

## 11. IMPLEMENTATION EXAMPLE

### 11.1 Integration into training loop
```python
from src.utils.training_logger import TrainingLogger

def train_one_epoch(model, criterion, loader, optimizer, scheduler, 
                     scaler, ema, device, cfg, epoch, writer):
    
    logger = TrainingLogger(writer, log_interval=cfg.log_interval)
    time_tracker = TimeTracker(writer)
    
    for step, (images, targets) in enumerate(loader):
        global_step = epoch * len(loader) + step
        
        # Forward pass
        t0 = time.time()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            preds = model(images)
            loss, items = criterion(preds, targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Log gradients BEFORE optimizer step
        logger.log_gradients(model, global_step)
        
        # Optimizer step
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        
        # Log weights and other stats AFTER optimizer step
        logger.log_weights(model, global_step)
        logger.log_learning_rate(optimizer, global_step, epoch)
        logger.log_losses(items, global_step, "train")
        logger.logEMA_stats(ema, global_step)
        
        # Time tracking
        batch_time = time.time() - t0
        time_tracker.log_batch_time(batch_time, global_step)
        
        # GPU memory
        logGPU_memory(writer, global_step)
```

---

## 12. TENSORBOARD VISUALIZATION COMMANDS

```bash
# Start TensorBoard
tensorboard --logdir runs

# With specific port
tensorboard --logdir runs --port 6006

# With host binding
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

---

## 13. SUMMARY OF TRACKED METRICS

| Category | Metrics | Frequency |
|----------|---------|-----------|
| Gradients | Histogram, RMS, mean, std, total norm | Every step |
| Weights | Histogram, mean, std, RMS, min, max | Every step |
| Updates | Update ratio, magnitude | Every step |
| Learning Rate | Per group | Every step |
| Loss Components | Total, o2m, o2o, iou, cls, dfl, n_pos | Every step |
| Loss Ratios | Component ratios | Every step |
| EMA | Decay, updates, warmup progress | Every step |
| System | GPU memory, batch time | Every step |
| Activations | Mean, std, histogram | Every N steps |
| BatchNorm | Running stats, gamma, beta | Every N steps |
