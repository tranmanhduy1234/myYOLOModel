# Hyperparameters Tracking for NMSFreeDetector Training

## Quick Reference Card

### Essential Metrics to Track (Per Step)

| Metric | Type | Purpose | TensorBoard Tag |
|--------|------|---------|-----------------|
| Gradient Total Norm | Scalar | Detect vanishing/exploding gradients | `Gradients/total_norm` |
| Gradient RMS (per layer) | Per-layer | Layer-wise gradient health | `Gradients_RMS/{layer}` |
| Gradient Histogram | Histogram | Distribution shape | `Gradients/{layer}` |
| Weight Mean/Std | Per-layer | Weight distribution tracking | `Weights_Stats/{layer}/mean` |
| Weight RMS | Per-layer | Weight magnitude | `Weights_Stats/{layer}/rms` |
| Update Ratio | Per-layer | Learning rate effectiveness | `Update_Ratio/{layer}` |
| Loss (total, o2m, o2o) | Scalar | Training progress | `train/loss_total` |
| Loss Components | Scalar | IoU, CLS, DFL per branch | `train/o2m_iou` etc |
| Positive Anchors | Scalar | Assignment quality | `train/o2m_n_pos` |
| Learning Rate | Scalar | LR schedule tracking | `Learning_Rate/group_0` |
| EMA Decay | Scalar | EMA warmup progress | `EMA/current_decay` |
| GPU Memory | Scalar | Resource monitoring | `System/GPU_memory_allocated_GB` |
| Batch Time | Scalar | Training throughput | `Time/batch_time_ms` |

---

## Integration Guide

### Step 1: Import and Initialize

```python
from torch.utils.tensorboard import SummaryWriter
from src.utils.training_logger import TrainingLogger

# Initialize writer
writer = SummaryWriter(log_dir="runs/experiment_name")

# Initialize logger
logger = TrainingLogger(
    writer=writer,
    log_interval=10,        # Log scalars every 10 steps
    histogram_interval=100  # Log histograms every 100 steps
)
```

### Step 2: Modify Training Loop

```python
def train_one_epoch(model, criterion, loader, optimizer, scheduler,
                     scaler, ema, device, cfg, epoch, writer):
    
    logger = TrainingLogger(writer, log_interval=cfg.log_interval)
    
    for step, (images, targets) in enumerate(loader):
        global_step = epoch * len(loader) + step
        
        # Move to device
        images, targets = move_batch(images, targets, device)
        
        # Store params BEFORE update (for update ratio tracking)
        prev_params = [p.data.clone().detach() for p in model.parameters() 
                       if p.requires_grad and p.grad is not None]
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with AMP
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        with torch.autocast(device_type=device_type, enabled=scaler is not None):
            preds = model(images)
            loss, items = criterion(preds, targets)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # === LOG GRADIENTS (before optimizer step) ===
        logger.log_gradients(model, global_step)
        
        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            skip_lr_sched = (scale_after < scale_before)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            skip_lr_sched = False
        
        # === LOG WEIGHTS (after optimizer step) ===
        logger.log_weights(model, global_step)
        logger.log_weight_updates(model, prev_params, optimizer.param_groups[0]['lr'], global_step)
        
        # Scheduler and EMA
        if not skip_lr_sched:
            scheduler.step()
        if ema is not None:
            ema.update(model=model)
        
        # === LOG OTHER METRICS ===
        logger.log_learning_rate(optimizer, global_step, epoch)
        logger.log_losses(items, global_step, "train")
        
        from src.utils.training_logger import logEMA_stats, logGPU_memory
        logEMA_stats(writer, ema, global_step)
        logGPU_memory(writer, global_step)
        
        # Console logging
        if step % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[epoch {epoch}] step {step}/{len(loader)} "
                f"loss={items['loss']:.4f} "
                f"lr={lr:.6f}"
            )
    
    return sum(items["loss"] for _ in range(len(loader))) / len(loader)
```

---

## Hyperparameters to Track

### Model Architecture Hyperparameters
```python
# From config.py - these should be logged at start
cfg = TrainConfig()

writer.add_text("Hyperparameters/Model", f"""
- nc (num classes): {cfg.nc}
- reg_max: {cfg.reg_max}
- backbone_w: {cfg.backbone_w}
- backbone_n: {cfg.backbone_n}
- neck_n: {cfg.neck_n}
- strides: {cfg.strides}
""")
```

### Training Hyperparameters
```python
writer.add_text("Hyperparameters/Training", f"""
- epochs: {cfg.epochs}
- batch_size: {cfg.batch_size}
- img_size: {cfg.img_size}
- lr0: {cfg.lr0}
- lr_min_factor: {cfg.lr_min_factor}
- warmup_epochs: {cfg.warmup_epochs}
- weight_decay: {cfg.weight_decay}
- grad_clip_norm: {cfg.grad_clip_norm}
- optimizer: {cfg.optimizer}
""")
```

### Loss Hyperparameters
```python
writer.add_text("Hyperparameters/Loss", f"""
- cls_gain: {cfg.cls_gain}
- box_gain: {cfg.box_gain}
- dfl_gain: {cfg.dfl_gain}
- w_o2o: {cfg.w_o2o}
- w_o2m: {cfg.w_o2m}
- topk_o2m: {cfg.topk_o2m}
- topk_o2o: {cfg.topk_o2o}
- alpha: {cfg.alpha}
- beta: {cfg.beta}
""")
```

### EMA Hyperparameters
```python
writer.add_text("Hyperparameters/EMA", f"""
- use_ema: {cfg.use_ema}
- ema_decay: {cfg.ema_decay}
- ema_warmup_updates: {cfg.ema_warmup_updates}
""")
```

---

## TensorBoard Viewing Commands

```bash
# Basic TensorBoard launch
tensorboard --logdir runs

# With specific port
tensorboard --logdir runs --port 6006

# With host binding for remote access
tensorboard --logdir runs --host 0.0.0.0 --port 6006

# With specific experiment
tensorboard --logdir runs/experiment_name

# With reload interval (for live training)
tensorboard --logdir runs --reload_interval 5
```

---

## Summary of All Tracked Metrics

### Per-Step Metrics (High Frequency)
| Category | Metrics | TensorBoard Tag Pattern |
|----------|---------|---------------------------|
| Gradients | Total norm | `Gradients/total_norm` |
| | Avg norm (moving) | `Gradients/avg_norm` |
| | Per-layer RMS | `Gradients_RMS/{layer_name}` |
| Weights | Per-layer mean | `Weights_Stats/{layer_name}/mean` |
| | Per-layer std | `Weights_Stats/{layer_name}/std` |
| | Per-layer RMS | `Weights_Stats/{layer_name}/rms` |
| | Update ratio | `Update_Ratio/{layer_name}` |
| Loss | Total | `train/loss_total` |
| | O2M branch | `train/loss_o2m` |
| | O2O branch | `train/loss_o2o` |
| | Components | `train/o2m_iou`, etc |
| Training | Learning rate | `Learning_Rate/group_{i}` |
| | Epoch | `Training/epoch` |
| EMA | Current decay | `EMA/current_decay` |
| | Updates | `EMA/updates` |
| System | GPU memory | `System/GPU_memory_allocated_GB` |
| | Batch time | `Time/batch_time_ms` |

### Per-N-Steps Metrics (Lower Frequency)
| Category | Metrics | Frequency |
|----------|---------|-----------|
| Gradients | Histograms | Every 100 steps |
| Weights | Histograms | Every 100 steps |
| Activations | Mean, std, histogram | Every 100 steps |
| BatchNorm | Running stats, gamma, beta | Every 100 steps |

---

## Final Checklist for Implementation

- [ ] Import `TrainingLogger` from `src.utils.training_logger`
- [ ] Initialize logger at start of training: `logger = TrainingLogger(writer)`
- [ ] Log gradients BEFORE optimizer step: `logger.log_gradients(model, step)`
- [ ] Store previous parameters BEFORE update for update ratio tracking
- [ ] Log weights AFTER optimizer step: `logger.log_weights(model, step)`
- [ ] Log learning rate: `logger.log_learning_rate(optimizer, step, epoch)`
- [ ] Log loss components: `logger.log_losses(items, step, "train")`
- [ ] Log EMA statistics: `logEMA_stats(writer, ema, step)`
- [ ] Log GPU memory: `logGPU_memory(writer, step)`
- [ ] Log timing: Use `TimeTracker` class
- [ ] Launch TensorBoard: `tensorboard --logdir runs`
