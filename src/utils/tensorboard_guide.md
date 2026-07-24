# TensorBoard Logging Guide - NMSFreeDetector

Tài liệu tham chiếu cho `src/utils/tb_logger.py`. 
## 1. Bảng tham chiếu nhanh

| Chỉ số | Method | Ghi khi nào | TensorBoard tag |
|---|---|---|---|
| Loss (total/o2m/o2o, 4 chart gộp) | `log_losses` | mỗi step | `{phase}/loss_total`, `loss_o2m_parts`, `loss_o2o_parts`, `n_pos` |
| Tỷ lệ đóng góp loss | `log_loss_ratios` | mỗi step (tuỳ chọn) | `{phase}/loss_ratios`, `o2m_component_ratios` |
| Gradient norm tổng/trung bình | `log_gradients` | mỗi `log_interval` step | `Gradients/total_norm`, `avg_norm` |
| Gradient RMS/histogram theo layer | `log_gradients` | RMS mỗi `log_interval`, histogram mỗi `histogram_interval` | `Gradients_RMS/{layer}`, `Gradients/{layer}` |
| Weight mean/std/rms/histogram | `log_weights` | như trên | `Weights_Stats/{layer}/*`, `Weights/{layer}` |
| Update ratio (delta_w / w) | `log_weight_updates` | mỗi `log_interval` step | `Update_Ratio/{layer}` |
| Learning rate / weight decay | `log_learning_rate` | mỗi `log_interval` step | `Learning_Rate/group_i` |
| EMA decay/warmup | `log_ema` | mỗi step (rẻ) | `EMA/current_decay`, `updates`, `warmup_progress` |
| GPU memory | `log_gpu_memory` | mỗi step (rẻ) | `System/GPU_memory_*` |
| BatchNorm running stats | `log_batchnorm` | mỗi `histogram_interval` step (đắt) | `BN/{layer}/*` |
| Hyperparameters (text) | `log_hparams` | 1 lần lúc khởi tạo run | `Hyperparameters/*` |

## 2. Cách dùng trong vòng lặp training

```python
from torch.utils.tensorboard import SummaryWriter
from src.utils.tb_logger import TrainingLogger

writer = SummaryWriter(log_dir="runs/experiment_name")
logger = TrainingLogger(writer, log_interval=10, histogram_interval=100)
logger.log_hparams(cfg)  # ghi 1 lần lúc bắt đầu

for step, (images, targets) in enumerate(loader):
    global_step = epoch * len(loader) + step

    # chụp tham số TRƯỚC optimizer.step() (dùng cho update ratio)
    prev_params = TrainingLogger.snapshot_params(model)

    with torch.autocast(device_type=device.type, enabled=scaler is not None):
        preds = model(images)
        loss, items = criterion(preds, targets)

    (scaler.scale(loss) if scaler else loss).backward()

    logger.log_gradients(model, global_step)          # TRƯỚC optimizer.step()

    if scaler is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

    logger.log_weights(model, global_step)             # SAU optimizer.step()
    logger.log_weight_updates(model, prev_params, global_step)
    logger.log_learning_rate(optimizer, global_step, epoch)
    logger.log_losses(items, global_step, phase="train")
    logger.log_ema(ema, global_step)
    logger.log_gpu_memory(global_step)

    if ema is not None:
        ema.update(model=model)
```

Lưu ý so với bản cũ: `snapshot_params()` giờ chụp theo **tên tham số** thay vì
vị trí trong `named_parameters()` — bản cũ dùng index nên nếu có tham số nào
`grad is None` (bị bỏ qua khi build `prev_params`), toàn bộ các cặp phía sau
sẽ bị lệch tên/giá trị khi ghi `Update_Ratio`.

## 3. Lệnh xem TensorBoard

```bash
tensorboard --logdir runs
tensorboard --logdir runs --port 6006
tensorboard --logdir runs --host 0.0.0.0 --port 6006       # truy cập từ máy khác
tensorboard --logdir runs/experiment_name
tensorboard --logdir runs --reload_interval 5               # theo dõi khi đang train
```

## 4. Checklist khi thêm logging vào training loop mới

- [ ] `logger = TrainingLogger(writer, log_interval=..., histogram_interval=...)`
- [ ] `logger.log_hparams(cfg)` — 1 lần lúc đầu run
- [ ] `prev_params = TrainingLogger.snapshot_params(model)` — trước backward
- [ ] `logger.log_gradients(model, step)` — sau backward, trước optimizer.step()
- [ ] `logger.log_weights(model, step)` + `log_weight_updates(...)` — sau optimizer.step()
- [ ] `logger.log_learning_rate(...)`, `log_losses(...)`, `log_ema(...)`, `log_gpu_memory(...)`
- [ ] (tuỳ chọn, đắt) `logger.log_batchnorm(model, step)` — chỉ nếu cần debug BN