"""
tb_logger.py
============
TensorBoard logging cho NMSFreeDetector - GOM VE 1 CHO (thay cho ban cu bi
trung lap giua tb_logger.py va training_logger.py).

Noi dung:
    - TrainingLogger : lop chinh, goi 1 lan/step, tu quan ly nhip do ghi log
      (log_interval cho scalar, histogram_interval cho histogram/BN vi cac
      thao tac nay ton chi phi hon).
    - TimeTracker, ActivationTracker : tien ich phu, tach rieng vi chung giu
      state rieng khong thuoc nhip do log_interval/histogram_interval chung.
    - LossSmoother, log_lr_schedule, log_activation_histograms : tien ich nho,
      dung khi can (khong bat buoc trong vong lap chinh).

Vi du su dung (xem them docs/TENSORBOARD_GUIDE.md):

    writer = SummaryWriter(log_dir="runs/experiment_name")
    logger = TrainingLogger(writer, log_interval=10, histogram_interval=100)

    for step, (images, targets) in enumerate(loader):
        prev_params = TrainingLogger.snapshot_params(model)   # truoc backward

        preds = model(images)
        loss, items = criterion(preds, targets)
        loss.backward()

        logger.log_gradients(model, global_step)               # TRUOC optimizer.step()
        optimizer.step()
        logger.log_weights(model, global_step)                 # SAU optimizer.step()
        logger.log_weight_updates(model, prev_params, global_step)

        logger.log_learning_rate(optimizer, global_step, epoch)
        logger.log_losses(items, global_step, phase="train")
        logger.log_ema(ema, global_step)
        logger.log_gpu_memory(global_step)
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Dung chung logger "train" voi logging_setup.py de canh bao NaN/Inf gradient
# hoac weight nam chung 1 file .log voi phan con lai cua qua trinh training.
# Neu setup_logging() chua duoc goi thi logger nay chua co handler - cac dong
# warning() se khong bien mat, chi la khong duoc ghi ra dau ca (Python logging
# van chay binh thuong, khong loi).
_log = logging.getLogger("train")


class TrainingLogger:
    """Logger TensorBoard tong hop: loss, gradient, weight, LR, EMA, GPU, BN."""

    def __init__(self, writer: SummaryWriter, log_interval: int = 10, histogram_interval: int = 100):
        """
        Args:
            writer: TensorBoard SummaryWriter.
            log_interval: so step giua 2 lan ghi scalar (gradient RMS, weight stats, LR...).
            histogram_interval: so step giua 2 lan ghi histogram / BN stats (dat > log_interval vi ton chi phi hon).
        """
        self.writer = writer
        self.log_interval = log_interval
        self.histogram_interval = histogram_interval
        self._grad_norm_buffer = []

    # ==========================================================================
    # 0. SNAPSHOT THAM SO (dung cho log_weight_updates)
    # ==========================================================================

    @staticmethod
    def snapshot_params(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Chup lai gia tri tham so TRUOC optimizer.step(), dung theo TEN (khong
        dung vi tri/index - tranh loi lech thu tu khi vai param chua co .grad)."""
        return {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    # ==========================================================================
    # 1. LOSS
    # ==========================================================================

    def log_losses(self, items: dict, step: int, phase: str = "train") -> None:
        """4 bieu do gop nhieu duong/1 chart - de so sanh truc quan. Ghi moi step
        (re, khong gate theo log_interval)."""
        assert phase in ("train", "val"), f"phase phai la 'train' hoac 'val', duoc '{phase}'"

        self.writer.add_scalars(f"{phase}/loss_total", {
            "total": items["loss"],
            "o2m": items["loss_o2m"],
            "o2o": items["loss_o2o"],
        }, step)

        self.writer.add_scalars(f"{phase}/loss_o2m_parts", {
            "iou": items["o2m/iou"],
            "cls": items["o2m/cls"],
            "dfl": items["o2m/dfl"],
        }, step)

        self.writer.add_scalars(f"{phase}/loss_o2o_parts", {
            "iou": items["o2o/iou"],
            "cls": items["o2o/cls"],
            "dfl": items["o2o/dfl"],
        }, step)

        # So luong anchor duong: tach rieng vi scale la SO NGUYEN, khac han
        # scale float nho cua loss - gop chung se lam bieu do loss bi det.
        self.writer.add_scalars(f"{phase}/n_pos", {
            "o2m": items["o2m/n_pos"],
            "o2o": items["o2o/n_pos"],
        }, step)

    def log_loss_ratios(self, items: dict, step: int, phase: str = "train") -> None:
        """Ty le dong gop cua tung nhanh / tung thanh phan vao tong loss."""
        total = items["loss"] + 1e-8
        self.writer.add_scalars(f"{phase}/loss_ratios", {
            "o2m_ratio": items["loss_o2m"] / total,
            "o2o_ratio": items["loss_o2o"] / total,
        }, step)

        o2m_total = items["loss_o2m"] + 1e-8
        self.writer.add_scalars(f"{phase}/o2m_component_ratios", {
            "iou": items["o2m/iou"] / o2m_total,
            "cls": items["o2m/cls"] / o2m_total,
            "dfl": items["o2m/dfl"] / o2m_total,
        }, step)

    # ==========================================================================
    # 2. GRADIENT
    # ==========================================================================

    def log_gradients(self, model: nn.Module, step: int) -> float:
        """Goi SAU loss.backward(), TRUOC optimizer.step() VA TRUOC clip_grad_norm_
        (de thay dung do lon that cua gradient, khong bi che boi clipping).
        Tra ve total_norm."""
        do_hist = step % self.histogram_interval == 0
        do_scalar = step % self.log_interval == 0

        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None or param.grad.numel() == 0:
                continue
            # torch's add_histogram tu crash (ValueError: histogram is empty) neu
            # tensor chua NaN/Inf - thuong la dau hieu gradient bi no (explode),
            # khong phai loi cua logging. Bo qua histogram cho param nay va canh
            # bao, thay vi de crash toan bo training.
            if do_hist:
                if torch.isfinite(param.grad).all():
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
                else:
                    _log.warning(
                        f"[TrainingLogger] Bo qua histogram gradient '{name}' o step {step}: "
                        f"gradient co gia tri NaN/Inf (co the do loss/learning rate dang phan ky)."
                    )
            if do_scalar:
                rms = param.grad.norm().item() / math.sqrt(param.data.numel())
                self.writer.add_scalar(f"Gradients_RMS/{name}", rms, step)
            total_norm_sq += param.grad.data.norm(2).item() ** 2

        total_norm = total_norm_sq ** 0.5
        if do_scalar:
            self.writer.add_scalar("Gradients/total_norm", total_norm, step)
            self._grad_norm_buffer.append(total_norm)
            if len(self._grad_norm_buffer) > 100:
                self._grad_norm_buffer.pop(0)
            avg_norm = sum(self._grad_norm_buffer) / len(self._grad_norm_buffer)
            self.writer.add_scalar("Gradients/avg_norm", avg_norm, step)
        return total_norm

    # ==========================================================================
    # 3. WEIGHT / BIAS
    # ==========================================================================

    def log_weights(self, model: nn.Module, step: int) -> None:
        """Goi SAU optimizer.step()."""
        if step % self.log_interval != 0:
            return
        do_hist = step % self.histogram_interval == 0

        for name, param in model.named_parameters():
            w = param.data
            if do_hist:
                if torch.isfinite(w).all():
                    self.writer.add_histogram(f"Weights/{name}", param, step)
                else:
                    _log.warning(
                        f"[TrainingLogger] Bo qua histogram weight '{name}' o step {step}: "
                        f"trong so co gia tri NaN/Inf."
                    )
            self.writer.add_scalar(f"Weights_Stats/{name}/mean", w.mean().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/std", w.std().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/rms", w.norm().item() / math.sqrt(w.numel()), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/max", w.max().item(), step)
            self.writer.add_scalar(f"Weights_Stats/{name}/min", w.min().item(), step)

    def log_weight_updates(self, model: nn.Module, prev_params: Dict[str, torch.Tensor], step: int) -> None:
        """update_ratio = |delta_w| / |w_truoc|. prev_params lay tu snapshot_params()."""
        if step % self.log_interval != 0:
            return
        for name, param in model.named_parameters():
            if name not in prev_params:
                continue
            prev_w = prev_params[name]
            update = param.data - prev_w
            ratio = (update.abs() / (prev_w.abs() + 1e-8)).mean().item()
            self.writer.add_scalar(f"Update_Ratio/{name}", ratio, step)
            self.writer.add_scalar(f"Update_Magnitude/{name}", update.norm().item(), step)

    # ==========================================================================
    # 4. LEARNING RATE
    # ==========================================================================

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int, epoch: Optional[int] = None) -> None:
        if step % self.log_interval != 0:
            return
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"Learning_Rate/group_{i}", param_group["lr"], step)
            if "weight_decay" in param_group:
                self.writer.add_scalar(f"Weight_Decay/group_{i}", param_group["weight_decay"], step)
        if epoch is not None:
            self.writer.add_scalar("Training/epoch", epoch, step)

    # ==========================================================================
    # 5. EMA
    # ==========================================================================

    def log_ema(self, ema, step: int) -> None:
        if ema is None:
            return
        current_decay = ema._current_decay()
        self.writer.add_scalar("EMA/current_decay", current_decay, step)
        self.writer.add_scalar("EMA/updates", ema.updates, step)
        warmup_progress = min(ema.updates / max(1, ema.warmup_updates), 1.0)
        self.writer.add_scalar("EMA/warmup_progress", warmup_progress, step)

    def log_ema_params(self, ema_model: Optional[nn.Module], step: int, prefix: str = "EMA") -> None:
        """Norm tong cua tham so model EMA - ton chi phi hon log_ema(), nen gate theo log_interval."""
        if ema_model is None or step % self.log_interval != 0:
            return
        total_norm_sq = 0.0
        param_count = 0
        for param in ema_model.parameters():
            if param.dtype.is_floating_point:
                total_norm_sq += param.data.norm(2).item() ** 2
                param_count += 1
        self.writer.add_scalar(f"{prefix}/param_norm", total_norm_sq ** 0.5, step)
        self.writer.add_scalar(f"{prefix}/param_count", param_count, step)

    # ==========================================================================
    # 6. HE THONG (GPU / BatchNorm)
    # ==========================================================================

    def log_gpu_memory(self, step: int) -> None:
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        self.writer.add_scalar("System/GPU_memory_allocated_GB", allocated, step)
        self.writer.add_scalar("System/GPU_memory_reserved_GB", reserved, step)
        self.writer.add_scalar("System/GPU_max_memory_allocated_GB", max_allocated, step)
        self.writer.add_scalar("System/GPU_memory_utilization", allocated / (reserved + 1e-8), step)

    def log_batchnorm(self, model: nn.Module, step: int) -> None:
        """Ton chi phi (duyet toan bo module) - chi ghi theo histogram_interval."""
        if step % self.histogram_interval != 0:
            return
        for name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            safe_name = name.replace(".", "/")
            self.writer.add_scalar(f"BN/{safe_name}/running_mean", module.running_mean.mean().item(), step)
            self.writer.add_scalar(f"BN/{safe_name}/running_var", module.running_var.mean().item(), step)
            if module.weight is not None:
                self.writer.add_scalar(f"BN/{safe_name}/gamma_mean", module.weight.mean().item(), step)
                self.writer.add_scalar(f"BN/{safe_name}/gamma_std", module.weight.std().item(), step)
            if module.bias is not None:
                self.writer.add_scalar(f"BN/{safe_name}/beta_mean", module.bias.mean().item(), step)
                self.writer.add_scalar(f"BN/{safe_name}/beta_std", module.bias.std().item(), step)

    # ==========================================================================
    # 7. HYPERPARAMETERS (ghi 1 lan luc bat dau run)
    # ==========================================================================

    def log_hparams(self, cfg, step: int = 0) -> None:
        """Ghi hyperparameter cua cfg (TrainConfig) duoi dang text, chia 4 nhom.
        Chi can goi 1 lan luc khoi tao run."""
        self.writer.add_text("Hyperparameters/Model", "\n".join([
            f"- nc (num classes): {cfg.nc}",
            f"- reg_max: {cfg.reg_max}",
            f"- backbone_w: {cfg.backbone_w}",
            f"- backbone_n: {cfg.backbone_n}",
            f"- neck_n: {cfg.neck_n}",
            f"- strides: {cfg.strides}",
        ]), step)

        self.writer.add_text("Hyperparameters/Training", "\n".join([
            f"- epochs: {cfg.epochs}",
            f"- batch_size: {cfg.batch_size}",
            f"- img_size: {cfg.img_size}",
            f"- lr0: {cfg.lr0}",
            f"- lr_min_factor: {cfg.lr_min_factor}",
            f"- warmup_epochs: {cfg.warmup_epochs}",
            f"- weight_decay: {cfg.weight_decay}",
            f"- grad_clip_norm: {cfg.grad_clip_norm}",
            f"- optimizer: {cfg.optimizer}",
        ]), step)

        self.writer.add_text("Hyperparameters/Loss", "\n".join([
            f"- cls_gain: {cfg.cls_gain}",
            f"- box_gain: {cfg.box_gain}",
            f"- dfl_gain: {cfg.dfl_gain}",
            f"- w_o2o: {cfg.w_o2o}",
            f"- w_o2m: {cfg.w_o2m}",
            f"- topk_o2m: {cfg.topk_o2m}",
            f"- topk_o2o: {cfg.topk_o2o}",
            f"- alpha: {cfg.alpha}",
            f"- beta: {cfg.beta}",
        ]), step)

        self.writer.add_text("Hyperparameters/EMA", "\n".join([
            f"- use_ema: {cfg.use_ema}",
            f"- ema_decay: {cfg.ema_decay}",
            f"- ema_warmup_updates: {cfg.ema_warmup_updates}",
        ]), step)


# ==============================================================================
# TIEN ICH DOC LAP (khong gan voi nhip do log_interval/histogram_interval chung)
# ==============================================================================

class TimeTracker:
    """Theo doi thoi gian moi batch va throughput."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.batch_times = []

    def log_batch_time(self, batch_time: float, step: int) -> None:
        self.batch_times.append(batch_time)
        if len(self.batch_times) > 100:
            self.batch_times.pop(0)
        avg_time = sum(self.batch_times) / len(self.batch_times)
        self.writer.add_scalar("Time/batch_time_ms", batch_time * 1000, step)
        self.writer.add_scalar("Time/avg_batch_time_ms", avg_time * 1000, step)

    def log_throughput(self, batch_size: int, batch_time: float, step: int) -> None:
        self.writer.add_scalar("Time/throughput_samples_per_sec", batch_size / batch_time, step)


class ActivationTracker:
    """Gan forward hook de theo doi thong ke activation (Conv2d/BatchNorm2d/SiLU)."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.hooks = []
        self.step = 0

    def register_hooks(self, model: nn.Module) -> None:
        def get_hook(name):
            def hook(module, inp, output):
                if isinstance(output, torch.Tensor):
                    self._log_activation(name, output, self.step)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.SiLU)):
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def _log_activation(self, name: str, tensor: torch.Tensor, step: int) -> None:
        if tensor.numel() == 0:
            return
        safe_name = name.replace(".", "/")
        self.writer.add_scalar(f"Activations/{safe_name}/mean", tensor.mean().item(), step)
        self.writer.add_scalar(f"Activations/{safe_name}/std", tensor.std().item(), step)
        self.writer.add_scalar(f"Activations/{safe_name}/max", tensor.max().item(), step)
        self.writer.add_scalar(f"Activations/{safe_name}/min", tensor.min().item(), step)

    def set_step(self, step: int) -> None:
        self.step = step

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class LossSmoother:
    """Trung binh truot (moving average) cua loss - de xem duong loss it giat hon
    tren TensorBoard, tach rieng khoi loss "song" tung step."""

    def __init__(self, window: int = 100):
        self.window = window
        self.buffer = []

    def update(self, loss: float) -> float:
        self.buffer.append(loss)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return sum(self.buffer) / len(self.buffer)

    def log_smoothed(self, writer: SummaryWriter, loss: float, step: int, tag: str = "loss/smoothed") -> float:
        smoothed = self.update(loss)
        writer.add_scalar(tag, smoothed, step)
        return smoothed


def log_lr_schedule(writer: SummaryWriter, scheduler, step: int) -> None:
    """Ghi LR hien tai truc tiep tu scheduler.get_last_lr() (thay vi tu optimizer),
    huu ich khi muon xac nhan scheduler hoat dong dung nhu ky vong."""
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar("LR_Schedule/current", current_lr, step)
    if hasattr(scheduler, "warmup_steps"):
        warmup_progress = min(step / scheduler.warmup_steps, 1.0)
        writer.add_scalar("LR_Schedule/warmup_progress", warmup_progress, step)


def log_activation_histograms(writer: SummaryWriter, activations: Dict[str, torch.Tensor], step: int) -> None:
    """Ghi histogram cho 1 dict {ten_layer: tensor} activation da thu thap san."""
    for name, activation in activations.items():
        if isinstance(activation, torch.Tensor):
            writer.add_histogram(f"Activations_hist/{name.replace('.', '/')}", activation, step)


# ==============================================================================
# VI DU SU DUNG / DEMO (chay: python -m src.utils.tb_logger)
# ==============================================================================
if __name__ == "__main__":
    import os
    import random
    import shutil

    log_dir = "runs/demo_loss_logging"
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logger = TrainingLogger(writer, log_interval=10, histogram_interval=100)

    dummy_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    global_step = 0
    for epoch in range(5):
        for batch_idx in range(20):
            fake_items = {
                "loss": 10.0 / (epoch + 1) + random.uniform(-0.3, 0.3),
                "loss_o2m": 6.0 / (epoch + 1) + random.uniform(-0.2, 0.2),
                "loss_o2o": 4.0 / (epoch + 1) + random.uniform(-0.2, 0.2),
                "o2m/iou": 2.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/cls": 3.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/dfl": 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/iou": 1.5 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/cls": 2.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/dfl": 0.5 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/n_pos": random.randint(50, 200),
                "o2o/n_pos": random.randint(5, 20),
            }
            logger.log_losses(fake_items, step=global_step, phase="train")

            prev_params = TrainingLogger.snapshot_params(dummy_model)
            x = torch.randn(4, 10)
            y = dummy_model(x).sum()
            dummy_model.zero_grad()
            y.backward()
            logger.log_gradients(dummy_model, global_step)

            # (khong co optimizer that trong demo nay, nen chi mo phong weight logging)
            logger.log_weights(dummy_model, global_step)
            logger.log_weight_updates(dummy_model, prev_params, global_step)

            global_step += 1

        fake_val_items = {
            "loss": 9.0 / (epoch + 1),
            "loss_o2m": 5.5 / (epoch + 1),
            "loss_o2o": 3.5 / (epoch + 1),
            "o2m/iou": 1.8 / (epoch + 1),
            "o2m/cls": 2.8 / (epoch + 1),
            "o2m/dfl": 0.9 / (epoch + 1),
            "o2o/iou": 1.3 / (epoch + 1),
            "o2o/cls": 1.8 / (epoch + 1),
            "o2o/dfl": 0.4 / (epoch + 1),
            "o2m/n_pos": random.randint(50, 200),
            "o2o/n_pos": random.randint(5, 20),
        }
        logger.log_losses(fake_val_items, step=epoch, phase="val")

    writer.close()
    print(f"Da ghi log demo vao '{log_dir}'. Chay: tensorboard --logdir {log_dir}")