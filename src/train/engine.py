import os
import math
import time
import logging

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.model import NMSFreeDetector
from src.train.loss import DetectionLoss
from src.train.ema import ModelEMA
from src.train.dataloader1_obj365 import build_dataloaders
from src.config import TrainConfig
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging_setup import setup_logging
from src.utils.tb_logger import TrainingLogger
import shutil
from tqdm import tqdm

# Logger text (ghi file .log) - dung chung 1 logger ten "train" xuyen suot module.
# logging.getLogger("train") tra ve CUNG 1 object moi lan goi, nen du duoc lay
# truoc khi setup_logging() gan handler (xem _ensure_text_logging), cac loi goi
# logger.info(...) ben duoi van tu dong co handler ngay khi setup_logging() chay.
logger = logging.getLogger("train")


def _ensure_text_logging(cfg: TrainConfig) -> None:
    """Khoi tao logging_setup 1 lan duy nhat. Neu noi goi khac (vd training.py)
    da goi setup_logging() truoc do roi thi bo qua, tranh mo 2 file .log khac
    timestamp cho cung 1 lan chay."""
    if logger.handlers:
        return
    setup_logging(
        log_dir=cfg.log_dir,
        run_name=cfg.run_name,
        also_stdout=False,
    )


def get_dataloader(cfg: TrainConfig):
    train_loader, val_loader, classes, num_classes = build_dataloaders(cfg)

    if cfg.nc is None:
        cfg.nc = num_classes
        logger.info(f"[Config] cfg.nc chưa được set -> tự động lấy từ dữ liệu: nc={cfg.nc}")
    elif cfg.nc != num_classes:
        logger.warning(f"[Config] cfg.nc={cfg.nc} KHÁC với số class thực tế trong dữ liệu "
                        f"({num_classes}). Model sẽ dùng cfg.nc={cfg.nc} theo đúng ý người dùng, "
                        f"nhưng hãy chắc chắn đây là chủ đích (vd: giữ chỗ cho các class sẽ thêm sau).")

    return train_loader, val_loader, classes, num_classes

def get_model(cfg: TrainConfig):
    return NMSFreeDetector(nc=cfg.nc, reg_max=cfg.reg_max,
                           backbone_w=cfg.backbone_w, backbone_n=cfg.backbone_n,
                           neck_n=cfg.neck_n, strides=cfg.strides)

def get_criterion(cfg: TrainConfig):
    return DetectionLoss(
        nc=cfg.nc,
        reg_max=cfg.reg_max,
        topk_o2m=getattr(cfg, "topk_o2m", 10),
        topk_o2o=getattr(cfg, "topk_o2o", 1),
        alpha=getattr(cfg, "alpha", 0.5),
        beta=getattr(cfg, "beta", 6.0),
        box_gain=getattr(cfg, "box_gain", 7.5),
        cls_gain=getattr(cfg, "cls_gain", 1.0),
        dfl_gain=getattr(cfg, "dfl_gain", 1.5),
        o2m_weight=getattr(cfg, "w_o2m", 1.0),
        o2o_weight=getattr(cfg, "w_o2o", 1.0)
    )


def get_optimizer(model: NMSFreeDetector, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg.lr0, betas=getattr(cfg, "betas", (0.9, 0.999)))
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg.lr0, momentum=cfg.momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    return opt


def lr_lambda_factory(cfg: TrainConfig, steps_per_epoch):
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, cfg.epochs * steps_per_epoch)

    def _lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * cosine

    return _lambda


def move_batch(images, targets, device):
    images = images.to(device, non_blocking=True)
    targets = [
        {
            "boxes": t["boxes"].to(device, non_blocking=True),
            "labels": t["labels"].to(device, non_blocking=True)
        }
        for t in targets
    ]
    return images, targets


def train_one_epoch(model: NMSFreeDetector,
                    criterion: DetectionLoss, loader, optimizer,
                    scheduler, scaler, ema: ModelEMA, device, cfg: TrainConfig, epoch: int,
                    tb_logger: TrainingLogger = None):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_batches = len(loader)
    use_amp = scaler is not None

    # Cac cong tac bat/tat tung loai log TensorBoard (giu nguyen y nghia nhu cfg
    # cu: cfg.log_gradients / cfg.log_weights); tan suat ghi (scalar vs histogram)
    # do TrainingLogger tu quan ly qua log_interval/histogram_interval.
    do_grad_log = tb_logger is not None and getattr(cfg, "log_gradients", True)
    do_weight_log = tb_logger is not None and getattr(cfg, "log_weights", True)

    
    pbar = tqdm(
            enumerate(loader),
            total=n_batches,
            desc=f"Epoch [{epoch + 1}/{cfg.epochs}]",
            ncols=100,
            leave=True
        )
    
    for step, (images, targets) in pbar:
        images, targets = move_batch(images, targets, device)
        global_step = epoch * n_batches + step

        # Chup tham so TRUOC khi update, de sau nay tinh Update_Ratio (delta_w / w).
        # Chi can chup khi thuc su se log weight, tranh clone() thua moi step.
        prev_params = TrainingLogger.snapshot_params(model) if do_weight_log else None

        optimizer.zero_grad(set_to_none=True)

        device_type = str(device).split(":")[0]
        with torch.autocast(device_type=device_type, enabled=use_amp):
            preds = model(images)
            loss, items = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Log gradient SAU unscale_ (phan anh dung scale that, khong bi AMP
            # scale factor lam sai lech) NHUNG TRUOC clip_grad_norm_ - neu log sau
            # khi clip, Gradients/total_norm se luon bi ep <= grad_clip_norm va
            # khong con thay duoc gradient thuc su lon bao nhieu (mat tin hieu
            # explosion, chinh la thu can thay de debug NaN/Inf).
            if do_grad_log:
                total_norm = tb_logger.log_gradients(model, global_step)
                if not math.isfinite(total_norm):
                    logger.warning(f"[epoch {epoch}] step {step}: gradient NaN/Inf truoc khi clip "
                                    f"(total_norm={total_norm}) - loss dang co the phan ky.")

            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            skip_lr_sched = (scale_after < scale_before)   # scale giảm => step vừa rồi đã bị skip
        else:
            loss.backward()

            if do_grad_log:
                total_norm = tb_logger.log_gradients(model, global_step)
                if not math.isfinite(total_norm):
                    logger.warning(f"[epoch {epoch}] step {step}: gradient NaN/Inf truoc khi clip "
                                    f"(total_norm={total_norm}) - loss dang co the phan ky.")

            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            skip_lr_sched = False

        # Log weight SAU optimizer.step() - luc nay trong so da thuc su duoc cap nhat.
        if do_weight_log:
            tb_logger.log_weights(model, global_step)
            tb_logger.log_weight_updates(model, prev_params, global_step)

        if not skip_lr_sched:
            scheduler.step()
        if ema is not None:
            ema.update(model=model)

        running_loss += items["loss"]

        if tb_logger is not None:
            tb_logger.log_losses(items, step=global_step, phase="train")
            tb_logger.log_learning_rate(optimizer, global_step, epoch)
            tb_logger.log_ema(ema, global_step)
            tb_logger.log_gpu_memory(global_step)

        
        lr = optimizer.param_groups[0]["lr"]
        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=f"{items['loss']:.4f}", lr=f"{lr:.1e}")
            elapsed = time.time() - t0
            logger.info(
                f"[epoch {epoch}] step {step}/{n_batches} "
                f"loss={items['loss']:.4f} "
                f"(o2m iou={items['o2m/iou']:.3f} cls={items['o2m/cls']:.3f} "
                f"dfl={items['o2m/dfl']:.3f} npos={items['o2m/n_pos']}) "
                f"(o2o iou={items['o2o/iou']:.3f} cls={items['o2o/cls']:.3f} "
                f"dfl={items['o2o/dfl']:.3f} npos={items['o2o/n_pos']}) "
                f"lr={lr:.6f} t={elapsed:.1f}s"
            )
    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate(model, criterion, loader, device, tb_logger: TrainingLogger = None, epoch: int = 0):
    model.eval()
    total = 0.0
    n = 0
    last_items = None
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        preds = model(images)
        loss, items = criterion(preds, targets)
        total += items["loss"]
        last_items = items
        n += 1

    if tb_logger is not None and last_items is not None:
        tb_logger.log_losses(last_items, step=epoch, phase="val")

    return total / max(1, n)


def run_training(cfg: TrainConfig):
    _ensure_text_logging(cfg)

    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    tb_log_dir = getattr(cfg, "tb_log_dir", "runs")
    writer = SummaryWriter(log_dir=tb_log_dir) if tb_log_dir else None
    tb_logger = None
    if writer is not None:
        tb_logger = TrainingLogger(
            writer,
            log_interval=cfg.log_interval,
            histogram_interval=getattr(cfg, "log_hist_interval", 100),
        )
        tb_logger.log_hparams(cfg)  # ghi hyperparameters 1 lan luc bat dau run

    device = cfg.device if torch.cuda.is_available() else "cpu"  # fallback về CPU nếu không có CUDA
    if device != cfg.device:
        logger.warning(f"'{cfg.device}' không khả dụng, fallback về '{device}'")

    train_loader, val_loader, classes, _ = get_dataloader(cfg)
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    logger.info(f"[data] train={len(train_loader.dataset)} val={n_val} classes={len(classes)}")

    model = get_model(cfg).to(device=device)
    criterion = get_criterion(cfg).to(device=device)

    optimizer = get_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    lr_lambda = lr_lambda_factory(cfg, steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates) if cfg.use_ema else None

    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, best_val = load_checkpoint(cfg.resume, model, optimizer, scheduler, ema, map_location=device)
        start_epoch += 1
        logger.info(f"[resume] tiếp tục từ epoch {start_epoch}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer,
                                     scheduler, scaler, ema, device, cfg, epoch, tb_logger=tb_logger)

        do_val = ((epoch + 1) % cfg.val_interval == 0) or (epoch == cfg.epochs - 1)
        val_loss = None
        if do_val and val_loader is None:
            logger.info(f"[epoch {epoch}] train_loss={train_loss:.4f} "
                        f"(bỏ qua validate: không có val_loader)")
        elif do_val:
            eval_model = ema.ema if ema is not None else model
            val_loss = validate(eval_model, criterion, val_loader, device, tb_logger=tb_logger, epoch=epoch)
            logger.info(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        else:
            logger.info(f"[epoch {epoch}] train_loss={train_loss:.4f}")

        is_best = (val_loss is not None) and (val_loss < best_val)
        if is_best:
            best_val = val_loss
            save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                            model, optimizer, scheduler, ema, epoch, best_val, cfg)
            (ema.ema if ema is not None else model).save_trunk(
                os.path.join(cfg.ckpt_dir, "best_trunk.pt")
            )
            logger.info(f"[epoch {epoch}] -> best checkpoint mới (val_loss={best_val:.4f})")

        if not cfg.save_best_only:
            save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                            model, optimizer, scheduler, ema, epoch, best_val, cfg)

    if writer is not None:
        writer.close()

    logger.info(f"Training xong. Best_val = {best_val}")
    logger.info(f"Checkpoint tốt nhất: {os.path.join(cfg.ckpt_dir, 'best.pt')}")
    logger.info(f"Trunk (backbone+neck) tốt nhất để đổi head sau này: {os.path.join(cfg.ckpt_dir, 'best_trunk.pt')}")
    return best_val


if __name__=="__main__":
    cfg = TrainConfig()
    _ensure_text_logging(cfg)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    set_seed(cfg.seed)

    train_loader, val_loader, classes, num_classes = get_dataloader(cfg)
    model = get_model(cfg).to(cfg.device)
    criterion = get_criterion(cfg).to(cfg.device)
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg, len(train_loader)))

    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates) if cfg.use_ema else None

    train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler,
                                 scaler, ema, device, cfg, epoch=0)