import os
import glob
import math
import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.model import NMSFreeDetector
from src.train.loss import DetectionLoss
from src.train.ema import ModelEMA
from src.train.dataloader1_obj365 import build_dataloaders
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint
from src.utils.logging_setup import setup_logging
from src.utils.tb_logger import TrainingLogger
from src.finetune.finetune_config import FinetuneConfig
from src.evaluation.mAPEvaluation import MetricAccumulator
from tqdm import tqdm

logger = logging.getLogger("finetune")

def build_finetune_model(cfg: FinetuneConfig) -> NMSFreeDetector:
    model = NMSFreeDetector.from_config(cfg.pretrain_architecture)

    if cfg.pretrained_ckpt:
        ckpt_path = cfg.pretrained_ckpt
        logger.info(f"Loading pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "backbone" in ckpt:
            model.load_feature_extractor(ckpt, strict=True)
            logger.info("Loaded backbone + neck from trunk checkpoint.")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            # Chuẩn hóa key để tránh lỗi module prefix (DDP / DataParallel)
            cleaned_state_dict = {
                k.replace("module.", "").replace("model.", ""): v 
                for k, v in state_dict.items()
            }
            filtered = {k: v for k, v in cleaned_state_dict.items()
                        if k.startswith("backbone.") or k.startswith("neck.")}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            logger.info(f"Loaded backbone+neck from full checkpoint. "
                        f"Missing keys (head expected): {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            raise ValueError(f"Checkpoint format không nhận diện được. Keys: {list(ckpt.keys())[:10]}")
    else:
        logger.warning("Không có pretrained_ckpt -> train from scratch (không khuyến khích cho finetune).")

    model.replace_head(nc=cfg.nc, reg_max=cfg.reg_max, strides=cfg.strides)
    logger.info(f"Đã thay DetectHead mới: nc={cfg.nc}, reg_max={cfg.reg_max}")
    return model

def build_optimizer(model: NMSFreeDetector, cfg: FinetuneConfig, phase: str = "frozen"):
    if phase == "frozen":
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr0,
                                     weight_decay=cfg.weight_decay,
                                     betas=cfg.betas)
        else:
            return torch.optim.SGD(params, lr=cfg.lr0,
                                   momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay,
                                   nesterov=True)
    else:
        trunk_decay, trunk_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_trunk = name.startswith("backbone.") or name.startswith("neck.")
            is_no_decay = (p.ndim <= 1 or name.endswith("bias"))

            if is_trunk:
                (trunk_no_decay if is_no_decay else trunk_decay).append(p)
            else:
                (head_no_decay if is_no_decay else head_decay).append(p)

        trunk_lr = cfg.lr0 * cfg.backbone_lr_scale
        groups = [
            {"params": trunk_decay, "lr": trunk_lr, "weight_decay": cfg.weight_decay},
            {"params": trunk_no_decay, "lr": trunk_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": cfg.lr0, "weight_decay": cfg.weight_decay},
            {"params": head_no_decay, "lr": cfg.lr0, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if len(g["params"]) > 0]

        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(groups, betas=cfg.betas)
        else:
            return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)

def build_scheduler(optimizer, cfg: FinetuneConfig, steps_per_epoch: int,
                    total_epochs: int, warmup_epochs: float = None):
    warmup_ep = warmup_epochs if warmup_epochs is not None else cfg.warmup_epochs
    warmup_steps = max(1, int(warmup_ep * steps_per_epoch))
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def _lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lambda)

def move_batch(images, targets, device):
    images = images.to(device, non_blocking=True)
    targets = [
        {
            "boxes": t["boxes"].to(device, non_blocking=True),
            "labels": t["labels"].to(device, non_blocking=True),
        }
        for t in targets
    ]
    return images, targets

def train_one_epoch(model, criterion, loader, optimizer, scheduler,
                    scaler, ema, device, cfg: FinetuneConfig, epoch, global_step,
                    best_val, val_loader=None, tb_logger=None):
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    use_amp = scaler is not None

    pbar = tqdm(
        enumerate(loader),
        total=n_batches,
        desc=f"Epoch [{epoch + 1}/{cfg.epochs}]",
        ncols=100,
        leave=True,
    )

    for step, (images, targets) in pbar:
        images, targets = move_batch(images, targets, device)
        global_step += 1

        optimizer.zero_grad(set_to_none=True)

        device_type = str(device).split(":")[0]
        with torch.autocast(device_type=device_type, enabled=use_amp):
            preds = model(images)
            loss, items = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
        if not math.isfinite(total_norm):
            logger.warning(f"[epoch {epoch + 1}] step {step} (global {global_step}): gradient NaN/Inf")

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_lr = scaler.get_scale() < scale_before
        else:
            optimizer.step()
            skip_lr = False

        if not skip_lr:
            scheduler.step()
            if ema is not None:
                ema.update(model=model)

        # Tránh rò rỉ VRAM bằng .item()
        loss_val = items["loss"].item() if isinstance(items["loss"], torch.Tensor) else items["loss"]
        running_loss += loss_val

        # TensorBoard logging
        if tb_logger is not None:
            tb_logger.log_losses(items, step=global_step, phase="finetune_train")
            tb_logger.log_learning_rate(optimizer, global_step, epoch)
            if ema is not None:
                tb_logger.log_ema(ema, global_step)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.1e}")

        # Log ngắn gọn ra file & stdout (sử dụng tqdm.write để không làm hỏng progress bar)
        if cfg.log_loss_interval > 0 and global_step % cfg.log_loss_interval == 0:
            msg = f"[epoch {epoch + 1}] step {step}/{n_batches} (global {global_step}) loss={loss_val:.4f} lr={lr:.6f}"
            tqdm.write(msg)
            logger.info(msg)

        if cfg.log_interval > 0 and global_step % cfg.log_interval == 0:
            mem = f" gpu={torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else ""
            msg = (
                f"[epoch {epoch + 1}] step {step}/{n_batches} (global {global_step}) "
                f"o2m(iou={items['o2m/iou']:.3f} cls={items['o2m/cls']:.3f} dfl={items['o2m/dfl']:.3f}) "
                f"o2o(iou={items['o2o/iou']:.3f} cls={items['o2o/cls']:.3f} dfl={items['o2o/dfl']:.3f}) "
                f"lr={lr:.6f}{mem}"
            )
            tqdm.write(msg)
            logger.info(msg)

        # Validate theo step
        if val_loader is not None and cfg.val_interval_steps > 0 and (global_step % cfg.val_interval_steps == 0):
            eval_model = ema.ema if ema is not None else model
            val_loss, val_metrics = validate(eval_model, criterion, val_loader, device,
                                             tb_logger=tb_logger, step=global_step, cfg=cfg)
            
            val_msg = (f"[VALIDATE step {global_step}] val_loss={val_loss:.4f} | "
                       f"mAP50={val_metrics['map_50']:.4f} | mAP50-95={val_metrics['map_50_95']:.4f} | "
                       f"P={val_metrics['precision']:.4f} | R={val_metrics['recall']:.4f}")
            tqdm.write(val_msg)
            logger.info(val_msg)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                                model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)
                (ema.ema if ema is not None else model).save_trunk(
                    os.path.join(cfg.ckpt_dir, "best_trunk.pt")
                )
                logger.info(f"[step {global_step}] -> best checkpoint mới (val_loss={best_val:.4f})")
            model.train()

        # Save checkpoint định kỳ
        if not cfg.save_best_only and cfg.save_ckpt_interval_steps > 0 and (global_step % cfg.save_ckpt_interval_steps == 0):
            step_path = os.path.join(cfg.ckpt_dir, f"ft_step{global_step:08d}.pt")
            save_checkpoint(step_path, model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)
            save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                            model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)
            
            keep = getattr(cfg, "ckpt_keep_last", 3)
            if keep > 0:
                old_ckpts = sorted(glob.glob(os.path.join(cfg.ckpt_dir, "ft_step*.pt")))
                for old in old_ckpts[:-keep]:
                    os.remove(old)
            logger.info(f"[step {global_step}] -> saved {step_path}")

    return running_loss / max(1, n_batches), global_step, best_val

@torch.no_grad()
def validate(model, criterion, loader, device, cfg: FinetuneConfig, tb_logger=None, step=0):
    model.eval()
    total = 0.0
    n = 0
    last_items = None
    acc = MetricAccumulator(nc=cfg.nc)

    pbar_val = tqdm(loader, desc="Validating", ncols=100, leave=False)

    for images, targets in pbar_val:
        images, targets = move_batch(images, targets, device)
        preds = model(images)
        loss, items = criterion(preds, targets)
        
        loss_val = items["loss"].item() if isinstance(items["loss"], torch.Tensor) else items["loss"]
        total += loss_val
        last_items = items
        n += 1

        acc.update(preds, targets)

    metrics = acc.compute()

    if tb_logger is not None:
        if last_items is not None:
            tb_logger.log_losses(last_items, step=step, phase="finetune_val")
        tb_logger.log_scalars({
            "map_50_95": metrics["map_50_95"],
            "map_50": metrics["map_50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }, step)

    return total / max(1, n), metrics

def run_finetune(cfg: FinetuneConfig):
    if not logger.handlers:
        setup_logging(log_dir=cfg.log_dir, run_name=cfg.run_name, also_stdout=False)

    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        logger.warning(f"'{cfg.device}' không khả dụng, fallback về '{device}'")

    train_loader, val_loader, classes, num_classes = build_dataloaders(cfg)
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    logger.info(f"[data] train={len(train_loader.dataset)} val={n_val} classes={len(classes)} (nc={num_classes})")

    if cfg.nc != num_classes:
        logger.warning(f"cfg.nc={cfg.nc} != num_classes từ dữ liệu={num_classes}. "
                       f"Sẽ dùng nc={cfg.nc} theo config.")

    # ---- Model ----
    model = build_finetune_model(cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[model] Total params: {n_total:,} | Trainable: {n_train:,}")

    # ---- Loss ----
    criterion = DetectionLoss(
        nc=cfg.nc,
        reg_max=cfg.reg_max,
        topk_o2m=cfg.topk_o2m,
        topk_o2o=cfg.topk_o2o,
        alpha=cfg.alpha,
        beta=cfg.beta,
        box_gain=cfg.box_gain,
        cls_gain=cfg.cls_gain,
        dfl_gain=cfg.dfl_gain,
        o2m_weight=cfg.w_o2m,
        o2o_weight=cfg.w_o2o,
    ).to(device)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=cfg.tb_log_dir) if cfg.tb_log_dir else None
    tb_logger = None
    if writer is not None:
        tb_logger = TrainingLogger(
            writer,
            log_interval=cfg.log_interval,
            histogram_interval=getattr(cfg, "log_hist_interval", 500),
        )
        tb_logger.log_hparams(cfg)

    # ---- AMP ----
    use_amp = cfg.amp and device.startswith("cuda") if isinstance(device, str) else False
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    steps_per_epoch = len(train_loader)
    global_step = 0
    best_val = float("inf")

    # ===== PHASE 1: Freeze trunk, train head =====
    phase1_epochs = min(cfg.freeze_epochs, cfg.epochs)
    logger.info(f"===== PHASE 1: Freeze trunk, train head ({phase1_epochs} epochs) =====")

    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.neck.parameters():
        p.requires_grad_(False)

    n_train_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[phase1] Trainable params: {n_train_p1:,}")

    optimizer = build_optimizer(model, cfg, phase="frozen")
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch,
                                total_epochs=max(phase1_epochs, 1), warmup_epochs=cfg.warmup_epochs)
    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates) if cfg.use_ema else None

    for epoch in range(phase1_epochs):
        train_loss, global_step, best_val = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            scaler, ema, device, cfg, epoch, global_step, best_val,
            val_loader=val_loader, tb_logger=tb_logger,
        )
        logger.info(f"[phase1][epoch {epoch + 1}] train_loss={train_loss:.4f} (global_step={global_step})")

    # ===== PHASE 2: Full model training =====
    phase2_epochs = cfg.epochs - phase1_epochs
    logger.info(f"===== PHASE 2: Full model training ({phase2_epochs} epochs) =====")

    for p in model.parameters():
        p.requires_grad_(True)

    n_train_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[phase2] Trainable params: {n_train_p2:,}")

    optimizer = build_optimizer(model, cfg, phase="full")
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch,
                                total_epochs=max(phase2_epochs, 1), warmup_epochs=cfg.phase2_warmup_epochs)
    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates) if cfg.use_ema else None

    for epoch in range(phase1_epochs, cfg.epochs):
        train_loss, global_step, best_val = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            scaler, ema, device, cfg, epoch, global_step, best_val,
            val_loader=val_loader, tb_logger=tb_logger,
        )
        logger.info(f"[phase2][epoch {epoch + 1}] train_loss={train_loss:.4f} (global_step={global_step})")

    if val_loader is not None:
        eval_model = ema.ema if ema is not None else model
        val_loss, val_metrics = validate(eval_model, criterion, val_loader, device,
                                         tb_logger=tb_logger, step=global_step, cfg=cfg)
        logger.info(f"[FINAL VAL] loss={val_loss:.4f} | mAP50={val_metrics['map_50']:.4f} | "
                    f"mAP50-95={val_metrics['map_50_95']:.4f} | P={val_metrics['precision']:.4f} | R={val_metrics['recall']:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                            model, optimizer, scheduler, ema, cfg.epochs - 1, global_step, best_val, cfg)
            (ema.ema if ema is not None else model).save_trunk(
                os.path.join(cfg.ckpt_dir, "best_trunk.pt")
            )

    save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                    model, optimizer, scheduler, ema, cfg.epochs - 1, global_step, best_val, cfg)
    (ema.ema if ema is not None else model).save_trunk(
        os.path.join(cfg.ckpt_dir, "last_trunk.pt")
    )

    if writer is not None:
        writer.close()

    logger.info(f"Finetune hoàn tất. Best val_loss = {best_val:.4f}")
    logger.info(f"Best checkpoint: {os.path.join(cfg.ckpt_dir, 'best.pt')}")
    logger.info(f"Last trunk: {os.path.join(cfg.ckpt_dir, 'last_trunk.pt')}")
    return best_val