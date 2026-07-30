import os
import glob
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
from src.utils.checkpoint import load_checkpoint, save_checkpoint, load_model_only
from src.utils.logging_setup import setup_logging
from src.utils.tb_logger import TrainingLogger
from tqdm import tqdm

logger = logging.getLogger("train")

def _ensure_text_logging(cfg: TrainConfig) -> None:
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
    model = NMSFreeDetector(nc=cfg.nc, reg_max=cfg.reg_max,
                           backbone_w=cfg.backbone_w, backbone_n=cfg.backbone_n,
                           neck_n=cfg.neck_n, strides=cfg.strides)
    # LOAD PRE MODEL
    # load_model_only(model=model, path="/run/media/tranmanhduy/Data/Ending/ckpt_step00160000.pt", map_location="cuda")
    # print("load last model successful")
    return model

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

def save_periodic_checkpoint(cfg: TrainConfig, model, optimizer, scheduler, ema, epoch, global_step, best_val):
    step_path = os.path.join(cfg.ckpt_dir, f"ckpt_step{global_step:08d}.pt")
    save_checkpoint(step_path, model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)
    save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                    model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)

    keep_last = getattr(cfg, "ckpt_keep_last", 3)
    if keep_last > 0:
        step_ckpts = sorted(glob.glob(os.path.join(cfg.ckpt_dir, "ckpt_step*.pt")))
        for old_path in step_ckpts[:-keep_last]:
            os.remove(old_path)

    logger.info(f"[step {global_step}] -> đã lưu {step_path} (và cập nhật last.pt)")
    return step_path

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
                    criterion: DetectionLoss, loader, val_loader, optimizer,
                    scheduler, scaler, ema: ModelEMA, device, cfg: TrainConfig, epoch: int,
                    global_step: int = 0, best_val: float = float("inf"),
                    tb_logger: TrainingLogger = None):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_batches = len(loader)
    use_amp = scaler is not None
    
    do_grad_log = tb_logger is not None and getattr(cfg, "log_gradients", True)
    do_weight_log = tb_logger is not None and getattr(cfg, "log_weights", True)
    
    val_interval_steps = getattr(cfg, "val_interval_steps", 500)
    save_ckpt_interval_steps = getattr(cfg, "save_ckpt_interval_steps", 1000)
    log_interval = getattr(cfg, "log_interval", 0)
    loss_log_interval = getattr(cfg, "log_loss_interval", 50)
    
    pbar = tqdm(
            enumerate(loader),
            total=n_batches,
            desc=f"Epoch [{epoch + 1}/{cfg.epochs}]",
            ncols=100,
            leave=True
        )
    
    for step, (images, targets) in pbar:
        images, targets = move_batch(images, targets, device)
        global_step += 1
        do_snapshot = do_weight_log and tb_logger.should_log_scalar(global_step)
        prev_params = TrainingLogger.snapshot_params(model) if do_snapshot else None

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
            logger.warning(f"[epoch {epoch}] step {step} (global {global_step}): gradient NaN/Inf "
                            f"(total_norm={total_norm}) - loss đang có thể phân kỳ.")
        if do_grad_log:
            tb_logger.log_gradients(model, global_step, total_norm=total_norm)

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_lr_sched = scaler.get_scale() < scale_before   # scale giảm => step vừa rồi bị skip
        else:
            optimizer.step()
            skip_lr_sched = False

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
        pbar.set_postfix(loss=f"{items['loss']:.4f}", lr=f"{lr:.1e}")

        if loss_log_interval > 0 and global_step % loss_log_interval == 0:
            logger.info(f"[epoch {epoch}] step {step}/{n_batches} (global {global_step}) loss={items['loss']:.4f} lr={lr:.6f}")

        if log_interval > 0 and global_step % log_interval == 0:
            elapsed = time.time() - t0
            ema_str = f" ema_decay={ema._current_decay():.5f}" if ema is not None else ""
            mem_str = f" gpu_mem={torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else ""
            logger.info(
                f"[epoch {epoch}] step {step}/{n_batches} (global {global_step}) "
                f"o2m(iou={items['o2m/iou']:.3f} cls={items['o2m/cls']:.3f} dfl={items['o2m/dfl']:.3f} npos={items['o2m/n_pos']}) "
                f"o2o(iou={items['o2o/iou']:.3f} cls={items['o2o/cls']:.3f} dfl={items['o2o/dfl']:.3f} npos={items['o2o/n_pos']}) "
                f"lr={lr:.6f}{ema_str}{mem_str} t={elapsed:.1f}s"
            )

        # Validate theo step
        if val_loader is not None and val_interval_steps > 0 and (global_step % val_interval_steps == 0):
            eval_model = ema.ema if ema is not None else model
            val_loss = validate(eval_model, criterion, val_loader, device, tb_logger=tb_logger, step=global_step)
            logger.info(f"[step {global_step}] (epoch {epoch}) train_loss={items['loss']:.4f} val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                                model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg)
                (ema.ema if ema is not None else model).save_trunk(
                    os.path.join(cfg.ckpt_dir, "best_trunk.pt")
                )
                logger.info(f"[step {global_step}] -> best checkpoint mới (val_loss={best_val:.4f})")

            model.train()

        if not cfg.save_best_only and save_ckpt_interval_steps > 0 and (global_step % save_ckpt_interval_steps == 0):
            save_periodic_checkpoint(cfg, model, optimizer, scheduler, ema, epoch, global_step, best_val)

    return running_loss / max(1, n_batches), global_step, best_val

@torch.no_grad()
def validate(model, criterion, loader, device, tb_logger: TrainingLogger = None, step: int = 0):
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
        tb_logger.log_losses(last_items, step=step, phase="val")

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
    global_step = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, global_step, best_val = load_checkpoint(cfg.resume, model, optimizer, scheduler, ema, map_location=device)
        logger.info(f"[resume] tiếp tục từ epoch {start_epoch}, global_step {global_step}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, global_step, best_val = train_one_epoch(
            model, criterion, train_loader, val_loader, optimizer,
            scheduler, scaler, ema, device, cfg, epoch,
            global_step=global_step, best_val=best_val, tb_logger=tb_logger
        )
        logger.info(f"[epoch {epoch}] train_loss={train_loss:.4f} (global_step={global_step})")
    
    # Kiểm tra validate & checkpoint lần cuối khi kết thúc training
    if val_loader is not None:
        eval_model = ema.ema if ema is not None else model
        val_loss = validate(eval_model, criterion, val_loader, device, tb_logger=tb_logger, step=global_step)
        logger.info(f"[final val] step {global_step}: val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                            model, optimizer, scheduler, ema, cfg.epochs - 1, global_step, best_val, cfg)
            (ema.ema if ema is not None else model).save_trunk(
                os.path.join(cfg.ckpt_dir, "best_trunk.pt")
            )
            logger.info(f"[final step {global_step}] -> best checkpoint mới (val_loss={best_val:.4f})")

    if not cfg.save_best_only:
        save_periodic_checkpoint(cfg, model, optimizer, scheduler, ema, cfg.epochs - 1, global_step, best_val)

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

    train_loss, global_step, best_val = train_one_epoch(
        model, criterion, train_loader, val_loader, optimizer, scheduler,
        scaler, ema, device, cfg, epoch=0, global_step=0, best_val=float("inf")
    )