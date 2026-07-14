import os
import math
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import NMSFreeDetector
from dataset import YOLOv10Dataset, collate_fn, split_dataset
from train.lossv1 import DetectionLoss
from train.ema import ModelEMA

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def build_dataloaders(cfg):
    """
    Tao 2 instance dataset tro chung data_dir: 1 ban augment=True (train),
    1 ban augment=False (val) - roi chia theo CUNG mot bo index de tranh
    leak du lieu train/val nhung van co augmentation dung cho tap train.
    """
    base_ds = YOLOv10Dataset(cfg.data_dir, cfg.annotations_file, cfg.classes_file,
                              img_size=cfg.img_size, augment=False)
    train_idx, val_idx = split_dataset(base_ds, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_ds_full = YOLOv10Dataset(cfg.data_dir, cfg.annotations_file, cfg.classes_file,
                                    img_size=cfg.img_size, augment=cfg.augment,
                                    hflip_p=cfg.hflip_p, color_jitter_p=cfg.color_jitter_p)
    val_ds_full = base_ds  # augment=False

    train_set = Subset(train_ds_full, train_idx)
    val_set = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate_fn,
                               drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_fn,
                             drop_last=False, pin_memory=True)
    return train_loader, val_loader, base_ds.classes


# ----------------------------------------------------------------------
# Model / optim / scheduler
# ----------------------------------------------------------------------
def build_model(cfg):
    return NMSFreeDetector(nc=cfg.nc, reg_max=cfg.reg_max,
                            backbone_w=cfg.backbone_w, backbone_n=cfg.backbone_n,
                            neck_n=cfg.neck_n, strides=cfg.strides)


def build_optimizer(model, cfg):
    """
    Tach param group: KHONG ap weight decay cho BatchNorm va bias (chuan YOLO),
    chi ap weight decay cho trong so conv/linear.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg.lr0, betas=(0.9, 0.999))
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg.lr0, momentum=cfg.momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    return opt


def lr_lambda_factory(cfg, steps_per_epoch):
    """
    Warmup tuyen tinh (warmup_epochs) roi cosine decay ve lr0*lr_min_factor.
    Tra ve ham lambda(step) nhan voi lr0 -> dung voi torch.optim.lr_scheduler.LambdaLR.
    """
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, cfg.epochs * steps_per_epoch)

    def _lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * cosine

    return _lambda


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_val, cfg):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "cfg": cfg.__dict__,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


# ----------------------------------------------------------------------
# Train / Val loops
# ----------------------------------------------------------------------
def move_batch(batch, device):
    return (batch["images"].to(device, non_blocking=True),
            batch["gt_boxes"].to(device, non_blocking=True),
            batch["gt_labels"].to(device, non_blocking=True),
            batch["gt_mask"].to(device, non_blocking=True))


def train_one_epoch(model, criterion, loader, optimizer, scheduler, scaler, ema, device, cfg, epoch):
    model.train()
    t0 = time.time()
    running = {"loss_total": 0.0}
    n_batches = len(loader)

    for step, batch in enumerate(loader):
        images, gt_boxes, gt_labels, gt_mask = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                             enabled=cfg.amp and device.startswith("cuda")):
            preds = model(images)
            loss, logs = criterion(preds, gt_boxes, gt_labels, gt_mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        running["loss_total"] += logs["loss_total"]

        if step % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"[epoch {epoch}] step {step}/{n_batches} "
                  f"loss={logs['loss_total']:.4f} "
                  f"(o2m cls={logs['o2m']['cls']:.3f} box={logs['o2m']['box']:.3f} dfl={logs['o2m']['dfl']:.3f} npos={logs['o2m']['n_pos']}) "
                  f"(o2o cls={logs['o2o']['cls']:.3f} box={logs['o2o']['box']:.3f} dfl={logs['o2o']['dfl']:.3f} npos={logs['o2o']['n_pos']}) "
                  f"lr={lr:.6f} t={elapsed:.1f}s")

    return running["loss_total"] / max(1, n_batches)


@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images, gt_boxes, gt_labels, gt_mask = move_batch(batch, device)
        preds = model(images)
        loss, logs = criterion(preds, gt_boxes, gt_labels, gt_mask)
        total += logs["loss_total"]
        n += 1
    return total / max(1, n)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_training(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"[warn] '{cfg.device}' khong kha dung, fallback ve '{device}'")

    train_loader, val_loader, classes = build_dataloaders(cfg)
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={len(classes)}")

    # nc lay theo so class thuc te trong du lieu neu vuot qua cfg.nc
    n_classes = max(classes.keys()) + 1 if classes else cfg.nc
    if n_classes != cfg.nc:
        print(f"[info] dat lai nc={n_classes} theo classes.jsonl (config dang la {cfg.nc})")
        cfg.nc = n_classes

    model = build_model(cfg).to(device)
    criterion = DetectionLoss(nc=cfg.nc, reg_max=cfg.reg_max,
                               w_cls=cfg.w_cls, w_box=cfg.w_box, w_dfl=cfg.w_dfl, w_o2o=cfg.w_o2o)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    lr_lambda = lr_lambda_factory(cfg, steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.startswith("cuda"))
    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates) if cfg.use_ema else None

    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, best_val = load_checkpoint(cfg.resume, model, optimizer, scheduler, ema, map_location=device)
        start_epoch += 1
        print(f"[resume] tiep tuc tu epoch {start_epoch}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler,
                                      scaler, ema, device, cfg, epoch)

        do_val = ((epoch + 1) % cfg.val_interval == 0) or (epoch == cfg.epochs - 1)
        val_loss = None
        if do_val:
            eval_model = ema.ema if ema is not None else model
            val_loss = validate(eval_model, criterion, val_loader, device)
            print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                                 model, optimizer, scheduler, ema, epoch, best_val, cfg)
                # luu rieng trunk (backbone+neck) cua ban BEST -> dung khi doi head sau nay
                (ema.ema if ema is not None else model).save_trunk(
                    os.path.join(cfg.ckpt_dir, "best_trunk.pt"))
                print(f"[epoch {epoch}] -> best checkpoint moi (val_loss={best_val:.4f})")

        if not cfg.save_best_only:
            save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                             model, optimizer, scheduler, ema, epoch, best_val, cfg)

    print("Training xong. best_val =", best_val)
    print(f"Checkpoint tot nhat: {os.path.join(cfg.ckpt_dir, 'best.pt')}")
    print(f"Trunk (backbone+neck) tot nhat de doi head sau nay: {os.path.join(cfg.ckpt_dir, 'best_trunk.pt')}")
    return best_val