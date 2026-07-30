import argparse
import logging
import math
import os
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from face_lmk_config import FaceLmkConfig
from train_config_face_lmk import TrainConfig
from model_face_lmk import FaceLmkDetector
from face_landmark_dataset_v3 import FaceLandmarkDataset, face_landmark_collate
from loss_face_landmark_v3 import FaceLandmarkDetectionLoss
logger = logging.getLogger('train_face_lmk')

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _ensure_text_logging(cfg: TrainConfig):
    if logger.handlers:
        return
    os.makedirs(cfg.log_dir, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_h = logging.FileHandler(os.path.join(cfg.log_dir, f'{cfg.run_name}.log'), encoding='utf-8')
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    logger.addHandler(file_h)
    logger.addHandler(stream_h)
    logger.setLevel(logging.INFO)

class ModelEMA:

    def __init__(self, model: nn.Module, decay: float=0.9998, warmup_updates: int=2000):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup_updates))
        msd = model.state_dict()
        with torch.no_grad():
            for (k, v) in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
                else:
                    v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

def save_checkpoint(path, model, optimizer, scaler, ema: Optional[ModelEMA], epoch, global_step, best_val):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if scaler is not None else None, 'ema': ema.state_dict() if ema is not None else None, 'epoch': epoch, 'global_step': global_step, 'best_val': best_val}, path)

def save_periodic_checkpoint(cfg: TrainConfig, model, optimizer, scaler, ema, epoch, global_step, best_val):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    step_path = os.path.join(cfg.ckpt_dir, f'ckpt_step{global_step:08d}.pt')
    last_path = os.path.join(cfg.ckpt_dir, 'last.pt')
    save_checkpoint(step_path, model, optimizer, scaler, ema, epoch, global_step, best_val)
    save_checkpoint(last_path, model, optimizer, scaler, ema, epoch, global_step, best_val)
    step_files = sorted((f for f in os.listdir(cfg.ckpt_dir) if f.startswith('ckpt_step') and f.endswith('.pt')))
    while len(step_files) > cfg.ckpt_keep_last:
        os.remove(os.path.join(cfg.ckpt_dir, step_files.pop(0)))

def load_checkpoint(path, model, optimizer=None, scaler=None, ema: Optional[ModelEMA]=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and ckpt.get('ema') is not None:
        ema.ema.load_state_dict(ckpt['ema'])
    return (ckpt.get('epoch', 0), ckpt.get('global_step', 0), ckpt.get('best_val', float('inf')))

def build_dataloaders(cfg: TrainConfig):
    train_ds = FaceLandmarkDataset(cfg.train_root_dir, jsonl_name=cfg.jsonl_name, image_size=cfg.image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=face_landmark_collate, pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None, drop_last=True)
    val_loader = None
    if cfg.val_root_dir:
        val_ds = FaceLandmarkDataset(cfg.val_root_dir, jsonl_name=cfg.jsonl_name, image_size=cfg.image_size)
        if val_ds.num_landmarks != train_ds.num_landmarks:
            raise ValueError(f'Số landmark tập val ({val_ds.num_landmarks}) khác tập train ({train_ds.num_landmarks}) - kiểm tra lại dữ liệu 2 tập trước khi train.')
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=face_landmark_collate, pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None, drop_last=False)
    return (train_loader, val_loader, train_ds.num_landmarks)

def build_optimizer(cfg: TrainConfig, model: FaceLmkDetector):
    trunk_params = [p for p in list(model.backbone.parameters()) + list(model.neck.parameters()) if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    param_groups = [{'params': head_params, 'lr': cfg.lr}]
    if trunk_params:
        param_groups.append({'params': trunk_params, 'lr': cfg.lr * cfg.trunk_lr_mult})
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(param_groups, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f'optimizer không hỗ trợ: {cfg.optimizer}')

def build_scheduler(cfg: TrainConfig, optimizer, steps_per_epoch: int, epochs: Optional[int]=None):
    epochs = epochs if epochs is not None else cfg.epochs
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(int(cfg.warmup_epochs * steps_per_epoch), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(cfg, model, loss_fn, loader, optimizer, scheduler, scaler, ema, device, epoch, global_step, writer):
    model.train()
    t_epoch = time.time()
    (running_loss, n_steps_done) = (0.0, 0)
    use_amp = cfg.amp and device.type == 'cuda'
    for (step, batch) in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['targets']
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
            (total_loss, items) = loss_fn(preds, targets)
        if not torch.isfinite(total_loss):
            logger.warning(f'[epoch {epoch} step {step}] loss không hữu hạn ({total_loss.item()}) - bỏ qua step.')
            continue
        skip_lr_sched = False
        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
            if not math.isfinite(total_norm):
                logger.warning(f'[epoch {epoch} step {step}] grad norm không hữu hạn ({total_norm}).')
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_lr_sched = scaler.get_scale() < prev_scale
        else:
            total_loss.backward()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
            optimizer.step()
        if not skip_lr_sched:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        global_step += 1
        running_loss += items['loss']
        n_steps_done += 1
        if global_step % cfg.log_loss_interval == 0:
            lr_head = optimizer.param_groups[0]['lr']
            logger.info(f"epoch {epoch} step {step}/{len(loader)} (global {global_step}) loss={items['loss']:.4f} lr={lr_head:.2e}")
            writer.add_scalar('train/loss_total', items['loss'], global_step)
            writer.add_scalar('train/lr', lr_head, global_step)
        if global_step % cfg.log_interval == 0:
            writer.add_scalars('train/loss_o2m', {'iou': items['o2m/iou'], 'cls': items['o2m/cls'], 'dfl': items['o2m/dfl'], 'lmk': items['o2m/lmk'], 'geo': items['o2m/geo']}, global_step)
            writer.add_scalars('train/loss_o2o', {'iou': items['o2o/iou'], 'cls': items['o2o/cls'], 'dfl': items['o2o/dfl'], 'lmk': items['o2o/lmk'], 'geo': items['o2o/geo']}, global_step)
            writer.add_scalars('train/n_pos', {'o2m': items['o2m/n_pos'], 'o2o': items['o2o/n_pos'], 'o2m_lmk': items['o2m/n_lmk_pos'], 'o2o_lmk': items['o2o/n_lmk_pos']}, global_step)
            if cfg.log_gradients:
                writer.add_scalar('train/grad_norm', total_norm, global_step)
            if device.type == 'cuda':
                writer.add_scalar('train/gpu_mem_gb', torch.cuda.max_memory_allocated() / 1000000000.0, global_step)
    dt = time.time() - t_epoch
    avg_loss = running_loss / max(n_steps_done, 1)
    logger.info(f'== epoch {epoch} xong: loss trung bình={avg_loss:.4f}, {n_steps_done} step, mất {dt:.1f}s ==')
    return global_step

@torch.no_grad()
def validate(cfg, model, loss_fn, loader, device, epoch, global_step, writer):
    if loader is None:
        return None
    model.eval()
    # Tạm bật training mode cho head để nó trả cả nhánh o2m (loss cần cả o2m lẫn o2o).
    # Lưu ý: @torch.no_grad() vẫn đảm bảo không tính gradient.
    model.head.train()
    (total, n) = (0.0, 0)
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['targets']
        preds = model(images)
        (_, items) = loss_fn(preds, targets)
        total += items['loss']
        n += 1
    model.eval()
    val_loss = total / max(n, 1)
    logger.info(f'== epoch {epoch} validate: loss={val_loss:.4f} ({n} batch) ==')
    writer.add_scalar('val/loss_total', val_loss, global_step)
    return val_loss

def run_training(cfg: TrainConfig):
    _ensure_text_logging(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    if cfg.device == 'cuda' and device.type == 'cpu':
        logger.warning("cfg.device='cuda' nhưng CUDA không sẵn có - chuyển sang CPU.")
    logger.info(f'TrainConfig: {cfg}')
    (train_loader, val_loader, num_landmarks) = build_dataloaders(cfg)
    cfg.face.sync_num_landmarks(num_landmarks)
    logger.info(f'Đồng bộ FaceLmkConfig.num_landmarks = {cfg.face.num_landmarks} (dò từ dữ liệu tại {cfg.train_root_dir})')
    model = FaceLmkDetector(cfg).to(device)
    if cfg.trunk_ckpt:
        model.load_trunk(cfg.trunk_ckpt, map_location=device)
        logger.info(f'Đã nạp trunk pretrained từ {cfg.trunk_ckpt}')
    else:
        logger.warning('Không có trunk_ckpt - backbone+neck khởi tạo NGẪU NHIÊN, mất hết lợi ích transfer-learning từ NMSFreeDetector. Chỉ hợp lý nếu bạn CHỦ Ý train from-scratch cho bài toán face landmark.')
    loss_fn = FaceLandmarkDetectionLoss(cfg.face).to(device)
    freeze_now = cfg.freeze_trunk_epochs > 0
    model.freeze_trunk(freeze_now)
    if freeze_now:
        logger.info(f'Đóng băng backbone+neck trong {cfg.freeze_trunk_epochs} epoch đầu (chỉ head landmark/box học từ đầu).')
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.run_name))
    (start_epoch, global_step, best_val) = (0, 0, float('inf'))
    if cfg.resume:
        (start_epoch, global_step, best_val) = load_checkpoint(cfg.resume, model, optimizer, scaler, ema, device=device)
        start_epoch += 1
        logger.info(f'Resume từ {cfg.resume}: bắt đầu epoch {start_epoch}, global_step={global_step}, best_val={best_val:.4f}')
    for epoch in range(start_epoch, cfg.epochs):
        should_be_frozen = epoch < cfg.freeze_trunk_epochs
        if should_be_frozen != freeze_now:
            freeze_now = should_be_frozen
            model.freeze_trunk(freeze_now)
            logger.info(f"epoch {epoch}: {('đóng băng lại' if freeze_now else 'MỞ KHOÁ')} backbone+neck - build lại optimizer/scheduler.")
            optimizer = build_optimizer(cfg, model)
            scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader), epochs=cfg.epochs - epoch)
        global_step = train_one_epoch(cfg, model, loss_fn, train_loader, optimizer, scheduler, scaler, ema, device, epoch, global_step, writer)
        val_loss = None
        if val_loader is not None and (epoch + 1) % cfg.val_interval == 0:
            eval_model = ema.ema if ema is not None else model
            val_loss = validate(cfg, eval_model, loss_fn, val_loader, device, epoch, global_step, writer)
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                save_checkpoint(os.path.join(cfg.ckpt_dir, 'best.pt'), model, optimizer, scaler, ema, epoch, global_step, best_val)
                logger.info(f'epoch {epoch}: best.pt mới, val_loss={best_val:.4f}')
        save_periodic_checkpoint(cfg, model, optimizer, scaler, ema, epoch, global_step, best_val)
    writer.close()
    logger.info('Training hoàn tất.')
    return (model, ema)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', type=str, required=True)
    parser.add_argument('--val-root', type=str, default=None)
    parser.add_argument('--trunk-ckpt', type=str, default='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze-trunk-epochs', type=int, default=3)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--run-name', type=str, default='face_lmk_train')
    parser.add_argument('--lmk-margin', type=float, default=0.15, help='chạy check_lmk_margin_coverage.py trước để chọn giá trị hợp lý')
    parser.add_argument('--lmk-loss-type', type=str, default='smooth_l1', choices=['smooth_l1', 'wing'])
    parser.add_argument('--geo-gain', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    cfg = TrainConfig(face=FaceLmkConfig(lmk_margin=args.lmk_margin, lmk_loss_type=args.lmk_loss_type, geo_gain=args.geo_gain), trunk_ckpt=args.trunk_ckpt, train_root_dir=args.train_root, val_root_dir=args.val_root, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.epochs, lr=args.lr, freeze_trunk_epochs=args.freeze_trunk_epochs, resume=args.resume, run_name=args.run_name, device=args.device)
    run_training(cfg)
if __name__ == '__main__':
    main()
