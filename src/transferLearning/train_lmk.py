import logging
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.transferLearning.config_lmk import TrainConfig
from src.transferLearning.dataloader_lmk import FaceLandmarkDataModule
from src.transferLearning.loss_lmk import FaceLandmarkDetectionLoss
from src.transferLearning.model_lmk import FaceLmkDetector

logger = logging.getLogger('train_face_lmk')

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelEMA:

    def __init__(self, model: nn.Module, decay: float, warmup_updates: int):
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
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
                else:
                    v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

class CheckpointManager:
    """Đóng gói toàn bộ logic save/load checkpoint theo 1 TrainConfig."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def save(self, path, model, optimizer, scaler, ema: Optional[ModelEMA], epoch, global_step, best_val):
        stage_name = self.cfg.stage_for_epoch(epoch)[0]
        torch.save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'ema': ema.state_dict() if ema is not None else None,
            'epoch': epoch, 'global_step': global_step, 'best_val': best_val,
            'stage': stage_name,
            'ema_updates': ema.updates if ema is not None else 0,
            'training_plan': self.cfg.checkpoint_training_plan(),
            'model_signature': self.cfg.checkpoint_model_signature(),
        }, path)

    def save_periodic(self, model, optimizer, scaler, ema, epoch, global_step, best_val):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self.save(os.path.join(self.cfg.ckpt_dir, f'ckpt_step{global_step:08d}.pt'), model,
                  optimizer, scaler, ema, epoch, global_step, best_val)
        self.save(os.path.join(self.cfg.ckpt_dir, 'last.pt'), model, optimizer,
                  scaler, ema, epoch, global_step, best_val)
        step_files = sorted(f for f in os.listdir(self.cfg.ckpt_dir) if f.startswith('ckpt_step') and f.endswith('.pt'))
        while len(step_files) > self.cfg.ckpt_keep_last:
            os.remove(os.path.join(self.cfg.ckpt_dir, step_files.pop(0)))

    def save_best(self, model, optimizer, scaler, ema, epoch, global_step, best_val):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self.save(os.path.join(self.cfg.ckpt_dir, 'best.pt'), model, optimizer, scaler, ema, epoch, global_step, best_val)

    def save_stage_final(self, stage_name, model, optimizer, scaler, ema, epoch, global_step, best_val):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self.save(
            os.path.join(self.cfg.ckpt_dir, f'{stage_name}_final.pt'),
            model, optimizer, scaler, ema, epoch, global_step, best_val,
        )

    def load(self, path, model, optimizer=None, scaler=None, ema: Optional[ModelEMA] = None, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        expected_signature = self.cfg.checkpoint_model_signature()
        saved_signature = ckpt.get('model_signature')
        if saved_signature is None:
            raise ValueError(
                'Checkpoint không có model_signature. Đây nhiều khả năng là checkpoint '
                'landmark bbox-relative cũ và không được phép resume vào HEAD anchor-relative.'
            )
        if saved_signature != expected_signature:
            raise ValueError(
                f'Model signature trong checkpoint khác cấu hình hiện tại: '
                f'checkpoint={saved_signature}, current={expected_signature}.'
            )

        expected_plan = self.cfg.checkpoint_training_plan()
        saved_plan = ckpt.get('training_plan')
        if saved_plan is not None and saved_plan != expected_plan:
            raise ValueError(
                f'Training plan trong checkpoint khác config hiện tại: '
                f'checkpoint={saved_plan}, current={expected_plan}.'
            )
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        if ema is not None and ckpt.get('ema') is not None:
            ema.ema.load_state_dict(ckpt['ema'])
            ema.updates = int(ckpt.get('ema_updates', 0))
        return ckpt.get('epoch', 0), ckpt.get('global_step', 0), ckpt.get('best_val', float('inf'))

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._setup_logging()
        set_seed(cfg.seed)

        self.device = torch.device(cfg.device if cfg.device == 'cpu' or torch.cuda.is_available() else 'cpu')
        if cfg.device == 'cuda' and self.device.type == 'cpu':
            logger.warning("cfg.device='cuda' nhưng CUDA không sẵn có - chuyển sang CPU.")

        self.train_dm = FaceLandmarkDataModule(cfg.dataset_config(cfg.train_root_dir, train=True))
        self.val_dm = None
        if cfg.val_root_dir:
            self.val_dm = FaceLandmarkDataModule(cfg.dataset_config(cfg.val_root_dir, train=False))
            if self.val_dm.num_landmarks != self.train_dm.num_landmarks:
                raise ValueError(f'Số landmark val ({self.val_dm.num_landmarks}) khác train ({self.train_dm.num_landmarks}).')

        cfg.face.sync_num_landmarks(self.train_dm.num_landmarks)
        logger.info(f'Đồng bộ FaceLmkConfig.num_landmarks = {cfg.face.num_landmarks}')

        self.model = FaceLmkDetector(cfg).to(self.device)
        if cfg.trunk_ckpt:
            self.model.load_trunk(cfg.trunk_ckpt, map_location=self.device)
            logger.info(f'Đã nạp trunk pretrained từ {cfg.trunk_ckpt}')
        elif cfg.require_pretrained_trunk:
            raise ValueError('Transfer learning hai giai đoạn yêu cầu trunk_ckpt pretrained.')
        else:
            logger.warning('Không có trunk_ckpt - backbone+neck khởi tạo ngẫu nhiên (mất transfer-learning).')

        self.loss_fn = FaceLandmarkDetectionLoss(cfg.face).to(self.device)
        self.ckpt = CheckpointManager(cfg)

        self.train_loader = self.train_dm.loader()
        self.val_loader = self.val_dm.loader() if self.val_dm is not None else None
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and self.device.type == 'cuda')
        self.ema = ModelEMA(
            self.model, decay=cfg.ema_decay, warmup_updates=cfg.ema_warmup_updates
        ) if cfg.use_ema else None

        os.makedirs(cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.run_name))

        self.start_epoch, self.global_step, self.best_val = 0, 0, float('inf')
        if cfg.resume:
            self.start_epoch, self.global_step, self.best_val = self.ckpt.load(
                cfg.resume, self.model, self.optimizer, self.scaler, self.ema, device=self.device)
            self.start_epoch += 1
            logger.info(f'Resume từ {cfg.resume}: epoch {self.start_epoch}, global_step={self.global_step}, best_val={self.best_val:.4f}')
        self._active_stage_name = None
        self._trunk_frozen = False
        if self.start_epoch < cfg.epochs:
            self._configure_stage(self.start_epoch, force=True)

    def _setup_logging(self):
        if logger.handlers:
            return
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_h = logging.FileHandler(os.path.join(self.cfg.log_dir, f'{self.cfg.run_name}.log'), encoding='utf-8')
        file_h.setFormatter(fmt)
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(fmt)
        logger.addHandler(file_h)
        logger.addHandler(stream_h)
        logger.setLevel(logging.INFO)

    def _build_optimizer(self):
        cfg = self.cfg
        # Luôn giữ đủ hai param group để optimizer state của head không bị mất khi chuyển stage.
        head_params = list(self.model.head.parameters())
        trunk_params = list(self.model.backbone.parameters()) + list(self.model.neck.parameters())
        param_groups = [
            {'params': head_params, 'lr': cfg.stage1.head_lr, 'name': 'head'},
            {'params': trunk_params, 'lr': 0.0, 'name': 'trunk'},
        ]
        if cfg.optimizer == 'adamw':
            return torch.optim.AdamW(param_groups, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        if cfg.optimizer == 'sgd':
            return torch.optim.SGD(
                param_groups,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.sgd_nesterov,
            )
        raise ValueError(f'optimizer không hỗ trợ: {cfg.optimizer}')

    def _configure_stage(self, epoch: int, force: bool = False) -> None:
        stage_name, stage_cfg, local_epoch = self.cfg.stage_for_epoch(epoch)
        if not force and stage_name == self._active_stage_name:
            return
        freeze_trunk = stage_name == 'stage1_head_only'
        self.model.freeze_trunk(freeze_trunk)
        self._trunk_frozen = freeze_trunk
        self._active_stage_name = stage_name
        trunk_trainable = sum(p.numel() for group in (self.model.backbone, self.model.neck)
                              for p in group.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.model.head.parameters() if p.requires_grad)
        logger.info(
            f'Bắt đầu {stage_name}: local_epoch={local_epoch}, epochs={stage_cfg.epochs}, '
            f'freeze_trunk={freeze_trunk}, head_lr={stage_cfg.head_lr:.2e}, '
            f'trunk_lr={stage_cfg.trunk_lr:.2e}, trainable head={head_trainable:,}, '
            f'trunk={trunk_trainable:,}.'
        )

    def _set_stage_learning_rates(self, epoch: int, step: int) -> tuple:
        _, stage_cfg, local_epoch = self.cfg.stage_for_epoch(epoch)
        steps_per_epoch = len(self.train_loader)
        local_step = local_epoch * steps_per_epoch + step
        total_steps = max(stage_cfg.epochs * steps_per_epoch, 1)
        warmup_steps = int(stage_cfg.warmup_epochs * steps_per_epoch)
        if warmup_steps > 0 and local_step < warmup_steps:
            factor = (local_step + 1) / warmup_steps
        else:
            decay_step = max(local_step - warmup_steps, 0)
            decay_steps = max(total_steps - warmup_steps - 1, 1)
            progress = min(decay_step / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = stage_cfg.min_lr_factor + (1.0 - stage_cfg.min_lr_factor) * cosine

        lr_head = stage_cfg.head_lr * factor
        lr_trunk = stage_cfg.trunk_lr * factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr_head if group['name'] == 'head' else lr_trunk
        return lr_head, lr_trunk

    def _train_one_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        self.model.train()
        t_epoch = time.time()
        running_loss, n_steps_done = 0.0, 0
        augmentation_counts = {}
        use_amp = cfg.amp and self.device.type == 'cuda'
        total_norm = 0.0

        train_progress = tqdm(
            self.train_loader,
            desc=f'Train epoch {epoch + 1}/{cfg.epochs}',
            unit='batch',
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_progress):
            lr_head, lr_trunk = self._set_stage_learning_rates(epoch, step)
            images = batch['image'].to(self.device, non_blocking=True)
            for aug_name in batch.get('geometric_aug', []):
                augmentation_counts[aug_name] = augmentation_counts.get(aug_name, 0) + 1
            targets = batch['targets']
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                preds = self.model(images)
                total_loss, items = self.loss_fn(preds, targets)
            if not torch.isfinite(total_loss):
                logger.warning(f'[epoch {epoch} step {step}] loss không hữu hạn ({total_loss.item()}) - bỏ qua step.')
                continue

            optimizer_step_succeeded = True
            if self.scaler.is_enabled():
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm).item()
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_succeeded = self.scaler.get_scale() >= prev_scale
            else:
                total_loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm).item()
                self.optimizer.step()

            if optimizer_step_succeeded:
                if self.ema is not None:
                    self.ema.update(self.model)

            self.global_step += 1
            running_loss += items['loss']
            n_steps_done += 1
            train_progress.set_postfix(
                loss=f"{items['loss']:.4f}",
                lr_head=f'{lr_head:.2e}',
                lr_trunk=f'{lr_trunk:.2e}',
            )

            if self.global_step % cfg.log_loss_interval == 0:
                logger.info(f"epoch {epoch} step {step}/{len(self.train_loader)} (global {self.global_step}) "
                            f"stage={self._active_stage_name} loss={items['loss']:.4f} "
                            f"lr_head={lr_head:.2e} lr_trunk={lr_trunk:.2e}")
                self.writer.add_scalar('train/loss_total', items['loss'], self.global_step)
                self.writer.add_scalar('train/lr_head', lr_head, self.global_step)
                self.writer.add_scalar('train/lr_trunk', lr_trunk, self.global_step)
            if self.global_step % cfg.log_interval == 0:
                self.writer.add_scalars('train/loss_o2m', {'iou': items['o2m/iou'], 'cls': items['o2m/cls'], 'dfl': items['o2m/dfl'], 'lmk': items['o2m/lmk']}, self.global_step)
                self.writer.add_scalars('train/loss_o2o', {'iou': items['o2o/iou'], 'cls': items['o2o/cls'], 'dfl': items['o2o/dfl'], 'lmk': items['o2o/lmk']}, self.global_step)
                self.writer.add_scalars('train/n_pos', {'o2m': items['o2m/n_pos'], 'o2o': items['o2o/n_pos'], 'o2m_lmk': items['o2m/n_lmk_pos'], 'o2o_lmk': items['o2o/n_lmk_pos']}, self.global_step)
                if cfg.log_gradients:
                    self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                if self.device.type == 'cuda':
                    self.writer.add_scalar('train/gpu_mem_gb', torch.cuda.max_memory_allocated() / 1e9, self.global_step)

        avg_loss = running_loss / max(n_steps_done, 1)
        logger.info(
            f'== epoch {epoch} xong: loss trung bình={avg_loss:.4f}, '
            f'{n_steps_done} step, {time.time() - t_epoch:.1f}s | '
            f'geometric_aug={augmentation_counts} =='
        )

    @torch.no_grad()
    def _validate(self, epoch: int) -> Optional[float]:
        if self.val_dm is None:
            return None
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        total, n = 0.0, 0
        val_progress = tqdm(
            self.val_loader,
            desc=f'Validate epoch {epoch + 1}/{self.cfg.epochs}',
            unit='batch',
            dynamic_ncols=True,
        )
        for batch in val_progress:
            images = batch['image'].to(self.device, non_blocking=True)
            preds = eval_model(images, return_o2m=True)
            _, items = self.loss_fn(preds, batch['targets'])
            total += items['loss']
            n += 1
            val_progress.set_postfix(val_loss=f'{total / n:.4f}')
        val_loss = total / max(n, 1)
        logger.info(f'== epoch {epoch} validate: loss={val_loss:.4f} ({n} batch) ==')
        self.writer.add_scalar('val/loss_total', val_loss, self.global_step)
        return val_loss

    def fit(self):
        cfg = self.cfg
        logger.info(f'TrainConfig: {cfg}')

        for epoch in range(self.start_epoch, cfg.epochs):
            self._configure_stage(epoch)
            self._train_one_epoch(epoch)

            if self.val_dm is not None and (epoch + 1) % cfg.val_interval == 0:
                val_loss = self._validate(epoch)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.ckpt.save_best(self.model, self.optimizer, self.scaler, self.ema, epoch, self.global_step, self.best_val)
                    logger.info(f'epoch {epoch}: best.pt mới, val_loss={self.best_val:.4f}')
            self.ckpt.save_periodic(self.model, self.optimizer, self.scaler, self.ema, epoch, self.global_step, self.best_val)
            if epoch + 1 == cfg.stage1.epochs:
                self.ckpt.save_stage_final(
                    'stage1_head_only', self.model, self.optimizer, self.scaler,
                    self.ema, epoch, self.global_step, self.best_val,
                )
            if epoch + 1 == cfg.epochs:
                self.ckpt.save_stage_final(
                    'stage2_finetune', self.model, self.optimizer, self.scaler,
                    self.ema, epoch, self.global_step, self.best_val,
                )

        self.writer.close()
        logger.info('Training hoàn tất.')
        return self.model, self.ema

if __name__ == '__main__':
    Trainer(TrainConfig()).fit()
