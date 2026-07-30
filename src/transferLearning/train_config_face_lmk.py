from dataclasses import dataclass, field
from typing import Optional, Tuple
from face_lmk_config import FaceLmkConfig

@dataclass
class TrainConfig:
    face: FaceLmkConfig = field(default_factory=FaceLmkConfig)
    trunk_ckpt: str = ''
    trunk_feat_channels: Tuple[int, int, int] = (224, 448, 640)
    trunk_backbone_w: Tuple[int, int, int, int, int] = (56, 112, 224, 448, 640)
    trunk_backbone_n: Tuple[int, int, int, int] = (3, 6, 6, 3)
    trunk_neck_n: int = 3
    freeze_trunk_epochs: int = 3
    trunk_lr_mult: float = 0.1
    train_root_dir: str = ''
    val_root_dir: Optional[str] = None
    jsonl_name: str = 'annotations_all.jsonl'
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    optimizer: str = 'adamw'
    lr: float = 0.001
    weight_decay: float = 0.0005
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.937
    eps: float = 1e-08
    warmup_epochs: float = 1.0
    epochs: int = 50
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9998
    grad_clip_norm: float = 10.0
    ckpt_dir: str = './checkpoints_face_lmk'
    ckpt_keep_last: int = 3
    resume: str = ''
    log_dir: str = './logs_face_lmk'
    run_name: str = 'face_lmk_train'
    log_loss_interval: int = 50
    log_interval: int = 500
    log_gradients: bool = True
    seed: int = 42
    device: str = 'cuda'
    val_interval: int = 1

    def __post_init__(self):
        assert self.log_interval > 0, 'log_interval phải > 0'
        assert self.log_loss_interval > 0, 'log_loss_interval phải > 0'
        assert self.epochs > 0, 'epochs phải > 0'
        assert self.freeze_trunk_epochs >= 0, 'freeze_trunk_epochs phải >= 0'
        if self.freeze_trunk_epochs > self.epochs:
            raise ValueError(f'freeze_trunk_epochs ({self.freeze_trunk_epochs}) > epochs ({self.epochs}) - trunk sẽ không bao giờ được unfreeze trong lần chạy này.')
