from dataclasses import dataclass
from typing import Tuple, Optional
from src.config import TrainConfig

@dataclass
class FinetuneConfig(TrainConfig):
    pretrained_ckpt: str = ""

    pretrain_nc: int = 80

    labels_root: str = ""
    images_root_dir: str = ""
    index_cache_dir: str = ""
    
    nc: int = 10

    freeze_backbone: bool = True
    freeze_neck: bool = True
    freeze_epochs: int = 5

    phase2_warmup_epochs: float = 1.0

    lr0: float = 1e-4
    lr_min_factor: float = 0.01
    epochs: int = 50
    warmup_epochs: float = 1.0
    batch_size: int = 8

    backbone_lr_scale: float = 0.1

    ckpt_dir: str = "./checkpoints_finetune"
    tb_log_dir: str = "runs_finetune"
    log_dir: str = "./logs_finetune"
    run_name: str = "finetune"
    save_ckpt_interval_steps: int = 500
    val_interval_steps: int = 2000
    log_interval: int = 200
    log_loss_interval: int = 20