from dataclasses import dataclass
from typing import Tuple, Optional
from src.config import TrainConfig

@dataclass
class FinetuneConfig(TrainConfig):
    pretrained_ckpt: str = "/workspace/compress2server/checkpoints/best.pt"
    pretrain_architecture: str = "/workspace/compress2server/checkpoints/model_manifest.json"
    labels_root: str = "/workspace/MSCOCO/labels"
    images_root_dir: str = "/workspace/MSCOCO/images"
    index_cache_dir: str = "/workspace/MSCOCO/images/cache"
    
    nc: int = 80
    freeze_epochs: int = 5
    warmup_epochs: float = 1.0
    
    phase2_warmup_epochs: float = 5
    epochs: int = 100
    
    lr0: float = 1e-3
    lr_min_factor: float = 0.1
    
    batch_size: int = 64
    backbone_lr_scale: float = 0.5

    use_ema: bool = True
    
    ckpt_dir: str = "./checkpoints_finetune"
    tb_log_dir: str = "runs_finetune"
    log_dir: str = "./logs_finetune"
    run_name: str = "finetune"
    save_ckpt_interval_steps: int = 1000
    val_interval_steps: int = 1804 * 5
    log_interval: int = 300
    log_loss_interval: int = 100
    log_hist_interval: int = 1000