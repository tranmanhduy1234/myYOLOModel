from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainConfig:
    # ---- Data ----
    data_dir: str = "./data"
    annotations_file: str = "annotations.jsonl"
    classes_file: str = "classes.jsonl"
    img_size: int = 640
    val_ratio: float = 0.1
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42

    # ---- Model ----
    nc: int = 80
    reg_max: int = 16
    backbone_w: Tuple[int, int, int, int, int] = (48, 96, 192, 384, 512)
    backbone_n: Tuple[int, int, int, int] = (2, 4, 4, 2)
    neck_n: int = 2
    strides: Tuple[int, int, int] = (8, 16, 32)

    # ---- Optim ----
    epochs: int = 100
    lr0: float = 1e-3            # LR sau warmup
    lr_min_factor: float = 0.01  # LR cuoi = lr0 * lr_min_factor (cosine)
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_bias_lr: float = 0.1
    momentum: float = 0.9         # dung cho SGD, khong dung neu optimizer=adamw
    optimizer: str = "adamw"      # "adamw" | "sgd"
    grad_clip_norm: float = 10.0

    # ---- Loss weights (truyen thang xuong DetectionLoss) ----
    w_cls: float = 1.0
    w_box: float = 7.5
    w_dfl: float = 1.5
    w_o2o: float = 1.0

    # ---- EMA ----
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup_updates: int = 2000

    # ---- Runtime ----
    device: str = "cuda"          # se tu fallback ve cpu neu khong co GPU
    amp: bool = True              # mixed precision
    log_interval: int = 20        # so step giua 2 lan log
    val_interval: int = 1         # so epoch giua 2 lan validate
    ckpt_dir: str = "./checkpoints"
    resume: str = ""              # path checkpoint de resume, rong = train tu dau
    save_best_only: bool = False  # False -> luu them checkpoint dinh ky

    # ---- Augmentation ----
    augment: bool = True
    hflip_p: float = 0.5
    color_jitter_p: float = 0.5