from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    # ---- Data: 2 GỐC ĐƯỜNG DẪN HOÀN TOÀN TÁCH BIỆT (Labels vs Images) ----
    # Theo đúng schema Object365 mà ObjectDetectionDataset (dataset.py) dùng:
    # train/val đã tách sẵn theo folder -> KHÔNG dùng val_ratio để tự chia
    # ngẫu nhiên nữa (khác bản config cũ trước đó dùng data_image_dir đơn lẻ).

    # 1) labels_root: chứa 2 thư mục con train/ và val/, mỗi thư mục có đủ bộ
    #    4 file jsonl (annotations/categories/images_info/file-map-path).
    labels_root: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/labels"
    train_subdir: str = "train"
    val_subdir: str = "val"
    images_info_filename: str = "images_info.jsonl"
    annotations_filename: str = "annotations.jsonl"
    categories_filename: str = "categories.jsonl"
    train_image_path_map_filename: str = "images_train.jsonl"
    val_image_path_map_filename: str = "images_val.jsonl"

    # 2) images_root_dir: chứa ảnh thật, CŨNG có 2 thư mục con train/ và val/
    #    (mỗi thư mục con lại chứa patch0/, patch1/, ...). Đường dẫn ảnh thật
    #    = images_root_dir / <images_train_subdir hoặc images_val_subdir> / path
    #    (path lấy từ file map ở trên).
    images_root_dir: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/images"
    images_train_subdir: str = "train"
    images_val_subdir: str = "val"

    index_cache_dir: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/images/cache"     # nơi lưu byte-offset index (pickle cache)
    rebuild_index: bool = False          # ép build lại index dù đã có cache

    skip_iscrowd: bool = True            # bỏ annotation iscrowd=1 khi train
    skip_isfake: bool = True             # bỏ annotation isfake=1 khi train
    include_images_without_annotations: bool = False  # có lấy ảnh "rỗng" (sau lọc top-80) làm sample hay không

    img_size: int = 480
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    seed: int = 28

    # ---- Data Augment ----
    horizontalFlip: float = 0.5
    shiftScaleRotate: tuple = (0.03, 0.03, 5, 0.3)   # shift_limit, scale_limit, rotate_limit, p
    randomBrightnessContrast: float = 0.15
    hueSaturationValue: tuple = (5, 8, 5, 0.1)       # hue_shift_limit, sat_shift_limit, val_shift_limit, p
    gaussNoise: tuple = (5.0, 15.0, 0.1)             # var_limit(a, b), p
    blur: tuple = (3, 0.05)                          # (blur_limit, p)
 
    # ---- Augmentation (cờ tổng - bật/tắt augment và 2 xác suất dùng riêng
    # ngoài bộ albumentations phía trên, giữ nguyên từ bản config bạn gửi) ----
    augment: bool = True
    hflip_p: float = 0.5
    color_jitter_p: float = 0.5

    # ---- Model ----
    nc: int = 80
    reg_max: int = 16
    backbone_w: Tuple = (56, 112, 224, 448, 640)
    backbone_n: Tuple = (3, 6, 6, 3)
    neck_n: int = 3
    strides: Tuple = (8, 16, 32)

    # ---- Optim ----
    epochs: int = 100
    lr0: float = 1e-3             # LR sau warmup
    lr_min_factor: float = 0.01   # LR cuối = lr0 * lr_min_factor (cosine)
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_bias_lr: float = 0.1
    momentum: float = 0.9         # dùng cho SGD, không dùng nếu optimizer=adamw
    optimizer: str = "adamw"      # "adamw" | "sgd"
    grad_clip_norm: float = 10.0
    esp: float = 1e-6             # eps cho optimizer (giữ nguyên tên như bản gốc)
    betas: tuple = (0.9, 0.98)

    # ---- Loss weights (truyền thẳng xuống DetectionLoss) ----
    cls_gain: float = 1.0
    box_gain: float = 7.5
    dfl_gain: float = 1.5
    w_o2o: float = 1.0
    w_o2m: float = 1.0
    topk_o2m: int = 10
    topk_o2o: int = 1
    alpha: float = 0.5
    beta: float = 6.0

    # ---- EMA ----
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup_updates: int = 2000

    # ---- Runtime ----
    device: str = "cuda"          # sẽ tự fallback về cpu nếu không có GPU
    amp: bool = True              # mixed precision
    scale: bool = True
    log_interval: int = 20        # số step giữa 2 lần log
    val_interval: int = 1         # số epoch giữa 2 lần validate
    ckpt_dir: str = "./checkpoints"
    resume: str = ""              # path checkpoint để resume, rỗng = train từ đầu
    save_best_only: bool = False  # False -> lưu thêm checkpoint định kỳ
    
    checkpoint_ema: str = "" # Save only 
    checkpoint: str = "" # Save checkpoint when training main model, save optimizer, scaler, ...etc
    weight_statedict_model: str = "" # Save weight parameters of only model, all function to take weight in model.py (all config model and parts)
    
    def __post_init__(self):
        assert len(self.shiftScaleRotate) == 4, \
            "shiftScaleRotate cần đúng 4 phần tử: (shift_limit, scale_limit, rotate_limit, p)"
        assert len(self.hueSaturationValue) == 4, \
            "hueSaturationValue cần đúng 4 phần tử: (hue_shift_limit, sat_shift_limit, val_shift_limit, p)"
        assert len(self.gaussNoise) == 3, \
            "gaussNoise cần đúng 3 phần tử: (var_min, var_max, p)"
        assert len(self.blur) == 2, \
            "blur cần đúng 2 phần tử: (blur_limit, p)"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.prefetch_factor is None or self.prefetch_factor >= 1, \
            "prefetch_factor phải >= 1 (hoặc None nếu num_workers=0)"

        if self.num_workers == 0:
            if self.persistent_workers:
                print("[Config][Warning] persistent_workers=True yêu cầu num_workers > 0. "
                      "Tự động đặt lại persistent_workers=False.")
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                print("[Config][Warning] prefetch_factor chỉ có tác dụng khi num_workers > 0. "
                      "Tự động đặt lại prefetch_factor=None.")
                self.prefetch_factor = None