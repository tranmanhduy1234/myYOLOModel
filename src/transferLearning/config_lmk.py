"""Nguồn cấu hình duy nhất cho pipeline face landmark transfer learning."""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


# MediaPipe Face Mesh 478: các vùng quan trọng cho bài toán mắt/miệng/mũi.
MEDIAPIPE_LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
MEDIAPIPE_RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
MEDIAPIPE_LEFT_IRIS = (468, 469, 470, 471, 472)
MEDIAPIPE_RIGHT_IRIS = (473, 474, 475, 476, 477)
MEDIAPIPE_LIPS = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)
MEDIAPIPE_NOSE_TIP = (1,)


@dataclass
class FaceLmkConfig:
    """Cấu hình model, assigner và loss; không chứa cấu hình dữ liệu."""

    nc: int = 1
    reg_max: int = 16
    strides: Tuple[int, ...] = (8, 16, 32)
    num_landmarks: Optional[int] = None
    lmk_margin: float = 0.05
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    lmk_gain: float = 2.0
    lmk_smooth_l1_beta: float = 0.05
    eye_landmark_indices: Tuple[int, ...] = MEDIAPIPE_LEFT_EYE + MEDIAPIPE_RIGHT_EYE + MEDIAPIPE_LEFT_IRIS + MEDIAPIPE_RIGHT_IRIS
    mouth_landmark_indices: Tuple[int, ...] = MEDIAPIPE_LIPS
    nose_tip_landmark_indices: Tuple[int, ...] = MEDIAPIPE_NOSE_TIP
    eye_landmark_weight: float = 3.0
    mouth_landmark_weight: float = 3.0
    nose_tip_landmark_weight: float = 4.0
    topk_o2m: int = 10
    topk_o2o: int = 1
    alpha: float = 0.5
    beta: float = 6.0
    o2m_weight: float = 1.0
    o2o_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.nc <= 0 or self.reg_max <= 1:
            raise ValueError('nc phải > 0 và reg_max phải > 1.')
        if not 0.0 <= self.lmk_margin <= 0.5:
            raise ValueError('lmk_margin phải nằm trong [0, 0.5].')
        weights = (self.lmk_gain, self.eye_landmark_weight, self.mouth_landmark_weight, self.nose_tip_landmark_weight)
        if any(weight <= 0 for weight in weights):
            raise ValueError('Các trọng số landmark phải > 0.')

    def sync_num_landmarks(self, dataset_num_landmarks: int) -> None:
        if self.num_landmarks is None:
            self.num_landmarks = dataset_num_landmarks
        elif self.num_landmarks != dataset_num_landmarks:
            raise ValueError(
                f'num_landmarks model ({self.num_landmarks}) khác dataset ({dataset_num_landmarks}).'
            )

    def require_num_landmarks(self) -> int:
        if self.num_landmarks is None:
            raise ValueError('Phải sync num_landmarks từ dataset trước khi tạo model/loss.')
        return self.num_landmarks


@dataclass(frozen=True)
class DatasetConfig:
    """Cấu hình một split, luôn được tạo từ :class:`TrainConfig`."""

    root_dir: str
    images_dir_name: str
    jsonl_name: str
    index_cache_dir: str
    image_size: int
    min_box_size_px: float
    normalized_coordinate_tolerance: float
    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    augment: bool
    strict_schema: bool
    allow_empty_targets: bool
    prefetch_factor: Optional[int]
    pin_memory: bool
    persistent_workers: bool
    brightness: float
    contrast: float
    saturation: float
    hue: float
    horizontal_flip_mode: str
    horizontal_flip_probability: float


@dataclass
class MarginCoverageConfig:
    root_dir: str = ''
    jsonl_name: str = 'merged_faces.jsonl'
    index_cache_dir: str = './cache_face_lmk_indices'
    sample_size: int = 5000
    margins: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.1, 0.15)
    seed: int = 0


@dataclass(frozen=True)
class TrainingStageConfig:
    """Learning-rate policy của một giai đoạn fine-tuning."""

    epochs: int
    head_lr: float
    trunk_lr: float
    warmup_epochs: float
    min_lr_factor: float

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError('Mỗi training stage phải có ít nhất 1 epoch.')
        if self.head_lr <= 0 or self.trunk_lr < 0:
            raise ValueError('head_lr phải > 0 và trunk_lr phải >= 0.')
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError('warmup_epochs phải nằm trong [0, epochs).')
        if not 0 < self.min_lr_factor <= 1:
            raise ValueError('min_lr_factor phải nằm trong (0, 1].')


@dataclass
class TrainConfig:
    """Cấu hình cấp cao duy nhất dùng bởi train, model, dataset và inference."""

    face: FaceLmkConfig = field(default_factory=FaceLmkConfig)

    # Trunk (feature channels được suy ra từ backbone_w[2:5], không khai báo lặp).
    trunk_ckpt: str = './checkpoints/best_trunk.pt'
    trunk_backbone_w: Tuple[int, int, int, int, int] = (56, 112, 224, 448, 640)
    trunk_backbone_n: Tuple[int, int, int, int] = (3, 6, 6, 3)
    trunk_neck_n: int = 3
    require_pretrained_trunk: bool = True

    # Hiện dùng toàn bộ tập chưa chia làm train; khi chia xong chỉ cần đổi hai root này.
    # Mỗi root chứa images/ và merged_faces.jsonl với cùng một cấu trúc.
    train_root_dir: str = '/run/media/tranmanhduy/Data/DataTransfer'
    val_root_dir: Optional[str] = None
    images_dir_name: str = 'images'
    jsonl_name: str = 'merged_faces.jsonl'
    # Cache byte-offset nằm ngoài dataset để mount dữ liệu có thể giữ read-only.
    index_cache_dir: str = './cache_face_lmk_indices'
    image_size: int = 480
    min_box_size_px: float = 2.0
    normalized_coordinate_tolerance: float = 0.0
    strict_dataset_schema: bool = True
    allow_empty_targets: bool = False
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: Optional[int] = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Chỉ photometric augmentation; không làm lệch bbox/landmark.
    train_brightness: float = 0.20
    train_contrast: float = 0.20
    train_saturation: float = 0.15
    train_hue: float = 0.02
    # 'paired' tạo cả ảnh gốc và ảnh lật trong mỗi epoch -> cân bằng trái/phải chính xác.
    # 'random' không tăng độ dài dataset nhưng chỉ cân bằng theo kỳ vọng; 'off' để tắt.
    train_horizontal_flip_mode: str = 'paired'
    train_horizontal_flip_probability: float = 0.5

    # Hai giai đoạn transfer learning. Optimizer được giữ nguyên giữa hai stage.
    # Stage 1: trunk frozen hoàn toàn, chỉ head có LR.
    stage1: TrainingStageConfig = field(default_factory=lambda: TrainingStageConfig(
        epochs=5,
        head_lr=1e-3,
        trunk_lr=0.0,
        warmup_epochs=0.5,
        min_lr_factor=0.20,
    ))
    # Stage 2: fine-tune toàn mạng; trunk LR mặc định bằng 1/10 head LR.
    stage2: TrainingStageConfig = field(default_factory=lambda: TrainingStageConfig(
        epochs=45,
        head_lr=3e-4,
        trunk_lr=3e-5,
        warmup_epochs=1.0,
        min_lr_factor=0.01,
    ))

    # Optimizer dùng chung.
    optimizer: str = 'adamw'
    weight_decay: float = 0.0005
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.937
    eps: float = 1e-8
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup_updates: int = 2000
    grad_clip_norm: float = 10.0

    # Checkpoint / logging.
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

    # Inference.
    inference_conf_threshold: float = 0.25
    inference_iou_threshold: float = 0.0  # o2o mặc định NMS-free
    inference_max_det: int = 100
    numpy_input_color: str = 'rgb'

    @property
    def trunk_feat_channels(self) -> Tuple[int, int, int]:
        return tuple(self.trunk_backbone_w[2:5])

    @property
    def epochs(self) -> int:
        return self.stage1.epochs + self.stage2.epochs

    def stage_for_epoch(self, epoch: int) -> Tuple[str, TrainingStageConfig, int]:
        """Trả về (tên stage, config stage, epoch cục bộ)."""
        if not 0 <= epoch < self.epochs:
            raise ValueError(f'epoch {epoch} nằm ngoài [0, {self.epochs}).')
        if epoch < self.stage1.epochs:
            return 'stage1_head_only', self.stage1, epoch
        return 'stage2_finetune', self.stage2, epoch - self.stage1.epochs

    def dataset_config(self, root_dir: str, *, train: bool) -> DatasetConfig:
        if not root_dir:
            raise ValueError('root_dir của split không được để trống.')
        workers_enabled = self.num_workers > 0
        return DatasetConfig(
            root_dir=root_dir,
            images_dir_name=self.images_dir_name,
            jsonl_name=self.jsonl_name,
            index_cache_dir=self.index_cache_dir,
            image_size=self.image_size,
            min_box_size_px=self.min_box_size_px,
            normalized_coordinate_tolerance=self.normalized_coordinate_tolerance,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=train,
            drop_last=train,
            augment=train,
            strict_schema=self.strict_dataset_schema,
            allow_empty_targets=self.allow_empty_targets,
            prefetch_factor=self.prefetch_factor if workers_enabled else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and workers_enabled,
            brightness=self.train_brightness if train else 0.0,
            contrast=self.train_contrast if train else 0.0,
            saturation=self.train_saturation if train else 0.0,
            hue=self.train_hue if train else 0.0,
            horizontal_flip_mode=self.train_horizontal_flip_mode if train else 'off',
            horizontal_flip_probability=self.train_horizontal_flip_probability if train else 0.0,
        )

    def checkpoint_training_plan(self) -> dict:
        """Các tham số phải giống nhau để resume giữ đúng ý nghĩa training."""
        return {
            'stage1': asdict(self.stage1),
            'stage2': asdict(self.stage2),
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'betas': tuple(self.betas),
            'momentum': self.momentum,
            'eps': self.eps,
            'amp': self.amp,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
            'ema_warmup_updates': self.ema_warmup_updates,
            'grad_clip_norm': self.grad_clip_norm,
        }

    def __post_init__(self) -> None:
        self.optimizer = self.optimizer.lower()
        if self.image_size != 480:
            raise ValueError('Pipeline này được chuẩn hoá cố định ở image_size=480.')
        if len(self.face.strides) != 3 or len(self.trunk_backbone_w) < 5:
            raise ValueError('Trunk phải xuất đúng ba feature map P3/P4/P5.')
        if self.num_workers < 0 or self.batch_size <= 0:
            raise ValueError('num_workers phải >= 0 và batch_size phải > 0.')
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError('prefetch_factor phải >= 1 hoặc None.')
        if self.min_box_size_px <= 0:
            raise ValueError('min_box_size_px phải > 0.')
        if not 0 <= self.normalized_coordinate_tolerance <= 0.5:
            raise ValueError('normalized_coordinate_tolerance phải nằm trong [0, 0.5].')
        if not self.index_cache_dir:
            raise ValueError('index_cache_dir không được để trống.')
        if self.optimizer.lower() not in {'adamw', 'sgd'}:
            raise ValueError("optimizer phải là 'adamw' hoặc 'sgd'.")
        if self.weight_decay < 0 or self.eps <= 0:
            raise ValueError('weight_decay phải >= 0 và eps phải > 0.')
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError('betas phải gồm hai giá trị trong [0, 1).')
        if not 0 <= self.momentum < 1:
            raise ValueError('momentum phải nằm trong [0, 1).')
        if not 0 < self.ema_decay < 1 or self.ema_warmup_updates <= 0:
            raise ValueError('ema_decay phải trong (0, 1), ema_warmup_updates phải > 0.')
        if self.grad_clip_norm <= 0:
            raise ValueError('grad_clip_norm phải > 0.')
        if min(self.ckpt_keep_last, self.log_loss_interval, self.log_interval, self.val_interval) <= 0:
            raise ValueError('Các interval và ckpt_keep_last phải > 0.')
        if any(value < 0 for value in (
            self.train_brightness, self.train_contrast, self.train_saturation,
        )) or not 0 <= self.train_hue <= 0.5:
            raise ValueError('ColorJitter phải không âm và hue phải nằm trong [0, 0.5].')
        if self.numpy_input_color.lower() not in {'rgb', 'bgr'}:
            raise ValueError("numpy_input_color phải là 'rgb' hoặc 'bgr'.")
        if self.train_horizontal_flip_mode not in {'off', 'random', 'paired'}:
            raise ValueError("train_horizontal_flip_mode phải là 'off', 'random' hoặc 'paired'.")
        if not 0 <= self.train_horizontal_flip_probability <= 1:
            raise ValueError('train_horizontal_flip_probability phải nằm trong [0, 1].')
        if self.inference_max_det <= 0:
            raise ValueError('inference_max_det phải > 0.')
        if not 0 <= self.inference_conf_threshold <= 1 or not 0 <= self.inference_iou_threshold <= 1:
            raise ValueError('Các threshold inference phải nằm trong [0, 1].')
        if self.stage1.trunk_lr != 0:
            raise ValueError('stage1.trunk_lr phải bằng 0 vì backbone+neck bị đóng băng hoàn toàn.')
        if not 0 < self.stage2.trunk_lr < self.stage2.head_lr:
            raise ValueError('Stage 2 yêu cầu 0 < trunk_lr < head_lr.')
