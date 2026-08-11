"""Nguồn cấu hình duy nhất cho pipeline face-landmark transfer learning.

Các module model, loss, dataloader, train, inference và demo chỉ đọc giá trị từ
file này. Các hằng thuật toán cố định (shape tensor, số tọa độ bbox, ...)
vẫn được giữ tại module sử dụng.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

# Kích thước đầu vào cố định của toàn pipeline.
PIPELINE_IMAGE_SIZE = 480
# Số điểm của MediaPipe Face Mesh kèm iris.
MEDIAPIPE_NUM_LANDMARKS = 478
# Màu RGB dùng để lấp phần trống khi letterbox/biến đổi ảnh.
DEFAULT_PADDING_COLOR = (114, 114, 114)

# Chỉ số MediaPipe thuộc từng vùng quan trọng trên khuôn mặt.
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
    # Số lớp, số bin DFL và stride của ba feature map P3/P4/P5.
    nc: int = 1
    reg_max: int = 16
    strides: Tuple[int, ...] = (8, 16, 32)
    # Số landmark và cách mã hóa offset so với tâm anchor.
    num_landmarks: Optional[int] = MEDIAPIPE_NUM_LANDMARKS
    landmark_encoding: str = 'anchor_offset_grid_v1'

    # Tỷ lệ giảm channel cho nhánh class, bbox và landmark.
    cls_channel_divisor: int = 2
    reg_channel_divisor: int = 4
    lmk_channel_divisor: int = 4
    # Giới hạn channel nhỏ nhất của head và lớn nhất của landmark head.
    head_min_channels: int = 64
    lmk_hidden_max_channels: int = 256
    # Tham số khởi tạo bias/trọng số của detection head.
    cls_prior_probability: float = 0.01
    stride_bias_expected_objects: float = 5.0
    reg_bias: float = 1.0
    lmk_weight_std: float = 0.001
    # Vị trí anchor trong mỗi ô lưới; 0.5 là tâm ô.
    anchor_offset: float = 0.5

    # Epsilon tránh chia cho 0 khi chuẩn hóa landmark/loss.
    lmk_scale_eps: float = 1.0
    loss_normalizer_eps: float = 1e-9
    # Trọng số của từng thành phần loss.
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    lmk_gain: float = 2.0
    # Ngưỡng chuyển tiếp giữa L1 và L2 trong Smooth L1.
    lmk_smooth_l1_beta: float = 0.05
    # Các landmark được tăng trọng số theo vùng quan trọng.
    eye_landmark_indices: Tuple[int, ...] = MEDIAPIPE_LEFT_EYE + MEDIAPIPE_RIGHT_EYE + MEDIAPIPE_LEFT_IRIS + MEDIAPIPE_RIGHT_IRIS
    mouth_landmark_indices: Tuple[int, ...] = MEDIAPIPE_LIPS
    nose_tip_landmark_indices: Tuple[int, ...] = MEDIAPIPE_NOSE_TIP

    # Mức ưu tiên loss cho mắt, miệng và chóp mũi.
    eye_landmark_weight: float = 3.0
    mouth_landmark_weight: float = 3.0
    nose_tip_landmark_weight: float = 4.0

    # Số anchor dương tính của nhánh one-to-many/one-to-one.
    topk_o2m: int = 10
    topk_o2o: int = 1
    # Hệ số cân bằng class score và IoU trong Task-Aligned Assigner.
    alpha: float = 0.5
    beta: float = 6.0

    # Trọng số loss cuối của hai nhánh o2m và o2o.
    o2m_weight: float = 1.0
    o2o_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.nc <= 0 or self.reg_max <= 1:
            raise ValueError('nc phải > 0 và reg_max phải > 1.')
        if not self.strides or any(stride <= 0 for stride in self.strides):
            raise ValueError('strides phải là dãy không rỗng gồm các giá trị > 0.')
        if self.num_landmarks is not None and self.num_landmarks <= 0:
            raise ValueError('num_landmarks phải là None hoặc số nguyên > 0.')
        if self.landmark_encoding != 'anchor_offset_grid_v1':
            raise ValueError(
                "landmark_encoding phải là 'anchor_offset_grid_v1' cho HEAD hiện tại."
            )
        if self.lmk_smooth_l1_beta <= 0:
            raise ValueError('lmk_smooth_l1_beta phải > 0.')
        if self.lmk_scale_eps <= 0:
            raise ValueError('lmk_scale_eps phải > 0.')
        channel_values = (
            self.cls_channel_divisor,
            self.reg_channel_divisor,
            self.lmk_channel_divisor,
            self.head_min_channels,
            self.lmk_hidden_max_channels,
        )
        if any(value <= 0 for value in channel_values):
            raise ValueError('Các tham số channel của detection head phải > 0.')
        if self.lmk_hidden_max_channels < self.head_min_channels:
            raise ValueError('lmk_hidden_max_channels phải >= head_min_channels.')
        if not 0 < self.cls_prior_probability < 1:
            raise ValueError('cls_prior_probability phải nằm trong (0, 1).')
        if min(
            self.stride_bias_expected_objects,
            self.reg_bias,
            self.lmk_weight_std,
            self.loss_normalizer_eps,
        ) <= 0:
            raise ValueError('Các tham số khởi tạo và epsilon phải > 0.')
        if not 0 <= self.anchor_offset <= 1:
            raise ValueError('anchor_offset phải nằm trong [0, 1].')
        weights = (
            self.box_gain,
            self.cls_gain,
            self.dfl_gain,
            self.lmk_gain,
            self.eye_landmark_weight,
            self.mouth_landmark_weight,
            self.nose_tip_landmark_weight,
            self.o2m_weight,
            self.o2o_weight,
        )
        if any(weight <= 0 for weight in weights):
            raise ValueError('Các gain và trọng số loss phải > 0.')
        if self.topk_o2m <= 0 or self.topk_o2o <= 0:
            raise ValueError('topk_o2m và topk_o2o phải > 0.')
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError('alpha và beta của assigner phải > 0.')

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
    """Cấu hình split đã resolve; mọi giá trị được cấp bởi TrainConfig."""

    # Vị trí dataset, thư mục ảnh và file nhãn JSONL.
    root_dir: str
    images_dir_name: str
    jsonl_name: str

    # Kích thước ảnh, batch và số tiến trình đọc dữ liệu.
    image_size: int
    batch_size: int
    num_workers: int

    # Cách lấy sample trong split.
    shuffle: bool
    drop_last: bool
    augment: bool

    # Tối ưu truyền dữ liệu CPU sang GPU.
    pin_memory: bool
    persistent_workers: bool

    # Chế độ lật ngang: off, random hoặc paired.
    horizontal_flip_mode: str

    # Cache index JSONL và tham số số học của biến đổi hình học.
    index_cache_dir: str
    padding_color: Tuple[int, int, int]
    geometry_eps: float
    bbox_perimeter_samples: int
    random_seed_upper_bound: int

    # Loại bbox quá nhỏ và dung sai cho tọa độ chuẩn hóa.
    min_box_size_px: float
    normalized_coordinate_tolerance: float

    # Kiểm tra schema nghiêm ngặt và cho phép ảnh không có face.
    strict_schema: bool
    allow_empty_targets: bool

    # Số batch mỗi worker nạp trước; None là dùng mặc định.
    prefetch_factor: Optional[int]

    # Biên độ ColorJitter.
    brightness: float
    contrast: float
    saturation: float
    hue: float

    # Xác suất lật ngang khi mode=random.
    horizontal_flip_probability: float

    # Xác suất chung và từng loại augmentation hình học.
    geometric_probability: float
    affine_probability: float
    perspective_probability: float
    radial_distortion_probability: float

    # Phạm vi scale, tịnh tiến, xoay, shear và phối cảnh.
    affine_scale_x: Tuple[float, float]
    affine_scale_y: Tuple[float, float]
    affine_translate: float
    affine_rotate_degrees: float
    affine_shear_degrees: float
    perspective_scale: float

    # Hệ số méo ống kính và độ lệch tâm méo.
    radial_k1: Tuple[float, float]
    radial_k2: Tuple[float, float]
    radial_center_jitter: float

    # Ngưỡng giữ bbox/landmark sau augmentation.
    min_face_visibility: float
    min_bbox_visibility: Optional[float]
    min_landmark_visibility: Optional[float]
    min_landmark_face_size_px: float
    require_all_landmarks_inside: bool

    def __post_init__(self) -> None:
        if not self.root_dir or not self.images_dir_name or not self.jsonl_name:
            raise ValueError('root_dir, images_dir_name và jsonl_name không được để trống.')
        if not self.index_cache_dir:
            raise ValueError('index_cache_dir không được để trống.')
        if self.image_size <= 0:
            raise ValueError(
                f"image_size phải > 0, nhận được {self.image_size}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size phải > 0, nhận được {self.batch_size}."
            )

        if self.num_workers < 0:
            raise ValueError(
                f"num_workers phải >= 0, nhận được {self.num_workers}."
            )
        if self.min_box_size_px <= 0:
            raise ValueError(
                "min_box_size_px phải > 0, "
                f"nhận được {self.min_box_size_px}."
            )
        if self.normalized_coordinate_tolerance < 0:
            raise ValueError(
                "normalized_coordinate_tolerance phải >= 0, "
                f"nhận được {self.normalized_coordinate_tolerance}."
            )
        if len(self.padding_color) != 3 or any(not 0 <= value <= 255 for value in self.padding_color):
            raise ValueError('padding_color phải gồm ba giá trị trong [0, 255].')
        if self.geometry_eps <= 0:
            raise ValueError('geometry_eps phải > 0.')
        if self.bbox_perimeter_samples < 2:
            raise ValueError('bbox_perimeter_samples phải >= 2.')
        if self.random_seed_upper_bound <= 0:
            raise ValueError('random_seed_upper_bound phải > 0.')
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError(
                "prefetch_factor phải là None hoặc số nguyên > 0, "
                f"nhận được {self.prefetch_factor}."
            )
        if self.horizontal_flip_mode not in {"off", "random", "paired"}:
            raise ValueError(
                "horizontal_flip_mode phải thuộc "
                "{'off', 'random', 'paired'}, "
                f"nhận được {self.horizontal_flip_mode!r}."
            )
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError(
                "horizontal_flip_probability phải nằm trong [0, 1], "
                f"nhận được {self.horizontal_flip_probability}."
            )
        if any(value < 0 for value in (self.brightness, self.contrast, self.saturation)):
            raise ValueError('brightness, contrast và saturation không được âm.')
        if not 0 <= self.hue <= 0.5:
            raise ValueError('hue phải nằm trong [0, 0.5].')
        probabilities = (
            self.geometric_probability,
            self.affine_probability,
            self.perspective_probability,
            self.radial_distortion_probability,
            self.min_face_visibility,
        )
        optional_probabilities = (
            self.min_bbox_visibility,
            self.min_landmark_visibility,
        )

        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError('Các xác suất geometric và min_face_visibility phải nằm trong [0, 1].')
        if any(value is not None and not 0.0 <= value <= 1.0 for value in optional_probabilities):
            raise ValueError('Các ngưỡng visibility tùy chọn phải nằm trong [0, 1].')
        if self.min_landmark_face_size_px < 0:
            raise ValueError('min_landmark_face_size_px không được âm.')
        for name, bounds in (
            ('affine_scale_x', self.affine_scale_x),
            ('affine_scale_y', self.affine_scale_y),
            ('radial_k1', self.radial_k1),
            ('radial_k2', self.radial_k2),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f'{name} phải là Tuple[min, max] hợp lệ.')
        if self.affine_scale_x[0] <= 0 or self.affine_scale_y[0] <= 0:
            raise ValueError('Scale affine phải dương.')
        if min(
            self.affine_translate,
            self.affine_rotate_degrees,
            self.affine_shear_degrees,
            self.perspective_scale,
            self.radial_center_jitter,
        ) < 0:
            raise ValueError('Biên độ geometric augmentation không được âm.')

        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError(
                "persistent_workers=True yêu cầu num_workers > 0."
            )


@dataclass(frozen=True)
class TrainingStageConfig:
    # Số epoch của stage.
    epochs: int
    # Learning rate riêng cho head và trunk.
    head_lr: float
    trunk_lr: float
    # Số epoch warmup và LR tối thiểu so với LR ban đầu.
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
    # Cấu hình model head, assigner và loss.
    face: FaceLmkConfig = field(default_factory=FaceLmkConfig)

    # Checkpoint pretrained và kiến trúc backbone/neck.
    trunk_ckpt: str = ""
    # Số channel mỗi tầng và số block lặp của backbone.
    trunk_backbone_w: Tuple[int, int, int, int, int] = (56, 112, 224, 448, 640)
    trunk_backbone_n: Tuple[int, int, int, int] = (3, 6, 6, 3)
    # Số block lặp trong neck; bật cờ để bắt buộc pretrained trunk.
    trunk_neck_n: int = 3
    require_pretrained_trunk: bool = True

    # Đường dẫn train/validation và tên thành phần dataset.
    train_root_dir: str = ""
    val_root_dir: Optional[str] = None
    images_dir_name: str = 'images'
    jsonl_name: str = 'merged_faces.jsonl'

    # Tiền xử ảnh, cache và độ ổn định số học.
    index_cache_dir: str = './cache_face_lmk_indices'
    image_size: int = PIPELINE_IMAGE_SIZE
    min_box_size_px: float = 48.0
    padding_color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR
    geometry_eps: float = 1e-6
    bbox_perimeter_samples: int = 9
    random_seed_upper_bound: int = 2**31 - 1

    # Kiểm tra dataset và cấu hình DataLoader.
    normalized_coordinate_tolerance: float = 0.0001
    strict_dataset_schema: bool = True
    allow_empty_targets: bool = False
    batch_size: int = 2
    num_workers: int = 4
    prefetch_factor: Optional[int] = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Cường độ ColorJitter khi train.
    train_brightness: float = 0.20
    train_contrast: float = 0.20
    train_saturation: float = 0.15
    train_hue: float = 0.02

    # Chế độ và xác suất lật ngang khi train.
    train_horizontal_flip_mode: str = 'paired'
    train_horizontal_flip_probability: float = 0.5

    # Xác suất chung và từng phép biến đổi hình học.
    train_geometric_probability: float = 0.65
    train_affine_probability: float = 0.45
    train_perspective_probability: float = 0.20
    train_radial_distortion_probability: float = 0.20

    # Phạm vi affine, perspective và méo ống kính khi train.
    train_affine_scale_x: Tuple[float, float] = (0.85, 1.15)
    train_affine_scale_y: Tuple[float, float] = (0.85, 1.15)
    train_affine_translate: float = 0.06
    train_affine_rotate_degrees: float = 12.0
    train_affine_shear_degrees: float = 6.0
    train_perspective_scale: float = 0.06
    train_radial_k1: Tuple[float, float] = (-0.18, 0.18)
    train_radial_k2: Tuple[float, float] = (-0.05, 0.05)

    # Độ lệch ngẫu nhiên của tâm méo ống kính.
    train_radial_center_jitter: float = 0.08
    # Ngưỡng giữ bbox/landmark sau augmentation.
    train_min_face_visibility: float = 0.60
    train_min_bbox_visibility: Optional[float] = None
    train_min_landmark_visibility: Optional[float] = None
    train_min_landmark_face_size_px: float = 0.0
    train_require_all_landmarks_inside: bool = True

    # Stage 1: đóng băng trunk, chỉ huấn luyện head.
    stage1: TrainingStageConfig = field(default_factory=lambda: TrainingStageConfig(
        epochs=5,
        head_lr=1e-3,
        trunk_lr=0.0,
        warmup_epochs=1,
        min_lr_factor=0.20
    ))

    # Stage 2: fine-tune cả trunk và head với LR trunk nhỏ hơn.
    stage2: TrainingStageConfig = field(default_factory=lambda: TrainingStageConfig(
        epochs=45,
        head_lr=3e-4,
        trunk_lr=3e-5,
        warmup_epochs=5.0,
        min_lr_factor=0.01
    ))

    # Optimizer và regularization.
    optimizer: str = 'adamw'
    weight_decay: float = 0.0005
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.937
    sgd_nesterov: bool = True
    eps: float = 1e-8
    # Mixed precision, EMA và giới hạn gradient.
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup_updates: int = 2000
    grad_clip_norm: float = 10.0

    # Thư mục checkpoint, số bản gần nhất cần giữ và file resume.
    ckpt_dir: str = './checkpoints_face_lmk'
    ckpt_keep_last: int = 3
    resume: str = ''
    # Thư mục/tên run và chu kỳ ghi loss, metric, gradient.
    log_dir: str = './logs_face_lmk'
    run_name: str = 'face_lmk_train'
    log_loss_interval: int = 50
    log_interval: int = 500
    log_gradients: bool = True
    # Seed tái lập, thiết bị train và chu kỳ validation theo epoch.
    seed: int = 42
    device: str = 'cuda'
    val_interval: int = 1

    # Ngưỡng lọc, NMS và số detection tối đa khi inference.
    inference_conf_threshold: float = 0.25
    inference_iou_threshold: float = 0.0  # o2o mặc định NMS-free
    inference_max_det: int = 100
    # Thứ tự màu của ảnh numpy đầu vào.
    numpy_input_color: str = 'rgb'
    # Kiểu vẽ bbox, landmark, text và kích thước hình hiển thị.
    inference_box_color: Tuple[int, int, int] = (0, 255, 0)
    inference_landmark_color: Tuple[int, int, int] = (255, 0, 0)
    inference_text_color: Tuple[int, int, int] = (0, 0, 0)
    inference_box_thickness: int = 2
    inference_landmark_radius: int = 1
    inference_text_scale: float = 0.5
    inference_text_thickness: int = 1
    inference_figure_size: Tuple[float, float] = (9.0, 9.0)

    # Số lượng sample/batch và kích thước plot trong demo dataloader.
    demo_batches: int = 5
    demo_max_images: int = 4
    demo_batch_size: int = 4
    demo_min_box_size_px: float = 48.0
    demo_plot_columns: int = 2
    demo_plot_cell_size: float = 7.0
    # Sample cần xem và dung sai khi kiểm tra lật hai lần.
    demo_sample_index: int = 0
    demo_flip_atol: float = 1e-6
    demo_flip_rtol: float = 0.0
    # Ảnh thử model và thông số tạo ảnh synthetic cho demo inference.
    demo_model_image_path: str = 'src/image_sketch/demo/image.png'
    demo_inference_image_shape: Tuple[int, int, int] = (
        PIPELINE_IMAGE_SIZE,
        640,
        3,
    )
    demo_inference_face_box: Tuple[int, int, int, int] = (200, 100, 440, 380)
    demo_inference_face_color: Tuple[int, int, int] = (255, 200, 180)

    @property
    def trunk_feat_channels(self) -> Tuple[int, int, int]:
        return tuple(self.trunk_backbone_w[2:5])

    @property
    def epochs(self) -> int:
        return self.stage1.epochs + self.stage2.epochs

    def stage_for_epoch(self, epoch: int) -> Tuple[str, TrainingStageConfig, int]:
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
            padding_color=self.padding_color,
            geometry_eps=self.geometry_eps,
            bbox_perimeter_samples=self.bbox_perimeter_samples,
            random_seed_upper_bound=self.random_seed_upper_bound,
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
            geometric_probability=self.train_geometric_probability if train else 0.0,
            affine_probability=self.train_affine_probability if train else 0.0,
            perspective_probability=self.train_perspective_probability if train else 0.0,
            radial_distortion_probability=self.train_radial_distortion_probability if train else 0.0,
            affine_scale_x=self.train_affine_scale_x,
            affine_scale_y=self.train_affine_scale_y,
            affine_translate=self.train_affine_translate,
            affine_rotate_degrees=self.train_affine_rotate_degrees,
            affine_shear_degrees=self.train_affine_shear_degrees,
            perspective_scale=self.train_perspective_scale,
            radial_k1=self.train_radial_k1,
            radial_k2=self.train_radial_k2,
            radial_center_jitter=self.train_radial_center_jitter,
            min_face_visibility=self.train_min_face_visibility,
            min_bbox_visibility=self.train_min_bbox_visibility,
            min_landmark_visibility=self.train_min_landmark_visibility,
            min_landmark_face_size_px=self.train_min_landmark_face_size_px,
            require_all_landmarks_inside=self.train_require_all_landmarks_inside,
        )

    def checkpoint_model_signature(self) -> dict:
        return {
            'format_version': 3,
            'landmark_encoding': self.face.landmark_encoding,
            'num_landmarks': self.face.require_num_landmarks(),
            'nc': self.face.nc,
            'reg_max': self.face.reg_max,
            'strides': tuple(self.face.strides),
            'cls_channel_divisor': self.face.cls_channel_divisor,
            'reg_channel_divisor': self.face.reg_channel_divisor,
            'lmk_channel_divisor': self.face.lmk_channel_divisor,
            'head_min_channels': self.face.head_min_channels,
            'lmk_hidden_max_channels': self.face.lmk_hidden_max_channels,
            'anchor_offset': self.face.anchor_offset,
            'trunk_backbone_w': tuple(self.trunk_backbone_w),
            'trunk_backbone_n': tuple(self.trunk_backbone_n),
            'trunk_neck_n': self.trunk_neck_n,
        }

    def checkpoint_training_plan(self) -> dict:
        return {
            'stage1': asdict(self.stage1),
            'stage2': asdict(self.stage2),
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'betas': tuple(self.betas),
            'momentum': self.momentum,
            'sgd_nesterov': self.sgd_nesterov,
            'eps': self.eps,
            'amp': self.amp,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
            'ema_warmup_updates': self.ema_warmup_updates,
            'grad_clip_norm': self.grad_clip_norm,
            'face_loss': {
                'box_gain': self.face.box_gain,
                'cls_gain': self.face.cls_gain,
                'dfl_gain': self.face.dfl_gain,
                'lmk_gain': self.face.lmk_gain,
                'lmk_smooth_l1_beta': self.face.lmk_smooth_l1_beta,
                'lmk_scale_eps': self.face.lmk_scale_eps,
                'topk_o2m': self.face.topk_o2m,
                'topk_o2o': self.face.topk_o2o,
                'alpha': self.face.alpha,
                'beta': self.face.beta,
                'o2m_weight': self.face.o2m_weight,
                'o2o_weight': self.face.o2o_weight,
                'eye_landmark_weight': self.face.eye_landmark_weight,
                'mouth_landmark_weight': self.face.mouth_landmark_weight,
                'nose_tip_landmark_weight': self.face.nose_tip_landmark_weight,
            },
        }

    def __post_init__(self) -> None:
        self.optimizer = self.optimizer.lower()
        if self.image_size != PIPELINE_IMAGE_SIZE:
            raise ValueError(
                f'Pipeline này được chuẩn hoá cố định ở '
                f'image_size={PIPELINE_IMAGE_SIZE}.'
            )
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
        if not self.images_dir_name or not self.jsonl_name:
            raise ValueError('images_dir_name và jsonl_name không được để trống.')
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
        if not (
            self.device == 'cpu'
            or self.device == 'cuda'
            or self.device.startswith('cuda:')
        ):
            raise ValueError("device phải là 'cpu', 'cuda' hoặc 'cuda:<index>'.")
        if self.train_horizontal_flip_mode not in {'off', 'random', 'paired'}:
            raise ValueError("train_horizontal_flip_mode phải là 'off', 'random' hoặc 'paired'.")
        if not 0 <= self.train_horizontal_flip_probability <= 1:
            raise ValueError('train_horizontal_flip_probability phải nằm trong [0, 1].')
        if self.inference_max_det <= 0:
            raise ValueError('inference_max_det phải > 0.')
        if not 0 <= self.inference_conf_threshold <= 1 or not 0 <= self.inference_iou_threshold <= 1:
            raise ValueError('Các threshold inference phải nằm trong [0, 1].')
        colors = (
            self.padding_color,
            self.inference_box_color,
            self.inference_landmark_color,
            self.inference_text_color,
            self.demo_inference_face_color,
        )
        if any(len(color) != 3 or any(not 0 <= value <= 255 for value in color) for color in colors):
            raise ValueError('Các cấu hình màu phải là tuple RGB gồm ba giá trị trong [0, 255].')
        if min(
            self.inference_box_thickness,
            self.inference_landmark_radius,
            self.inference_text_scale,
            self.inference_text_thickness,
            *self.inference_figure_size,
            self.demo_batches,
            self.demo_max_images,
            self.demo_batch_size,
            self.demo_min_box_size_px,
            self.demo_plot_columns,
            self.demo_plot_cell_size,
            self.demo_flip_atol,
        ) <= 0:
            raise ValueError('Các tham số hiển thị và demo phải > 0.')
        if self.demo_sample_index < 0 or self.demo_flip_rtol < 0:
            raise ValueError('demo_sample_index và demo_flip_rtol không được âm.')
        if len(self.demo_inference_image_shape) != 3 or any(value <= 0 for value in self.demo_inference_image_shape):
            raise ValueError('demo_inference_image_shape phải là tuple (H, W, C) dương.')
        if self.stage1.trunk_lr != 0:
            raise ValueError('stage1.trunk_lr phải bằng 0 vì backbone+neck bị đóng băng hoàn toàn.')
        if not 0 < self.stage2.trunk_lr < self.stage2.head_lr:
            raise ValueError('Stage 2 yêu cầu 0 < trunk_lr < head_lr.')
        # Dùng chính DatasetConfig để không lặp lại validation của data pipeline.
        self.dataset_config('__config_validation__', train=True)
