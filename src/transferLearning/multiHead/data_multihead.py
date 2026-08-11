"""Dataloader va hinh hoc crop dung chung cho multi-head landmark.

Moi record anh tao bon input:

* ``full_face``: anh RGB letterbox ve 480x480 cho HEAD 4.
* ``left_eye`` / ``right_eye``: crop truc tiep tu anh goc ve 128x128.
* ``mouth``: crop truc tiep tu anh goc ve 160x160.

Ma tran trong :class:`RegionCropTransform` la nguon su that duy nhat de bien
doi toa do. Inference phai goi lai :func:`build_region_crop` va dung
``map_points_to_original`` thay vi tu tinh scale/padding lan nua.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transferLearning.config_lmk import (
    DEFAULT_PADDING_COLOR,
    MEDIAPIPE_NUM_LANDMARKS,
    PIPELINE_IMAGE_SIZE,
)
from src.transferLearning.multiHead.model_multihead import (
    LEFT_EYE,
    MOUTH,
    REGION_HEAD_SPECS,
    RIGHT_EYE,
    SPECIALIST_NAMES,
    RegionHeadSpec,
)


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class RegionCropConfig:
    """Cau hinh crop nhe, dung giong nhau o train va inference."""

    eye_crop_scale: float = 2.0
    mouth_crop_scale: float = 1.75
    min_crop_side_px: float = 16.0
    train_center_jitter_fraction: float = 0.06
    train_scale_jitter: Tuple[float, float] = (0.92, 1.10)
    target_bbox_padding_fraction: float = 0.025
    # Dam bao bbox mat nham khong suy bien thanh mot duong thang.
    target_min_bbox_fraction: float = 0.12
    padding_color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR

    def __post_init__(self) -> None:
        if min(self.eye_crop_scale, self.mouth_crop_scale, self.min_crop_side_px) <= 0:
            raise ValueError('Crop scale va min_crop_side_px phai > 0.')
        if not 0 <= self.train_center_jitter_fraction < 0.5:
            raise ValueError('train_center_jitter_fraction phai nam trong [0, 0.5).')
        if (
            len(self.train_scale_jitter) != 2
            or self.train_scale_jitter[0] <= 0
            or self.train_scale_jitter[0] > self.train_scale_jitter[1]
        ):
            raise ValueError('train_scale_jitter phai la (min, max), 0 < min <= max.')
        if not 0 <= self.target_bbox_padding_fraction < 0.5:
            raise ValueError('target_bbox_padding_fraction phai nam trong [0, 0.5).')
        if not 0 < self.target_min_bbox_fraction <= 1:
            raise ValueError('target_min_bbox_fraction phai nam trong (0, 1].')
        if (
            len(self.padding_color) != 3
            or any(not 0 <= int(value) <= 255 for value in self.padding_color)
        ):
            raise ValueError('padding_color phai la tuple RGB gom ba gia tri [0, 255].')

    def crop_scale_for(self, name: str) -> float:
        if name in (LEFT_EYE, RIGHT_EYE):
            return self.eye_crop_scale
        if name == MOUTH:
            return self.mouth_crop_scale
        raise KeyError(f'Khong co crop scale cho region {name!r}.')


@dataclass(frozen=True)
class RegionCropTransform:
    """Bien doi affine giua pixel crop va pixel anh goc.

    Hai ma tran 3x3 dung toa do homogeneous theo quy uoc ``[x, y, 1]``.
    ``crop_box_original`` co the nam ngoai bien anh; phan do duoc padding.
    ``padding_ltrb`` duoc tinh bang don vi pixel tren anh goc.
    """

    region_name: str
    output_size: int
    original_size: Tuple[int, int]
    original_to_crop: np.ndarray
    crop_to_original: np.ndarray
    crop_box_original: np.ndarray
    padding_ltrb: np.ndarray

    def __post_init__(self) -> None:
        if self.region_name not in SPECIALIST_NAMES:
            raise ValueError(f'Region khong hop le: {self.region_name!r}.')
        if self.output_size <= 0:
            raise ValueError('output_size phai > 0.')
        original_w, original_h = self.original_size
        if original_w <= 0 or original_h <= 0:
            raise ValueError('original_size phai gom width/height > 0.')

        matrices = ('original_to_crop', 'crop_to_original')
        for name in matrices:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (3, 3) or not np.isfinite(value).all():
                raise ValueError(f'{name} phai co shape [3,3] va huu han.')
            object.__setattr__(self, name, value.copy())
        for name in ('crop_box_original', 'padding_ltrb'):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (4,) or not np.isfinite(value).all():
                raise ValueError(f'{name} phai co shape [4] va huu han.')
            object.__setattr__(self, name, value.copy())

        identity = self.original_to_crop @ self.crop_to_original
        if not np.allclose(identity, np.eye(3), atol=2e-4, rtol=2e-4):
            raise ValueError('original_to_crop va crop_to_original khong nghich dao.')

    @staticmethod
    def _map(points: ArrayLike, matrix: np.ndarray) -> ArrayLike:
        if isinstance(points, torch.Tensor):
            if points.shape[-1] != 2:
                raise ValueError('Points phai co shape [...,2].')
            flat = points.reshape(-1, 2)
            matrix_t = torch.as_tensor(matrix, dtype=points.dtype, device=points.device)
            ones = torch.ones((len(flat), 1), dtype=points.dtype, device=points.device)
            homogeneous = torch.cat((flat, ones), dim=1)
            mapped = homogeneous @ matrix_t.transpose(0, 1)
            denominator = mapped[:, 2:3]
            eps = torch.finfo(points.dtype).eps
            denominator = torch.where(
                denominator.abs() < eps,
                torch.full_like(denominator, eps),
                denominator,
            )
            return (mapped[:, :2] / denominator).reshape(points.shape)

        points_np = np.asarray(points)
        if points_np.shape[-1] != 2:
            raise ValueError('Points phai co shape [...,2].')
        output_dtype = points_np.dtype if np.issubdtype(points_np.dtype, np.floating) else np.float32
        flat = points_np.reshape(-1, 2).astype(np.float64, copy=False)
        homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
        mapped = homogeneous @ matrix.astype(np.float64).T
        denominator = mapped[:, 2:3]
        denominator[np.abs(denominator) < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
        return (mapped[:, :2] / denominator).reshape(points_np.shape).astype(output_dtype)

    def map_points_to_original(self, local_points: ArrayLike) -> ArrayLike:
        """Doi landmark pixel tren crop ve pixel anh goc."""
        return self._map(local_points, self.crop_to_original)

    def map_points_to_crop(self, original_points: ArrayLike) -> ArrayLike:
        """Doi landmark pixel tren anh goc sang pixel crop."""
        return self._map(original_points, self.original_to_crop)

    def tensor_metadata(self) -> Dict[str, torch.Tensor]:
        """Metadata co the stack truc tiep trong collate."""
        return {
            'crop_to_original': torch.from_numpy(self.crop_to_original.copy()),
            'original_to_crop': torch.from_numpy(self.original_to_crop.copy()),
            'crop_box_original': torch.from_numpy(self.crop_box_original.copy()),
            'padding_ltrb': torch.from_numpy(self.padding_ltrb.copy()),
        }


@dataclass(frozen=True)
class RegionCrop:
    """Ket qua duy nhat cua shared crop geometry."""

    image_rgb: np.ndarray
    local_landmarks: np.ndarray
    transform: RegionCropTransform

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb)
        landmarks = np.asarray(self.local_landmarks, dtype=np.float32)
        size = self.transform.output_size
        if image.shape != (size, size, 3) or image.dtype != np.uint8:
            raise ValueError(
                f'image_rgb phai la uint8 [{size},{size},3], nhan {image.shape}/{image.dtype}.'
            )
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError('local_landmarks phai co shape [K,2].')
        if not np.isfinite(landmarks).all():
            raise ValueError('local_landmarks chua NaN/Inf.')
        object.__setattr__(self, 'image_rgb', np.ascontiguousarray(image))
        object.__setattr__(self, 'local_landmarks', landmarks.copy())


def _resolve_spec(spec: Union[str, RegionHeadSpec]) -> RegionHeadSpec:
    if isinstance(spec, str):
        try:
            return REGION_HEAD_SPECS[spec]
        except KeyError as exc:
            raise KeyError(
                f'Region {spec!r} khong hop le; chon {SPECIALIST_NAMES}.'
            ) from exc
    if not isinstance(spec, RegionHeadSpec):
        raise TypeError('spec phai la ten region hoac RegionHeadSpec.')
    canonical = REGION_HEAD_SPECS.get(spec.name)
    if canonical != spec:
        raise ValueError(f'spec {spec.name!r} khong khop REGION_HEAD_SPECS.')
    return spec


def _validate_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'image_rgb phai co shape [H,W,3], nhan {image.shape}.')
    if image.dtype != np.uint8:
        raise TypeError(f'image_rgb phai co dtype uint8, nhan {image.dtype}.')
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError('image_rgb rong.')
    return np.ascontiguousarray(image)


def build_region_crop(
    image_rgb: np.ndarray,
    landmarks_478_px: ArrayLike,
    spec: Union[str, RegionHeadSpec],
    crop_cfg: Optional[RegionCropConfig] = None,
    *,
    training: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> RegionCrop:
    """Crop mot region tu anh RGB goc va tao mapping crop -> anh goc.

    Args:
        image_rgb: Anh RGB goc ``uint8 [H,W,3]``; khong duoc truyen anh da
            letterbox 480.
        landmarks_478_px: Toa do pixel MediaPipe ``[478,2]`` tren anh goc.
        spec: ``left_eye``, ``right_eye``, ``mouth`` hoac spec tu model.
        crop_cfg: Cung mot config phai duoc luu trong checkpoint va dung lai
            o inference.
        training: Neu True, ap dung center/scale jitter nhe.
        rng: Generator bat buoc neu ``training=True`` de ket qua co the lap lai.
    """
    image = _validate_rgb_image(image_rgb)
    resolved_spec = _resolve_spec(spec)
    cfg = crop_cfg or RegionCropConfig()
    landmarks = (
        landmarks_478_px.detach().cpu().numpy()
        if isinstance(landmarks_478_px, torch.Tensor)
        else np.asarray(landmarks_478_px)
    )
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        raise ValueError('landmarks_478_px phai co shape [478,2] (hoac nhieu hon 2 cot).')
    if landmarks.shape[0] != MEDIAPIPE_NUM_LANDMARKS:
        raise ValueError(
            f'Can dung {MEDIAPIPE_NUM_LANDMARKS} landmark, nhan {landmarks.shape[0]}.'
        )
    landmarks_xy = landmarks[:, :2].astype(np.float32, copy=False)
    if not np.isfinite(landmarks_xy).all():
        raise ValueError('landmarks_478_px chua NaN/Inf.')
    target_points = landmarks_xy[
        np.asarray(resolved_spec.global_landmark_indices)
    ]
    # Iris la output can sua, khong phai diem neo crop on dinh. Voi hai mat,
    # spec chi dung 16 diem mi/goc mat de HEAD4 iris outlier khong keo lech ROI.
    crop_anchor_points = landmarks_xy[
        np.asarray(resolved_spec.crop_anchor_indices)
    ]

    point_min = crop_anchor_points.min(axis=0)
    point_max = crop_anchor_points.max(axis=0)
    center = (point_min + point_max) * 0.5
    base_side = max(float((point_max - point_min).max()), cfg.min_crop_side_px)
    side = base_side * cfg.crop_scale_for(resolved_spec.name)

    if training:
        if rng is None:
            raise ValueError('Phai truyen rng khi training=True de jitter deterministic.')
        jitter = cfg.train_center_jitter_fraction * side
        center += rng.uniform(-jitter, jitter, size=2).astype(np.float32)
        side *= float(rng.uniform(*cfg.train_scale_jitter))

    if not np.isfinite(center).all() or not math.isfinite(side) or side <= 0:
        raise ValueError('Khong the tao crop tu landmark hien tai.')

    output_size = resolved_spec.input_size
    x1 = float(center[0] - side * 0.5)
    y1 = float(center[1] - side * 0.5)
    x2 = x1 + side
    y2 = y1 + side
    scale = output_size / side
    original_to_crop = np.array(
        [[scale, 0.0, -x1 * scale], [0.0, scale, -y1 * scale], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    crop_to_original = np.linalg.inv(original_to_crop).astype(np.float32)

    # OpenCV nhan ma tran src -> dst va tu nghich dao khi sampling. Border
    # constant ho tro crop vuot bien ma khong can cat thu cong.
    cropped = cv2.warpAffine(
        image,
        original_to_crop[:2],
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in cfg.padding_color),
    )
    orig_h, orig_w = image.shape[:2]
    transform = RegionCropTransform(
        region_name=resolved_spec.name,
        output_size=output_size,
        original_size=(orig_w, orig_h),
        original_to_crop=original_to_crop,
        crop_to_original=crop_to_original,
        crop_box_original=np.asarray((x1, y1, x2, y2), dtype=np.float32),
        padding_ltrb=np.asarray(
            (
                max(0.0, -x1),
                max(0.0, -y1),
                max(0.0, x2 - orig_w),
                max(0.0, y2 - orig_h),
            ),
            dtype=np.float32,
        ),
    )
    local_landmarks = transform.map_points_to_crop(target_points).astype(np.float32)
    return RegionCrop(cropped, local_landmarks, transform)


def _rgb_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image_rgb)
    return torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).div_(255.0)


def letterbox_full_face(
    image_rgb: np.ndarray,
    size: int = PIPELINE_IMAGE_SIZE,
    padding_color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Letterbox anh full-face va tra hai matrix 3x3.

    Matrix dau la ``original_to_full_face``; matrix sau la nghich dao.
    """
    image = _validate_rgb_image(image_rgb)
    if size <= 0:
        raise ValueError('size phai > 0.')
    orig_h, orig_w = image.shape[:2]
    uniform_scale = min(size / orig_w, size / orig_h)
    new_w = max(1, round(orig_w * uniform_scale))
    new_h = max(1, round(orig_h * uniform_scale))
    # HEAD 4 cu duoc train bang PIL bilinear. Giu dung implementation nay de
    # ``full_face`` khong bi lech vai gia tri pixel so voi pipeline goc.
    pil_image = Image.fromarray(image)
    if (new_w, new_h) != (orig_w, orig_h):
        pil_image = pil_image.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR,
        )
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas_image = Image.new('RGB', (size, size), tuple(padding_color))
    canvas_image.paste(pil_image, (left, top))
    canvas = np.asarray(canvas_image, dtype=np.uint8).copy()

    # Dung scale x/y sau rounding de matrix mo ta chinh xac anh da tao.
    sx, sy = new_w / orig_w, new_h / orig_h
    original_to_full = np.array(
        [[sx, 0.0, left], [0.0, sy, top], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    full_to_original = np.linalg.inv(original_to_full).astype(np.float32)
    return _rgb_to_tensor(canvas), original_to_full, full_to_original


def build_specialist_target(
    crop: RegionCrop,
    crop_cfg: Optional[RegionCropConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Tao target detection mot object cho specialist hien tai."""
    cfg = crop_cfg or RegionCropConfig()
    points = crop.local_landmarks
    size = float(crop.transform.output_size)
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    center = (point_min + point_max) * 0.5
    extent = point_max - point_min
    extent += size * (2.0 * cfg.target_bbox_padding_fraction)
    min_extent = size * cfg.target_min_bbox_fraction
    extent = np.maximum(extent, min_extent)
    box = np.concatenate((center - extent * 0.5, center + extent * 0.5))
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, size)
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, size)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'BBox specialist {crop.transform.region_name} bi suy bien.')

    landmarks_inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= size)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= size)
    ).all()
    return {
        'boxes': torch.from_numpy(box.reshape(1, 4).astype(np.float32)),
        'labels': torch.zeros(1, dtype=torch.long),
        'landmarks': torch.from_numpy(points[None].astype(np.float32)),
        'landmarks_valid': torch.tensor([bool(landmarks_inside)], dtype=torch.bool),
    }


@dataclass(frozen=True)
class MultiHeadDatasetConfig:
    """Cau hinh toi thieu cho dataset va hai DataLoader."""

    root_dir: str
    images_dir_name: str = 'images'
    jsonl_name: str = 'annotations.jsonl'
    batch_size: int = 4
    num_workers: int = 4
    val_ratio: float = 0.15
    seed: int = 42
    full_face_size: int = PIPELINE_IMAGE_SIZE
    pin_memory: bool = True
    # False giup trainer reseed crop jitter theo stage/epoch khi resume.
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = 2
    train_drop_last: bool = True
    strict_schema: bool = True
    normalized_coordinate_tolerance: float = 1e-3
    crop: RegionCropConfig = field(default_factory=RegionCropConfig)

    def __post_init__(self) -> None:
        if not self.root_dir or not self.images_dir_name or not self.jsonl_name:
            raise ValueError('root_dir/images_dir_name/jsonl_name khong duoc rong.')
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError('batch_size phai > 0 va num_workers phai >= 0.')
        if self.seed < 0:
            raise ValueError('seed phai >= 0.')
        if not 0 < self.val_ratio < 1:
            raise ValueError('val_ratio phai nam trong (0,1).')
        if self.full_face_size != PIPELINE_IMAGE_SIZE:
            raise ValueError(f'HEAD 4 bat buoc input {PIPELINE_IMAGE_SIZE}.')
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError('prefetch_factor phai > 0 hoac None.')
        if not 0 <= self.normalized_coordinate_tolerance <= 0.5:
            raise ValueError('normalized_coordinate_tolerance phai nam trong [0,0.5].')


def _jsonl_offsets(jsonl_path: str) -> np.ndarray:
    offsets: List[int] = []
    with open(jsonl_path, 'rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    if len(offsets) < 2:
        raise ValueError('Dataset can it nhat 2 record de chia train/validation.')
    return np.asarray(offsets, dtype=np.int64)


def _read_record(handle: BinaryIO, offset: int) -> dict:
    handle.seek(int(offset))
    try:
        value = json.loads(handle.readline().decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'JSONL loi tai byte offset {offset}: {exc}.') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Record tai byte offset {offset} phai la object JSON.')
    return value


def _primary_face(record: Mapping, context: str) -> Mapping:
    faces = record.get('faces')
    if not isinstance(faces, list) or not faces:
        raise ValueError(f'{context}: faces phai la list khong rong.')
    primary_index = record.get('primary_face_index', 0)
    if isinstance(primary_index, bool) or not isinstance(primary_index, int):
        raise ValueError(f'{context}: primary_face_index phai la so nguyen.')
    if not 0 <= primary_index < len(faces):
        raise ValueError(
            f'{context}: primary_face_index={primary_index} ngoai [0,{len(faces) - 1}].'
        )
    face = faces[primary_index]
    if not isinstance(face, Mapping):
        raise ValueError(f'{context}: primary face phai la object.')
    return face


def _normalized_landmarks(face: Mapping, context: str, tolerance: float) -> np.ndarray:
    raw = face.get('landmarks_normalized')
    if not isinstance(raw, list) or len(raw) != MEDIAPIPE_NUM_LANDMARKS:
        received = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ValueError(
            f'{context}: can {MEDIAPIPE_NUM_LANDMARKS} landmark, nhan {received}.'
        )
    try:
        points = np.asarray([[float(p['x']), float(p['y'])] for p in raw], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{context}: landmark schema loi ({exc}).') from exc
    if points.shape != (MEDIAPIPE_NUM_LANDMARKS, 2) or not np.isfinite(points).all():
        raise ValueError(f'{context}: landmark shape sai hoac co NaN/Inf.')
    if (points < -tolerance).any() or (points > 1.0 + tolerance).any():
        raise ValueError(f'{context}: landmark normalized nam ngoai [0,1].')
    return points


def _validate_bbox(face: Mapping, context: str, tolerance: float) -> None:
    raw = face.get('bounding_box_normalized')
    if not isinstance(raw, Mapping):
        raise ValueError(f'{context}: thieu bounding_box_normalized.')
    try:
        box = np.asarray(
            [float(raw['xmin']), float(raw['ymin']), float(raw['xmax']), float(raw['ymax'])],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{context}: bbox schema loi ({exc}).') from exc
    if not np.isfinite(box).all():
        raise ValueError(f'{context}: bbox co NaN/Inf.')
    if (box < -tolerance).any() or (box > 1.0 + tolerance).any():
        raise ValueError(f'{context}: bbox normalized nam ngoai [0,1].')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'{context}: bbox xyxy bi suy bien.')


class MultiHeadFaceRegionDataset(Dataset):
    """Moi record chi dung primary face va khong horizontal flip."""

    def __init__(
        self,
        cfg: MultiHeadDatasetConfig,
        record_indices: Sequence[int],
        *,
        training: bool,
        offsets: Optional[np.ndarray] = None,
    ):
        # Khoi tao handle truoc moi validation de __del__ luon an toan.
        self._file_handle: Optional[BinaryIO] = None
        self._file_handle_pid: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: Optional[int] = None
        self._augmentation_seed = int(cfg.seed)
        self.cfg = cfg
        self.training = bool(training)
        self.images_dir = os.path.realpath(os.path.join(cfg.root_dir, cfg.images_dir_name))
        self.jsonl_path = os.path.realpath(os.path.join(cfg.root_dir, cfg.jsonl_name))
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f'Khong tim thay thu muc anh {self.images_dir}.')
        if not os.path.isfile(self.jsonl_path):
            raise FileNotFoundError(f'Khong tim thay annotation {self.jsonl_path}.')

        self.offsets = np.asarray(offsets if offsets is not None else _jsonl_offsets(self.jsonl_path))
        if self.offsets.ndim != 1 or self.offsets.dtype.kind not in 'iu':
            raise ValueError('offsets phai la mang so nguyen 1 chieu.')
        self.record_indices = np.asarray(record_indices, dtype=np.int64)
        if self.record_indices.ndim != 1 or len(self.record_indices) == 0:
            raise ValueError('record_indices phai la mang 1 chieu khong rong.')
        if (
            self.record_indices.min() < 0
            or self.record_indices.max() >= len(self.offsets)
            or len(np.unique(self.record_indices)) != len(self.record_indices)
        ):
            raise ValueError('record_indices ngoai pham vi hoac bi trung.')
        if cfg.strict_schema:
            self._validate_records()

    def _image_path(self, file_name: str) -> str:
        if not isinstance(file_name, str) or not file_name:
            raise ValueError('file_name phai la chuoi khong rong.')
        path = os.path.realpath(os.path.join(self.images_dir, file_name))
        if os.path.commonpath((self.images_dir, path)) != self.images_dir:
            raise ValueError(f'file_name vuot ra ngoai images/: {file_name!r}.')
        return path

    def _validate_record(self, record: Mapping, record_index: int) -> None:
        context = f'record {record_index}'
        file_name = record.get('file_name')
        image_path = self._image_path(file_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f'{context}: thieu anh {image_path}.')
        try:
            width = int(record['image_width'])
            height = int(record['image_height'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{context}: image_width/image_height loi.') from exc
        if width <= 0 or height <= 0:
            raise ValueError(f'{context}: kich thuoc annotation phai > 0.')
        faces = record.get('faces')
        if not isinstance(faces, list) or not faces:
            raise ValueError(f'{context}: faces phai la list khong rong.')
        declared_num_faces = record.get('num_faces')
        if declared_num_faces is not None and declared_num_faces != len(faces):
            raise ValueError(f'{context}: num_faces khong khop len(faces).')
        primary = _primary_face(record, context)
        _validate_bbox(primary, context, self.cfg.normalized_coordinate_tolerance)
        _normalized_landmarks(primary, context, self.cfg.normalized_coordinate_tolerance)

    def _validate_records(self) -> None:
        with open(self.jsonl_path, 'rb') as handle:
            for record_index in self.record_indices:
                record = _read_record(handle, int(self.offsets[record_index]))
                self._validate_record(record, int(record_index))

    def __len__(self) -> int:
        return len(self.record_indices)

    def _get_file(self) -> BinaryIO:
        pid = os.getpid()
        handle = self._file_handle
        if handle is None or handle.closed or self._file_handle_pid != pid:
            self.close()
            self._file_handle = open(self.jsonl_path, 'rb')
            self._file_handle_pid = pid
        return self._file_handle

    def close(self) -> None:
        handle = getattr(self, '_file_handle', None)
        if handle is not None and not handle.closed:
            handle.close()
        self._file_handle = None
        self._file_handle_pid = None

    def __del__(self):
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_file_handle'] = None
        state['_file_handle_pid'] = None
        state['_rng'] = None
        state['_rng_pid'] = None
        return state

    def _get_rng(self) -> np.random.Generator:
        """RNG rieng tung worker, co seed lap lai nhung tien trien moi batch.

        ``torch.initial_seed`` duoc DataLoader sinh deterministically tu
        generator cua loader. Khac voi seed co dinh theo record, generator
        nay tiep tuc tien qua moi lan ``__getitem__`` nen crop jitter thay doi
        giua cac epoch.
        """
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            seed = (
                int(torch.initial_seed()) + self._augmentation_seed
            ) % (2 ** 63 - 1)
            self._rng = np.random.default_rng(seed)
            self._rng_pid = pid
        return self._rng

    def set_augmentation_seed(self, seed: int) -> None:
        """Reset crop jitter stream truoc moi stage/epoch.

        Voi ``persistent_workers=False`` (mac dinh), workers moi se nhan seed
        nay khi DataLoader bat dau iterator tiep theo.
        """
        self._augmentation_seed = int(seed)
        self._rng = None
        self._rng_pid = None

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = int(self.record_indices[index])
        record = _read_record(self._get_file(), int(self.offsets[record_index]))
        context = f'record {record_index}'
        face = _primary_face(record, context)
        normalized = _normalized_landmarks(
            face, context, self.cfg.normalized_coordinate_tolerance
        )
        image_path = self._image_path(record.get('file_name'))
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f'Khong doc duoc anh {image_path}.')
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        annotated_size = (int(record['image_width']), int(record['image_height']))
        if annotated_size != (orig_w, orig_h):
            raise ValueError(
                f'{context}: annotation size={annotated_size}, anh that={(orig_w, orig_h)}.'
            )
        landmarks_original = normalized * np.asarray((orig_w, orig_h), dtype=np.float32)

        full_face, original_to_full, full_to_original = letterbox_full_face(
            image_rgb,
            size=self.cfg.full_face_size,
            padding_color=self.cfg.crop.padding_color,
        )
        regions = {}
        worker_rng = self._get_rng() if self.training else None
        for name in SPECIALIST_NAMES:
            crop = build_region_crop(
                image_rgb,
                landmarks_original,
                REGION_HEAD_SPECS[name],
                self.cfg.crop,
                training=self.training,
                rng=worker_rng,
            )
            regions[name] = {
                'image': _rgb_to_tensor(crop.image_rgb),
                'target': build_specialist_target(crop, self.cfg.crop),
                'transform': crop.transform,
            }

        return {
            'full_face': full_face,
            'regions': regions,
            'file_name': record['file_name'],
            'record_index': record_index,
            'original_size': torch.tensor((orig_w, orig_h), dtype=torch.long),
            'original_to_full_face': torch.from_numpy(original_to_full),
            'full_face_to_original': torch.from_numpy(full_to_original),
        }


def multihead_collate(batch: List[dict]) -> dict:
    """Collate dung cho moi specialist va trainer train tuan tu."""
    if not batch:
        raise ValueError('Batch rong.')
    target_keys = ('boxes', 'labels', 'landmarks', 'landmarks_valid')
    regions: Dict[str, dict] = {}
    for name in SPECIALIST_NAMES:
        samples = [sample['regions'][name] for sample in batch]
        metadata = [sample['transform'].tensor_metadata() for sample in samples]
        regions[name] = {
            'images': torch.stack([sample['image'] for sample in samples]),
            'targets': [
                {key: sample['target'][key] for key in target_keys}
                for sample in samples
            ],
            'crop_to_original': torch.stack([item['crop_to_original'] for item in metadata]),
            'original_to_crop': torch.stack([item['original_to_crop'] for item in metadata]),
            'crop_box_original': torch.stack([item['crop_box_original'] for item in metadata]),
            'padding_ltrb': torch.stack([item['padding_ltrb'] for item in metadata]),
        }
    return {
        'full_face': torch.stack([sample['full_face'] for sample in batch]),
        'regions': regions,
        'file_name': [sample['file_name'] for sample in batch],
        'record_index': torch.tensor([sample['record_index'] for sample in batch], dtype=torch.long),
        'original_size': torch.stack([sample['original_size'] for sample in batch]),
        'original_to_full_face': torch.stack(
            [sample['original_to_full_face'] for sample in batch]
        ),
        'full_face_to_original': torch.stack(
            [sample['full_face_to_original'] for sample in batch]
        ),
    }


@dataclass(frozen=True)
class RecordSplit:
    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.train_indices or not self.val_indices:
            raise ValueError('Train/val split khong duoc rong.')
        if set(self.train_indices) & set(self.val_indices):
            raise ValueError('Train va val bi overlap record.')


def deterministic_record_split(
    num_records: int,
    val_ratio: float,
    seed: int,
) -> RecordSplit:
    """Chia theo record goc truoc khi tao bat ky crop nao."""
    if num_records < 2:
        raise ValueError('Can it nhat 2 record.')
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio phai nam trong (0,1).')
    num_val = max(1, min(num_records - 1, round(num_records * val_ratio)))
    permutation = np.random.default_rng(seed).permutation(num_records)
    val_indices = tuple(sorted(int(value) for value in permutation[:num_val]))
    train_indices = tuple(sorted(int(value) for value in permutation[num_val:]))
    return RecordSplit(train_indices=train_indices, val_indices=val_indices)


@dataclass(frozen=True)
class MultiHeadDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: MultiHeadFaceRegionDataset
    val_dataset: MultiHeadFaceRegionDataset
    split: RecordSplit

    @property
    def train(self) -> DataLoader:
        return self.train_loader

    @property
    def val(self) -> DataLoader:
        return self.val_loader


def _make_loader(
    dataset: MultiHeadFaceRegionDataset,
    cfg: MultiHeadDatasetConfig,
    *,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    workers_enabled = cfg.num_workers > 0
    kwargs = {
        'dataset': dataset,
        'batch_size': cfg.batch_size,
        'shuffle': shuffle,
        'num_workers': cfg.num_workers,
        'collate_fn': multihead_collate,
        'pin_memory': cfg.pin_memory and torch.cuda.is_available(),
        'persistent_workers': cfg.persistent_workers and workers_enabled,
        'drop_last': drop_last,
        'generator': torch.Generator().manual_seed(cfg.seed + (0 if shuffle else 1)),
    }
    if workers_enabled and cfg.prefetch_factor is not None:
        kwargs['prefetch_factor'] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def build_multihead_loaders(cfg: MultiHeadDatasetConfig) -> MultiHeadDataLoaders:
    """Validate, chia record deterministic va tao train/validation loaders."""
    jsonl_path = os.path.realpath(os.path.join(cfg.root_dir, cfg.jsonl_name))
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f'Khong tim thay annotation {jsonl_path}.')
    offsets = _jsonl_offsets(jsonl_path)
    split = deterministic_record_split(len(offsets), cfg.val_ratio, cfg.seed)
    train_dataset = MultiHeadFaceRegionDataset(
        cfg, split.train_indices, training=True, offsets=offsets
    )
    val_dataset = MultiHeadFaceRegionDataset(
        cfg, split.val_indices, training=False, offsets=offsets
    )
    train_loader = _make_loader(
        train_dataset,
        cfg,
        shuffle=True,
        drop_last=cfg.train_drop_last and len(train_dataset) >= cfg.batch_size,
    )
    val_loader = _make_loader(val_dataset, cfg, shuffle=False, drop_last=False)
    return MultiHeadDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        split=split,
    )


__all__ = (
    'RegionCropConfig',
    'RegionCropTransform',
    'RegionCrop',
    'build_region_crop',
    'letterbox_full_face',
    'build_specialist_target',
    'MultiHeadDatasetConfig',
    'MultiHeadFaceRegionDataset',
    'multihead_collate',
    'RecordSplit',
    'deterministic_record_split',
    'MultiHeadDataLoaders',
    'build_multihead_loaders',
)
