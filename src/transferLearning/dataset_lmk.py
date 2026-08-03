import json
import hashlib
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

try:
    from .config_lmk import DatasetConfig, MarginCoverageConfig
    from .mediapipe_478 import MEDIAPIPE_478_FLIP_PERMUTATION
except ImportError:  # Cho phép chạy trực tiếp file trong thư mục này.
    from config_lmk import DatasetConfig, MarginCoverageConfig
    from mediapipe_478 import MEDIAPIPE_478_FLIP_PERMUTATION


def _build_or_load_offsets(jsonl_path: str, index_cache_dir: str) -> np.ndarray:
    source_key = os.path.realpath(jsonl_path).encode('utf-8')
    digest = hashlib.sha256(source_key).hexdigest()[:16]
    os.makedirs(index_cache_dir, exist_ok=True)
    idx_path = os.path.join(index_cache_dir, f'{os.path.basename(jsonl_path)}.{digest}.idx.npy')
    if os.path.exists(idx_path) and os.path.getmtime(idx_path) >= os.path.getmtime(jsonl_path):
        return np.load(idx_path)
    print(f'[Dataset] Đang xây index cho {jsonl_path}...')
    t0 = time.time()
    offsets: List[int] = []
    with open(jsonl_path, 'rb') as f:
        offset = f.tell()
        for line in f:
            if line.strip():
                offsets.append(offset)
            offset = f.tell()
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    try:
        np.save(idx_path, offsets_arr)
    except OSError as exc:
        print(f'[Dataset] Không thể ghi cache index {idx_path}: {exc}. Tiếp tục dùng index trong RAM.')
    print(f'[Dataset] Xong: {len(offsets_arr)} record, {time.time() - t0:.1f}s.')
    return offsets_arr


def _read_record(file_handle, offset: int) -> dict:
    file_handle.seek(offset)
    return json.loads(file_handle.readline().decode('utf-8'))


def _validate_landmark_schema(jsonl_path: str, offsets: np.ndarray, cfg: DatasetConfig) -> Tuple[int, dict]:
    """Quét toàn bộ JSONL để không âm thầm trộn schema 468/478 hoặc bỏ positive."""
    expected_k: Optional[int] = None
    stats = {'records': len(offsets), 'faces': 0, 'empty_records': 0}
    errors: List[str] = []
    error_count = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 20:
            errors.append(message)

    with open(jsonl_path, 'rb') as f:
        for row_idx, offset in enumerate(offsets):
            try:
                record = _read_record(f, int(offset))
                file_name = record['file_name']
                if not os.path.isfile(os.path.join(cfg.root_dir, cfg.images_dir_name, file_name)):
                    add_error(f'dòng {row_idx + 1}: thiếu ảnh {file_name}')
                orig_w = float(record['image_width'])
                orig_h = float(record['image_height'])
                if not np.isfinite((orig_w, orig_h)).all() or orig_w <= 0 or orig_h <= 0:
                    add_error(f'dòng {row_idx + 1} ({file_name}): kích thước ảnh không hợp lệ')
                    continue
                letterbox_scale = min(cfg.image_size / orig_w, cfg.image_size / orig_h)
                faces = record.get('faces', [])
                if not faces:
                    stats['empty_records'] += 1
                    if not cfg.allow_empty_targets:
                        add_error(f'dòng {row_idx + 1} ({file_name}) không có face')
                for face_idx, face in enumerate(faces):
                    pts = face['landmarks_normalized']
                    k = len(pts)
                    expected_k = k if expected_k is None else expected_k
                    if k <= 0:
                        add_error(f'dòng {row_idx + 1}, face {face_idx}: landmark rỗng')
                    if k != expected_k:
                        add_error(
                            f'dòng {row_idx + 1}, face {face_idx}: K={k}, expected={expected_k}'
                        )
                    bb = face['bounding_box_normalized']
                    bbox_values = np.asarray(
                        [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']], dtype=np.float64
                    )
                    if not np.isfinite(bbox_values).all():
                        add_error(f'dòng {row_idx + 1}, face {face_idx}: bbox chứa NaN/Inf')
                        continue
                    tolerance = cfg.normalized_coordinate_tolerance
                    if (bbox_values < -tolerance).any() or (bbox_values > 1 + tolerance).any():
                        add_error(f'dòng {row_idx + 1}, face {face_idx}: bbox ngoài miền chuẩn hóa [0, 1]')
                    bw = (bbox_values[2] - bbox_values[0]) * orig_w * letterbox_scale
                    bh = (bbox_values[3] - bbox_values[1]) * orig_h * letterbox_scale
                    if bw < cfg.min_box_size_px or bh < cfg.min_box_size_px:
                        add_error(
                            f'dòng {row_idx + 1}, face {face_idx}: bbox quá nhỏ ({bw:.2f}x{bh:.2f}px)'
                        )
                    point_values = np.asarray(
                        [[point['x'], point['y']] for point in pts], dtype=np.float64
                    )
                    if not np.isfinite(point_values).all():
                        add_error(f'dòng {row_idx + 1}, face {face_idx}: landmark chứa NaN/Inf')
                    elif (point_values < -tolerance).any() or (point_values > 1 + tolerance).any():
                        add_error(f'dòng {row_idx + 1}, face {face_idx}: landmark ngoài miền chuẩn hóa [0, 1]')
                    stats['faces'] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                add_error(f'dòng {row_idx + 1}: schema lỗi ({exc})')

    if expected_k is None:
        raise ValueError(f'{jsonl_path} không chứa landmark hợp lệ nào.')
    if error_count:
        message = '\n  - '.join(errors)
        if cfg.strict_schema:
            raise ValueError(
                f'Dataset có {error_count} lỗi (hiển thị tối đa 20):\n  - {message}'
            )
        print(f'[Dataset] CẢNH BÁO: có {error_count} lỗi schema:\n  - {message}')
    return expected_k, stats


def _letterbox_pil(image: Image.Image, size: int, fill: int = 114) -> Tuple[Image.Image, float, int, int]:
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    right = size - new_w - left
    bottom = size - new_h - top
    return ImageOps.expand(image, (left, top, right, bottom), fill=(fill, fill, fill)), scale, left, top


def horizontal_flip_targets(
    boxes: torch.Tensor,
    landmarks: torch.Tensor,
    image_size: int,
    permutation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lật xyxy và remap semantic landmark; không sửa tensor đầu vào."""
    flipped_boxes = boxes.clone()
    flipped_landmarks = landmarks.clone()
    flipped_boxes[:, 0] = image_size - boxes[:, 2]
    flipped_boxes[:, 2] = image_size - boxes[:, 0]
    flipped_landmarks[..., 0] = image_size - flipped_landmarks[..., 0]
    flipped_landmarks = flipped_landmarks[:, permutation, :]
    return flipped_boxes, flipped_landmarks


class FaceLandmarkDataset(Dataset):

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.images_dir = os.path.join(cfg.root_dir, cfg.images_dir_name)
        self.jsonl_path = os.path.join(cfg.root_dir, cfg.jsonl_name)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f'Không tìm thấy thư mục ảnh {self.images_dir}.')
        if not os.path.isfile(self.jsonl_path):
            raise FileNotFoundError(f'Không tìm thấy annotation {self.jsonl_path}.')
        self.offsets = _build_or_load_offsets(self.jsonl_path, cfg.index_cache_dir)
        self.num_landmarks, self.validation_stats = _validate_landmark_schema(
            self.jsonl_path, self.offsets, cfg
        )
        if cfg.horizontal_flip_mode != 'off' and self.num_landmarks != 478:
            raise ValueError(
                f'Horizontal flip semantic chỉ hỗ trợ MediaPipe 478, dataset có K={self.num_landmarks}.'
            )
        self.image_size = cfg.image_size
        self.flip_permutation = torch.tensor(MEDIAPIPE_478_FLIP_PERMUTATION, dtype=torch.long)
        self._file_handle = None
        self.photometric = (
            T.ColorJitter(cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)
            if cfg.augment and any((cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue))
            else None
        )
        self.to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        print(
            f"[Dataset] {cfg.root_dir}: {len(self.offsets)} record, "
            f"{self.validation_stats['faces']} face, K={self.num_landmarks}."
        )

    def __len__(self) -> int:
        multiplier = 2 if self.cfg.horizontal_flip_mode == 'paired' else 1
        return len(self.offsets) * multiplier

    def _resolve_index_and_flip(self, index: int) -> Tuple[int, bool]:
        if self.cfg.horizontal_flip_mode == 'paired':
            return index // 2, bool(index % 2)
        if self.cfg.horizontal_flip_mode == 'random':
            should_flip = bool(torch.rand(()) < self.cfg.horizontal_flip_probability)
            return index, should_flip
        return index, False

    def _get_file(self):
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.jsonl_path, 'rb')
        return self._file_handle

    def close(self) -> None:
        file_handle = getattr(self, '_file_handle', None)
        if file_handle is not None and not file_handle.closed:
            file_handle.close()
        self._file_handle = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file_handle'] = None
        return state

    def __getitem__(self, i: int):
        record_index, should_flip = self._resolve_index_and_flip(i)
        record = _read_record(self._get_file(), int(self.offsets[record_index]))
        image_path = os.path.join(self.images_dir, record['file_name'])
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        orig_w, orig_h = image.size
        image, scale, pad_x, pad_y = _letterbox_pil(image, self.image_size)
        if should_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.photometric is not None:
            image = self.photometric(image)

        boxes, labels, landmarks = [], [], []
        for face_idx, face in enumerate(record.get('faces', [])):
            pts = face['landmarks_normalized']
            if len(pts) != self.num_landmarks:
                raise ValueError(
                    f"{record['file_name']} face {face_idx}: K={len(pts)} khác {self.num_landmarks}."
                )
            bb = face['bounding_box_normalized']
            x1 = np.clip(float(bb['xmin']) * orig_w * scale + pad_x, 0, self.image_size)
            y1 = np.clip(float(bb['ymin']) * orig_h * scale + pad_y, 0, self.image_size)
            x2 = np.clip(float(bb['xmax']) * orig_w * scale + pad_x, 0, self.image_size)
            y2 = np.clip(float(bb['ymax']) * orig_h * scale + pad_y, 0, self.image_size)
            if x2 - x1 < self.cfg.min_box_size_px or y2 - y1 < self.cfg.min_box_size_px:
                raise ValueError(f"{record['file_name']} face {face_idx}: bbox không hợp lệ sau letterbox.")
            lm = [
                [
                    np.clip(float(p['x']) * orig_w * scale + pad_x, 0, self.image_size),
                    np.clip(float(p['y']) * orig_h * scale + pad_y, 0, self.image_size),
                ]
                for p in pts
            ]
            boxes.append([x1, y1, x2, y2])
            labels.append(0)
            landmarks.append(lm)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            landmarks_t = torch.tensor(landmarks, dtype=torch.float32)
            valid_t = torch.ones(len(boxes), dtype=torch.bool)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
            landmarks_t = torch.zeros((0, self.num_landmarks, 2), dtype=torch.float32)
            valid_t = torch.zeros((0,), dtype=torch.bool)

        if should_flip:
            boxes_t, landmarks_t = horizontal_flip_targets(
                boxes_t, landmarks_t, self.image_size, self.flip_permutation
            )

        return {
            'image': self.to_tensor(image),
            'boxes': boxes_t,
            'labels': labels_t,
            'landmarks': landmarks_t,
            'landmarks_valid': valid_t,
            'file_name': record['file_name'],
            'orig_size': torch.tensor([orig_w, orig_h], dtype=torch.long),
            'was_flipped': should_flip,
        }


def face_landmark_collate(batch):
    return {
        'image': torch.stack([b['image'] for b in batch], dim=0),
        'targets': [
            {key: b[key] for key in ('boxes', 'labels', 'landmarks', 'landmarks_valid')}
            for b in batch
        ],
        'file_name': [b['file_name'] for b in batch],
        'orig_size': torch.stack([b['orig_size'] for b in batch], dim=0),
        'was_flipped': torch.tensor([b['was_flipped'] for b in batch], dtype=torch.bool),
    }


class FaceLandmarkDataModule:

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.dataset = FaceLandmarkDataset(cfg)

    @property
    def num_landmarks(self) -> int:
        return self.dataset.num_landmarks

    def loader(self) -> DataLoader:
        cfg = self.cfg
        kwargs = dict(
            dataset=self.dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            collate_fn=face_landmark_collate,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            persistent_workers=cfg.persistent_workers,
            drop_last=cfg.drop_last,
        )
        if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
            kwargs['prefetch_factor'] = cfg.prefetch_factor
        return DataLoader(**kwargs)


class LandmarkMarginCoverageChecker:
    """Chỉ hữu ích khi bbox độc lập với landmark; lấy mẫu bằng offset, không đọc cả file vào RAM."""

    def __init__(self, cfg: MarginCoverageConfig):
        self.cfg = cfg

    def run(self) -> None:
        cfg = self.cfg
        jsonl_path = os.path.join(cfg.root_dir, cfg.jsonl_name)
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f'Không tìm thấy {jsonl_path}.')
        offsets = _build_or_load_offsets(jsonl_path, cfg.index_cache_dir)
        rng = np.random.default_rng(cfg.seed)
        count = min(cfg.sample_size, len(offsets))
        chosen = rng.choice(offsets, size=count, replace=False) if count else []
        n_points_total = 0
        n_outside = {m: 0 for m in cfg.margins}
        overflow_fracs, n_faces_checked = [], 0
        with open(jsonl_path, 'rb') as f:
            for offset in chosen:
                record = _read_record(f, int(offset))
                for face in record.get('faces', []):
                    bb = face['bounding_box_normalized']
                    x1, y1, x2, y2 = map(float, (bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']))
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    n_faces_checked += 1
                    for p in face['landmarks_normalized']:
                        n_points_total += 1
                        px, py = float(p['x']), float(p['y'])
                        overflow = max((x1 - px) / w, (px - x2) / w, (y1 - py) / h, (py - y2) / h, 0.0)
                        overflow_fracs.append(overflow)
                        for margin in cfg.margins:
                            if not (x1 - margin * w <= px <= x2 + margin * w and y1 - margin * h <= py <= y2 + margin * h):
                                n_outside[margin] += 1
        if not n_points_total:
            print('[MarginCoverageChecker] Không tìm thấy landmark nào trong mẫu.')
            return
        print('[MarginCoverageChecker] Lưu ý: bbox sinh từ min/max landmark sẽ làm coverage gần như luôn 100%.')
        print(f'Số mặt: {n_faces_checked} | số điểm: {n_points_total}')
        for margin in cfg.margins:
            print(f'  margin={margin:.3f}: outside={100 * n_outside[margin] / n_points_total:.4f}%')
        overflow_arr = np.asarray(overflow_fracs)
        for pct in (95, 99, 99.9, 100):
            print(f'  percentile {pct}%: margin >= {np.percentile(overflow_arr, pct):.4f}')
