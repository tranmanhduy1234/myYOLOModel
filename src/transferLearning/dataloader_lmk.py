import hashlib
import json
import os
import time
from typing import BinaryIO, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from src.transferLearning.config_lmk import DatasetConfig, MEDIAPIPE_NUM_LANDMARKS
from src.transferLearning.mediapipe_478 import MEDIAPIPE_478_FLIP_PERMUTATION

def _build_or_load_offsets(jsonl_path: str, cache_dir: str) -> np.ndarray:
    """Tạo cache byte-offset để đọc ngẫu nhiên từng dòng JSONL."""
    digest = hashlib.sha256(os.path.realpath(jsonl_path).encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(jsonl_path)}.{digest}.idx.npy")

    if (
        os.path.isfile(cache_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(jsonl_path)
    ):
        return np.load(cache_path)

    print(f"[Dataset] Đang xây index cho {jsonl_path}...")
    started = time.time()
    offsets: List[int] = []
    with open(jsonl_path, "rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)

    result = np.asarray(offsets, dtype=np.int64)
    try:
        np.save(cache_path, result)
    except OSError as exc:
        print(f"[Dataset] Không thể ghi cache {cache_path}: {exc}")
    print(f"[Dataset] Xong: {len(result)} record, {time.time() - started:.1f}s.")
    return result


def _read_record(file: BinaryIO, offset: int) -> dict:
    file.seek(offset)
    return json.loads(file.readline().decode("utf-8"))


def _validate_landmark_schema(
    jsonl_path: str,
    offsets: np.ndarray,
    cfg: DatasetConfig,
) -> Tuple[int, dict]:
    """Kiểm tra ảnh, bbox và landmark trước khi khởi tạo DataLoader."""
    expected_k: Optional[int] = None
    stats = {"records": len(offsets), "faces": 0, "empty_records": 0}
    errors: List[str] = []
    error_count = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 20:
            errors.append(message)

    tolerance = cfg.normalized_coordinate_tolerance
    image_root = os.path.join(cfg.root_dir, cfg.images_dir_name)

    with open(jsonl_path, "rb") as file:
        for row_idx, offset in enumerate(offsets, start=1):
            try:
                record = _read_record(file, int(offset))
                file_name = record["file_name"]
                image_path = os.path.join(image_root, file_name)
                if not os.path.isfile(image_path):
                    add_error(f"dòng {row_idx}: thiếu ảnh {file_name}")

                orig_w = float(record["image_width"])
                orig_h = float(record["image_height"])
                if not np.isfinite((orig_w, orig_h)).all() or orig_w <= 0 or orig_h <= 0:
                    add_error(f"dòng {row_idx} ({file_name}): kích thước ảnh không hợp lệ")
                    continue

                scale = min(cfg.image_size / orig_w, cfg.image_size / orig_h)
                faces = record.get("faces", [])
                if not faces:
                    stats["empty_records"] += 1
                    if not cfg.allow_empty_targets:
                        add_error(f"dòng {row_idx} ({file_name}): không có face")

                for face_idx, face in enumerate(faces):
                    points = face["landmarks_normalized"]
                    k = len(points)
                    if k <= 0:
                        add_error(f"dòng {row_idx}, face {face_idx}: landmark rỗng")
                        continue
                    if expected_k is None:
                        expected_k = k
                    elif k != expected_k:
                        add_error(
                            f"dòng {row_idx}, face {face_idx}: K={k}, expected={expected_k}"
                        )

                    bbox = face["bounding_box_normalized"]
                    bbox_values = np.asarray(
                        [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
                        dtype=np.float64,
                    )
                    if not np.isfinite(bbox_values).all():
                        add_error(f"dòng {row_idx}, face {face_idx}: bbox chứa NaN/Inf")
                        continue
                    if (bbox_values < -tolerance).any() or (bbox_values > 1 + tolerance).any():
                        add_error(f"dòng {row_idx}, face {face_idx}: bbox ngoài [0,1]")

                    width = (bbox_values[2] - bbox_values[0]) * orig_w * scale
                    height = (bbox_values[3] - bbox_values[1]) * orig_h * scale
                    if width < cfg.min_box_size_px or height < cfg.min_box_size_px:
                        add_error(
                            f"dòng {row_idx}, face {face_idx}: bbox quá nhỏ "
                            f"({width:.2f}x{height:.2f}px)"
                        )

                    point_values = np.asarray(
                        [[point["x"], point["y"]] for point in points],
                        dtype=np.float64,
                    )
                    if not np.isfinite(point_values).all():
                        add_error(f"dòng {row_idx}, face {face_idx}: landmark chứa NaN/Inf")
                    elif (point_values < -tolerance).any() or (point_values > 1 + tolerance).any():
                        add_error(f"dòng {row_idx}, face {face_idx}: landmark ngoài [0,1]")
                    stats["faces"] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                add_error(f"dòng {row_idx}: schema lỗi ({exc})")

    if expected_k is None:
        raise ValueError(f"{jsonl_path} không chứa landmark hợp lệ.")
    if error_count:
        message = "\n  - ".join(errors)
        if cfg.strict_schema:
            raise ValueError(
                f"Dataset có {error_count} lỗi (hiển thị tối đa 20):\n  - {message}"
            )
        print(f"[Dataset] CẢNH BÁO: có {error_count} lỗi:\n  - {message}")
    return expected_k, stats


def _letterbox(
    image: Image.Image,
    size: int,
    padding_color: Tuple[int, int, int],
) -> Tuple[Image.Image, float, int, int]:
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    right = size - new_w - left
    bottom = size - new_h - top
    return ImageOps.expand(image, (left, top, right, bottom), fill=padding_color), scale, left, top


def _flip_targets(
    boxes: torch.Tensor,
    landmarks: torch.Tensor,
    image_size: int,
    permutation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flipped_boxes = boxes.clone()
    flipped_landmarks = landmarks.clone()
    flipped_boxes[:, 0] = image_size - boxes[:, 2]
    flipped_boxes[:, 2] = image_size - boxes[:, 0]
    flipped_landmarks[..., 0] = image_size - landmarks[..., 0]
    return flipped_boxes, flipped_landmarks[:, permutation, :]


def _transform_points(points: np.ndarray, matrix: np.ndarray, eps: float) -> np.ndarray:
    """Áp dụng homography 3x3 lên mảng điểm có shape [...,2]."""
    if points.size == 0:
        return points.astype(np.float32, copy=True)
    shape = points.shape
    flat = points.reshape(-1, 2).astype(np.float32)
    homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float32)))
    warped = homogeneous @ matrix.T
    w = warped[:, 2:3]
    w = np.where(np.abs(w) < eps, np.where(w < 0, -eps, eps), w)
    return (warped[:, :2] / w).reshape(shape).astype(np.float32)

def _bbox_perimeters(boxes: np.ndarray, samples_per_edge: int) -> np.ndarray:
    """Biểu diễn bbox bằng các điểm trên chu vi để hỗ trợ méo phi tuyến."""
    if boxes.size == 0:
        return np.empty((0, samples_per_edge * 4, 2), dtype=np.float32)

    t = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)
    x1, y1, x2, y2 = boxes.astype(np.float32).T
    lerp_x = x1[:, None] + (x2 - x1)[:, None] * t
    lerp_y = y1[:, None] + (y2 - y1)[:, None] * t

    top = np.stack((lerp_x, np.broadcast_to(y1[:, None], lerp_x.shape)), axis=-1)
    right = np.stack((np.broadcast_to(x2[:, None], lerp_y.shape), lerp_y), axis=-1)
    bottom = np.stack((lerp_x[:, ::-1], np.broadcast_to(y2[:, None], lerp_x.shape)), axis=-1)
    left = np.stack((np.broadcast_to(x1[:, None], lerp_y.shape), lerp_y[:, ::-1]), axis=-1)
    return np.concatenate((top, right, bottom, left), axis=1)


def _boxes_from_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate((points.min(axis=1), points.max(axis=1)), axis=-1).astype(np.float32)


def _sample_affine(cfg, size: int, rng: np.random.Generator) -> np.ndarray:
    center = size * 0.5
    sx = rng.uniform(*cfg.affine_scale_x)
    sy = rng.uniform(*cfg.affine_scale_y)
    angle = np.deg2rad(rng.uniform(-cfg.affine_rotate_degrees, cfg.affine_rotate_degrees))
    shear_x = np.tan(np.deg2rad(rng.uniform(-cfg.affine_shear_degrees, cfg.affine_shear_degrees)))
    shear_y = np.tan(np.deg2rad(rng.uniform(-cfg.affine_shear_degrees, cfg.affine_shear_degrees)))
    tx = rng.uniform(-cfg.affine_translate, cfg.affine_translate) * size
    ty = rng.uniform(-cfg.affine_translate, cfg.affine_translate) * size

    to_origin = np.array([[1, 0, -center], [0, 1, -center], [0, 0, 1]], np.float32)
    scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float32)
    shear = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], np.float32)
    rotate = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        np.float32,
    )
    restore = np.array([[1, 0, center + tx], [0, 1, center + ty], [0, 0, 1]], np.float32)
    return restore @ rotate @ shear @ scale @ to_origin


def _sample_perspective(cfg, size: int, rng: np.random.Generator) -> np.ndarray:
    magnitude = cfg.perspective_scale * size
    src = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    dst = src + rng.uniform(-magnitude, magnitude, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _distort_points_radial(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32, copy=True)

    shape = points.shape
    flat = points.reshape(-1, 2).astype(np.float32)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    normalized = np.column_stack(((flat[:, 0] - cx) / fx, (flat[:, 1] - cy) / fy))
    object_points = np.column_stack((normalized, np.ones(len(flat), np.float32))).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros(3, np.float32),
        np.zeros(3, np.float32),
        camera_matrix,
        distortion,
    )
    return projected.reshape(shape).astype(np.float32)


def _apply_radial_distortion(
    image: np.ndarray,
    boxes: np.ndarray,
    landmarks: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = image.shape[0]
    jitter = cfg.radial_center_jitter * size
    cx = size * 0.5 + rng.uniform(-jitter, jitter)
    cy = size * 0.5 + rng.uniform(-jitter, jitter)
    camera_matrix = np.array(
        [[size, 0, cx], [0, size, cy], [0, 0, 1]],
        dtype=np.float32,
    )
    distortion = np.array(
        [rng.uniform(*cfg.radial_k1), rng.uniform(*cfg.radial_k2), 0, 0, 0],
        dtype=np.float32,
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
    )
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    source_map = cv2.undistortPoints(
        distorted_pixels,
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(size, size, 2)
    image = cv2.remap(
        image,
        source_map[..., 0],
        source_map[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=cfg.padding_color,
    )

    perimeter = _distort_points_radial(
        _bbox_perimeters(boxes, cfg.bbox_perimeter_samples),
        camera_matrix,
        distortion,
    )
    return (
        image,
        _boxes_from_points(perimeter),
        _distort_points_radial(landmarks, camera_matrix, distortion),
    )


def _sanitize_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    landmarks: np.ndarray,
    size: int,
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if boxes.size == 0:
        k = landmarks.shape[1] if landmarks.ndim == 3 else 0
        return (
            np.empty((0, 4), np.float32),
            np.empty((0,), np.int64),
            np.empty((0, k, 2), np.float32),
            np.empty((0,), np.bool_),
        )

    unclipped = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, size)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    original_area = np.maximum(
        (unclipped[:, 2] - unclipped[:, 0]) * (unclipped[:, 3] - unclipped[:, 1]),
        cfg.geometry_eps,
    )
    visibility = np.maximum(width, 0) * np.maximum(height, 0) / original_area
    min_bbox_visibility = (
        cfg.min_face_visibility
        if cfg.min_bbox_visibility is None
        else cfg.min_bbox_visibility
    )

    keep = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(landmarks).all(axis=(1, 2))
        & (width >= cfg.min_box_size_px)
        & (height >= cfg.min_box_size_px)
        & (visibility >= min_bbox_visibility)
    )
    boxes, labels, landmarks = boxes[keep], labels[keep], landmarks[keep]
    if len(boxes) == 0:
        return boxes, labels, landmarks, np.empty((0,), np.bool_)

    inside = (
        (landmarks[..., 0] >= 0)
        & (landmarks[..., 0] <= size)
        & (landmarks[..., 1] >= 0)
        & (landmarks[..., 1] <= size)
    )
    min_lmk_visibility = (
        cfg.min_face_visibility
        if cfg.min_landmark_visibility is None
        else cfg.min_landmark_visibility
    )
    valid = inside.all(axis=1) if cfg.require_all_landmarks_inside else inside.mean(axis=1) >= min_lmk_visibility

    min_lmk_size = cfg.min_landmark_face_size_px
    if min_lmk_size > 0:
        valid &= (width[keep] >= min_lmk_size) & (height[keep] >= min_lmk_size)
    return boxes, labels, landmarks, valid.astype(np.bool_)


def apply_geometric_augmentation(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    landmarks: torch.Tensor,
    cfg,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Áp dụng affine, perspective và radial distortion đồng bộ."""
    valid = torch.ones(len(boxes), dtype=torch.bool)
    if not cfg.augment or cfg.geometric_probability <= 0:
        return image, boxes, labels, landmarks, valid, "none"

    rng = np.random.default_rng(
        int(torch.randint(0, cfg.random_seed_upper_bound, ()).item())
    )
    if rng.random() >= cfg.geometric_probability:
        return image, boxes, labels, landmarks, valid, "none"

    image_np = np.asarray(image, dtype=np.uint8)
    boxes_np = boxes.numpy().astype(np.float32, copy=True)
    labels_np = labels.numpy().astype(np.int64, copy=True)
    landmarks_np = landmarks.numpy().astype(np.float32, copy=True)
    applied: List[str] = []

    matrix = np.eye(3, dtype=np.float32)
    if rng.random() < cfg.affine_probability:
        matrix = _sample_affine(cfg, cfg.image_size, rng) @ matrix
        applied.append("affine")
    if rng.random() < cfg.perspective_probability:
        matrix = _sample_perspective(cfg, cfg.image_size, rng) @ matrix
        applied.append("perspective")

    if not np.allclose(matrix, np.eye(3, dtype=np.float32)):
        image_np = cv2.warpPerspective(
            image_np,
            matrix,
            (cfg.image_size, cfg.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=cfg.padding_color,
        )
        boxes_np = _boxes_from_points(
            _transform_points(
                _bbox_perimeters(boxes_np, cfg.bbox_perimeter_samples),
                matrix,
                cfg.geometry_eps,
            )
        )
        landmarks_np = _transform_points(landmarks_np, matrix, cfg.geometry_eps)

    if rng.random() < cfg.radial_distortion_probability:
        image_np, boxes_np, landmarks_np = _apply_radial_distortion(
            image_np, boxes_np, landmarks_np, cfg, rng
        )
        applied.append("radial")

    boxes_np, labels_np, landmarks_np, valid_np = _sanitize_targets(
        boxes_np, labels_np, landmarks_np, cfg.image_size, cfg
    )
    if len(boxes_np) == 0 and not cfg.allow_empty_targets:
        return image, boxes, labels, landmarks, valid, "fallback_original"

    return (
        Image.fromarray(image_np),
        torch.from_numpy(boxes_np),
        torch.from_numpy(labels_np),
        torch.from_numpy(landmarks_np),
        torch.from_numpy(valid_np),
        "+".join(applied) if applied else "none",
    )

class FaceLandmarkDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.images_dir = os.path.join(cfg.root_dir, cfg.images_dir_name)
        self.jsonl_path = os.path.join(cfg.root_dir, cfg.jsonl_name)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh {self.images_dir}.")
        if not os.path.isfile(self.jsonl_path):
            raise FileNotFoundError(f"Không tìm thấy annotation {self.jsonl_path}.")

        self.offsets = _build_or_load_offsets(self.jsonl_path, cfg.index_cache_dir)
        self.num_landmarks, self.validation_stats = _validate_landmark_schema(
            self.jsonl_path, self.offsets, cfg
        )
        if cfg.horizontal_flip_mode != "off" and self.num_landmarks != MEDIAPIPE_NUM_LANDMARKS:
            raise ValueError(
                f"Horizontal flip chỉ hỗ trợ MediaPipe {MEDIAPIPE_NUM_LANDMARKS}, "
                f"dataset có K={self.num_landmarks}."
            )

        self.flip_permutation = torch.tensor(
            MEDIAPIPE_478_FLIP_PERMUTATION,
            dtype=torch.long,
        )
        self._file_handle: Optional[BinaryIO] = None
        self._file_handle_pid: Optional[int] = None
        self.photometric = (
            T.ColorJitter(cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)
            if cfg.augment and any((cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue))
            else None
        )
        self.to_tensor = T.Compose((T.ToImage(), T.ToDtype(torch.float32, scale=True)))
        print(
            f"[Dataset] {cfg.root_dir}: {len(self.offsets)} record, "
            f"{self.validation_stats['faces']} face, K={self.num_landmarks}."
        )

    def __len__(self) -> int:
        return len(self.offsets) * (2 if self.cfg.horizontal_flip_mode == "paired" else 1)

    def _resolve_index_and_flip(self, index: int) -> Tuple[int, bool]:
        mode = self.cfg.horizontal_flip_mode
        if mode == "paired":
            return index // 2, bool(index % 2)
        if mode == "random":
            return index, bool(torch.rand(()) < self.cfg.horizontal_flip_probability)
        return index, False

    def _get_file(self) -> BinaryIO:
        pid = os.getpid()
        if self._file_handle is None or self._file_handle.closed or self._file_handle_pid != pid:
            self.close()
            self._file_handle = open(self.jsonl_path, "rb")
            self._file_handle_pid = pid
        return self._file_handle

    def close(self) -> None:
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
        self._file_handle = None
        self._file_handle_pid = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        state["_file_handle_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        record_index, should_flip = self._resolve_index_and_flip(index)
        record = _read_record(self._get_file(), int(self.offsets[record_index]))

        image_path = os.path.join(self.images_dir, record["file_name"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        orig_w, orig_h = image.size
        image, scale, pad_x, pad_y = _letterbox(
            image,
            self.cfg.image_size,
            self.cfg.padding_color,
        )

        boxes, landmarks = [], []
        for face_idx, face in enumerate(record.get("faces", [])):
            points = face["landmarks_normalized"]
            if len(points) != self.num_landmarks:
                raise ValueError(
                    f"{record['file_name']} face {face_idx}: K={len(points)} "
                    f"khác {self.num_landmarks}."
                )

            bbox = face["bounding_box_normalized"]
            box = np.array(
                [
                    float(bbox["xmin"]) * orig_w * scale + pad_x,
                    float(bbox["ymin"]) * orig_h * scale + pad_y,
                    float(bbox["xmax"]) * orig_w * scale + pad_x,
                    float(bbox["ymax"]) * orig_h * scale + pad_y,
                ],
                dtype=np.float32,
            )
            box[[0, 2]] = np.clip(box[[0, 2]], 0, self.cfg.image_size)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, self.cfg.image_size)
            if box[2] - box[0] < self.cfg.min_box_size_px or box[3] - box[1] < self.cfg.min_box_size_px:
                raise ValueError(f"{record['file_name']} face {face_idx}: bbox quá nhỏ sau letterbox.")

            point_array = np.asarray(
                [[float(p["x"]) * orig_w * scale + pad_x,
                  float(p["y"]) * orig_h * scale + pad_y] for p in points],
                dtype=np.float32,
            )
            boxes.append(box)
            landmarks.append(point_array)

        boxes_t = torch.as_tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
        labels_t = torch.zeros(len(boxes_t), dtype=torch.long)
        landmarks_t = torch.as_tensor(
            np.asarray(landmarks, dtype=np.float32).reshape(-1, self.num_landmarks, 2)
        )

        if should_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes_t, landmarks_t = _flip_targets(
                boxes_t,
                landmarks_t,
                self.cfg.image_size,
                self.flip_permutation,
            )

        image, boxes_t, labels_t, landmarks_t, valid_t, augmentation = (
            apply_geometric_augmentation(image, boxes_t, labels_t, landmarks_t, self.cfg)
        )
        if self.photometric is not None:
            image = self.photometric(image)

        return {
            "image": self.to_tensor(image),
            "boxes": boxes_t.float(),
            "labels": labels_t.long(),
            "landmarks": landmarks_t.float(),
            "landmarks_valid": valid_t.bool(),
            "file_name": record["file_name"],
            "orig_size": torch.tensor((orig_w, orig_h), dtype=torch.long),
            "was_flipped": should_flip,
            "geometric_aug": augmentation,
        }

def face_landmark_collate(batch: List[dict]) -> dict:
    target_keys = ("boxes", "labels", "landmarks", "landmarks_valid")
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "targets": [{key: sample[key] for key in target_keys} for sample in batch],
        "file_name": [sample["file_name"] for sample in batch],
        "orig_size": torch.stack([sample["orig_size"] for sample in batch]),
        "was_flipped": torch.tensor([sample["was_flipped"] for sample in batch]),
        "geometric_aug": [sample["geometric_aug"] for sample in batch],
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
        kwargs = {
            "dataset": self.dataset,
            "batch_size": cfg.batch_size,
            "shuffle": cfg.shuffle,
            "num_workers": cfg.num_workers,
            "collate_fn": face_landmark_collate,
            "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
            "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
            "drop_last": cfg.drop_last,
        }
        if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return DataLoader(**kwargs)

# {
#     "image": Tensor[B, 3, H, W],     # float32, giá trị trong [0, 1].
#                                       # Ảnh đã letterbox, geometric augmentation,
#                                       # photometric augmentation và chuyển sang tensor.
#                                       # Thường H = W = cfg.image_size, ví dụ 480.

#     "targets": List[Dict],            # Danh sách dài B.
#                                       # Mỗi phần tử tương ứng với một ảnh và có cấu trúc:
#                                       #
#                                       # {
#                                       #   "boxes": Tensor[Ni, 4],           float32,
#                                       #            bbox dạng xyxy theo pixel trên ảnh H×W;
#                                       #
#                                       #   "labels": Tensor[Ni],             int64,
#                                       #             nhãn class của từng face;
#                                       #
#                                       #   "landmarks": Tensor[Ni, K, 2],    float32,
#                                       #                tọa độ (x, y) theo pixel trên ảnh H×W;
#                                       #
#                                       #   "landmarks_valid": Tensor[Ni],    bool,
#                                       #                      True nếu landmark của face đó
#                                       #                      được phép tham gia landmark loss.
#                                       # }
#                                       #
#                                       # Ni có thể khác nhau giữa các ảnh nên targets
#                                       # không được stack thành một tensor duy nhất.

#     "file_name": List[str],           # Danh sách dài B, chứa tên file ảnh gốc.

#     "orig_size": Tensor[B, 2],        # int64, kích thước ảnh trước letterbox/augmentation.
#                                       # Thứ tự hiện tại là [orig_width, orig_height].

#     "was_flipped": Tensor[B],         # bool, True nếu sample đã được horizontal flip.
#                                       # Với MediaPipe 478, landmark trái/phải cũng đã
#                                       # được remap semantic tương ứng.

#     "geometric_aug": List[str],       # Danh sách dài B, mô tả geometric augmentation:
#                                       # "none"
#                                       # "affine"
#                                       # "perspective"
#                                       # "radial"
#                                       # "affine+perspective"
#                                       # "affine+perspective+radial"
#                                       # "fallback_original"
# }
