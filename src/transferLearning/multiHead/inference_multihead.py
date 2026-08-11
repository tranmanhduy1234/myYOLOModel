"""Inference cho mô hình face-landmark multi-head chuyên biệt.

Luồng suy luận cố ý giữ HEAD4 làm nguồn kết quả an toàn:

1. ``FaceLandmarkInferencer`` chạy đúng một lượt trên toàn ảnh bằng checkpoint
   478 điểm đã fine-tune.
2. Từ landmark thô của HEAD4, ``build_region_crop`` tạo crop mắt trái, mắt
   phải và miệng bằng đúng phép biến đổi dùng trong dataloader multi-head.
3. Ba mini-detector chạy theo batch (mỗi specialist tối đa một forward/frame).
4. Chỉ các index thuộc specialist được hiệu chỉnh khi candidate đạt cả ngưỡng
   confidence lẫn kiểm tra hình học. Kết quả được trộn với HEAD4 theo từng vùng
   để tránh specialist mắt kéo landmark đi quá xa; mọi bất thường giữ HEAD4.

File này không dùng CLI. Chỉnh cấu hình trực tiếp trong khối ``__main__`` để
chạy demo ảnh hoặc camera; toàn bộ phần hiển thị dùng Matplotlib.
"""

from __future__ import annotations

import os
import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch

from src.transferLearning.config_lmk import TrainConfig
from src.transferLearning.inference import FaceLandmarkInferencer
from src.transferLearning.multiHead.data_multihead import (
    RegionCropConfig,
    build_region_crop,
)
from src.transferLearning.multiHead.model_multihead import (
    REGION_HEAD_SPECS,
    SPECIALIST_NAMES,
    RegionHeadSpec,
    SpecializedMultiHeadFaceLandmark,
)


ImageInput = Union[str, os.PathLike, PIL.Image.Image, np.ndarray]
SpecialistWeights = Union[
    str,
    os.PathLike,
    Mapping[str, Union[str, os.PathLike]],
]

# Topology MediaPipe theo index global 478. Moi contour duoc tach rieng de
# khong bao gio noi diem cuoi cua vong nay sang diem dau cua vong khac.
_LEFT_EYE_CONNECTIONS = (
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158),
    (158, 157), (157, 173), (173, 133),
)
_RIGHT_EYE_CONNECTIONS = (
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373),
    (373, 390), (390, 249), (249, 263),
    (362, 398), (398, 384), (384, 385), (385, 386), (386, 387),
    (387, 388), (388, 466), (466, 263),
)
_LEFT_IRIS_CONNECTIONS = ((469, 470), (470, 471), (471, 472), (472, 469))
_RIGHT_IRIS_CONNECTIONS = ((474, 475), (475, 476), (476, 477), (477, 474))
_LIP_CONNECTIONS = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
)
_EYEBROW_CONNECTIONS = (
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
)
_RENDER_GROUPS = (
    ('left_eye', _LEFT_EYE_CONNECTIONS, (30, 220, 255)),
    ('right_eye', _RIGHT_EYE_CONNECTIONS, (255, 190, 30)),
    ('left_iris', _LEFT_IRIS_CONNECTIONS, (80, 255, 255)),
    ('right_iris', _RIGHT_IRIS_CONNECTIONS, (255, 235, 80)),
    ('mouth', _LIP_CONNECTIONS, (255, 70, 170)),
    ('eyebrow', _EYEBROW_CONNECTIONS, (80, 255, 120)),
)


def _state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash canonical de doi chieu dung trong so HEAD4, khong chi kien truc."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        if tensor.numel():
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def _is_tensor_state_dict(value: Any) -> bool:
    """Nhận diện state_dict thuần, không nhầm với checkpoint container."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(tensor, (torch.Tensor, torch.nn.Parameter)) for tensor in value.values())
    )


def _canonical_metadata(value: Any) -> Any:
    """Chuẩn hóa list/tuple để so metadata checkpoint ổn định."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_metadata(item) for item in value)
    return value


def _normalize_specialist_state(
    raw_state: Mapping[str, torch.Tensor],
    name: str,
    expected_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Bỏ prefix đóng gói nhưng vẫn bắt buộc khớp state mini-detector 100%."""
    expected_keys = set(expected_state)
    normalized: Dict[str, torch.Tensor] = {}
    region_markers = (
        f'specialists.{name}.',
        f'specialist.{name}.',
        f'{name}.',
    )

    for original_key, value in raw_state.items():
        clean = original_key
        while clean.startswith(('module.', 'model.')):
            clean = clean.split('.', 1)[1]

        matched_region_prefix = False
        for marker in region_markers:
            if clean.startswith(marker):
                clean = clean[len(marker):]
                matched_region_prefix = True
                break

        # Khi nhận full multi-head state, bỏ qua HEAD4 và hai specialist khác.
        if not matched_region_prefix and clean not in expected_keys:
            continue
        if clean in normalized:
            raise KeyError(
                f'Trùng key {clean!r} sau khi chuẩn hóa checkpoint {name!r}.'
            )
        normalized[clean] = value.detach() if isinstance(value, torch.nn.Parameter) else value

    missing = sorted(expected_keys - set(normalized))
    unexpected = sorted(set(normalized) - expected_keys)
    bad_shapes = sorted(
        key
        for key in expected_keys.intersection(normalized)
        if tuple(normalized[key].shape) != tuple(expected_state[key].shape)
    )
    if missing or unexpected or bad_shapes:
        raise RuntimeError(
            f'Checkpoint specialist {name!r} không tương thích strict: '
            f'missing={len(missing)}, unexpected={len(unexpected)}, '
            f'sai shape={len(bad_shapes)}. Ví dụ missing={missing[:5]}, '
            f'unexpected={unexpected[:5]}, sai shape={bad_shapes[:5]}.'
        )
    return normalized


def _find_specialist_state(payload: Any, name: str) -> Mapping[str, torch.Tensor]:
    """Trích state một specialist từ checkpoint combined hoặc per-region."""
    if isinstance(payload, torch.nn.Module):
        payload = payload.state_dict()
    if _is_tensor_state_dict(payload):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(
            f'Checkpoint {name!r} phải là dict/nn.Module, nhận {type(payload).__name__}.'
        )

    # Checkpoint combined: chỉ chứa ba state specialist, không chứa HEAD4.
    for container_key in ('specialists', 'specialist_states', 'specialist_state_dicts'):
        container = payload.get(container_key)
        if isinstance(container, Mapping) and name in container:
            return _find_specialist_state(container[name], name)

    # Checkpoint riêng của một stage/specialist.
    saved_name = payload.get('specialist_name', payload.get('active_specialist'))
    if saved_name is not None and saved_name != name:
        raise ValueError(
            f'Checkpoint khai báo specialist={saved_name!r}, nhưng đang nạp {name!r}.'
        )
    for container_key in (
        'ema',
        'specialist_state',
        'specialist_state_dict',
        'model',
        'state_dict',
    ):
        container = payload.get(container_key)
        if container is not None:
            try:
                return _find_specialist_state(container, name)
            except (KeyError, TypeError, ValueError):
                continue

    raise KeyError(
        f'Không tìm thấy state_dict của specialist {name!r} trong checkpoint. '
        'Các key hiện có: '
        f'{list(payload)[:15]}.'
    )


class MultiHeadLandmarkInferencer:
    """HEAD4 toàn mặt + ba specialist mắt trái, mắt phải và miệng."""

    def __init__(
        self,
        global_weights_path: Union[str, os.PathLike],
        specialist_weights: Optional[SpecialistWeights] = None,
        *,
        cfg: Optional[TrainConfig] = None,
        crop_cfg: Any = None,
        device: Optional[str] = None,
        require_checkpoint_metadata: bool = True,
        global_confidence: Optional[float] = None,
        global_iou_threshold: Optional[float] = None,
        specialist_confidence: Union[float, Mapping[str, float]] = 0.25,
        specialist_blend: Optional[Union[float, Mapping[str, float]]] = None,
        run_specialist_heads: bool = True,
        max_specialist_faces: int = 1,
        minimum_inside_ratio: float = 0.80,
        minimum_scale_ratio: float = 0.35,
        maximum_scale_ratio: float = 2.25,
        maximum_median_shift_ratio: float = 0.60,
        crop_tolerance_ratio: float = 0.15,
        render_all_landmarks: bool = False,
    ) -> None:
        if max_specialist_faces <= 0:
            raise ValueError('max_specialist_faces phải > 0.')
        if not 0.0 <= minimum_inside_ratio <= 1.0:
            raise ValueError('minimum_inside_ratio phải nằm trong [0, 1].')
        if not 0.0 < minimum_scale_ratio <= maximum_scale_ratio:
            raise ValueError(
                'Cần 0 < minimum_scale_ratio <= maximum_scale_ratio.'
            )
        if maximum_median_shift_ratio <= 0.0:
            raise ValueError('maximum_median_shift_ratio phải > 0.')
        if crop_tolerance_ratio < 0.0:
            raise ValueError('crop_tolerance_ratio phải >= 0.')

        resolved_global_weights = os.path.abspath(
            os.path.expanduser(os.fspath(global_weights_path))
        )
        inference_cfg = cfg or TrainConfig(require_pretrained_trunk=False)
        self.global_inferencer = FaceLandmarkInferencer(
            weights_path=resolved_global_weights,
            cfg=inference_cfg,
            device=device,
            conf_threshold=global_confidence,
            iou_threshold=global_iou_threshold,
            enhance_details=False,
            refine_eye_mouth=False,
            max_refine_faces=1,
        )
        self.device = self.global_inferencer.device
        self.global_weights_path = resolved_global_weights
        self._global_state_sha256 = _state_sha256(
            self.global_inferencer.model.state_dict()
        )
        self._global_reference_verified = False
        self.require_checkpoint_metadata = bool(require_checkpoint_metadata)
        if crop_cfg is not None and not isinstance(crop_cfg, RegionCropConfig):
            raise TypeError('crop_cfg phải là RegionCropConfig hoặc None.')
        self.crop_cfg: Optional[RegionCropConfig] = crop_cfg
        self.max_specialist_faces = int(max_specialist_faces)
        self.minimum_inside_ratio = float(minimum_inside_ratio)
        self.minimum_scale_ratio = float(minimum_scale_ratio)
        self.maximum_scale_ratio = float(maximum_scale_ratio)
        self.maximum_median_shift_ratio = float(maximum_median_shift_ratio)
        self.crop_tolerance_ratio = float(crop_tolerance_ratio)
        self.specialist_confidence = self._resolve_confidences(
            specialist_confidence
        )
        self.specialist_blend = self._resolve_blend_weights(specialist_blend)
        self.run_specialist_heads = bool(run_specialist_heads)
        self.render_all_landmarks = bool(render_all_landmarks)

        self.model: Optional[SpecializedMultiHeadFaceLandmark] = None
        if self.run_specialist_heads:
            if specialist_weights is None:
                raise ValueError(
                    'run_specialist_heads=True thì phải truyền specialist_weights.'
                )
            self.model = SpecializedMultiHeadFaceLandmark(
                self.global_inferencer.model
            ).to(self.device)
            self.model.activate_specialist(None)
            self.model.eval()
            self._load_specialists(specialist_weights)
        else:
            # HEAD4-only không tạo ba CNN và không đọc checkpoint multi-head.
            self.crop_cfg = self.crop_cfg or RegionCropConfig()
            print('[MultiHead] Chế độ HEAD4-only: đã bỏ qua specialist.')

    @staticmethod
    def _resolve_confidences(
        value: Union[float, Mapping[str, float]],
    ) -> Dict[str, float]:
        if isinstance(value, Mapping):
            missing = set(SPECIALIST_NAMES) - set(value)
            unexpected = set(value) - set(SPECIALIST_NAMES)
            if missing or unexpected:
                raise KeyError(
                    'specialist_confidence phải chứa đúng '
                    f'{SPECIALIST_NAMES}; missing={sorted(missing)}, '
                    f'unexpected={sorted(unexpected)}.'
                )
            resolved = {name: float(value[name]) for name in SPECIALIST_NAMES}
        else:
            resolved = {name: float(value) for name in SPECIALIST_NAMES}
        if any(not 0.0 <= threshold <= 1.0 for threshold in resolved.values()):
            raise ValueError('Mọi specialist confidence phải nằm trong [0, 1].')
        return resolved

    @staticmethod
    def _resolve_blend_weights(
        value: Optional[Union[float, Mapping[str, float]]],
    ) -> Dict[str, float]:
        """Muc specialist thay doi HEAD4; eye co tinh bao thu hon mouth."""
        if value is None:
            resolved = {
                'left_eye': 0.25,
                'right_eye': 0.25,
                'mouth': 0.75,
            }
        elif isinstance(value, Mapping):
            missing = set(SPECIALIST_NAMES) - set(value)
            unexpected = set(value) - set(SPECIALIST_NAMES)
            if missing or unexpected:
                raise KeyError(
                    'specialist_blend phai chua dung ba region; '
                    f'missing={sorted(missing)}, unexpected={sorted(unexpected)}.'
                )
            resolved = {name: float(value[name]) for name in SPECIALIST_NAMES}
        else:
            resolved = {name: float(value) for name in SPECIALIST_NAMES}
        if any(not 0.0 <= weight <= 1.0 for weight in resolved.values()):
            raise ValueError('Moi specialist_blend phai nam trong [0,1].')
        return resolved

    def _validate_checkpoint_metadata(self, payload: Any, path: str) -> None:
        """Nếu checkpoint có signature thì bắt buộc trùng kiến trúc hiện tại."""
        if self.model is None:
            raise RuntimeError('Specialist model chưa được khởi tạo.')
        if not isinstance(payload, Mapping):
            if self.require_checkpoint_metadata:
                raise TypeError(
                    'Checkpoint specialist strict phai la mapping co metadata.'
                )
            return
        checkpoint_kind = payload.get('kind')
        if checkpoint_kind is None and self.require_checkpoint_metadata:
            raise ValueError(
                'Checkpoint specialist khong co kind/metadata de xac minh. '
                'Hay dung multihead_final.pt do trainer moi tao; chi dat '
                'require_checkpoint_metadata=False khi chu dong nap legacy state.'
            )
        if checkpoint_kind is not None and checkpoint_kind != 'multihead_specialists':
            raise ValueError(
                f'Checkpoint {path} có kind={checkpoint_kind!r}, '
                "cần 'multihead_specialists'."
            )
        if checkpoint_kind == 'multihead_specialists':
            required_keys = {
                'format_version',
                'global_checkpoint',
                'architecture_signature',
                'crop_config',
                'specs',
                'specialists',
                'completed_stages',
                'training_state',
            }
            missing_keys = sorted(required_keys - set(payload))
            if missing_keys:
                raise KeyError(
                    'Checkpoint multi-head duoc nhan dien nhung thieu metadata '
                    f'bat buoc: {missing_keys}.'
                )
        format_version = payload.get('format_version')
        if format_version is not None and int(format_version) != 1:
            raise ValueError(
                f'Checkpoint specialist format_version={format_version!r}; '
                'inference hiện hỗ trợ version=1.'
            )
        if checkpoint_kind == 'multihead_specialists':
            completed = tuple(payload.get('completed_stages', ()))
            training_state = payload.get('training_state') or {}
            active_stage = (
                training_state.get('active_stage')
                if isinstance(training_state, Mapping)
                else None
            )
            if completed != tuple(SPECIALIST_NAMES) or active_stage is not None:
                raise ValueError(
                    'Checkpoint specialist chua train xong ca ba stage: '
                    f'completed={completed}, active={active_stage!r}. '
                    'Hay dung multihead_final.pt (hoac last.pt sau khi fit xong).'
                )

        saved_specs = payload.get('specs')
        if saved_specs is not None:
            if not isinstance(saved_specs, Mapping) or set(saved_specs) != set(SPECIALIST_NAMES):
                raise ValueError(
                    'Checkpoint phải chứa specs cho đúng ba vùng '
                    f'{SPECIALIST_NAMES}.'
                )
            for name in SPECIALIST_NAMES:
                spec = REGION_HEAD_SPECS[name]
                expected_spec = {
                    'name': name,
                    'input_size': spec.input_size,
                    'num_landmarks': spec.num_landmarks,
                    'global_landmark_indices': tuple(spec.global_landmark_indices),
                    'crop_anchor_indices': tuple(spec.crop_anchor_indices),
                }
                saved_spec = saved_specs[name]
                if not isinstance(saved_spec, Mapping):
                    raise TypeError(f'specs[{name!r}] phải là mapping.')
                comparable = {
                    key: saved_spec.get(key)
                    for key in expected_spec
                }
                if _canonical_metadata(comparable) != _canonical_metadata(expected_spec):
                    raise ValueError(
                        f'Spec {name!r} trong checkpoint không khớp model hiện tại: '
                        f'checkpoint={comparable}, current={expected_spec}.'
                    )

        global_metadata = payload.get('global_checkpoint')
        if isinstance(global_metadata, Mapping):
            saved_global_signature = global_metadata.get('model_signature')
            current_global_signature = (
                self.global_inferencer.train_cfg.checkpoint_model_signature()
            )
            if (
                saved_global_signature is not None
                and _canonical_metadata(saved_global_signature)
                != _canonical_metadata(current_global_signature)
            ):
                raise ValueError(
                    'HEAD4 đang nạp không cùng kiến trúc với HEAD4 dùng khi train '
                    f'specialist: {path}.'
                )
            saved_state_sha256 = global_metadata.get('state_sha256')
            if (
                saved_state_sha256 is not None
                and saved_state_sha256 != self._global_state_sha256
            ):
                raise ValueError(
                    'Checkpoint specialist duoc train voi bo trong so HEAD4 '
                    'khac file global dang nap. Tu choi ghep de tranh crop sai.'
                )
            if saved_state_sha256 is not None and not self._global_reference_verified:
                print(
                    '[MultiHead] HEAD4 state SHA256 khop checkpoint train: '
                    f'{self._global_state_sha256[:16]}...'
                )
                self._global_reference_verified = True

        saved_signature = payload.get(
            'architecture_signature',
            payload.get('multihead_signature'),
        )
        if saved_signature is None:
            return
        expected_signature = self.model.architecture_signature()
        if _canonical_metadata(saved_signature) != _canonical_metadata(expected_signature):
            raise ValueError(
                'Kiến trúc trong checkpoint specialist không khớp model hiện tại: '
                f'{path}.'
            )

    def _load_checkpoint(self, path: Union[str, os.PathLike]) -> Any:
        resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f'Không tìm thấy checkpoint specialist: {resolved}'
            )
        payload = torch.load(
            resolved,
            map_location=self.device,
            weights_only=True,
        )
        self._validate_checkpoint_metadata(payload, resolved)
        return payload

    def _load_specialists(self, weights: SpecialistWeights) -> None:
        if self.model is None:
            raise RuntimeError('Không thể load specialist ở chế độ HEAD4-only.')
        if isinstance(weights, Mapping):
            received = set(weights)
            expected = set(SPECIALIST_NAMES)
            if received != expected:
                raise KeyError(
                    f'Cần checkpoint cho đúng {sorted(expected)}, '
                    f'nhận {sorted(received)}.'
                )
            payloads = {
                name: self._load_checkpoint(weights[name])
                for name in SPECIALIST_NAMES
            }
        else:
            payload = self._load_checkpoint(weights)
            payloads = {name: payload for name in SPECIALIST_NAMES}

        saved_crop_configs = []
        for payload in payloads.values():
            if isinstance(payload, Mapping) and payload.get('crop_config') is not None:
                saved_crop_configs.append(payload['crop_config'])
        if saved_crop_configs:
            first_saved = saved_crop_configs[0]
            if any(
                _canonical_metadata(value) != _canonical_metadata(first_saved)
                for value in saved_crop_configs[1:]
            ):
                raise ValueError('Các checkpoint dùng RegionCropConfig khác nhau.')
            if not isinstance(first_saved, Mapping):
                raise TypeError('checkpoint["crop_config"] phải là mapping.')
            checkpoint_crop_cfg = RegionCropConfig(**dict(first_saved))
            if (
                self.crop_cfg is not None
                and _canonical_metadata(asdict(self.crop_cfg))
                != _canonical_metadata(asdict(checkpoint_crop_cfg))
            ):
                raise ValueError(
                    'crop_cfg inference khác crop_config dùng khi train specialist.'
                )
            self.crop_cfg = checkpoint_crop_cfg
        elif self.crop_cfg is None:
            self.crop_cfg = RegionCropConfig()

        for name in SPECIALIST_NAMES:
            specialist = self.model.specialists[name]
            expected_state = specialist.state_dict()
            raw_state = _find_specialist_state(payloads[name], name)
            normalized = _normalize_specialist_state(
                raw_state,
                name,
                expected_state,
            )
            specialist.load_state_dict(normalized, strict=True)
            specialist.to(self.device).eval()
            print(
                f'[MultiHead] Đã nạp strict {name}: '
                f'{len(normalized)} tensor, '
                f'{sum(t.numel() for t in normalized.values()):,} phần tử.'
            )
        self.model.activate_specialist(None)
        self.model.eval()

    @staticmethod
    def _to_crop_tensor(crop_image_rgb: np.ndarray) -> torch.Tensor:
        crop = np.asarray(crop_image_rgb)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(
                f'RegionCrop.image_rgb phải có shape [H,W,3], nhận {crop.shape}.'
            )
        if crop.size == 0 or not np.isfinite(crop).all():
            raise ValueError('Crop rỗng hoặc chứa NaN/Inf.')
        if np.issubdtype(crop.dtype, np.floating):
            max_value = float(crop.max())
            if max_value <= 1.0:
                crop = crop * 255.0
        crop = np.clip(crop, 0, 255).astype(np.uint8, copy=False)
        return (
            torch.from_numpy(np.ascontiguousarray(crop))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )

    def _candidate_geometry_is_valid(
        self,
        local_points: np.ndarray,
        mapped_points: np.ndarray,
        coarse_points: np.ndarray,
        spec: RegionHeadSpec,
        image_shape: Tuple[int, int],
    ) -> bool:
        """Chặn candidate bám sai cấu trúc trước khi thay landmark HEAD4."""
        if (
            local_points.shape != (spec.num_landmarks, 2)
            or mapped_points.shape != coarse_points.shape
            or not np.isfinite(local_points).all()
            or not np.isfinite(mapped_points).all()
            or not np.isfinite(coarse_points).all()
        ):
            return False

        tolerance = spec.input_size * self.crop_tolerance_ratio
        local_inside = (
            (local_points[:, 0] >= -tolerance)
            & (local_points[:, 0] <= spec.input_size + tolerance)
            & (local_points[:, 1] >= -tolerance)
            & (local_points[:, 1] <= spec.input_size + tolerance)
        )
        if float(local_inside.mean()) < self.minimum_inside_ratio:
            return False

        image_h, image_w = image_shape
        image_tolerance = max(image_h, image_w) * 0.01
        original_inside = (
            (mapped_points[:, 0] >= -image_tolerance)
            & (mapped_points[:, 0] <= image_w + image_tolerance)
            & (mapped_points[:, 1] >= -image_tolerance)
            & (mapped_points[:, 1] <= image_h + image_tolerance)
        )
        if float(original_inside.mean()) < self.minimum_inside_ratio:
            return False

        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(
                spec.global_landmark_indices
            )
        }
        anchor_local_indices = np.asarray(
            [global_to_local[index] for index in spec.crop_anchor_indices],
            dtype=np.int64,
        )
        # Kiem tra dich chuyen/scale tren cung tap diem neo da tao ROI. Iris
        # outlier cua HEAD4 khong duoc phep lam specialist tot bi reject.
        coarse_geometry = coarse_points[anchor_local_indices]
        mapped_geometry = mapped_points[anchor_local_indices]
        coarse_span = np.ptp(coarse_geometry, axis=0)
        mapped_span = np.ptp(mapped_geometry, axis=0)
        coarse_diagonal = max(float(np.linalg.norm(coarse_span)), 1.0)
        mapped_diagonal = float(np.linalg.norm(mapped_span))
        scale_ratio = mapped_diagonal / coarse_diagonal
        if not self.minimum_scale_ratio <= scale_ratio <= self.maximum_scale_ratio:
            return False

        median_shift = float(np.median(
            np.linalg.norm(mapped_geometry - coarse_geometry, axis=1)
        ))
        return median_shift / coarse_diagonal <= self.maximum_median_shift_ratio

    def _select_candidate(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch_index: int,
        spec: RegionHeadSpec,
        transform: Any,
        coarse_points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], float, str]:
        branch = predictions['o2o']
        scores = torch.sigmoid(branch['cls'][batch_index]).squeeze(-1)
        boxes = branch['box'][batch_index]
        landmarks = branch['lmk'][batch_index]
        if landmarks.ndim != 3 or landmarks.shape[-2:] != (spec.num_landmarks, 2):
            raise RuntimeError(
                f'{spec.name} trả landmark shape {tuple(landmarks.shape)}, '
                f'cần [A,{spec.num_landmarks},2].'
            )

        finite = (
            torch.isfinite(scores)
            & torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(landmarks).all(dim=(1, 2))
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        eligible = torch.nonzero(
            finite & (scores >= self.specialist_confidence[spec.name]),
            as_tuple=False,
        ).squeeze(1)
        best_observed = (
            float(scores[finite].max().item()) if bool(finite.any()) else float('nan')
        )
        if eligible.numel() == 0:
            return None, best_observed, 'low_confidence'

        ordered = eligible[torch.argsort(scores[eligible], descending=True)]
        for candidate_index in ordered.tolist():
            local_points = (
                landmarks[candidate_index]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            mapped_points = np.asarray(
                transform.map_points_to_original(local_points),
                dtype=np.float32,
            )
            if self._candidate_geometry_is_valid(
                local_points,
                mapped_points,
                coarse_points,
                spec,
                image_shape,
            ):
                return (
                    mapped_points,
                    float(scores[candidate_index].item()),
                    'used',
                )
        return None, best_observed, 'geometry_rejected'

    @staticmethod
    def _empty_usage(num_faces: int) -> Dict[str, Any]:
        return {
            'names': tuple(SPECIALIST_NAMES),
            'used': np.zeros((num_faces, len(SPECIALIST_NAMES)), dtype=np.bool_),
            'scores': np.full(
                (num_faces, len(SPECIALIST_NAMES)),
                np.nan,
                dtype=np.float32,
            ),
            'reasons': np.full(
                (num_faces, len(SPECIALIST_NAMES)),
                'not_selected',
                dtype=object,
            ),
        }

    @torch.inference_mode()
    def _apply_specialists(
        self,
        image_rgb: np.ndarray,
        global_detections: Mapping[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Batch theo specialist rồi ghép có fallback về landmark HEAD4."""
        if self.model is None:
            raise RuntimeError('Specialist model chưa được khởi tạo.')
        refined = {
            'boxes': np.asarray(global_detections['boxes']).copy(),
            'scores': np.asarray(global_detections['scores']).copy(),
            'landmarks': np.asarray(global_detections['landmarks']).copy(),
        }
        num_faces = len(refined['scores'])
        usage = self._empty_usage(num_faces)
        if num_faces == 0:
            return refined, usage

        # Crop luôn lấy từ kết quả HEAD4 ban đầu; ba vùng không ảnh hưởng nhau.
        coarse_landmarks = refined['landmarks'].copy()
        face_order = np.argsort(-refined['scores'])[:self.max_specialist_faces]
        image_shape = tuple(image_rgb.shape[:2])

        for specialist_column, name in enumerate(SPECIALIST_NAMES):
            spec = REGION_HEAD_SPECS[name]
            crop_tensors = []
            transforms = []
            valid_face_indices = []
            for face_index in face_order.tolist():
                try:
                    region_crop = build_region_crop(
                        image_rgb,
                        coarse_landmarks[face_index],
                        spec,
                        self.crop_cfg,
                        training=False,
                    )
                    crop_tensor = self._to_crop_tensor(region_crop.image_rgb)
                    if tuple(crop_tensor.shape[-2:]) != (
                        spec.input_size,
                        spec.input_size,
                    ):
                        raise RuntimeError(
                            f'Crop {name} có size {tuple(crop_tensor.shape[-2:])}, '
                            f'cần {(spec.input_size, spec.input_size)}.'
                        )
                except (RuntimeError, TypeError, ValueError, IndexError):
                    usage['reasons'][face_index, specialist_column] = 'crop_failed'
                    continue
                crop_tensors.append(crop_tensor)
                transforms.append(region_crop.transform)
                valid_face_indices.append(face_index)

            if not crop_tensors:
                continue
            batch = torch.stack(crop_tensors, dim=0).to(
                self.device,
                non_blocking=True,
            )
            predictions = self.model.forward_specialist(
                name,
                batch,
                return_o2m=False,
            )
            for batch_index, face_index in enumerate(valid_face_indices):
                global_indices = np.asarray(
                    spec.global_landmark_indices,
                    dtype=np.int64,
                )
                replacement, score, reason = self._select_candidate(
                    predictions,
                    batch_index,
                    spec,
                    transforms[batch_index],
                    coarse_landmarks[face_index, global_indices],
                    image_shape,
                )
                usage['scores'][face_index, specialist_column] = score
                usage['reasons'][face_index, specialist_column] = reason
                if replacement is None:
                    continue
                image_h, image_w = image_shape
                coarse_region = coarse_landmarks[face_index, global_indices]
                blend = self.specialist_blend[name]
                replacement = (
                    (1.0 - blend) * coarse_region + blend * replacement
                ).astype(np.float32, copy=False)
                replacement[:, 0] = np.clip(
                    replacement[:, 0], 0.0, max(image_w - 1, 0)
                )
                replacement[:, 1] = np.clip(
                    replacement[:, 1], 0.0, max(image_h - 1, 0)
                )
                # Gán đúng K index; box, score và landmark ngoài vùng bất biến.
                refined['landmarks'][face_index, global_indices] = replacement
                usage['used'][face_index, specialist_column] = True

        return refined, usage

    @staticmethod
    def usage_summary(usage: Mapping[str, Any]) -> Dict[str, int]:
        used = np.asarray(usage['used'], dtype=np.bool_)
        return {
            name: int(used[:, column].sum())
            for column, name in enumerate(usage['names'])
        }

    @classmethod
    def format_usage(cls, usage: Mapping[str, Any]) -> str:
        summary = cls.usage_summary(usage)
        return ' | '.join(f'{name}: {count}' for name, count in summary.items())

    def _draw_detections(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Vẽ contour MediaPipe đúng topology, không nối tuần tự theo index.

        Mỗi đoạn trong ``_RENDER_GROUPS`` là một cạnh MediaPipe global rõ
        ràng. Vì vậy không có cạnh giả nối mắt với iris, hai vòng môi hoặc hai
        lông mày như khi nối toàn bộ danh sách điểm theo thứ tự tensor.
        """
        output = np.ascontiguousarray(image_rgb).copy()
        image_h, image_w = output.shape[:2]
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        landmarks = np.asarray(landmarks)
        if landmarks.ndim != 3 or landmarks.shape[-2:] != (478, 2):
            raise ValueError(
                'landmarks render phải có shape [N,478,2], nhận '
                f'{landmarks.shape}.'
            )
        if len(boxes) != len(scores) or len(scores) != len(landmarks):
            raise ValueError('Số box, score và bộ landmark phải bằng nhau.')

        for face_index, (box, score, face_landmarks) in enumerate(
            zip(boxes, scores, landmarks)
        ):
            if not np.isfinite(face_landmarks).all():
                continue
            if np.asarray(box).shape == (4,) and np.isfinite(box).all():
                x1, y1, x2, y2 = np.rint(box).astype(np.int32).tolist()
                x1 = int(np.clip(x1, 0, max(image_w - 1, 0)))
                x2 = int(np.clip(x2, 0, max(image_w - 1, 0)))
                y1 = int(np.clip(y1, 0, max(image_h - 1, 0)))
                y2 = int(np.clip(y2, 0, max(image_h - 1, 0)))
                cv2.rectangle(output, (x1, y1), (x2, y2), (40, 255, 70), 1)
                label_y = max(y1 - 5, 12)
                cv2.putText(
                    output,
                    f'face {face_index + 1}: {float(score):.2f}',
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (40, 255, 70),
                    1,
                    cv2.LINE_AA,
                )

            rendered_indices = set()
            for _, connections, color in _RENDER_GROUPS:
                for start_index, end_index in connections:
                    start = face_landmarks[start_index]
                    end = face_landmarks[end_index]
                    start_xy = tuple(np.rint(start).astype(np.int32).tolist())
                    end_xy = tuple(np.rint(end).astype(np.int32).tolist())
                    cv2.line(
                        output,
                        start_xy,
                        end_xy,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    rendered_indices.update((start_index, end_index))

            # Iris center không nằm trên vòng contour nhưng vẫn cần hiển thị.
            rendered_indices.update((468, 473))
            for landmark_index in sorted(rendered_indices):
                point = tuple(
                    np.rint(face_landmarks[landmark_index])
                    .astype(np.int32)
                    .tolist()
                )
                cv2.circle(output, point, 1, (255, 255, 255), -1, cv2.LINE_AA)

            if self.render_all_landmarks:
                for point_xy in face_landmarks:
                    point = tuple(np.rint(point_xy).astype(np.int32).tolist())
                    cv2.circle(output, point, 1, (145, 145, 145), -1, cv2.LINE_AA)
        return output

    @torch.inference_mode()
    def predict(
        self,
        image_input: ImageInput,
        *,
        show: bool = True,
        global_confidence: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Trả ảnh render đúng kích thước gốc cùng detection đã ghép."""
        confidence = (
            self.global_inferencer.conf_threshold
            if global_confidence is None
            else float(global_confidence)
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('global_confidence phải nằm trong [0, 1].')

        if isinstance(image_input, os.PathLike):
            image_input = os.fspath(image_input)
        image_rgb = self.global_inferencer._prepare_image(image_input)
        # Chính xác một lượt full-image HEAD4; enhancement/refinement cũ đã tắt.
        global_detections, letterbox_info = (
            self.global_inferencer._infer_single_pass(image_rgb, confidence)
        )
        if self.run_specialist_heads:
            detections, usage = self._apply_specialists(
                image_rgb,
                global_detections,
            )
        else:
            detections = {
                'boxes': np.asarray(global_detections['boxes']).copy(),
                'scores': np.asarray(global_detections['scores']).copy(),
                'landmarks': np.asarray(global_detections['landmarks']).copy(),
            }
            usage = self._empty_usage(len(detections['scores']))
            usage['reasons'][:] = 'disabled'
        detections['specialist_usage'] = usage
        detections['inference_mode'] = (
            'multihead' if self.run_specialist_heads else 'head4_only'
        )

        output_rgb = self._draw_detections(
            image_rgb,
            detections['boxes'],
            detections['scores'],
            detections['landmarks'],
        )
        if output_rgb.shape != image_rgb.shape:
            raise RuntimeError(
                f'Output {output_rgb.shape} khác kích thước ảnh gốc {image_rgb.shape}.'
            )
        if show:
            self.global_inferencer.show_matplotlib(
                output_rgb,
                f'{detections["inference_mode"]} face landmark | '
                f'Faces: {len(detections["scores"])} | '
                f'{self.format_usage(usage)}',
            )
        return output_rgb, detections, letterbox_info


def demo_image(
    inferencer: MultiHeadLandmarkInferencer,
    image_input: ImageInput,
    *,
    global_confidence: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Demo một ảnh, hiển thị duy nhất bằng Matplotlib."""
    output, detections, info = inferencer.predict(
        image_input,
        show=True,
        global_confidence=global_confidence,
    )
    usage = detections['specialist_usage']
    print(
        f'[Image] output={output.shape}, faces={len(detections["scores"])}, '
        f'{inferencer.format_usage(usage)}, '
        f'scale={info["scale"]:.6f}, '
        f'padding=({info["left"]}, {info["top"]}, '
        f'{info["right"]}, {info["bottom"]})'
    )
    return output, detections, info

def demo_camera(
    inferencer: MultiHeadLandmarkInferencer,
    *,
    camera_index: int = 0,
    global_confidence: Optional[float] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> None:
    """Camera realtime bằng Matplotlib; đóng figure hoặc Ctrl+C để dừng."""
    import matplotlib.pyplot as plt

    capture = cv2.VideoCapture(camera_index)
    if frame_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
    if frame_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Không thể mở camera index={camera_index}.')

    plt.ion()
    fig, axis = plt.subplots(
        figsize=inferencer.global_inferencer.train_cfg.inference_figure_size
    )
    axis.axis('off')
    artist = None
    plt.show(block=False)

    try:
        while plt.fignum_exists(fig.number):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError('Không đọc được frame từ camera.')
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            output, detections, _ = inferencer.predict(
                PIL.Image.fromarray(frame_rgb),
                show=False,
                global_confidence=global_confidence,
            )
            elapsed = max(time.perf_counter() - started, 1e-9)
            usage_text = inferencer.format_usage(
                detections['specialist_usage']
            )

            if artist is None:
                artist = axis.imshow(output)
            else:
                artist.set_data(output)
            axis.set_title(
                f'Camera {camera_index} | Faces: {len(detections["scores"])} '
                f'| FPS: {1.0 / elapsed:.1f}\n{usage_text}'
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print('\n[Camera] Đã dừng bởi người dùng.')
    finally:
        capture.release()
        plt.ioff()
        plt.close(fig)


def main(
    *,
    mode: str,
    image_input: Optional[ImageInput],
    global_weights_path: Union[str, os.PathLike],
    specialist_weights: Optional[SpecialistWeights] = None,
    device: Optional[str] = None,
    require_checkpoint_metadata: bool = True,
    global_confidence: Optional[float] = None,
    specialist_confidence: Union[float, Mapping[str, float]] = 0.25,
    specialist_blend: Optional[Union[float, Mapping[str, float]]] = None,
    run_specialist_heads: bool = True,
    max_specialist_faces: int = 1,
    render_all_landmarks: bool = False,
    camera_index: int = 0,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
    """Chạy trực tiếp trong Python, không dùng argparse/CLI."""
    selected_mode = mode.lower().strip()
    if selected_mode not in {'image', 'camera'}:
        raise ValueError("mode phải là 'image' hoặc 'camera'.")
    if camera_width is not None and camera_width <= 0:
        raise ValueError('camera_width phải > 0.')
    if camera_height is not None and camera_height <= 0:
        raise ValueError('camera_height phải > 0.')

    config = TrainConfig(require_pretrained_trunk=False)
    inferencer = MultiHeadLandmarkInferencer(
        global_weights_path=global_weights_path,
        specialist_weights=specialist_weights,
        cfg=config,
        device=device,
        require_checkpoint_metadata=require_checkpoint_metadata,
        global_confidence=global_confidence,
        specialist_confidence=specialist_confidence,
        specialist_blend=specialist_blend,
        run_specialist_heads=run_specialist_heads,
        max_specialist_faces=max_specialist_faces,
        render_all_landmarks=render_all_landmarks,
    )
    if selected_mode == 'image':
        resolved_image = image_input or config.demo_model_image_path
        return demo_image(
            inferencer,
            resolved_image,
            global_confidence=global_confidence,
        )
    demo_camera(
        inferencer,
        camera_index=camera_index,
        global_confidence=global_confidence,
        frame_width=camera_width,
        frame_height=camera_height,
    )
    return None


if __name__ == '__main__':
    # Chỉnh trực tiếp tại đây; không có CLI.
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    GLOBAL_BEST_PT = PROJECT_ROOT / 'checkpoints_face_lmk_finetune' / 'best.pt'
    SPECIALIST_BEST_PT = PROJECT_ROOT / 'checkpoints_multihead' / 'multihead_final.pt'

    main(
        mode='camera',  # 'image' hoặc 'camera'
        image_input=PROJECT_ROOT / 'src' / 'image_sketch' / 'demo' / 'image.png',
        global_weights_path=GLOBAL_BEST_PT,
        specialist_weights=SPECIALIST_BEST_PT,
        device='cuda',
        global_confidence=0.25,
        specialist_confidence={
            'left_eye': 0.25,
            'right_eye': 0.25,
            'mouth': 0.25,
        },
        # Eye blend thấp để HEAD mới chỉ tinh chỉnh HEAD4; mouth tin cậy hơn.
        specialist_blend={
            'left_eye': 0.25,
            'right_eye': 0.25,
            'mouth': 0.75,
        },
        # True: HEAD4 + ba specialist. False: chỉ HEAD4, không load specialist.
        run_specialist_heads=True,
        max_specialist_faces=1,
        render_all_landmarks=True,
        camera_index=0,
        camera_width=None,
        camera_height=None,
    )
