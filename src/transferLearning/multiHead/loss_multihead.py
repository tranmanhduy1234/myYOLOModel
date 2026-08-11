"""Loss cho các mini-detector mắt và miệng.

Mỗi specialist vẫn học đủ detection objective cũ (class, bbox, DFL và
landmark anchor-offset). Ngoài ra, loss này thêm một ràng buộc hình học liên
tục, bất biến theo scale:

* mắt: Eye Aspect Ratio (EAR) của hai cặp mí và margin trạng thái
  nhắm/mở;
* miệng: tỷ lệ độ mở của ba cặp môi trong.

Các topology dưới đây được khai báo bằng index MediaPipe *toàn cục*, sau đó
được ánh xạ sang slot local của từng ``RegionHeadSpec``. Không có index 478
nào được dùng trực tiếp để truy cập output K=21/K=40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.transferLearning.loss_lmk import FaceLandmarkDetectionLoss
from src.transferLearning.multiHead.model_multihead import (
    LEFT_EYE,
    MOUTH,
    RIGHT_EYE,
    SPECIALIST_NAMES,
    RegionHeadSpec,
)


@dataclass(frozen=True)
class RegionGeometryLossConfig:
    """Hệ số loss hình học; đặt gain bằng 0 để tắt riêng một loại."""

    eye_aperture_gain: float = 8.0
    eye_state_gain: float = 20.0
    eye_closed_threshold: float = 0.15
    eye_open_threshold: float = 0.20
    mouth_aperture_gain: float = 6.0
    smooth_l1_beta: float = 0.02
    # Khoang cach toi thieu theo pixel crop khi chuan hoa aperture.
    distance_eps: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.eye_aperture_gain,
            self.eye_state_gain,
            self.mouth_aperture_gain,
        ) < 0:
            raise ValueError('Geometry gain không được âm.')
        if not 0 < self.eye_closed_threshold < self.eye_open_threshold:
            raise ValueError(
                'Cần 0 < eye_closed_threshold < eye_open_threshold.'
            )
        if self.smooth_l1_beta <= 0 or self.distance_eps <= 0:
            raise ValueError('smooth_l1_beta và distance_eps phải > 0.')


# (góc ngoài, mí trên ngoài, mí trên trong, góc trong, mí dưới trong,
# mí dưới ngoài). Đây là index MediaPipe global, chỉ dùng để tạo local map.
_EYE_EAR_GLOBAL = {
    LEFT_EYE: (33, 160, 158, 133, 153, 144),
    RIGHT_EYE: (362, 385, 387, 263, 373, 380),
}

# Chiều rộng miệng và ba cặp môi trong (trên, dưới), index global.
_MOUTH_WIDTH_GLOBAL = (78, 308)
_MOUTH_OPENING_PAIRS_GLOBAL = ((13, 14), (82, 87), (312, 317))


def _global_indices_to_local(
    spec: RegionHeadSpec,
    global_indices: Sequence[int],
) -> tuple[int, ...]:
    """Ánh xạ topology MediaPipe global sang đúng slot output local."""
    global_to_local = {
        global_index: local_index
        for local_index, global_index in enumerate(spec.global_landmark_indices)
    }
    missing = [index for index in global_indices if index not in global_to_local]
    if missing:
        raise ValueError(
            f'{spec.name} thiếu landmark global {missing} cần cho geometry loss.'
        )
    return tuple(global_to_local[index] for index in global_indices)


class RegionLandmarkDetectionLoss(FaceLandmarkDetectionLoss):
    """Detection loss cũ cộng geometry loss riêng cho một vùng.

    Contract trả về vẫn là ``(total_loss, items)``. Toàn bộ key cũ trong
    ``items`` được giữ nguyên; các key ``geometry`` và ``*/geometry*`` chỉ là
    thông tin bổ sung để theo dõi chi tiết.
    """

    def __init__(
        self,
        spec: RegionHeadSpec,
        face_cfg,
        geometry_cfg: RegionGeometryLossConfig | None = None,
    ) -> None:
        super().__init__(face_cfg)
        if spec.name not in SPECIALIST_NAMES:
            raise ValueError(f'Region không hỗ trợ: {spec.name!r}.')
        if self.num_landmarks != spec.num_landmarks:
            raise ValueError(
                f'{spec.name}: FaceLmkConfig có K={self.num_landmarks}, '
                f'nhưng spec cần K={spec.num_landmarks}.'
            )
        self.spec = spec
        self.geometry_cfg = geometry_cfg or RegionGeometryLossConfig()
        self._geometry_calls: list[dict[str, float]] = []

        if spec.name in _EYE_EAR_GLOBAL:
            self._eye_ear_local = _global_indices_to_local(
                spec,
                _EYE_EAR_GLOBAL[spec.name],
            )
            self._mouth_width_local = ()
            self._mouth_pairs_local = ()
        elif spec.name == MOUTH:
            self._eye_ear_local = ()
            self._mouth_width_local = _global_indices_to_local(
                spec,
                _MOUTH_WIDTH_GLOBAL,
            )
            self._mouth_pairs_local = tuple(
                _global_indices_to_local(spec, pair)
                for pair in _MOUTH_OPENING_PAIRS_GLOBAL
            )
        else:  # pragma: no cover - spec validation above is exhaustive.
            raise ValueError(f'Chưa định nghĩa geometry cho {spec.name!r}.')

    def _geometry_reference_width(
        self,
        target_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """Do rong GT on dinh dung chung cho prediction va target.

        Khong dung do rong *prediction* lam mau so: luc khoi tao cac diem du
        doan gan trung nhau, mau so do gan 0 va tung lam geometry loss tang
        den hang chuc nghin trong log thuc te.
        """
        if self.spec.name in _EYE_EAR_GLOBAL:
            p1, _, _, p4, _, _ = self._eye_ear_local
            endpoints = (p1, p4)
        else:
            endpoints = self._mouth_width_local
        start, end = endpoints
        return torch.linalg.vector_norm(
            target_landmarks[:, start] - target_landmarks[:, end],
            dim=-1,
        ).detach().clamp_min(self.geometry_cfg.distance_eps)

    def _geometry_values(
        self,
        landmarks: torch.Tensor,
        reference_width: torch.Tensor,
    ) -> torch.Tensor:
        """Tra aperture ratio voi mau so lay tu GT, shape [N,1] hoac [N,3]."""
        if reference_width.ndim != 1 or reference_width.shape[0] != landmarks.shape[0]:
            raise ValueError('reference_width phai co shape [N].')
        if self.spec.name in _EYE_EAR_GLOBAL:
            _, p2, p3, _, p5, p6 = self._eye_ear_local
            outer = torch.linalg.vector_norm(
                landmarks[:, p2] - landmarks[:, p6], dim=-1
            )
            inner = torch.linalg.vector_norm(
                landmarks[:, p3] - landmarks[:, p5], dim=-1
            )
            return ((outer + inner) / (2.0 * reference_width)).unsqueeze(-1)

        openings = [
            torch.linalg.vector_norm(
                landmarks[:, upper] - landmarks[:, lower], dim=-1
            ) / reference_width
            for upper, lower in self._mouth_pairs_local
        ]
        return torch.stack(openings, dim=-1)

    def _aperture_gain(self) -> float:
        if self.spec.name in (LEFT_EYE, RIGHT_EYE):
            return self.geometry_cfg.eye_aperture_gain
        return self.geometry_cfg.mouth_aperture_gain

    def _eye_state_loss(
        self,
        predicted_ear: torch.Tensor,
        target_ear: torch.Tensor,
        assignment_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Margin loss phạt mạnh khi nhầm trạng thái nhắm/mở.

        EAR trong khoảng chuyển tiếp không bị ép nhãn rời rạc; nó
        vẫn được học qua aperture loss liên tục.
        """
        closed = target_ear <= self.geometry_cfg.eye_closed_threshold
        opened = target_ear >= self.geometry_cfg.eye_open_threshold
        state_valid = closed | opened
        closed_violation = F.relu(
            predicted_ear - self.geometry_cfg.eye_closed_threshold
        )
        open_violation = F.relu(
            self.geometry_cfg.eye_open_threshold - predicted_ear
        )
        violation = torch.where(
            closed,
            closed_violation,
            torch.where(opened, open_violation, torch.zeros_like(predicted_ear)),
        )
        per_value = F.smooth_l1_loss(
            violation,
            torch.zeros_like(violation),
            beta=self.geometry_cfg.smooth_l1_beta,
            reduction='none',
        )
        state_weight = assignment_weight * state_valid.to(predicted_ear.dtype)
        return (per_value * state_weight).sum() / state_weight.sum().clamp_min(
            self.cfg.loss_normalizer_eps
        )

    def _landmark_loss(
        self,
        lmk_raw,
        anchors,
        strides,
        target_bboxes_pixel,
        target_scores,
        gt_landmarks,
        gt_lmk_valid,
        target_gt_idx,
        fg_mask,
    ):
        base_loss, num_positive = super()._landmark_loss(
            lmk_raw,
            anchors,
            strides,
            target_bboxes_pixel,
            target_scores,
            gt_landmarks,
            gt_lmk_valid,
            target_gt_idx,
            fg_mask,
        )
        target_landmarks, _, pos_b, pos_a = self._gather_landmark_targets(
            gt_landmarks,
            gt_lmk_valid,
            target_gt_idx,
            fg_mask,
        )
        aperture_gain = self._aperture_gain()
        state_gain = (
            self.geometry_cfg.eye_state_gain
            if self.spec.name in (LEFT_EYE, RIGHT_EYE)
            else 0.0
        )
        if pos_b.numel() == 0 or (aperture_gain == 0 and state_gain == 0):
            aperture_raw = lmk_raw.sum() * 0.0
            state_raw = lmk_raw.sum() * 0.0
        else:
            pred_offsets = (
                lmk_raw.transpose(1, 2)[pos_b, pos_a]
                .reshape(-1, self.num_landmarks, 2)
            )
            selected_anchors = anchors[pos_a].to(dtype=pred_offsets.dtype)
            selected_strides = strides[pos_a].to(dtype=pred_offsets.dtype)
            pred_landmarks = (
                selected_anchors.unsqueeze(1) + pred_offsets
            ) * selected_strides.unsqueeze(1)

            reference_width = self._geometry_reference_width(target_landmarks)
            pred_geometry = self._geometry_values(
                pred_landmarks,
                reference_width,
            )
            target_geometry = self._geometry_values(
                target_landmarks,
                reference_width,
            )
            per_value = F.smooth_l1_loss(
                pred_geometry,
                target_geometry,
                beta=self.geometry_cfg.smooth_l1_beta,
                reduction='none',
            )
            assignment_weight = target_scores.sum(-1)[pos_b, pos_a].view(-1, 1)
            normalizer = assignment_weight.sum() * per_value.shape[1]
            aperture_raw = (per_value * assignment_weight).sum() / (
                normalizer.clamp_min(self.cfg.loss_normalizer_eps)
            )
            if state_gain > 0:
                state_raw = self._eye_state_loss(
                    pred_geometry,
                    target_geometry,
                    assignment_weight,
                )
            else:
                state_raw = lmk_raw.sum() * 0.0

        # FaceLandmarkDetectionLoss.forward nhân lmk_loss với lmk_gain. Chia
        # ở đây để gain geometry có đúng ý nghĩa trên total loss cuối cùng.
        aperture_contribution = aperture_raw * aperture_gain
        state_contribution = state_raw * state_gain
        geometry_contribution = aperture_contribution + state_contribution
        combined_loss = base_loss + geometry_contribution / self.lmk_gain
        self._geometry_calls.append({
            'base_lmk': float(base_loss.detach().item()),
            'aperture_raw': float(aperture_raw.detach().item()),
            'state_raw': float(state_raw.detach().item()),
            'aperture_contribution': float(
                aperture_contribution.detach().item()
            ),
            'state_contribution': float(state_contribution.detach().item()),
            'contribution': float(geometry_contribution.detach().item()),
        })
        return combined_loss, num_positive

    def forward(self, preds, targets):
        if not isinstance(targets, (list, tuple)):
            raise TypeError(
                f'{self.spec.name}: targets phải là list/tuple Dict theo batch.'
            )
        self._geometry_calls = []
        total, items = super().forward(preds, targets)
        if len(self._geometry_calls) != 2:
            raise RuntimeError(
                f'{self.spec.name}: cần đúng 2 geometry call cho o2m/o2o, '
                f'nhận {len(self._geometry_calls)}.'
            )

        # Thứ tự gọi trong FaceLandmarkDetectionLoss là o2m rồi o2o.
        for branch, values in zip(('o2m', 'o2o'), self._geometry_calls):
            items[f'{branch}/lmk_base'] = values['base_lmk']
            items[f'{branch}/geometry_aperture_raw'] = values['aperture_raw']
            items[f'{branch}/geometry_state_raw'] = values['state_raw']
            items[f'{branch}/geometry_aperture'] = values[
                'aperture_contribution'
            ]
            items[f'{branch}/geometry_state'] = values['state_contribution']
            items[f'{branch}/geometry'] = values['contribution']
        items['geometry_aperture'] = (
            self.o2m_weight * items['o2m/geometry_aperture']
            + self.o2o_weight * items['o2o/geometry_aperture']
        )
        items['geometry_state'] = (
            self.o2m_weight * items['o2m/geometry_state']
            + self.o2o_weight * items['o2o/geometry_state']
        )
        geometry_total = (
            self.o2m_weight * items['o2m/geometry']
            + self.o2o_weight * items['o2o/geometry']
        )
        items['geometry'] = geometry_total
        items['loss_detection'] = items['loss'] - geometry_total
        return total, items


class MultiHeadRegionLoss(nn.Module):
    """Bộ loss của cả ba vùng; trainer chọn đúng loss theo stage hiện tại."""

    def __init__(
        self,
        specialists: Mapping[str, nn.Module],
        specs: Mapping[str, RegionHeadSpec],
        geometry_cfg: RegionGeometryLossConfig | None = None,
    ) -> None:
        super().__init__()
        if set(specialists) != set(SPECIALIST_NAMES):
            raise ValueError(
                f'specialists phải chứa đúng {SPECIALIST_NAMES}, '
                f'nhận {tuple(specialists)}.'
            )
        self.criteria = nn.ModuleDict({
            name: RegionLandmarkDetectionLoss(
                specs[name],
                specialists[name].face_cfg,
                geometry_cfg,
            )
            for name in SPECIALIST_NAMES
        })

    def forward(self, name: str, preds, targets):
        if name not in self.criteria:
            raise KeyError(f'Specialist không hợp lệ: {name!r}.')
        return self.criteria[name](preds, targets)


__all__ = (
    'RegionGeometryLossConfig',
    'RegionLandmarkDetectionLoss',
    'MultiHeadRegionLoss',
)
