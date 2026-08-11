"""Kiến trúc multi-head chuyên biệt cho landmark mắt và miệng.

Thiết kế:
    - HEAD 4: ``FaceLmkDetector`` đã fine-tune, giữ nguyên toàn bộ trọng số.
    - HEAD 1: mini-detector cho mắt trái + iris (21 điểm).
    - HEAD 2: mini-detector cho mắt phải + iris (21 điểm).
    - HEAD 3: mini-detector cho môi (40 điểm).

Ba specialist là các CNN độc lập, nhẹ hơn nhiều model toàn mặt. Backbone,
neck và HEAD 4 cũ luôn frozen/eval; mỗi lần chỉ một mini-detector được train.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.backbone_neck import Backbone, PAFPN
from src.transferLearning.config_lmk import (
    MEDIAPIPE_LEFT_EYE,
    MEDIAPIPE_LEFT_IRIS,
    MEDIAPIPE_LIPS,
    MEDIAPIPE_NUM_LANDMARKS,
    MEDIAPIPE_RIGHT_EYE,
    MEDIAPIPE_RIGHT_IRIS,
    FaceLmkConfig,
)
from src.transferLearning.model_lmk import DetectHeadFaceLmk, FaceLmkDetector
from src.utils.init_weights import initialize_weights


FULL_FACE_INPUT_KEY = 'full_face'
LEFT_EYE = 'left_eye'
RIGHT_EYE = 'right_eye'
MOUTH = 'mouth'
SPECIALIST_NAMES = (LEFT_EYE, RIGHT_EYE, MOUTH)

# Van nhe hon model toan mat nhung du capacity cho bien dang mi/iris/moi.
# Ban cu chi ~0.39M tham so/region va plateau ca train/val quanh 6-7.
LITE_BACKBONE_WIDTHS = (24, 32, 48, 72, 96)
LITE_BACKBONE_DEPTHS = (1, 2, 2, 1)
LITE_NECK_DEPTH = 2
REGION_HEAD_MIN_CHANNELS = 32
REGION_LMK_HIDDEN_MAX_CHANNELS = 96


@dataclass(frozen=True)
class RegionHeadSpec:
    """Đặc tả input/output của một specialist mini-detector."""

    name: str
    global_landmark_indices: Tuple[int, ...]
    input_size: int
    crop_anchor_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in SPECIALIST_NAMES:
            raise ValueError(
                f'name phải thuộc {SPECIALIST_NAMES}, nhận {self.name!r}.'
            )
        if not self.global_landmark_indices:
            raise ValueError('global_landmark_indices không được rỗng.')
        if len(set(self.global_landmark_indices)) != len(self.global_landmark_indices):
            raise ValueError(f'{self.name} chứa landmark index bị trùng.')
        if min(self.global_landmark_indices) < 0:
            raise ValueError('Landmark index không được âm.')
        if max(self.global_landmark_indices) >= MEDIAPIPE_NUM_LANDMARKS:
            raise ValueError(
                f'Landmark index phải nhỏ hơn {MEDIAPIPE_NUM_LANDMARKS}.'
            )
        crop_anchors = tuple(
            self.crop_anchor_indices or self.global_landmark_indices
        )
        if len(set(crop_anchors)) != len(crop_anchors):
            raise ValueError(f'{self.name} chứa crop anchor bị trùng.')
        if not set(crop_anchors).issubset(self.global_landmark_indices):
            raise ValueError(
                f'crop_anchor_indices của {self.name} phải là tập con '
                'của global_landmark_indices.'
            )
        object.__setattr__(self, 'crop_anchor_indices', crop_anchors)
        if self.input_size <= 0 or self.input_size % 32 != 0:
            raise ValueError('input_size phải là số dương chia hết cho 32.')

    @property
    def num_landmarks(self) -> int:
        return len(self.global_landmark_indices)

    @property
    def local_landmark_indices(self) -> Tuple[int, ...]:
        """Index output cục bộ; không được nhầm với index MediaPipe 478."""
        return tuple(range(self.num_landmarks))


REGION_HEAD_SPECS: Dict[str, RegionHeadSpec] = {
    LEFT_EYE: RegionHeadSpec(
        name=LEFT_EYE,
        global_landmark_indices=MEDIAPIPE_LEFT_EYE + MEDIAPIPE_LEFT_IRIS,
        input_size=128,
        crop_anchor_indices=MEDIAPIPE_LEFT_EYE,
    ),
    RIGHT_EYE: RegionHeadSpec(
        name=RIGHT_EYE,
        global_landmark_indices=MEDIAPIPE_RIGHT_EYE + MEDIAPIPE_RIGHT_IRIS,
        input_size=128,
        crop_anchor_indices=MEDIAPIPE_RIGHT_EYE,
    ),
    MOUTH: RegionHeadSpec(
        name=MOUTH,
        global_landmark_indices=MEDIAPIPE_LIPS,
        input_size=160,
        crop_anchor_indices=MEDIAPIPE_LIPS,
    ),
}


def make_region_face_config(
    spec: RegionHeadSpec,
    global_cfg: FaceLmkConfig,
) -> FaceLmkConfig:
    """Tạo config detect cũ với số landmark cục bộ K=21 hoặc K=40.

    Specialist đã chỉ chứa một vùng nên point weight được để đồng đều. Không
    truyền index MediaPipe 478 vào loss có K nhỏ vì sẽ weight nhầm local slot.
    """
    return FaceLmkConfig(
        nc=global_cfg.nc,
        reg_max=global_cfg.reg_max,
        strides=tuple(global_cfg.strides),
        num_landmarks=spec.num_landmarks,
        landmark_encoding=global_cfg.landmark_encoding,
        cls_channel_divisor=global_cfg.cls_channel_divisor,
        reg_channel_divisor=global_cfg.reg_channel_divisor,
        lmk_channel_divisor=global_cfg.lmk_channel_divisor,
        head_min_channels=REGION_HEAD_MIN_CHANNELS,
        lmk_hidden_max_channels=REGION_LMK_HIDDEN_MAX_CHANNELS,
        cls_prior_probability=global_cfg.cls_prior_probability,
        stride_bias_expected_objects=global_cfg.stride_bias_expected_objects,
        reg_bias=global_cfg.reg_bias,
        lmk_weight_std=global_cfg.lmk_weight_std,
        anchor_offset=global_cfg.anchor_offset,
        lmk_scale_eps=global_cfg.lmk_scale_eps,
        loss_normalizer_eps=global_cfg.loss_normalizer_eps,
        box_gain=global_cfg.box_gain,
        cls_gain=global_cfg.cls_gain,
        dfl_gain=global_cfg.dfl_gain,
        lmk_gain=global_cfg.lmk_gain,
        lmk_smooth_l1_beta=global_cfg.lmk_smooth_l1_beta,
        eye_landmark_indices=(),
        mouth_landmark_indices=(),
        nose_tip_landmark_indices=(),
        eye_landmark_weight=1.0,
        mouth_landmark_weight=1.0,
        nose_tip_landmark_weight=1.0,
        topk_o2m=global_cfg.topk_o2m,
        topk_o2o=global_cfg.topk_o2o,
        alpha=global_cfg.alpha,
        beta=global_cfg.beta,
        o2m_weight=global_cfg.o2m_weight,
        o2o_weight=global_cfg.o2o_weight,
    )


class LiteRegionLandmarkDetector(nn.Module):
    """Mini CNN độc lập, giữ nguyên contract detection landmark hiện tại."""

    def __init__(
        self,
        spec: RegionHeadSpec,
        global_cfg: FaceLmkConfig,
    ):
        super().__init__()
        self.spec = spec
        self.face_cfg = make_region_face_config(spec, global_cfg)
        self.num_landmarks = spec.num_landmarks
        self.input_size = spec.input_size

        self.backbone = Backbone(
            w=LITE_BACKBONE_WIDTHS,
            n=LITE_BACKBONE_DEPTHS,
        )
        feature_channels = tuple(LITE_BACKBONE_WIDTHS[2:5])
        self.neck = PAFPN(
            chs=feature_channels,
            n=LITE_NECK_DEPTH,
        )
        self.head = DetectHeadFaceLmk(
            chs=feature_channels,
            cfg=self.face_cfg,
            image_size=self.input_size,
        )
        initialize_weights(self)

    def set_trainable(self, trainable: bool) -> None:
        """Mở/khóa mini-detector nhưng luôn giữ DFL là fixed projection."""
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
        self.head.dfl.requires_grad_(False)
        self.train(trainable)

    def forward(
        self,
        crops: torch.Tensor,
        return_o2m: bool = True,
    ):
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError(
                'Crop phải có shape [B, 3, H, W], nhận '
                f'{tuple(crops.shape)}.'
            )
        expected_shape = (self.input_size, self.input_size)
        if tuple(crops.shape[-2:]) != expected_shape:
            raise ValueError(
                f'{self.spec.name} cần crop {expected_shape}, nhận '
                f'{tuple(crops.shape[-2:])}.'
            )
        features = self.neck(*self.backbone(crops))
        return self.head(features, return_o2m=return_o2m)


class SpecializedMultiHeadFaceLandmark(nn.Module):
    """HEAD 4 frozen + ba mini-detector train tuần tự trên crop riêng."""

    def __init__(
        self,
        global_detector: FaceLmkDetector,
        specs: Optional[Mapping[str, RegionHeadSpec]] = None,
    ):
        super().__init__()
        global_num_landmarks = int(global_detector.head.num_landmarks)
        if global_num_landmarks != MEDIAPIPE_NUM_LANDMARKS:
            raise ValueError(
                'HEAD 4 phải là model MediaPipe 478 landmark, nhận '
                f'{global_num_landmarks}.'
            )

        self.global_detector = global_detector
        self.specs = dict(specs or REGION_HEAD_SPECS)
        if set(self.specs) != set(SPECIALIST_NAMES):
            raise ValueError(
                f'specs phải chứa đúng {SPECIALIST_NAMES}, nhận {tuple(self.specs)}.'
            )

        global_cfg = global_detector.cfg.face
        self.specialists = nn.ModuleDict({
            name: LiteRegionLandmarkDetector(spec, global_cfg)
            for name, spec in self.specs.items()
        })
        self._active_specialist: Optional[str] = None
        self._freeze_global_detector()
        self.activate_specialist(None)

    def _freeze_global_detector(self) -> None:
        """Khóa cả backbone, neck và HEAD 4 để giữ checkpoint fine-tune."""
        for parameter in self.global_detector.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.global_detector.eval()

    @property
    def active_specialist(self) -> Optional[str]:
        return self._active_specialist

    def activate_specialist(self, name: Optional[str]) -> None:
        """Chỉ mở gradient cho HEAD 1, 2 hoặc 3 được chọn."""
        if name is not None and name not in self.specialists:
            raise KeyError(
                f'Specialist không hợp lệ {name!r}; chọn {SPECIALIST_NAMES} hoặc None.'
            )
        self._active_specialist = name
        for specialist_name, specialist in self.specialists.items():
            specialist.set_trainable(specialist_name == name)
        self._freeze_global_detector()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_global_detector()
        for name, specialist in self.specialists.items():
            specialist.train(mode and name == self._active_specialist)
        return self

    def trainable_parameters(self):
        """Iterator dùng trực tiếp để tạo optimizer cho stage hiện tại."""
        if self._active_specialist is None:
            raise RuntimeError('Chưa activate specialist để train.')
        return (
            parameter
            for parameter in self.specialists[
                self._active_specialist
            ].parameters()
            if parameter.requires_grad
        )

    def forward_global(
        self,
        full_images: torch.Tensor,
        return_o2m: bool = True,
    ):
        """HEAD 4 chỉ forward tham chiếu/inference, không tạo gradient."""
        with torch.no_grad():
            return self.global_detector(
                full_images,
                return_o2m=return_o2m,
            )

    def forward_specialist(
        self,
        name: str,
        crops: torch.Tensor,
        return_o2m: bool = True,
    ):
        if name not in self.specialists:
            raise KeyError(f'Specialist không hợp lệ: {name!r}.')
        if self.training and self._active_specialist != name:
            raise RuntimeError(
                f'Đang train {self._active_specialist!r}, không được forward '
                f'specialist {name!r} trong cùng stage.'
            )
        return self.specialists[name](
            crops,
            return_o2m=return_o2m,
        )

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        return_o2m: bool = True,
    ) -> Dict[str, dict]:
        """Forward các phần được truyền; trainer chỉ truyền HEAD đang học."""
        unknown = set(inputs) - {FULL_FACE_INPUT_KEY, *SPECIALIST_NAMES}
        if unknown:
            raise KeyError(f'Input chứa key không hỗ trợ: {sorted(unknown)}.')
        if not inputs:
            raise ValueError('inputs không được rỗng.')

        outputs = {}
        if FULL_FACE_INPUT_KEY in inputs:
            outputs[FULL_FACE_INPUT_KEY] = self.forward_global(
                inputs[FULL_FACE_INPUT_KEY],
                return_o2m=return_o2m,
            )
        for name in SPECIALIST_NAMES:
            if name in inputs:
                outputs[name] = self.forward_specialist(
                    name,
                    inputs[name],
                    return_o2m=return_o2m,
                )
        return outputs

    def parameter_summary(self) -> Dict[str, int]:
        """Số parameter theo HEAD để kiểm soát ngân sách mô hình."""
        summary = {
            'head4_global_frozen': sum(
                parameter.numel()
                for parameter in self.global_detector.parameters()
            ),
        }
        summary.update({
            name: sum(
                parameter.numel()
                for parameter in specialist.parameters()
            )
            for name, specialist in self.specialists.items()
        })
        summary['specialists_total'] = sum(
            summary[name] for name in SPECIALIST_NAMES
        )
        summary['trainable_now'] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return summary

    def architecture_signature(self) -> dict:
        """Metadata bắt buộc để kiểm tra checkpoint specialist tương thích."""
        return {
            'format_version': 1,
            'global_model_signature': (
                self.global_detector.cfg.checkpoint_model_signature()
            ),
            'lite_backbone_widths': tuple(LITE_BACKBONE_WIDTHS),
            'lite_backbone_depths': tuple(LITE_BACKBONE_DEPTHS),
            'lite_neck_depth': LITE_NECK_DEPTH,
            'region_head_min_channels': REGION_HEAD_MIN_CHANNELS,
            'region_lmk_hidden_max_channels': REGION_LMK_HIDDEN_MAX_CHANNELS,
            'regions': {
                name: {
                    'input_size': spec.input_size,
                    'num_landmarks': spec.num_landmarks,
                    'global_landmark_indices': tuple(
                        spec.global_landmark_indices
                    ),
                    'crop_anchor_indices': tuple(spec.crop_anchor_indices),
                    'nc': self.specialists[name].face_cfg.nc,
                    'reg_max': self.specialists[name].face_cfg.reg_max,
                    'strides': tuple(
                        self.specialists[name].face_cfg.strides
                    ),
                    'cls_channel_divisor': (
                        self.specialists[name].face_cfg.cls_channel_divisor
                    ),
                    'reg_channel_divisor': (
                        self.specialists[name].face_cfg.reg_channel_divisor
                    ),
                    'lmk_channel_divisor': (
                        self.specialists[name].face_cfg.lmk_channel_divisor
                    ),
                    'head_min_channels': (
                        self.specialists[name].face_cfg.head_min_channels
                    ),
                    'lmk_hidden_max_channels': (
                        self.specialists[name].face_cfg.lmk_hidden_max_channels
                    ),
                    'anchor_offset': (
                        self.specialists[name].face_cfg.anchor_offset
                    ),
                    'landmark_encoding': (
                        self.specialists[name].face_cfg.landmark_encoding
                    ),
                }
                for name, spec in self.specs.items()
            },
        }

    def specialist_state_dicts(self) -> Dict[str, dict]:
        """State nhẹ của HEAD 1–3; không chứa model global/HEAD 4."""
        return {
            name: specialist.state_dict()
            for name, specialist in self.specialists.items()
        }

    def load_specialist_state_dicts(
        self,
        state_dicts: Mapping[str, dict],
        *,
        strict: bool = True,
    ) -> None:
        """Nạp đủ ba specialist và giữ chúng frozen sau khi nạp."""
        received = set(state_dicts)
        expected = set(SPECIALIST_NAMES)
        if received != expected:
            raise KeyError(
                'Checkpoint specialist phải chứa đúng '
                f'{sorted(expected)}, nhận {sorted(received)}.'
            )
        for name in SPECIALIST_NAMES:
            self.specialists[name].load_state_dict(
                state_dicts[name],
                strict=strict,
            )
        self.activate_specialist(None)
