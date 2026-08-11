import logging
import math
import torch
import torch.nn as nn

from src.blocks import Conv, DWConv, DFL
from src.model import NMSFreeDetector
from src.transferLearning.config_lmk import (
    PIPELINE_IMAGE_SIZE,
    FaceLmkConfig,
    TrainConfig,
    TrainingStageConfig,
)
from src.utils.init_weights import initialize_detection_head

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
logger = logging.getLogger('train_face_lmk')

class ScaleHeadFaceLmk(nn.Module):
    def __init__(self, c_in: int, cfg: FaceLmkConfig):
        super().__init__()
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        self.cfg = cfg
        c_cls = max(c_in // cfg.cls_channel_divisor, cfg.head_min_channels)
        c_reg = max(c_in // cfg.reg_channel_divisor, cfg.head_min_channels)
        c_lmk = max(c_in // cfg.lmk_channel_divisor, cfg.head_min_channels)
        c_lmk_hidden = max(
            c_lmk,
            min(self.num_landmarks * 2, cfg.lmk_hidden_max_channels),
        )

        def cls_stem():
            return nn.Sequential(DWConv(c_in, c_in, 3, 1), Conv(c_in, c_cls, 1, 1),
                                  DWConv(c_cls, c_cls, 3, 1), Conv(c_cls, c_cls, 1, 1))

        def reg_stem():
            return nn.Sequential(Conv(c_in, c_reg, 3, 1), Conv(c_reg, c_reg, 3, 1))

        def lmk_stem():
            return nn.Sequential(Conv(c_in, c_lmk, 3, 1), Conv(c_lmk, c_lmk, 3, 1), Conv(c_lmk, c_lmk_hidden, 1, 1))

        self.cls_stem_o2m, self.reg_stem_o2m, self.lmk_stem_o2m = cls_stem(), reg_stem(), lmk_stem()
        self.cls_o2m = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2m = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)

        self.cls_stem_o2o, self.reg_stem_o2o, self.lmk_stem_o2o = cls_stem(), reg_stem(), lmk_stem()
        self.cls_o2o = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2o = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2o = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)
        self._init_bias()

    def _init_bias(self):
        probability = self.cfg.cls_prior_probability
        prior = -math.log((1 - probability) / probability)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, prior)
        for m in (self.reg_o2m, self.reg_o2o):
            nn.init.constant_(m.bias, self.cfg.reg_bias)
        for m in (self.lmk_o2m, self.lmk_o2o):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.lmk_weight_std)
            nn.init.constant_(m.bias, 0.0)

    def init_stride_bias(self, stride, img_size=PIPELINE_IMAGE_SIZE):
        value = math.log(
            self.cfg.stride_bias_expected_objects
            / self.nc
            / (img_size / stride) ** 2
        )
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, value)

    def forward(self, x, return_o2m: bool = True):
        cf_o2o, rf_o2o, lf_o2o = self.cls_stem_o2o(x), self.reg_stem_o2o(x), self.lmk_stem_o2o(x)
        out_o2o = (self.cls_o2o(cf_o2o), self.reg_o2o(rf_o2o), self.lmk_o2o(lf_o2o))
        if not self.training and not return_o2m:
            return None, out_o2o
        cf_o2m, rf_o2m, lf_o2m = self.cls_stem_o2m(x), self.reg_stem_o2m(x), self.lmk_stem_o2m(x)
        out_o2m = (self.cls_o2m(cf_o2m), self.reg_o2m(rf_o2m), self.lmk_o2m(lf_o2m))
        return out_o2m, out_o2o

class DetectHeadFaceLmk(nn.Module):
    def __init__(
        self,
        chs=None,
        cfg: FaceLmkConfig = None,
        image_size: int = PIPELINE_IMAGE_SIZE,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError('DetectHeadFaceLmk cần cfg: FaceLmkConfig dùng chung với FaceLandmarkDetectionLoss.')
        if chs is None:
            raise ValueError('DetectHeadFaceLmk cần danh sách channel P3/P4/P5 từ trunk.')
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.num_landmarks = cfg.require_num_landmarks()
        self.heads = nn.ModuleList(ScaleHeadFaceLmk(c, cfg) for c in chs)
        self.dfl = DFL(cfg.reg_max)
        for head, stride in zip(self.heads, self.strides):
            head.init_stride_bias(stride, image_size)

    def parameter_summary(self) -> list:
        """Trả về số tham số của từng layer có parameter trong HEAD."""
        rows = []
        seen_parameters = set()
        for layer_name, module in self.named_modules():
            if not layer_name:
                continue

            parameters = []
            for parameter in module.parameters(recurse=False):
                parameter_id = id(parameter)
                if parameter_id not in seen_parameters:
                    parameters.append(parameter)
                    seen_parameters.add(parameter_id)
            if not parameters:
                continue

            total = sum(parameter.numel() for parameter in parameters)
            trainable = sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )
            rows.append({
                'layer': layer_name,
                'type': module.__class__.__name__,
                'total': total,
                'trainable': trainable,
                'frozen': total - trainable,
            })
        return rows

    def log_parameter_summary(self, target_logger=None) -> dict:
        """Log bảng tham số theo layer và tổng tham số của HEAD."""
        target_logger = target_logger or logger
        rows = self.parameter_summary()
        total = sum(row['total'] for row in rows)
        trainable = sum(row['trainable'] for row in rows)
        frozen = total - trainable

        lines = [
            'HEAD PARAMETER SUMMARY',
            f'{"Layer":<58} {"Type":<16} {"Total":>14} '
            f'{"Trainable":>14} {"Frozen":>14}',
            '-' * 122,
        ]
        lines.extend(
            f'{row["layer"]:<58} {row["type"]:<16} '
            f'{row["total"]:>14,} {row["trainable"]:>14,} '
            f'{row["frozen"]:>14,}'
            for row in rows
        )
        lines.extend((
            '-' * 122,
            f'{"TOTAL HEAD":<75} {total:>14,} {trainable:>14,} {frozen:>14,}',
        ))
        target_logger.info('\n%s', '\n'.join(lines))
        return {
            'layers': rows,
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
        }

    def make_anchors(self, feats, strides, offset=None):
        if offset is None:
            offset = self.cfg.anchor_offset
        anchor_points, stride_tensor = [], []
        for (h, w), s in zip([f.shape[-2:] for f in feats], strides):
            sy = torch.arange(h, device=feats[0].device) + offset
            sx = torch.arange(w, device=feats[0].device) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def decode_box(self, reg, anchors, stride):
        ltrb = self.dfl(reg)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]
        anchors_t = anchors.transpose(0, 1).unsqueeze(0)
        xyxy = torch.cat([anchors_t - lt, anchors_t + rb], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)

    def decode_landmarks(self, lmk_raw, anchors, strides):
        """Decode signed landmark offsets relative to anchor centers.
        Args:
            lmk_raw: Tensor[B, 2*K, A], signed offsets in feature-grid units.
            anchors: Tensor[A, 2], anchor centers in feature-grid units.
            strides: Tensor[A, 1], pixel stride for each anchor.
        Returns:
            Tensor[B, A, K, 2] containing landmark coordinates in pixels.

        Landmark geometry is independent of the predicted bounding box. The
        box branch is still used by the assigner, but it no longer translates
        or rescales the final landmark coordinates.
        """
        if lmk_raw.ndim != 3:
            raise ValueError(f'lmk_raw phải có shape [B, 2K, A], nhận {tuple(lmk_raw.shape)}.')

        batch_size, channels, num_anchors = lmk_raw.shape
        expected_channels = self.num_landmarks * 2
        if channels != expected_channels:
            raise ValueError(
                f'lmk_raw có {channels} channel, cần {expected_channels} '
                f'cho {self.num_landmarks} landmark.'
            )
        if anchors.shape != (num_anchors, 2):
            raise ValueError(
                f'anchors phải có shape {(num_anchors, 2)}, nhận {tuple(anchors.shape)}.'
            )
        if strides.shape != (num_anchors, 1):
            raise ValueError(
                f'strides phải có shape {(num_anchors, 1)}, nhận {tuple(strides.shape)}.'
            )

        offsets_grid = (
            lmk_raw.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_anchors, self.num_landmarks, 2)
        )
        anchor_grid = anchors.to(dtype=offsets_grid.dtype).view(1, num_anchors, 1, 2)
        stride_pixel = strides.to(dtype=offsets_grid.dtype).view(1, num_anchors, 1, 1)
        return (anchor_grid + offsets_grid) * stride_pixel

    def forward(self, feats, return_o2m: bool = True):
        o2m_cls, o2m_reg, o2m_lmk = [], [], []
        o2o_cls, o2o_reg, o2o_lmk = [], [], []
        for feat, head in zip(feats, self.heads):
            out_o2m, (c_o, r_o, l_o) = head(feat, return_o2m=return_o2m)
            if out_o2m is not None:
                c_m, r_m, l_m = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))
                o2m_lmk.append(l_m.flatten(2))
            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))
            o2o_lmk.append(l_o.flatten(2))

        anchors, stride_t = self.make_anchors(feats, self.strides)
        o2o_cls_c = torch.cat(o2o_cls, 2).transpose(1, 2)
        o2o_reg_c = torch.cat(o2o_reg, 2)
        o2o_lmk_raw = torch.cat(o2o_lmk, 2)
        o2o_box = self.decode_box(o2o_reg_c, anchors, stride_t)

        if not self.training and not return_o2m:
            o2o_lmk_dec = self.decode_landmarks(o2o_lmk_raw, anchors, stride_t)
            return {'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk': o2o_lmk_dec, 'lmk_raw': o2o_lmk_raw},
                    'anchors': anchors, 'strides': stride_t}

        o2m_cls_c = torch.cat(o2m_cls, 2).transpose(1, 2)
        o2m_reg_c = torch.cat(o2m_reg, 2)
        o2m_lmk_raw = torch.cat(o2m_lmk, 2)
        o2m_box = self.decode_box(o2m_reg_c, anchors, stride_t)
        return {'o2m': {'cls': o2m_cls_c, 'box': o2m_box, 'reg_raw': o2m_reg_c, 'lmk_raw': o2m_lmk_raw},
                'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk_raw': o2o_lmk_raw},
                'anchors': anchors, 'strides': stride_t}

class FaceLmkDetector(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self._trunk_frozen = False
        scaffold = NMSFreeDetector(nc=cfg.face.nc, reg_max=cfg.face.reg_max, backbone_w=cfg.trunk_backbone_w,
                                    backbone_n=cfg.trunk_backbone_n, neck_n=cfg.trunk_neck_n, strides=cfg.face.strides)
        self.backbone = scaffold.backbone
        self.neck = scaffold.neck
        neck_channels = tuple(scaffold.neck_chs)
        if neck_channels != cfg.trunk_feat_channels:
            raise ValueError(
                f'Output channel của trunk {neck_channels} khác cấu hình suy ra {cfg.trunk_feat_channels}.'
            )
        if hasattr(scaffold, 'head'):
            del scaffold.head
        del scaffold
        self.head = DetectHeadFaceLmk(chs=neck_channels, cfg=cfg.face, image_size=cfg.image_size)
        initialize_detection_head(self.head, image_size=cfg.image_size)
        self.head.log_parameter_summary(logger)

    def load_trunk(self, path: str, map_location='cpu', strict: bool = True):
        """Nạp full checkpoint rồi chỉ lấy ``backbone`` và ``neck``.

        Hỗ trợ checkpoint training dạng::

            {
                'epoch': ...,
                'global_step': ...,
                'model': model.state_dict(),
                'optimizer': ...,
                'scheduler': ...,
                'best_val': ...,
                'cfg': cfg.__dict__,
            }

        Vẫn tương thích với trunk checkpoint cũ có hai key
        ``backbone`` và ``neck``.
        """
        # PyTorch >= 2.6 mặc định weights_only=True. cfg.__dict__ của pipeline
        # chứa các dataclass lồng, vì vậy chỉ allowlist đúng các kiểu config nội
        # bộ thay vì tắt cơ chế unpickle an toàn cho toàn bộ checkpoint.
        safe_config_types = [FaceLmkConfig, TrainingStageConfig]
        logger.info(
            "[TRUNK LOAD] START | file='%s' | map_location=%s | strict=%s",
            path,
            map_location,
            strict,
        )
        try:
            with torch.serialization.safe_globals(safe_config_types):
                checkpoint = torch.load(
                    path,
                    map_location=map_location,
                    weights_only=True,
                )
        except Exception as exc:
            logger.error(
                "[TRUNK LOAD] FAIL khi đọc file='%s' | %s: %s",
                path,
                type(exc).__name__,
                exc,
            )
            raise
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint '{path}' phải là dict.")

        checkpoint_keys = list(checkpoint)
        logger.info(
            "[TRUNK LOAD] file='%s' | map_location=%s | strict=%s | "
            'top-level keys=%s%s',
            path,
            map_location,
            strict,
            checkpoint_keys[:12],
            f' (+{len(checkpoint_keys) - 12} key)' if len(checkpoint_keys) > 12 else '',
        )

        saved_cfg = checkpoint.get('cfg')
        if saved_cfg is not None and not isinstance(saved_cfg, dict) and hasattr(saved_cfg, '__dict__'):
            saved_cfg = saved_cfg.__dict__
        if isinstance(saved_cfg, dict):
            saved_face_cfg = saved_cfg.get('face')
            if (
                saved_face_cfg is not None
                and not isinstance(saved_face_cfg, dict)
                and hasattr(saved_face_cfg, '__dict__')
            ):
                saved_face_cfg = saved_face_cfg.__dict__
            if not isinstance(saved_face_cfg, dict):
                saved_face_cfg = {}
            saved_architecture = {
                'backbone_w': saved_cfg.get('backbone_w', saved_cfg.get('trunk_backbone_w')),
                'backbone_n': saved_cfg.get('backbone_n', saved_cfg.get('trunk_backbone_n')),
                'neck_n': saved_cfg.get('neck_n', saved_cfg.get('trunk_neck_n')),
                'strides': saved_cfg.get('strides', saved_face_cfg.get('strides')),
            }
            target_architecture = {
                'backbone_w': tuple(self.cfg.trunk_backbone_w),
                'backbone_n': tuple(self.cfg.trunk_backbone_n),
                'neck_n': self.cfg.trunk_neck_n,
                'strides': tuple(self.cfg.face.strides),
            }
            mismatched_architecture = {}
            for key, saved_value in saved_architecture.items():
                if saved_value is None:
                    continue
                target_value = target_architecture[key]
                normalized_saved = tuple(saved_value) if isinstance(saved_value, (list, tuple)) else saved_value
                if normalized_saved != target_value:
                    mismatched_architecture[key] = {
                        'checkpoint': normalized_saved,
                        'current': target_value,
                    }
            logger.info(
                '[TRUNK ARCH] checkpoint=%s | current=%s | match=%s',
                saved_architecture,
                target_architecture,
                not mismatched_architecture,
            )
            if mismatched_architecture:
                logger.warning(
                    '[TRUNK ARCH] Kiến trúc không khớp: %s. '
                    'Preflight key/shape sẽ quyết định có thể load hay không.',
                    mismatched_architecture,
                )

        def require_state_dict(candidate, source_name: str) -> dict:
            if isinstance(candidate, nn.Module):
                candidate = candidate.state_dict()
            if not isinstance(candidate, dict):
                raise TypeError(
                    f"checkpoint[{source_name!r}] phải là nn.Module hoặc state_dict."
                )
            if not candidate:
                raise ValueError(f"checkpoint[{source_name!r}] là state_dict rỗng.")
            if not all(isinstance(key, str) for key in candidate):
                raise TypeError(f"checkpoint[{source_name!r}] chứa key không phải str.")
            return candidate

        # Format trunk cũ: {'backbone': state_dict, 'neck': state_dict}.
        if 'backbone' in checkpoint and 'neck' in checkpoint:
            parts = {
                'backbone': require_state_dict(checkpoint['backbone'], 'backbone'),
                'neck': require_state_dict(checkpoint['neck'], 'neck'),
            }
            source_name = 'trunk checkpoint'
            ignored_head_tensors = 0
            ignored_other_tensors = 0
        else:
            # Full training checkpoint: ưu tiên model.state_dict() đúng như
            # format save_checkpoint; không nhầm với optimizer/scheduler/cfg.
            container_key = next(
                (key for key in ('model', 'state_dict', 'ema') if key in checkpoint),
                None,
            )
            if container_key is None:
                # Cho phép truyền raw model.state_dict() để tương thích ngược.
                state_dict = require_state_dict(checkpoint, 'raw_state_dict')
                source_name = 'raw state_dict'
            else:
                state_dict = checkpoint[container_key]
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                if isinstance(state_dict, dict) and 'ema' in state_dict:
                    state_dict = state_dict['ema']
                state_dict = require_state_dict(state_dict, container_key)
                source_name = f'checkpoint[{container_key!r}]'

            normalized = {}
            for key, value in state_dict.items():
                clean = key
                while clean.startswith(('module.', 'model.', 'ema.', '_orig_mod.')):
                    clean = clean.split('.', 1)[1]
                normalized[clean] = value
            parts = {
                name: {k[len(name) + 1:]: v for k, v in normalized.items() if k.startswith(name + '.')}
                for name in ('backbone', 'neck')
            }
            ignored_head_tensors = sum(key.startswith('head.') for key in normalized)
            ignored_other_tensors = len(normalized) - sum(len(part) for part in parts.values()) - ignored_head_tensors

            logger.info(
                "Đã đọc %s từ '%s' (epoch=%s, global_step=%s); "
                'tách backbone=%d tensor, neck=%d tensor; '
                'bỏ qua head=%d tensor, key khác=%d.',
                source_name,
                path,
                checkpoint.get('epoch', 'N/A'),
                checkpoint.get('global_step', 'N/A'),
                len(parts['backbone']),
                len(parts['neck']),
                ignored_head_tensors,
                ignored_other_tensors,
            )

        # Preflight cả hai component trước khi thay đổi model để tránh trạng
        # thái chỉ load được backbone nhưng neck lại thất bại.
        component_preflight = {}
        for name, module in (('backbone', self.backbone), ('neck', self.neck)):
            if not parts[name]:
                logger.error("[TRUNK PREFLIGHT] checkpoint không chứa '%s'.", name)
                raise KeyError(f"Checkpoint '{path}' không chứa trọng số '{name}'.")

            source_state = parts[name]
            target_state = module.state_dict()
            source_keys = set(source_state)
            target_keys = set(target_state)
            matched_keys = sorted(source_keys & target_keys)
            missing_preflight = sorted(target_keys - source_keys)
            unexpected_preflight = sorted(source_keys - target_keys)
            non_tensor_keys = sorted(
                key for key in source_keys
                if not isinstance(source_state[key], torch.Tensor)
            )
            shape_mismatches = sorted(
                (
                    key,
                    tuple(source_state[key].shape),
                    tuple(target_state[key].shape),
                )
                for key in matched_keys
                if isinstance(source_state[key], torch.Tensor)
                and source_state[key].shape != target_state[key].shape
            )
            matched_elements = sum(
                source_state[key].numel()
                for key in matched_keys
                if isinstance(source_state[key], torch.Tensor)
                and source_state[key].shape == target_state[key].shape
            )
            target_elements = sum(value.numel() for value in target_state.values())

            logger.info(
                '[TRUNK PREFLIGHT] %s | source=%d tensor | target=%d tensor | '
                'matched=%d | elements=%s/%s | missing=%d | unexpected=%d | '
                'bad_shape=%d | non_tensor=%d',
                name,
                len(source_state),
                len(target_state),
                len(matched_keys),
                f'{matched_elements:,}',
                f'{target_elements:,}',
                len(missing_preflight),
                len(unexpected_preflight),
                len(shape_mismatches),
                len(non_tensor_keys),
            )
            if missing_preflight:
                logger.warning('[TRUNK PREFLIGHT] %s missing (tối đa 10): %s', name, missing_preflight[:10])
            if unexpected_preflight:
                logger.warning('[TRUNK PREFLIGHT] %s unexpected (tối đa 10): %s', name, unexpected_preflight[:10])
            if shape_mismatches:
                logger.error('[TRUNK PREFLIGHT] %s sai shape (tối đa 10): %s', name, shape_mismatches[:10])
                raise RuntimeError(
                    f"Không thể load {name}: có {len(shape_mismatches)} tensor sai shape."
                )
            if non_tensor_keys:
                logger.error('[TRUNK PREFLIGHT] %s có giá trị không phải tensor: %s', name, non_tensor_keys[:10])
                raise TypeError(
                    f"Không thể load {name}: state_dict chứa giá trị không phải tensor."
                )
            if strict and (missing_preflight or unexpected_preflight):
                raise RuntimeError(
                    f'Không thể strict-load {name}: missing={len(missing_preflight)}, '
                    f'unexpected={len(unexpected_preflight)}.'
                )
            component_preflight[name] = (
                module,
                source_state,
                target_state,
                matched_keys,
            )

        all_missing, all_unexpected = [], []
        total_verified_tensors = 0
        total_verified_elements = 0
        for name in ('backbone', 'neck'):
            module, source_state, target_state, matched_keys = component_preflight[name]
            missing, unexpected = module.load_state_dict(source_state, strict=strict)
            all_missing += [f'{name}.{k}' for k in missing]
            all_unexpected += [f'{name}.{k}' for k in unexpected]

            loaded_state = module.state_dict()
            verification_failures = []
            verified_elements = 0
            for key in matched_keys:
                source_tensor = source_state[key]
                if source_tensor.shape != loaded_state[key].shape:
                    continue
                expected = source_tensor.detach().to(
                    device=loaded_state[key].device,
                    dtype=loaded_state[key].dtype,
                )
                if not torch.equal(loaded_state[key], expected):
                    verification_failures.append(key)
                else:
                    verified_elements += expected.numel()
            if verification_failures:
                logger.error(
                    '[TRUNK VERIFY] %s không khớp sau load (tối đa 10): %s',
                    name,
                    verification_failures[:10],
                )
                raise RuntimeError(
                    f'Xác minh sau load thất bại cho {len(verification_failures)} tensor {name}.'
                )
            total_verified_tensors += len(matched_keys)
            total_verified_elements += verified_elements
            logger.info(
                '[TRUNK VERIFY] %s PASS | exact tensors=%d/%d | exact elements=%s',
                name,
                len(matched_keys),
                len(target_state),
                f'{verified_elements:,}',
            )
        if all_missing or all_unexpected:
            logger.warning(
                'load_trunk: missing=%s, unexpected=%s',
                all_missing,
                all_unexpected,
            )
        logger.info(
            "[TRUNK LOAD] PASS | source=%s | file='%s' | strict=%s | "
            'verified=%d tensor (%s phần tử) | HEAD không được nạp.',
            source_name,
            path,
            strict,
            total_verified_tensors,
            f'{total_verified_elements:,}',
        )
        return all_missing, all_unexpected

    def freeze_trunk(self, freeze: bool = True):
        self._trunk_frozen = freeze
        for p in list(self.backbone.parameters()) + list(self.neck.parameters()):
            p.requires_grad = not freeze
        for m in list(self.backbone.modules()) + list(self.neck.modules()):
            if isinstance(m, _BN_TYPES):
                m.eval() if freeze else m.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._trunk_frozen:
            for m in list(self.backbone.modules()) + list(self.neck.modules()):
                if isinstance(m, _BN_TYPES):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor, return_o2m: bool = True):
        feats = self.neck(*self.backbone(images))
        return self.head(feats, return_o2m=return_o2m)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cfg = TrainConfig(require_pretrained_trunk=False)
    image_path = cfg.demo_model_image_path
    requested_device = cfg.device
    device = torch.device(
        requested_device
        if requested_device == 'cpu' or torch.cuda.is_available()
        else 'cpu'
    )
    model = FaceLmkDetector(cfg).to(device).eval()

    image = Image.open(image_path).convert("RGB").resize((cfg.image_size, cfg.image_size))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    image_tensor = image_tensor.float().div(255.0).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor)
    for branch in ("o2m", "o2o"):
        for name, value in output[branch].items():
            print(f"{branch}.{name}:",
                value.shape if hasattr(value, "shape") else [x.shape for x in value])

    print("anchors:", output["anchors"].shape)
    print("strides:", output["strides"].shape)
    print("=" * 80)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Input: {tuple(image_tensor.shape)}")
    print(f"Class output: {tuple(output['o2o']['cls'].shape)}")
    print(f"Box output: {tuple(output['o2o']['box'].shape)}")
    print(f"Landmark raw output: {tuple(output['o2o']['lmk_raw'].shape)}")

    plt.imshow(image)
    plt.axis("off")
    plt.show()
