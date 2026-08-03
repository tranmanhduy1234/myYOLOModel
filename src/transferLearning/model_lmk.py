import math
import torch
import torch.nn as nn

from src.blocks import Conv, DWConv, DFL
from src.model import NMSFreeDetector
try:
    from .config_lmk import FaceLmkConfig, TrainConfig
except ImportError:
    from config_lmk import FaceLmkConfig, TrainConfig

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class ScaleHeadFaceLmk(nn.Module):

    def __init__(self, c_in: int, cfg: FaceLmkConfig):
        super().__init__()
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        c_cls = max(c_in // 2, 64)
        c_reg = max(c_in // 4, 64)
        c_lmk = max(c_in // 4, 64)
        c_lmk_hidden = max(c_lmk, min(self.num_landmarks * 2, 256))

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
        prior = -math.log((1 - 0.01) / 0.01)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, prior)
        for m in (self.reg_o2m, self.reg_o2o):
            nn.init.constant_(m.bias, 1.0)
        for m in (self.lmk_o2m, self.lmk_o2o):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.constant_(m.bias, 0.0)

    def init_stride_bias(self, stride, img_size=480):
        value = math.log(5 / self.nc / (img_size / stride) ** 2)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, value)

    def forward(self, x, return_o2m: bool = False):
        cf_o2o, rf_o2o, lf_o2o = self.cls_stem_o2o(x), self.reg_stem_o2o(x), self.lmk_stem_o2o(x)
        out_o2o = (self.cls_o2o(cf_o2o), self.reg_o2o(rf_o2o), self.lmk_o2o(lf_o2o))
        if not self.training and not return_o2m:
            return None, out_o2o
        cf_o2m, rf_o2m, lf_o2m = self.cls_stem_o2m(x), self.reg_stem_o2m(x), self.lmk_stem_o2m(x)
        out_o2m = (self.cls_o2m(cf_o2m), self.reg_o2m(rf_o2m), self.lmk_o2m(lf_o2m))
        return out_o2m, out_o2o


class DetectHeadFaceLmk(nn.Module):

    def __init__(self, chs=(128, 256, 512), cfg: FaceLmkConfig = None, img_size: int = 480):
        super().__init__()
        if cfg is None:
            raise ValueError('DetectHeadFaceLmk cần cfg: FaceLmkConfig dùng chung với FaceLandmarkDetectionLoss.')
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.num_landmarks = cfg.require_num_landmarks()
        self.lmk_margin = cfg.lmk_margin
        self.heads = nn.ModuleList(ScaleHeadFaceLmk(c, cfg) for c in chs)
        self.dfl = DFL(cfg.reg_max)
        for head, stride in zip(self.heads, self.strides):
            head.init_stride_bias(stride, img_size)

    @staticmethod
    def make_anchors(feats, strides, offset=0.5):
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

    def decode_landmarks(self, lmk_raw, box_pixel, margin=None):
        margin = self.lmk_margin if margin is None else margin
        B, C, A = lmk_raw.shape
        K = self.num_landmarks
        t = torch.sigmoid(lmk_raw).transpose(1, 2).view(B, A, K, 2)
        x1, y1, x2, y2 = box_pixel.unbind(-1)
        w, h = x2 - x1, y2 - y1
        x1e = (x1 - margin * w).unsqueeze(-1)
        y1e = (y1 - margin * h).unsqueeze(-1)
        we = (w * (1 + 2 * margin)).unsqueeze(-1).clamp(min=0.001)
        he = (h * (1 + 2 * margin)).unsqueeze(-1).clamp(min=0.001)
        return torch.stack([x1e + t[..., 0] * we, y1e + t[..., 1] * he], dim=-1)

    def forward(self, feats, return_o2m: bool = False):
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
            o2o_lmk_dec = self.decode_landmarks(o2o_lmk_raw, o2o_box)
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
        self.head = DetectHeadFaceLmk(chs=neck_channels, cfg=cfg.face, img_size=cfg.image_size)

    def load_trunk(self, path: str, map_location='cpu', strict: bool = True):
        checkpoint = torch.load(path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint '{path}' phải là dict/state_dict.")

        # Format gốc: {'backbone': state_dict, 'neck': state_dict}.
        if isinstance(checkpoint.get('backbone'), dict) and isinstance(checkpoint.get('neck'), dict):
            parts = {'backbone': checkpoint['backbone'], 'neck': checkpoint['neck']}
        else:
            state_dict = checkpoint
            for container_key in ('ema', 'model', 'state_dict'):
                candidate = state_dict.get(container_key) if isinstance(state_dict, dict) else None
                if candidate is not None:
                    if isinstance(candidate, nn.Module):
                        candidate = candidate.state_dict()
                    if isinstance(candidate, dict) and 'ema' in candidate and isinstance(candidate['ema'], dict):
                        candidate = candidate['ema']
                    state_dict = candidate
                    break
            if not isinstance(state_dict, dict):
                raise TypeError(f"Không tìm thấy state_dict hợp lệ trong '{path}'.")

            normalized = {}
            for key, value in state_dict.items():
                clean = key
                while clean.startswith(('module.', 'model.', 'ema.')):
                    clean = clean.split('.', 1)[1]
                normalized[clean] = value
            parts = {
                name: {k[len(name) + 1:]: v for k, v in normalized.items() if k.startswith(name + '.')}
                for name in ('backbone', 'neck')
            }

        all_missing, all_unexpected = [], []
        for name, module in (('backbone', self.backbone), ('neck', self.neck)):
            if not parts[name]:
                raise KeyError(f"Checkpoint '{path}' không chứa trọng số '{name}'.")
            missing, unexpected = module.load_state_dict(parts[name], strict=strict)
            all_missing += [f'{name}.{k}' for k in missing]
            all_unexpected += [f'{name}.{k}' for k in unexpected]
        if all_missing or all_unexpected:
            print(f'[FaceLmkDetector] load_trunk: missing={all_missing}, unexpected={all_unexpected}')
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

    def forward(self, images: torch.Tensor, return_o2m: bool = False):
        feats = self.neck(*self.backbone(images))
        return self.head(feats, return_o2m=return_o2m)
