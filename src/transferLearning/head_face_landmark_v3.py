import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL
from face_lmk_config import FaceLmkConfig

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

        def build_cls_stem():
            return nn.Sequential(DWConv(c_in, c_in, 3, 1), Conv(c_in, c_cls, 1, 1), DWConv(c_cls, c_cls, 3, 1), Conv(c_cls, c_cls, 1, 1))

        def build_reg_stem():
            return nn.Sequential(Conv(c_in, c_reg, 3, 1), Conv(c_reg, c_reg, 3, 1))

        def build_lmk_stem():
            return nn.Sequential(Conv(c_in, c_lmk, 3, 1), Conv(c_lmk, c_lmk, 3, 1), Conv(c_lmk, c_lmk_hidden, 1, 1))
        self.cls_stem_o2m = build_cls_stem()
        self.reg_stem_o2m = build_reg_stem()
        self.lmk_stem_o2m = build_lmk_stem()
        self.cls_o2m = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2m = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)
        self.cls_stem_o2o = build_cls_stem()
        self.reg_stem_o2o = build_reg_stem()
        self.lmk_stem_o2o = build_lmk_stem()
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
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 0.0)

    def init_stride_bias(self, stride, img_size=640):
        value = math.log(5 / self.nc / (img_size / stride) ** 2)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, value)

    def forward(self, x):
        cf_o2o = self.cls_stem_o2o(x)
        rf_o2o = self.reg_stem_o2o(x)
        lf_o2o = self.lmk_stem_o2o(x)
        out_o2o = (self.cls_o2o(cf_o2o), self.reg_o2o(rf_o2o), self.lmk_o2o(lf_o2o))
        if self.training:
            cf_o2m = self.cls_stem_o2m(x)
            rf_o2m = self.reg_stem_o2m(x)
            lf_o2m = self.lmk_stem_o2m(x)
            out_o2m = (self.cls_o2m(cf_o2m), self.reg_o2m(rf_o2m), self.lmk_o2m(lf_o2m))
            return (out_o2m, out_o2o)
        return (None, out_o2o)

class DetectHeadFaceLmk(nn.Module):

    def __init__(self, chs=(128, 256, 512), cfg: FaceLmkConfig=None):
        super().__init__()
        if cfg is None:
            raise ValueError('DetectHeadFaceLmk giờ BẮT BUỘC nhận cfg: FaceLmkConfig (dùng chung với FaceLandmarkDetectionLoss) thay vì các tham số num_landmarks/lmk_margin rời rạc như bản trước - xem face_lmk_config.py.')
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.num_landmarks = cfg.require_num_landmarks()
        self.lmk_margin = cfg.lmk_margin
        self.heads = nn.ModuleList((ScaleHeadFaceLmk(c, cfg) for c in chs))
        self.dfl = DFL(cfg.reg_max)
        for (head, s) in zip(self.heads, self.strides):
            head.init_stride_bias(s)

    @staticmethod
    def make_anchors(feats, strides, offset=0.5):
        (anchor_points, stride_tensor) = ([], [])
        for ((h, w), s) in zip([f.shape[-2:] for f in feats], strides):
            sy = torch.arange(h, device=feats[0].device) + offset
            sx = torch.arange(w, device=feats[0].device) + offset
            (gy, gx) = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))
        return (torch.cat(anchor_points), torch.cat(stride_tensor))

    def decode_box(self, reg, anchors, stride):
        ltrb = self.dfl(reg)
        (lt, rb) = (ltrb[:, :2], ltrb[:, 2:])
        anchors_t = anchors.transpose(0, 1).unsqueeze(0)
        x1y1 = anchors_t - lt
        x2y2 = anchors_t + rb
        xyxy = torch.cat([x1y1, x2y2], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)

    def decode_landmarks(self, lmk_raw, box_pixel, margin=None):
        if margin is None:
            margin = self.lmk_margin
        (B, C, A) = lmk_raw.shape
        K = self.num_landmarks
        t = torch.sigmoid(lmk_raw).transpose(1, 2).view(B, A, K, 2)
        (x1, y1, x2, y2) = box_pixel.unbind(-1)
        (w, h) = (x2 - x1, y2 - y1)
        x1e = (x1 - margin * w).unsqueeze(-1)
        y1e = (y1 - margin * h).unsqueeze(-1)
        we = (w * (1 + 2 * margin)).unsqueeze(-1).clamp(min=0.001)
        he = (h * (1 + 2 * margin)).unsqueeze(-1).clamp(min=0.001)
        px = x1e + t[..., 0] * we
        py = y1e + t[..., 1] * he
        return torch.stack([px, py], dim=-1)

    def forward(self, feats):
        (o2m_cls, o2m_reg, o2m_lmk_raw_list) = ([], [], [])
        (o2o_cls, o2o_reg, o2o_lmk_raw_list) = ([], [], [])
        for (feat, head) in zip(feats, self.heads):
            (out_o2m, (c_o, r_o, l_o)) = head(feat)
            if out_o2m is not None:
                (c_m, r_m, l_m) = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))
                o2m_lmk_raw_list.append(l_m.flatten(2))
            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))
            o2o_lmk_raw_list.append(l_o.flatten(2))
        (anchors, stride_t) = self.make_anchors(feats, self.strides)
        o2o_cls_c = torch.cat(o2o_cls, 2).transpose(1, 2)
        o2o_reg_c = torch.cat(o2o_reg, 2)
        o2o_lmk_raw = torch.cat(o2o_lmk_raw_list, 2)
        o2o_box = self.decode_box(o2o_reg_c, anchors, stride_t)
        if not self.training:
            o2o_lmk = self.decode_landmarks(o2o_lmk_raw, o2o_box)
            return {'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk': o2o_lmk, 'lmk_raw': o2o_lmk_raw}, 'anchors': anchors, 'strides': stride_t}
        o2m_cls_c = torch.cat(o2m_cls, 2).transpose(1, 2)
        o2m_reg_c = torch.cat(o2m_reg, 2)
        o2m_lmk_raw = torch.cat(o2m_lmk_raw_list, 2)
        o2m_box = self.decode_box(o2m_reg_c, anchors, stride_t)
        return {'o2m': {'cls': o2m_cls_c, 'box': o2m_box, 'reg_raw': o2m_reg_c, 'lmk_raw': o2m_lmk_raw}, 'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk_raw': o2o_lmk_raw}, 'anchors': anchors, 'strides': stride_t}
