import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL

class ScaleHead(nn.Module):
    """
    Head cho 1 scale, dùng chung feature-extraction convs, nhưng có 2 bộ
    predictor cuối: one-to-many (o2m, dùng lúc train để học phong phú) và
    one-to-one (o2o, dùng lúc inference -> KHÔNG CẦN NMS).
    Ý tưởng lấy từ YOLOv10 (dual label assignment).
    """

    def __init__(self, c_in, nc, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        c_cls = max(c_in // 2, 64)
        c_reg = max(c_in // 4, 64)

        # nhánh cls: dùng depthwise-separable để nhẹ tham số ("head xịn" nhưng rẻ)
        self.cls_stem = nn.Sequential(
            DWConv(c_in, c_in, 3, 1), Conv(c_in, c_cls, 1, 1),
            DWConv(c_cls, c_cls, 3, 1), Conv(c_cls, c_cls, 1, 1),
        )
        self.reg_stem = nn.Sequential(
            Conv(c_in, c_reg, 3, 1),
            Conv(c_reg, c_reg, 3, 1),
        )

        # predictor one-to-many
        self.cls_o2m = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * reg_max, 1)

        # predictor one-to-one (nhẹ hơn, chỉ 1x1 conv riêng trên cùng feature)
        self.cls_o2o = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2o = nn.Conv2d(c_reg, 4 * reg_max, 1)

        self._init_bias()

    def _init_bias(self):
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, -math.log((1 - 0.01) / 0.01))  # prior prob 0.01

    def forward(self, x):
        cf = self.cls_stem(x)
        rf = self.reg_stem(x)
        out_o2m = (self.cls_o2m(cf), self.reg_o2m(rf))
        out_o2o = (self.cls_o2o(cf), self.reg_o2o(rf))
        return out_o2m, out_o2o


class DetectHead(nn.Module):
    def __init__(self, chs=(128, 256, 512), nc=80, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.heads = nn.ModuleList(ScaleHead(c, nc, reg_max) for c in chs)
        self.dfl = DFL(reg_max)

    @staticmethod
    def make_anchors(feats, strides, offset=0.5):
        anchor_points, stride_tensor = [], []
        for (h, w), s in zip([f.shape[-2:] for f in feats], strides):
            sy, sx = torch.arange(h, device=feats[0].device) + offset, torch.arange(w, device=feats[0].device) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def decode_box(self, reg, anchors, stride):
        # reg: (B, 4*reg_max, A) -> ltrb (B, 4, A) qua DFL -> xyxy
        ltrb = self.dfl(reg)  # (B,4,A)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]
        anchors = anchors.transpose(0, 1).unsqueeze(0)  # (1,2,A)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        xyxy = torch.cat([x1y1, x2y2], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)  # (B, A, 4)

    def forward(self, feats):
        o2m_cls, o2m_reg, o2o_cls, o2o_reg = [], [], [], []
        for feat, head in zip(feats, self.heads):
            (c_m, r_m), (c_o, r_o) = head(feat)
            b = feat.shape[0]
            o2m_cls.append(c_m.flatten(2))
            o2m_reg.append(r_m.flatten(2))
            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))

        anchors, stride_t = self.make_anchors(feats, self.strides)

        o2m_cls = torch.cat(o2m_cls, 2).transpose(1, 2)   # (B, A, nc)
        o2o_cls = torch.cat(o2o_cls, 2).transpose(1, 2)   # (B, A, nc)
        o2m_reg = torch.cat(o2m_reg, 2)                   # (B, 4*reg_max, A)
        o2o_reg = torch.cat(o2o_reg, 2)

        o2m_box = self.decode_box(o2m_reg, anchors, stride_t)  # (B,A,4) xyxy pixel
        o2o_box = self.decode_box(o2o_reg, anchors, stride_t)

        return {
            "o2m": {"cls": o2m_cls, "box": o2m_box, "reg_raw": o2m_reg},
            "o2o": {"cls": o2o_cls, "box": o2o_box, "reg_raw": o2o_reg},
            "anchors": anchors, "strides": stride_t,
        }