import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL

class ScaleHead(nn.Module):
    def __init__(self, c_in, nc, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        c_cls = max(c_in // 2, 64)
        c_reg = max(c_in // 4, 64)

        def build_cls_stem():
            return nn.Sequential(
                DWConv(c_in, c_in, 3, 1), Conv(c_in, c_cls, 1, 1),
                DWConv(c_cls, c_cls, 3, 1), Conv(c_cls, c_cls, 1, 1),
            )

        def build_reg_stem():
            return nn.Sequential(
                Conv(c_in, c_reg, 3, 1),
                Conv(c_reg, c_reg, 3, 1),
            )

        # nhánh one-to-many (chỉ dùng khi training)
        self.cls_stem_o2m = build_cls_stem()
        self.reg_stem_o2m = build_reg_stem()
        self.cls_o2m = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * reg_max, 1)

        # nhánh one-to-one (dùng cho cả train lẫn inference, NMS-free)
        self.cls_stem_o2o = build_cls_stem()
        self.reg_stem_o2o = build_reg_stem()
        self.cls_o2o = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2o = nn.Conv2d(c_reg, 4 * reg_max, 1)

        self._init_bias()

    def _init_bias(self):
        prior = -math.log((1 - 0.01) / 0.01)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, prior)
        for m in (self.reg_o2m, self.reg_o2o):
            nn.init.constant_(m.bias, 1.0)

    def init_stride_bias(self, stride, img_size=640):
        """Init bias cls theo đúng công thức chuẩn YOLOv8/v10:
        log(5 / nc / (img_size/stride)^2) - phản ánh mật độ object kỳ vọng
        khác nhau ở từng scale (P3 dày object nhỏ hơn P5 nhiều). Được
        DetectHead gọi sau khi biết stride của scale này."""
        value = math.log(5 / self.nc / (img_size / stride) ** 2)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, value)

    def forward(self, x):
        # nhánh o2o luôn tính (dùng cho cả train lẫn inference)
        cf_o2o = self.cls_stem_o2o(x)
        rf_o2o = self.reg_stem_o2o(x)
        out_o2o = (self.cls_o2o(cf_o2o), self.reg_o2o(rf_o2o))

        # if self.training:
        cf_o2m = self.cls_stem_o2m(x)
        rf_o2m = self.reg_stem_o2m(x)
        out_o2m = (self.cls_o2m(cf_o2m), self.reg_o2m(rf_o2m))
        return out_o2m, out_o2o

        # inference: bỏ qua hoàn toàn stem o2m để tiết kiệm tài nguyên
        # return output format:
        # cls[batch size, num_class, 80, 80] or cls[batch size, num_class, 40, 40] or cls[batch size, num_class, 20, 20]
        # reg[batch size, reg_max * 4, 80, 80] or reg[batch size, reg_max * 4, 40, 40] or reg[batch size, reg_max * 4, 20, 20]
        return None, out_o2o

class DetectHead(nn.Module):
    def __init__(self, chs=(128, 256, 512), nc=80, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.nc = nc  # so luong num class
        self.reg_max = reg_max  # so luong num bin.
        self.strides = strides
        self.heads = nn.ModuleList(ScaleHead(c, nc, reg_max) for c in chs)  # Cac module scale head
        self.dfl = DFL(reg_max)

        # init bias cls theo stride cho từng scale (đúng chuẩn YOLOv8/v10,
        # thay vì hằng số 0.01 cố định như bản trước)
        for head, s in zip(self.heads, self.strides):
            head.init_stride_bias(s)

    @staticmethod
    def make_anchors(feats, strides, offset=0.5):
        anchor_points, stride_tensor = [], []
        # [p3, p4, p5] =
        # P3 - torch.Size([5, 192, 80, 80])
        # P4 - torch.Size([5, 384, 40, 40])
        # P5 - torch.Size([5, 512, 20, 20])

        for (h, w), s in zip([f.shape[-2:] for f in feats], strides):
            sy, sx = torch.arange(h, device=feats[0].device) + offset, torch.arange(w, device=feats[0].device) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")

            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))  # shape [6400, 2] or [1600, 2] or [400, 2]
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))

        # anchor_points: [8400, 2]
        # stride_tensor: [8400, 1]
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def decode_box(self, reg, anchors, stride):
        # reg: [B, 4*reg_max, A] (batch size, 64, 8400)
        # anchors: [8400, 2]
        # stride_t: [8400]
        
        # reg: (B, 4*reg_max, A) -> ltrb (B, 4, A) qua DFL -> xyxy
        ltrb = self.dfl(reg)  # (B,4,A)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]  # # (B,2,A)
        anchors = anchors.transpose(0, 1).unsqueeze(0)  # (1,2,A)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        xyxy = torch.cat([x1y1, x2y2], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)  # (B, A, 4)

    def forward(self, feats):
        o2m_cls, o2m_reg, o2o_cls, o2o_reg = [], [], [], []

        for feat, head in zip(feats, self.heads):
            out_o2m, (c_o, r_o) = head(feat)
            # cls: [batch size, num_class, 80, 80]
            # reg: [batch size, reg_max * 4, 80, 80]

            # Chỉ xử lý và gom nhóm nhánh o2m nếu nó tồn tại (đang trong trạng thái Train)
            if out_o2m is not None:
                c_m, r_m = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))

            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))

        # Khởi tạo ma trận điểm neo
        anchors, stride_t = self.make_anchors(feats, self.strides)
        # anchor_points: [8400, 2]
        # stride_tensor: [8400, 1]

        o2o_cls = torch.cat(o2o_cls, 2).transpose(1, 2)  # (B, A, nc)
        o2o_reg = torch.cat(o2o_reg, 2)  # (B, 4*reg_max, A) ~ (batch size, 64, 8400)
        o2o_box = self.decode_box(o2o_reg, anchors, stride_t)  # (B, A, 4)

        # if not self.training:
        #     return {
        #         "o2o": {"cls": o2o_cls, "box": o2o_box, "reg_raw": o2o_reg},
        #         "anchors": anchors,
        #         "strides": stride_t,
        #     }

        o2m_cls = torch.cat(o2m_cls, 2).transpose(1, 2)  # (B, A, nc)
        o2m_reg = torch.cat(o2m_reg, 2)  # (B, 4*reg_max, A)
        o2m_box = self.decode_box(o2m_reg, anchors, stride_t)  # (B, A, 4)

        return {
            "o2m": {"cls": o2m_cls, "box": o2m_box, "reg_raw": o2m_reg},
            "o2o": {"cls": o2o_cls, "box": o2o_box, "reg_raw": o2o_reg},
            "anchors": anchors,
            "strides": stride_t,
        }