"""
head_face_landmark.py  (v2)
============================
HEAD cho bai toan "Face Detection + Facial Landmarks", transfer tu
ScaleHead/DetectHead goc (kieu YOLOv10, dual-label-assignment o2m/o2o,
NMS-free o nhanh o2o).

THAY DOI SO VOI v1 (theo phan bien: "landmark khong duoc dam bao nam
trong box"):
------------------------------------------------------------------------
v1: landmark = anchor_point + offset_tho (offset khong bi chan, co the
    +10, +50 grid units bat ky) -> KHONG co gi ngan landmark bay ra
    ngoai box, chi clamp o buoc POST-PROCESS (khong anh huong training).

v2: landmark duoc encode dang TOA DO CHUAN HOA THEO BBOX (giong nhieu
    face detector hien dai vd RetinaFace-style box-relative encoding):

        Anchor --(nhanh reg + DFL)--> BBox (x1,y1,x2,y2)
              --(nhanh lmk + sigmoid)--> t = (tx,ty) in (0,1)
        landmark_pixel = (x1,y1) + t * (w,h)          [+ margin, xem duoi]

    Vi sigmoid() luon tra ve gia tri trong (0,1), landmark_pixel LUON
    LUON nam trong bbox (hoac bbox mo rong them margin) -- day la mot
    RANG BUOC CAU TRUC (structural), dung ngay trong kien truc mang,
    khong phai hy vong tu loss hay clamp hau xu ly nhu v1.

    margin (mac dinh 0.15) cho phep landmark nam ngoai bbox "chat" mot
    chut (vd diem tai/vien ham thuong ngoai bbox mat "tight"), nhung
    van la mot vung BI CHAN RO RANG quanh box, khong con la "bay tu do"
    nhu v1.

    Uu diem phu: t nam trong khong gian [0,1] KHONG PHU THUOC kich thuoc
    mat (scale-invariant) -> hoc de hon, transfer tot hon giua mat to/nho,
    va la tien de tu nhien de ap them Geometric Consistency Loss (xem
    loss_face_landmark.py) vi cac rang buoc hinh hoc (mat trai/phai,
    mat tren mui tren mieng...) co the ap TRUC TIEP len t ma khong can
    biet kich thuoc box thuc te.

CO CHE DAM BAO LANDMARK DUNG NGUOI (khong doi so voi v1):
------------------------------------------------------------------------
Landmark head van dung CHUNG 1 luoi anchor + CHUNG 1 Task-Aligned
Assigner voi box head. Anchor duong cho GT X chi ton tai khi tam anchor
nam trong bbox cua X (select_candidates_in_gts), va target_gt_idx do
assigner tra ve duoc dung CHUNG cho ca box lan landmark trong loss. Vi
vay box va landmark tai 1 anchor luon xuat phat tu CUNG 1 GT, khong the
"le" sang mat ben canh khi anh co nhieu mat.
"""

import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL


class ScaleHeadFaceLmk(nn.Module):
    """
    ScaleHead goc + nhanh landmark (chuan hoa theo bbox, xem docstring
    dau file). Giu nguyen toan bo thiet ke box/cls (stem doc lap o2m/o2o).

    num_landmarks: so diem landmark / mat (5, 68, 98...).
    """

    def __init__(self, c_in, nc, reg_max=16, num_landmarks=5):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.num_landmarks = num_landmarks

        c_cls = max(c_in // 2, 64)
        c_reg = max(c_in // 4, 64)
        c_lmk = max(c_in // 4, 64)

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

        def build_lmk_stem():
            return nn.Sequential(
                Conv(c_in, c_lmk, 3, 1),
                Conv(c_lmk, c_lmk, 3, 1),
            )

        # ---- nhanh one-to-many (chi dung khi training) ----
        self.cls_stem_o2m = build_cls_stem()
        self.reg_stem_o2m = build_reg_stem()
        self.lmk_stem_o2m = build_lmk_stem()
        self.cls_o2m = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * reg_max, 1)
        self.lmk_o2m = nn.Conv2d(c_lmk, num_landmarks * 2, 1)  # logit truoc sigmoid

        # ---- nhanh one-to-one (dung ca train lan inference, NMS-free) ----
        self.cls_stem_o2o = build_cls_stem()
        self.reg_stem_o2o = build_reg_stem()
        self.lmk_stem_o2o = build_lmk_stem()
        self.cls_o2o = nn.Conv2d(c_cls, nc, 1)
        self.reg_o2o = nn.Conv2d(c_reg, 4 * reg_max, 1)
        self.lmk_o2o = nn.Conv2d(c_lmk, num_landmarks * 2, 1)

        self._init_bias()

    def _init_bias(self):
        prior = -math.log((1 - 0.01) / 0.01)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, prior)
        for m in (self.reg_o2m, self.reg_o2o):
            nn.init.constant_(m.bias, 1.0)
        # landmark: weight=0, bias=0 => sigmoid(0)=0.5 => du doan ban dau
        # la "landmark nam DUNG TAM bbox" cho moi diem - diem khoi tao
        # an toan va trung lap (khong thien vi diem nao) truoc khi hoc.
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
            return out_o2m, out_o2o

        # inference: bo qua hoan toan stem o2m
        # out_o2o = (cls[B,nc,H,W], reg[B,4*reg_max,H,W], lmk_logit[B,K*2,H,W])
        return None, out_o2o


class DetectHeadFaceLmk(nn.Module):
    """
    Giong DetectHead goc, THEM buoc decode landmark THEO BBOX (khong con
    theo anchor+stride nhu v1). Landmark cua 1 anchor duoc decode dua
    tren CHINH bbox ma anchor do du doan -> luon nam trong (hoac gan sat)
    bbox tuong ung, khong can buoc "match lai" o hau xu ly.
    """

    def __init__(self, chs=(128, 256, 512), nc=1, reg_max=16, strides=(8, 16, 32),
                 num_landmarks=5, lmk_margin=0.15):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.num_landmarks = num_landmarks
        self.lmk_margin = lmk_margin  # % w/h moi ben cho phep landmark "tran" ra ngoai bbox
        self.heads = nn.ModuleList(
            ScaleHeadFaceLmk(c, nc, reg_max, num_landmarks) for c in chs
        )
        self.dfl = DFL(reg_max)

        for head, s in zip(self.heads, self.strides):
            head.init_stride_bias(s)

    @staticmethod
    def make_anchors(feats, strides, offset=0.5):
        anchor_points, stride_tensor = [], []
        for (h, w), s in zip([f.shape[-2:] for f in feats], strides):
            sy = torch.arange(h, device=feats[0].device) + offset
            sx = torch.arange(w, device=feats[0].device) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def decode_box(self, reg, anchors, stride):
        # reg: (B, 4*reg_max, A) -> ltrb qua DFL -> xyxy pixel, giong ban goc
        ltrb = self.dfl(reg)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]
        anchors_t = anchors.transpose(0, 1).unsqueeze(0)
        x1y1 = anchors_t - lt
        x2y2 = anchors_t + rb
        xyxy = torch.cat([x1y1, x2y2], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)  # (B, A, 4)

    def decode_landmarks(self, lmk_raw, box_pixel, margin=None):
        """
        lmk_raw  : (B, K*2, A) logit THO (chua qua sigmoid)
        box_pixel: (B, A, 4) xyxy pixel - CHINH la box da decode cua
                   CUNG branch (o2m hoac o2o), dam bao landmark va box
                   luon dong bo voi nhau (cung anchor, cung GT).
        margin   : % w/h mo rong bbox lam "khung chua" landmark. None =>
                   dung self.lmk_margin.

        Decode:
            t = sigmoid(lmk_raw) in (0,1)          # (B,A,K,2)
            box mo rong: x1e = x1 - margin*w, we = w*(1+2*margin) (tuong tu y)
            landmark_pixel = (x1e, y1e) + t * (we, he)

        -> (B, A, K, 2) toa do pixel, LUON nam trong [x1e,x2e] x [y1e,y2e],
           tuc la nam trong bbox (mo rong them margin).
        """
        if margin is None:
            margin = self.lmk_margin
        B, C, A = lmk_raw.shape
        K = self.num_landmarks

        t = torch.sigmoid(lmk_raw).transpose(1, 2).view(B, A, K, 2)  # (B,A,K,2) in (0,1)

        x1, y1, x2, y2 = box_pixel.unbind(-1)     # (B,A) moi tensor
        w, h = (x2 - x1), (y2 - y1)
        x1e = (x1 - margin * w).unsqueeze(-1)     # (B,A,1)
        y1e = (y1 - margin * h).unsqueeze(-1)
        we = (w * (1 + 2 * margin)).unsqueeze(-1).clamp(min=1e-3)
        he = (h * (1 + 2 * margin)).unsqueeze(-1).clamp(min=1e-3)

        px = x1e + t[..., 0] * we   # (B,A,K)
        py = y1e + t[..., 1] * he   # (B,A,K)
        return torch.stack([px, py], dim=-1)  # (B,A,K,2)

    def forward(self, feats):
        o2m_cls, o2m_reg, o2m_lmk = [], [], []
        o2o_cls, o2o_reg, o2o_lmk = [], [], []

        for feat, head in zip(feats, self.heads):
            out_o2m, (c_o, r_o, l_o) = head(feat)

            if out_o2m is not None:
                c_m, r_m, l_m = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))
                o2m_lmk.append(l_m.flatten(2))

            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))
            o2o_lmk.append(l_o.flatten(2))

        anchors, stride_t = self.make_anchors(feats, self.strides)

        # ---- nhanh o2o (luon chay) ----
        o2o_cls_c = torch.cat(o2o_cls, 2).transpose(1, 2)      # (B,A,nc)
        o2o_reg_c = torch.cat(o2o_reg, 2)                      # (B,4*reg_max,A)
        o2o_lmk_raw = torch.cat(o2o_lmk, 2)                    # (B,K*2,A) logit
        o2o_box = self.decode_box(o2o_reg_c, anchors, stride_t)
        # luu y: dung box DA DECODE (khong detach) chi de tra ve pixel
        # tham khao/visualize; loss KHONG dung truong "lmk" nay de tinh
        # gradient (xem loss_face_landmark.py - loss dung lmk_raw truc
        # tiep + GT box, khong phu thuoc box du doan).
        o2o_lmk = self.decode_landmarks(o2o_lmk_raw, o2o_box)

        if not self.training:
            return {
                "o2o": {
                    "cls": o2o_cls_c, "box": o2o_box,
                    "reg_raw": o2o_reg_c,
                    "lmk": o2o_lmk, "lmk_raw": o2o_lmk_raw,
                },
                "anchors": anchors, "strides": stride_t,
            }

        # ---- nhanh o2m (chi khi training) ----
        o2m_cls_c = torch.cat(o2m_cls, 2).transpose(1, 2)
        o2m_reg_c = torch.cat(o2m_reg, 2)
        o2m_lmk_raw = torch.cat(o2m_lmk, 2)
        o2m_box = self.decode_box(o2m_reg_c, anchors, stride_t)
        o2m_lmk = self.decode_landmarks(o2m_lmk_raw, o2m_box)

        return {
            "o2m": {
                "cls": o2m_cls_c, "box": o2m_box,
                "reg_raw": o2m_reg_c,
                "lmk": o2m_lmk, "lmk_raw": o2m_lmk_raw,
            },
            "o2o": {
                "cls": o2o_cls_c, "box": o2o_box,
                "reg_raw": o2o_reg_c,
                "lmk": o2o_lmk, "lmk_raw": o2o_lmk_raw,
            },
            "anchors": anchors, "strides": stride_t,
        }