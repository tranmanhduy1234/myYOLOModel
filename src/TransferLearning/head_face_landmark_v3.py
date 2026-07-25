"""
head_face_landmark.py  (v3)
============================
HEAD cho bài toán "Face Detection + Facial Landmarks", transfer từ
ScaleHead/DetectHead gốc (kiểu YOLOv10, dual-label-assignment o2m/o2o,
NMS-free ở nhánh o2o).

Kế thừa nguyên vẹn ý tưởng cốt lõi của v2 (landmark = tọa độ chuẩn hoá
THEO BBOX, sigmoid() đảm bảo landmark luôn nằm trong bbox+margin — xem
lại docstring v2 nếu cần ôn lại lý do). File này (v3) SỬA 3 vấn đề phát
hiện khi review v2:

1. ĐỒNG BỘ num_landmarks / lmk_margin qua FaceLmkConfig dùng chung
   -----------------------------------------------------------------
   Trước đây `num_landmarks` và `lmk_margin` là tham số riêng của
   DetectHeadFaceLmk, độc lập hoàn toàn với FaceLandmarkDetectionLoss.
   Default trùng nhau (5 và 0.15) khiến việc lệch nhau không lộ ra ngay,
   nhưng chỉ cần 1 trong 2 nơi bị sửa mà quên sửa nơi kia là encode lúc
   train và decode lúc inference lệch nhau ÂM THẦM. Giờ cả head lẫn loss
   đều nhận `cfg: FaceLmkConfig` — chỉ 1 nguồn duy nhất để sửa.

2. BỎ decode_landmarks() THỪA lúc training
   -----------------------------------------------------------------
   Ở v2, cả 2 nhánh o2m và o2o đều gọi decode_landmarks() (sigmoid + nhân
   với box đã decode) ngay cả khi đang training — nhưng
   FaceLandmarkDetectionLoss KHÔNG BAO GIỜ dùng field "lmk" (pixel đã
   decode) trong lúc train, nó tự tính lại `sigmoid(lmk_raw)` và so với
   target chuẩn hoá theo GT box. Vậy decode_landmarks() lúc train là
   compute + memory phí hoàn toàn (đặc biệt tốn với K lớn, ví dụ 478
   điểm MediaPipe: tensor (B, A, 478, 2) cho mỗi nhánh). v3 CHỈ decode
   landmark ra pixel lúc `not self.training` (khi cần trả kết quả cho
   người dùng / inference thực sự, lúc đó phải dùng box DỰ ĐOÁN vì
   không có GT).

3. TĂNG capacity nhánh landmark khi K lớn
   -----------------------------------------------------------------
   v2: lmk_stem chỉ 2 conv rồi chiếu thẳng 1 lớp 1x1 ra K*2 kênh. Với
   K nhỏ (5 điểm) thì ổn, nhưng với K lớn (478 điểm mesh) thì 1 lớp 1x1
   duy nhất từ ~64 kênh ra tới 956 kênh là một "cổ chai" khá hẹp so với
   độ khó của bài toán hồi quy dày đặc. v3 thêm 1 lớp hidden trung gian,
   độ rộng tỉ lệ với K (chặn trần ở 256 để tránh nổ tham số khi K rất
   lớn) TRƯỚC lớp chiếu cuối cùng.

CƠ CHẾ ĐẢM BẢO LANDMARK ĐÚNG NGƯỜI (không đổi so với v1/v2):
------------------------------------------------------------------------
Landmark head vẫn dùng CHUNG 1 lưới anchor + CHUNG 1 Task-Aligned
Assigner với box head. Anchor dương cho GT X chỉ tồn tại khi tâm anchor
nằm trong bbox của X, và target_gt_idx do assigner trả về được dùng
CHUNG cho cả box lẫn landmark trong loss. Vì vậy box và landmark tại 1
anchor luôn xuất phát từ CÙNG 1 GT, không thể "lệch" sang mặt bên cạnh
khi ảnh có nhiều mặt.

LƯU Ý QUAN TRỌNG (đọc kèm face_landmark_dataset_v3.py):
------------------------------------------------------------------------
Toàn bộ "đảm bảo cấu trúc" landmark nằm trong bbox+margin chỉ đúng NẾU
bbox nguồn (bounding_box_normalized trong data) thực sự bao trọn (hoặc
gần trọn) các điểm landmark. Nếu bbox đến từ 1 face-detector riêng
(thường tight hơn vùng mesh, đặc biệt với các điểm viền hàm/trán/tai của
mesh 478 điểm), một phần landmark GT có thể rơi ra ngoài margin và bị
CLAMP về biên trong loss (xem loss_face_landmark_v3.py) — không phải bug
code, nhưng cần kiểm tra bằng data thật (xem check_lmk_margin_coverage.py
đi kèm) để chọn lmk_margin phù hợp trước khi train lâu dài.
"""

import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL

# Sửa đường dẫn import cho khớp cấu trúc project thực tế.
from face_lmk_config import FaceLmkConfig


class ScaleHeadFaceLmk(nn.Module):
    """
    ScaleHead gốc + nhánh landmark (chuẩn hoá theo bbox, xem docstring
    đầu file). Giữ nguyên toàn bộ thiết kế box/cls (stem độc lập o2m/o2o).
    """

    def __init__(self, c_in: int, cfg: FaceLmkConfig):
        super().__init__()
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()

        c_cls = max(c_in // 2, 64)
        c_reg = max(c_in // 4, 64)
        c_lmk = max(c_in // 4, 64)
        # (fix #3) lớp hidden trước khi chiếu ra K*2 kênh, độ rộng tỉ lệ
        # với K, chặn trần 256 để không nổ tham số khi K rất lớn (478...).
        c_lmk_hidden = max(c_lmk, min(self.num_landmarks * 2, 256))

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
                Conv(c_lmk, c_lmk_hidden, 1, 1),   # lớp hidden mới (fix #3)
            )

        # ---- nhánh one-to-many (chỉ dùng khi training) ----
        self.cls_stem_o2m = build_cls_stem()
        self.reg_stem_o2m = build_reg_stem()
        self.lmk_stem_o2m = build_lmk_stem()
        self.cls_o2m = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2m = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)  # logit trước sigmoid

        # ---- nhánh one-to-one (dùng cả train lẫn inference, NMS-free) ----
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
        # landmark: weight=0, bias=0 => sigmoid(0)=0.5 => dự đoán ban đầu
        # là "landmark nằm ĐÚNG TÂM bbox" cho mọi điểm - điểm khởi tạo
        # an toàn và trung lập (không thiên vị điểm nào) trước khi học.
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

        # inference: bỏ qua hoàn toàn stem o2m
        # out_o2o = (cls[B,nc,H,W], reg[B,4*reg_max,H,W], lmk_logit[B,K*2,H,W])
        return None, out_o2o


class DetectHeadFaceLmk(nn.Module):
    """
    Giống DetectHead gốc, THÊM bước decode landmark THEO BBOX (không còn
    theo anchor+stride như v1). Landmark của 1 anchor được decode dựa
    trên CHÍNH bbox mà anchor đó dự đoán -> luôn nằm trong (hoặc gần sát)
    bbox tương ứng, không cần bước "match lại" ở hậu xử lý.
    """

    def __init__(self, chs=(128, 256, 512), cfg: FaceLmkConfig = None):
        super().__init__()
        if cfg is None:
            raise ValueError(
                "DetectHeadFaceLmk giờ BẮT BUỘC nhận cfg: FaceLmkConfig "
                "(dùng chung với FaceLandmarkDetectionLoss) thay vì các "
                "tham số num_landmarks/lmk_margin rời rạc như bản trước - "
                "xem face_lmk_config.py."
            )
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.num_landmarks = cfg.require_num_landmarks()
        self.lmk_margin = cfg.lmk_margin  # % w/h mỗi bên cho phép landmark "tràn" ra ngoài bbox
        self.heads = nn.ModuleList(
            ScaleHeadFaceLmk(c, cfg) for c in chs
        )
        self.dfl = DFL(cfg.reg_max)

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
        # reg: (B, 4*reg_max, A) -> ltrb qua DFL -> xyxy pixel, giống bản gốc
        ltrb = self.dfl(reg)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]
        anchors_t = anchors.transpose(0, 1).unsqueeze(0)
        x1y1 = anchors_t - lt
        x2y2 = anchors_t + rb
        xyxy = torch.cat([x1y1, x2y2], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)  # (B, A, 4)

    def decode_landmarks(self, lmk_raw, box_pixel, margin=None):
        """
        lmk_raw  : (B, K*2, A) logit THÔ (chưa qua sigmoid)
        box_pixel: (B, A, 4) xyxy pixel - CHÍNH là box đã decode của
                   CÙNG branch, đảm bảo landmark và box luôn đồng bộ với
                   nhau (cùng anchor, cùng GT).
        margin   : % w/h mở rộng bbox làm "khung chứa" landmark. None =>
                   dùng self.lmk_margin.

        CHỈ gọi hàm này khi cần toạ độ PIXEL thật sự (inference). Lúc
        training, loss dùng lmk_raw + GT box trực tiếp, không cần qua
        hàm này (xem forward() bên dưới và loss_face_landmark_v3.py).
        """
        if margin is None:
            margin = self.lmk_margin
        B, C, A = lmk_raw.shape
        K = self.num_landmarks

        t = torch.sigmoid(lmk_raw).transpose(1, 2).view(B, A, K, 2)  # (B,A,K,2) in (0,1)

        x1, y1, x2, y2 = box_pixel.unbind(-1)     # (B,A) mỗi tensor
        w, h = (x2 - x1), (y2 - y1)
        x1e = (x1 - margin * w).unsqueeze(-1)     # (B,A,1)
        y1e = (y1 - margin * h).unsqueeze(-1)
        we = (w * (1 + 2 * margin)).unsqueeze(-1).clamp(min=1e-3)
        he = (h * (1 + 2 * margin)).unsqueeze(-1).clamp(min=1e-3)

        px = x1e + t[..., 0] * we   # (B,A,K)
        py = y1e + t[..., 1] * he   # (B,A,K)
        return torch.stack([px, py], dim=-1)  # (B,A,K,2)

    def forward(self, feats):
        o2m_cls, o2m_reg, o2m_lmk_raw_list = [], [], []
        o2o_cls, o2o_reg, o2o_lmk_raw_list = [], [], []

        for feat, head in zip(feats, self.heads):
            out_o2m, (c_o, r_o, l_o) = head(feat)

            if out_o2m is not None:
                c_m, r_m, l_m = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))
                o2m_lmk_raw_list.append(l_m.flatten(2))

            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))
            o2o_lmk_raw_list.append(l_o.flatten(2))

        anchors, stride_t = self.make_anchors(feats, self.strides)

        # ---- nhánh o2o (luôn chạy) ----
        o2o_cls_c = torch.cat(o2o_cls, 2).transpose(1, 2)      # (B,A,nc)
        o2o_reg_c = torch.cat(o2o_reg, 2)                      # (B,4*reg_max,A)
        o2o_lmk_raw = torch.cat(o2o_lmk_raw_list, 2)            # (B,K*2,A) logit
        o2o_box = self.decode_box(o2o_reg_c, anchors, stride_t)

        if not self.training:
            # (fix #2) chỉ decode landmark ra pixel lúc inference thực sự
            # cần dùng nó - lúc train field này không được loss đụng tới.
            o2o_lmk = self.decode_landmarks(o2o_lmk_raw, o2o_box)
            return {
                "o2o": {
                    "cls": o2o_cls_c, "box": o2o_box,
                    "reg_raw": o2o_reg_c,
                    "lmk": o2o_lmk, "lmk_raw": o2o_lmk_raw,
                },
                "anchors": anchors, "strides": stride_t,
            }

        # ---- nhánh o2m (chỉ khi training) ----
        o2m_cls_c = torch.cat(o2m_cls, 2).transpose(1, 2)
        o2m_reg_c = torch.cat(o2m_reg, 2)
        o2m_lmk_raw = torch.cat(o2m_lmk_raw_list, 2)
        o2m_box = self.decode_box(o2m_reg_c, anchors, stride_t)
        # (fix #2) KHÔNG gọi decode_landmarks() ở đây (cho cả o2m lẫn o2o):
        # FaceLandmarkDetectionLoss chỉ đọc "lmk_raw" (+ GT box) trong lúc
        # train, không bao giờ đọc "lmk" (pixel) - decode ở đây là compute
        # + memory phí, đặc biệt tốn với K lớn.

        return {
            "o2m": {
                "cls": o2m_cls_c, "box": o2m_box,
                "reg_raw": o2m_reg_c,
                "lmk_raw": o2m_lmk_raw,
            },
            "o2o": {
                "cls": o2o_cls_c, "box": o2o_box,
                "reg_raw": o2o_reg_c,
                "lmk_raw": o2o_lmk_raw,
            },
            "anchors": anchors, "strides": stride_t,
        }
