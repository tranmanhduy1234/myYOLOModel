"""
loss_face_landmark.py  (v3)
=============================
Loss cho bài toán "Face Detection + Facial Landmarks", transfer từ
DetectionLoss gốc (TAL + CIoU + DFL + BCE, 2 nhánh o2m/o2o).

THAY ĐỔI SO VỚI v2 (fix theo review):
------------------------------------------------------------------------
1. Nhận `cfg: FaceLmkConfig` thay vì các tham số num_landmarks/lmk_margin
   rời rạc. Đây là NGUỒN DUY NHẤT cho num_landmarks/lmk_margin, dùng
   CHUNG với head_face_landmark_v3.py -> không còn khả năng 2 bên lệch
   nhau âm thầm (xem face_lmk_config.py để biết lý do).
2. preds["o2m"]/["o2o"] từ head v3 lúc training KHÔNG còn field "lmk"
   (pixel đã decode) nữa, chỉ còn "cls"/"box"/"reg_raw"/"lmk_raw" - loss
   vốn dĩ CHƯA BAO GIỜ đọc field "lmk" lúc train (chỉ đọc lmk_raw), nên
   phần này không cần sửa logic, chỉ cần xác nhận lại (xem _branch_loss).

Định dạng targets (không đổi so với v1/v2):
  targets = [
      {
        "boxes":  (N,4) xyxy pixel,
        "labels": (N,)  long,
        "landmarks":       (N, K, 2) xyxy pixel toạ độ từng điểm landmark,
        "landmarks_valid": (N,)  bool,  # True = GT này có nhãn landmark
      },
      ...
  ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tái sử dụng hàm hình học + TAL assigner + BboxLoss từ bản gốc, không
# viết lại. Sửa đường dẫn import cho khớp cấu trúc project thực tế.
from src.train.loss import bbox_iou, dist2bbox, bbox2dist, TaskAlignedAssigner, BboxLoss  # noqa: F401
from face_lmk_config import FaceLmkConfig

# ------------------------------------------------------------------------------
# 1. Landmark loss trên không gian CHUẨN HOÁ THEO BBOX (bị chặn trong [0,1])
# ------------------------------------------------------------------------------
def normalized_wing_loss(pred, target, w=0.10, epsilon=0.02):
    """
    Biến thể Wing Loss (Feng et al., 2018) cho không gian CHUẨN HOÁ
    [0,1] (khác bản pixel ở v1): w/epsilon nhỏ hơn nhiều vì biên độ sai
    số tối đa chỉ là 1.0 (thay vì hàng chục/trăm pixel). Nhạy hơn với sai
    số nhỏ (độ chính xác sub-pixel-ratio) nhưng vẫn tuyến tính với outlier.
    """
    diff = (pred - target).abs()
    import math
    C = w - w * math.log(1 + w / epsilon)
    return torch.where(diff < w, w * torch.log(1 + diff / epsilon), diff - C)

def landmark_regression_loss(pred_norm, target_norm, loss_type="smooth_l1", beta=0.05):
    """
    pred_norm, target_norm: cùng shape (..., ) trong [0,1] (xấp xỉ).
    loss_type: "smooth_l1" (mặc định, ổn định, phù hợp miền giá trị bị
               chặn) hoặc "wing" (nhạy hơn với sai số nhỏ).
    """
    if loss_type == "wing":
        return normalized_wing_loss(pred_norm, target_norm)
    return F.smooth_l1_loss(pred_norm, target_norm, beta=beta, reduction="none")


# ------------------------------------------------------------------------------
# 2. Geometric Consistency Loss (TUỲ CHỌN) - ràng buộc thứ tự hình học
# ------------------------------------------------------------------------------
def geometric_consistency_loss(pred_norm, constraints, margin=0.02):
    """
    pred_norm: (N, K, 2) toạ độ landmark ĐÃ sigmoid (chuẩn hoá theo bbox),
               N = số anchor dương đang xét (KHÔNG cần landmark GT, đây
               là regularizer thuần tuý trên dự đoán).
    constraints: list[(idx_a, idx_b, axis, sign)]
        axis: 0 = trục x, 1 = trục y
        sign = +1  => yêu cầu pred[idx_a, axis] + margin <= pred[idx_b, axis]
                      (vd ràng buộc "mắt trái (idx_a) nằm TRÊN mũi (idx_b)"
                      nếu idx_a, idx_b là index điểm mắt trái/mũi và axis=1)
        sign = -1  => chiều ngược lại
        Vi phạm thì phạt hinge loss = relu(vi_pham).

    Ví dụ constraints cho sơ đồ 5-điểm RetinaFace-style
    [left_eye=0, right_eye=1, nose=2, left_mouth=3, right_mouth=4]
    (trái/phải ở đây là trái/phải TRÊN ẢNH, không phải trái/phải của
    NGƯỜI trong ảnh - cần đối chiếu đúng với annotation scheme của bạn):
        constraints = [
            (0, 1, 0, +1),   # left_eye.x + margin <= right_eye.x
            (3, 4, 0, +1),   # left_mouth.x + margin <= right_mouth.x
            (0, 2, 1, +1),   # left_eye.y + margin <= nose.y  (mắt trên mũi)
            (1, 2, 1, +1),   # right_eye.y + margin <= nose.y
            (2, 3, 1, +1),   # nose.y + margin <= left_mouth.y (mũi trên miệng)
            (2, 4, 1, +1),
        ]
    """
    if pred_norm.numel() == 0 or not constraints:
        return pred_norm.sum() * 0
    terms = []
    for idx_a, idx_b, axis, sign in constraints:
        a = pred_norm[:, idx_a, axis]
        b = pred_norm[:, idx_b, axis]
        viol = (a - b + margin) if sign > 0 else (b - a + margin)
        terms.append(F.relu(viol))
    return torch.stack(terms, dim=0).mean()


# ------------------------------------------------------------------------------
# 3. FaceLandmarkDetectionLoss: DetectionLoss gốc + landmark + geo (tuỳ chọn)
# ------------------------------------------------------------------------------
class FaceLandmarkDetectionLoss(nn.Module):
    def __init__(self, cfg: FaceLmkConfig):
        super().__init__()
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        self.box_gain, self.cls_gain, self.dfl_gain, self.lmk_gain = (
            cfg.box_gain, cfg.cls_gain, cfg.dfl_gain, cfg.lmk_gain
        )
        self.o2m_weight, self.o2o_weight = cfg.o2m_weight, cfg.o2o_weight
        self.lmk_margin = cfg.lmk_margin
        self.lmk_loss_type = cfg.lmk_loss_type
        self.geo_constraints = cfg.geo_constraints or []
        self.geo_gain = cfg.geo_gain
        self.geo_margin = cfg.geo_margin

        self.assigner_o2m = TaskAlignedAssigner(topk=cfg.topk_o2m, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=cfg.topk_o2o, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.bbox_loss = BboxLoss(cfg.reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    # ---- tiền xử lý GT: list[dict] -> tensor có padding ----
    def preprocess_targets(self, targets, batch_size, device):
        n_max = max((t["boxes"].shape[0] for t in targets), default=0)
        n_max = max(n_max, 1)
        K = self.num_landmarks

        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)
        gt_landmarks = torch.zeros(batch_size, n_max, K, 2, device=device)
        gt_lmk_valid = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)

        for i, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = t["boxes"].to(device)
            gt_labels[i, :n, 0] = t["labels"].to(device)
            mask_gt[i, :n, 0] = True
            if "landmarks" in t and t["landmarks"] is not None and n > 0:
                lm = t["landmarks"]
                if lm.shape[1] != K:
                    raise ValueError(
                        f"target['landmarks'] có K={lm.shape[1]} điểm nhưng "
                        f"FaceLandmarkDetectionLoss được cấu hình num_landmarks="
                        f"{K} (qua cfg.sync_num_landmarks). Kiểm tra lại "
                        "cfg dùng cho head/loss có được sync đúng với "
                        "dataset.num_landmarks hay không."
                    )
                gt_landmarks[i, :n] = lm.to(device)
                if "landmarks_valid" in t and t["landmarks_valid"] is not None:
                    gt_lmk_valid[i, :n] = t["landmarks_valid"].to(device)
                else:
                    gt_lmk_valid[i, :n] = True

        return gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid

    # ---- gather target landmark PIXEL cho từng anchor theo target_gt_idx ----
    @staticmethod
    def _gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        """
        -> target_landmarks_pixel (bs, A, K, 2), target_lmk_mask (bs, A) bool
           (True = anchor là foreground VÀ GT tương ứng có nhãn landmark)
        """
        bs, M = gt_landmarks.shape[0], gt_landmarks.shape[1]
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_landmarks.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * M

        target_landmarks = gt_landmarks.view(-1, gt_landmarks.shape[2], 2)[flat_idx]  # (bs,A,K,2)
        target_lmk_has_label = gt_lmk_valid.view(-1)[flat_idx]                         # (bs,A)

        target_lmk_mask = fg_mask & target_lmk_has_label
        return target_landmarks, target_lmk_mask

    # ---- encode target landmark PIXEL -> chuẩn hoá theo GT box (mở rộng margin) ----
    def _encode_landmark_targets(self, target_landmarks_pixel, target_bboxes_pixel):
        """
        target_landmarks_pixel: (bs, A, K, 2) pixel
        target_bboxes_pixel   : (bs, A, 4) xyxy pixel - GT box ĐÃ MATCH
                                 (từ assigner), ỔN ĐỊNH vì lấy từ dữ liệu
                                 thật, không phải box dự đoán.
        -> target_norm (bs, A, K, 2) trong [0,1] (đã clamp để an toàn với
           các điểm hiếm gặp vượt ra ngoài vùng margin, tránh loss bùng
           nổ ở vài outlier annotation - xem check_lmk_margin_coverage.py
           để đo tỉ lệ điểm bị clamp này trên data thật của bạn).
        """
        x1, y1, x2, y2 = target_bboxes_pixel.unbind(-1)   # (bs,A)
        w, h = (x2 - x1), (y2 - y1)
        m = self.lmk_margin
        x1e = (x1 - m * w).unsqueeze(-1)                   # (bs,A,1)
        y1e = (y1 - m * h).unsqueeze(-1)
        we = (w * (1 + 2 * m)).unsqueeze(-1).clamp(min=1e-3)
        he = (h * (1 + 2 * m)).unsqueeze(-1).clamp(min=1e-3)

        tx = (target_landmarks_pixel[..., 0] - x1e) / we   # (bs,A,K)
        ty = (target_landmarks_pixel[..., 1] - y1e) / he
        target_norm = torch.stack([tx, ty], dim=-1).clamp(0.0, 1.0)
        return target_norm

    # ---- tính loss cho 1 nhánh (o2m hoặc o2o) ----
    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, lmk_raw, anchors, strides,
                      gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid):
        """
        lmk_raw: (bs, K*2, A) logit THÔ (trước sigmoid), đầu ra trực tiếp
                 từ lmk_o2m/lmk_o2o trong head. Đây là field DUY NHẤT của
                 landmark mà loss đọc lúc train (không đọc "lmk" pixel -
                 head v3 thậm chí không còn trả field đó lúc training).
        """
        bs = cls_raw.shape[0]
        A = cls_raw.shape[1]
        K = self.num_landmarks
        stride_b = strides.unsqueeze(0)               # (1, A, 1)
        anchors_pixel = anchors * strides              # (A, 2)

        pred_dist = reg_raw.transpose(1, 2).contiguous()          # (bs,A,4*reg_max) grid
        pred_lmk_norm = torch.sigmoid(lmk_raw).transpose(1, 2).view(bs, A, K, 2)  # (bs,A,K,2) in (0,1)

        with torch.no_grad():
            pd_scores_sig = cls_raw.detach().sigmoid()

        target_labels, target_bboxes_pixel, target_scores, fg_mask, target_gt_idx = assigner(
            pd_scores_sig, box_pixel.detach(), anchors_pixel,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum().item(), 1)

        # --- classification loss: giống hệt bản gốc ---
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum

        # --- box + dfl loss: giống hệt bản gốc, quy về không gian grid ---
        pred_bboxes_grid = box_pixel / stride_b
        target_bboxes_grid = target_bboxes_pixel / stride_b
        loss_iou, loss_dfl = self.bbox_loss(
            pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid,
            target_scores, target_scores_sum, fg_mask,
        )

        # --- landmark loss: không gian chuẩn hoá theo GT box (ổn định) ---
        target_landmarks_pixel, target_lmk_mask = self._gather_landmark_targets(
            gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask
        )
        n_lmk_pos = target_lmk_mask.sum().item()

        if n_lmk_pos == 0:
            loss_lmk = pred_lmk_norm.sum() * 0
        else:
            target_norm = self._encode_landmark_targets(target_landmarks_pixel, target_bboxes_pixel)

            pred_sel = pred_lmk_norm[target_lmk_mask]      # (n_lmk_pos, K, 2)
            target_sel = target_norm[target_lmk_mask]      # (n_lmk_pos, K, 2)
            weight_sel = target_scores.sum(-1)[target_lmk_mask].unsqueeze(-1).unsqueeze(-1)  # (n,1,1)

            per_point = landmark_regression_loss(pred_sel, target_sel, self.lmk_loss_type)
            loss_lmk = (per_point * weight_sel).sum() / (weight_sel.sum() * K * 2 + 1e-9)

        # --- geometric consistency loss (TUỲ CHỌN, áp lên MỌI anchor
        #     dương, không cần landmark GT - xem docstring đầu file) ---
        if self.geo_gain > 0 and self.geo_constraints and fg_mask.any():
            pred_fg = pred_lmk_norm[fg_mask]  # (n_fg, K, 2)
            loss_geo = geometric_consistency_loss(pred_fg, self.geo_constraints, self.geo_margin)
        else:
            loss_geo = pred_lmk_norm.sum() * 0

        n_pos = fg_mask.sum().item()
        return loss_iou, loss_cls, loss_dfl, loss_lmk, loss_geo, n_pos, n_lmk_pos

    def forward(self, preds, targets):
        """
        preds: dict từ DetectHeadFaceLmk ở chế độ train:
               {"o2m": {"cls","box","reg_raw","lmk_raw"}, "o2o": {...},
                "anchors", "strides"}
               (lưu ý: v3 không còn field "lmk" pixel lúc training, xem
               head_face_landmark_v3.py)
        targets: list[dict], xem docstring đầu file.
        """
        device = preds["anchors"].device
        batch_size = preds["o2o"]["cls"].shape[0]
        anchors, strides = preds["anchors"], preds["strides"]

        gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid = self.preprocess_targets(
            targets, batch_size, device
        )

        iou_m, cls_m, dfl_m, lmk_m, geo_m, npos_m, nlmk_m = self._branch_loss(
            self.assigner_o2m,
            preds["o2m"]["cls"], preds["o2m"]["box"], preds["o2m"]["reg_raw"], preds["o2m"]["lmk_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid,
        )
        iou_o, cls_o, dfl_o, lmk_o, geo_o, npos_o, nlmk_o = self._branch_loss(
            self.assigner_o2o,
            preds["o2o"]["cls"], preds["o2o"]["box"], preds["o2o"]["reg_raw"], preds["o2o"]["lmk_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid,
        )

        loss_o2m = (self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m
                    + self.lmk_gain * lmk_m + self.geo_gain * geo_m)
        loss_o2o = (self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o
                    + self.lmk_gain * lmk_o + self.geo_gain * geo_o)
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        items = {
            "loss": total.detach().item(),
            "loss_o2m": loss_o2m.detach().item(),
            "loss_o2o": loss_o2o.detach().item(),
            "o2m/iou": iou_m.detach().item(), "o2m/cls": cls_m.detach().item(),
            "o2m/dfl": dfl_m.detach().item(), "o2m/lmk": lmk_m.detach().item(),
            "o2m/geo": geo_m.detach().item(),
            "o2o/iou": iou_o.detach().item(), "o2o/cls": cls_o.detach().item(),
            "o2o/dfl": dfl_o.detach().item(), "o2o/lmk": lmk_o.detach().item(),
            "o2o/geo": geo_o.detach().item(),
            "o2m/n_pos": npos_m, "o2o/n_pos": npos_o,
            "o2m/n_lmk_pos": nlmk_m, "o2o/n_lmk_pos": nlmk_o,
        }
        return total, items
