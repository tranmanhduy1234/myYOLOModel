"""
loss_face_landmark.py  (v2)
=============================
Loss cho bai toan "Face Detection + Facial Landmarks", transfer tu
DetectionLoss goc (TAL + CIoU + DFL + BCE, 2 nhanh o2m/o2o).

THAY DOI SO VOI v1:
------------------------------------------------------------------------
1. Landmark loss chuyen sang KHONG GIAN CHUAN HOA THEO BBOX (khop voi
   head v2 - xem head_face_landmark.py):

       pred_norm  = sigmoid(lmk_raw)                       in (0,1)
       target_norm = (target_landmark_pixel - box1e) / boxwe   [box la
                     GT box DA MATCH boi assigner, mo rong margin]

   Diem quan trong: target_norm duoc tinh tu GT BOX (bien tu du lieu
   that, luon dung va on dinh ngay tu buoc dau training) - KHONG dung
   box du doan cua model. Nho vay loss landmark:
     - Khong phu thuoc/khong lam nhieu gradient cua box regression (2
       nhanh hoc doc lap, tranh vong lap bat on dinh "box sai -> target
       landmark sai -> landmark sai -> box sai hon").
     - On dinh tu epoch dau tien, khac voi phuong an "chuan hoa theo box
       du doan" se rat nhieu luc box con te.
   Luc INFERENCE (khong co GT), head moi dung box DU DOAN de decode
   landmark ra pixel - luc do khong con van de gradient (forward-only).

   Loss dung tren khong gian [0,1] bi chan nen chuyen tu Wing-loss-tren-
   pixel (v1) sang Smooth L1 (Huber) tren khong gian chuan hoa - phu hop
   hon voi mien gia tri nho, bi chan. Wing loss van duoc giu lam option
   (voi w/epsilon da chinh lai cho thang do [0,1]).

2. THEM Geometric Consistency Loss (tuy chon, mac dinh TAT - can khai
   bao constraints vi index diem landmark phu thuoc dataset/annotation
   scheme cua ban, vd 5-point RetinaFace, 68-point iBUG, 98-point WFLW
   deu khac thu tu). Loss nay ap dang hinge len CHINH pred_norm (khong
   can GT) nen ap dung duoc tren MOI anchor duong, ke ca anh khong co
   nhan landmark day du - vai tro nhu 1 regularizer "biet truoc cau truc
   khuon mat" (mat luon o tren mui, mui luon o tren mieng, mat trai luon
   ben trai mat phai...).

CO CHE DAM BAO LANDMARK/BOX DUNG NGUOI (khong doi so voi v1):
------------------------------------------------------------------------
TaskAlignedAssigner tra ve `target_gt_idx` (bs, A) - id GT ma tung anchor
duoc gan. Ca box lan landmark cua 1 anchor deu duoc lay tu CUNG 1 GT qua
CUNG 1 index nay => khong the "le" landmark cua mat nay sang mat khac
khi anh co nhieu mat.

Dinh dang targets (giong v1):
  targets = [
      {
        "boxes":  (N,4) xyxy pixel,
        "labels": (N,)  long,
        "landmarks":       (N, K, 2) xyxy pixel toa do tung diem landmark,
        "landmarks_valid": (N,)  bool,  # True = GT nay co nhan landmark
      },
      ...
  ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tai su dung ham hinh hoc + TAL assigner + BboxLoss tu ban goc, khong
# viet lai. Sua duong dan import cho khop cau truc project thuc te.
from  src.train.loss import bbox_iou, dist2bbox, bbox2dist, TaskAlignedAssigner, BboxLoss  # noqa: F401

# ------------------------------------------------------------------------------
# 1. Landmark loss tren khong gian CHUAN HOA THEO BBOX (bi chan trong [0,1])
# ------------------------------------------------------------------------------
def normalized_wing_loss(pred, target, w=0.10, epsilon=0.02):
    """
    Bien the Wing Loss (Feng et al., 2018) cho khong gian CHUAN HOA
    [0,1] (khac ban pixel o v1): w/epsilon nho hon nhieu vi bien do sai
    so toi da chi la 1.0 (thay vi hang chuc/tram pixel). Nhay hon voi sai
    so nho (do chinh xac sub-pixel-ratio) nhung van tuyen tinh voi outlier.
    """
    diff = (pred - target).abs()
    import math
    C = w - w * math.log(1 + w / epsilon)
    return torch.where(diff < w, w * torch.log(1 + diff / epsilon), diff - C)

def landmark_regression_loss(pred_norm, target_norm, loss_type="smooth_l1", beta=0.05):
    """
    pred_norm, target_norm: cung shape (..., ) trong [0,1] (xap xi).
    loss_type: "smooth_l1" (mac dinh, on dinh, phu hop mien gia tri bi
               chan) hoac "wing" (nhay hon voi sai so nho).
    """
    if loss_type == "wing":
        return normalized_wing_loss(pred_norm, target_norm)
    return F.smooth_l1_loss(pred_norm, target_norm, beta=beta, reduction="none")


# ------------------------------------------------------------------------------
# 2. Geometric Consistency Loss (TUY CHON) - rang buoc thu tu hinh hoc
# ------------------------------------------------------------------------------
def geometric_consistency_loss(pred_norm, constraints, margin=0.02):
    """
    pred_norm: (N, K, 2) toa do landmark DA sigmoid (chuan hoa theo bbox),
               N = so anchor duong dang xet (KHONG can landmark GT, day
               la regularizer thuan tuy tren du doan).
    constraints: list[(idx_a, idx_b, axis, sign)]
        axis: 0 = truc x, 1 = truc y
        sign = +1  => yeu cau pred[idx_a, axis] + margin <= pred[idx_b, axis]
                      (vd rang buoc "mat trai (idx_a) nam TREN mui (idx_b)"
                      neu idx_a, idx_b la index diem mat trai/mui va axis=1)
        sign = -1  => chieu nguoc lai
        Vi phaply thi phat hinge loss = relu(vi_pham).

    Vi du constraints cho so do 5-diem RetinaFace-style
    [left_eye=0, right_eye=1, nose=2, left_mouth=3, right_mouth=4]
    (trai/phai o day la trai/phai TREN ANH, khong phai trai/phai cua
    NGUOI trong anh - can doi chieu dung voi annotation scheme cua ban):
        constraints = [
            (0, 1, 0, +1),   # left_eye.x + margin <= right_eye.x
            (3, 4, 0, +1),   # left_mouth.x + margin <= right_mouth.x
            (0, 2, 1, +1),   # left_eye.y + margin <= nose.y  (mat tren mui)
            (1, 2, 1, +1),   # right_eye.y + margin <= nose.y
            (2, 3, 1, +1),   # nose.y + margin <= left_mouth.y (mui tren mieng)
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
# 3. FaceLandmarkDetectionLoss: DetectionLoss goc + landmark + geo (tuy chon)
# ------------------------------------------------------------------------------
class FaceLandmarkDetectionLoss(nn.Module):
    def __init__(
        self, nc=1, reg_max=16, num_landmarks=5,
        topk_o2m=10, topk_o2o=1,
        alpha=0.5, beta=6.0,
        box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, lmk_gain=1.0,
        o2m_weight=1.0, o2o_weight=1.0,
        lmk_margin=0.15,                # phai KHOP voi lmk_margin cua head khi inference
        lmk_loss_type="smooth_l1",      # "smooth_l1" hoac "wing"
        geo_constraints=None,           # list[(idx_a, idx_b, axis, sign)], None/[] = tat
        geo_gain=0.0,                   # >0 de bat geometric consistency loss
        geo_margin=0.02,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.num_landmarks = num_landmarks
        self.box_gain, self.cls_gain, self.dfl_gain, self.lmk_gain = (
            box_gain, cls_gain, dfl_gain, lmk_gain
        )
        self.o2m_weight, self.o2o_weight = o2m_weight, o2o_weight
        self.lmk_margin = lmk_margin
        self.lmk_loss_type = lmk_loss_type
        self.geo_constraints = geo_constraints or []
        self.geo_gain = geo_gain
        self.geo_margin = geo_margin

        self.assigner_o2m = TaskAlignedAssigner(topk=topk_o2m, num_classes=nc, alpha=alpha, beta=beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=topk_o2o, num_classes=nc, alpha=alpha, beta=beta)
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    # ---- tien xu ly GT: list[dict] -> tensor co padding ----
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
                gt_landmarks[i, :n] = t["landmarks"].to(device)
                if "landmarks_valid" in t and t["landmarks_valid"] is not None:
                    gt_lmk_valid[i, :n] = t["landmarks_valid"].to(device)
                else:
                    gt_lmk_valid[i, :n] = True

        return gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid

    # ---- gather target landmark PIXEL cho tung anchor theo target_gt_idx ----
    @staticmethod
    def _gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        """
        -> target_landmarks_pixel (bs, A, K, 2), target_lmk_mask (bs, A) bool
           (True = anchor la foreground VA GT tuong ung co nhan landmark)
        """
        bs, M = gt_landmarks.shape[0], gt_landmarks.shape[1]
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_landmarks.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * M

        target_landmarks = gt_landmarks.view(-1, gt_landmarks.shape[2], 2)[flat_idx]  # (bs,A,K,2)
        target_lmk_has_label = gt_lmk_valid.view(-1)[flat_idx]                         # (bs,A)

        target_lmk_mask = fg_mask & target_lmk_has_label
        return target_landmarks, target_lmk_mask

    # ---- encode target landmark PIXEL -> chuan hoa theo GT box (mo rong margin) ----
    def _encode_landmark_targets(self, target_landmarks_pixel, target_bboxes_pixel):
        """
        target_landmarks_pixel: (bs, A, K, 2) pixel
        target_bboxes_pixel   : (bs, A, 4) xyxy pixel - GT box DA MATCH
                                 (tu assigner), ON DINH vi lay tu du lieu
                                 that, khong phai box du doan.
        -> target_norm (bs, A, K, 2) trong [0,1] (da clamp de an toan so
           voi cac diem hiem gap vuot ra ngoai vung margin, tranh loss
           bung no o vai outlier annotation).
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

    # ---- tinh loss cho 1 nhanh (o2m hoac o2o) ----
    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, lmk_raw, anchors, strides,
                      gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid):
        """
        lmk_raw: (bs, K*2, A) logit THO (truoc sigmoid), dau ra truc tiep
                 tu lmk_o2m/lmk_o2o trong head.
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

        # --- classification loss: giong het ban goc ---
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum

        # --- box + dfl loss: giong het ban goc, quy ve khong gian grid ---
        pred_bboxes_grid = box_pixel / stride_b
        target_bboxes_grid = target_bboxes_pixel / stride_b
        loss_iou, loss_dfl = self.bbox_loss(
            pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid,
            target_scores, target_scores_sum, fg_mask,
        )

        # --- landmark loss: khong gian chuan hoa theo GT box (on dinh) ---
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

        # --- geometric consistency loss (TUY CHON, ap len MOI anchor
        #     duong, khong can landmark GT - xem docstring dau file) ---
        if self.geo_gain > 0 and self.geo_constraints and fg_mask.any():
            pred_fg = pred_lmk_norm[fg_mask]  # (n_fg, K, 2)
            loss_geo = geometric_consistency_loss(pred_fg, self.geo_constraints, self.geo_margin)
        else:
            loss_geo = pred_lmk_norm.sum() * 0

        n_pos = fg_mask.sum().item()
        return loss_iou, loss_cls, loss_dfl, loss_lmk, loss_geo, n_pos, n_lmk_pos

    def forward(self, preds, targets):
        """
        preds: dict tu DetectHeadFaceLmk o che do train:
               {"o2m": {"cls","box","reg_raw","lmk","lmk_raw"}, "o2o": {...},
                "anchors", "strides"}
        targets: list[dict], xem docstring dau file.
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