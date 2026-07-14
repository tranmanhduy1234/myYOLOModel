"""
loss.py
=======
Ham loss cho NMSFreeDetector (kieu YOLOv10): Task-Aligned Assigner (TAL) +
CIoU loss + Distribution Focal Loss (DFL) + BCE classification loss, ap dung
rieng cho CA HAI nhanh du doan cua head:
  - o2m (one-to-many): topk lon (mac dinh 10) -> nhieu anchor duong cho 1 GT,
    giup training on dinh & hoi tu nhanh (giong YOLOv8).
  - o2o (one-to-one)  : topk = 1 -> moi GT chi co DUY NHAT 1 anchor duong,
    day la nhanh giup model bo duoc NMS luc inference (dac trung cua YOLOv10).

Tong loss = o2m_weight * loss_o2m + o2o_weight * loss_o2o
    (moi loss_o2x = box_gain*loss_iou + cls_gain*loss_cls + dfl_gain*loss_dfl)

Dinh dang input mong doi:
  preds  : dict tra ve boi DetectHead.forward() khi model.train() dang bat,
           tuc la {"o2m": {"cls","box","reg_raw"}, "o2o": {...},
                    "anchors": (A,2), "strides": (A,1)}
  targets: list do dai = batch_size, moi phan tu la dict:
                {"boxes": FloatTensor (N,4) xyxy o khong gian PIXEL cua anh dau vao,
                 "labels": LongTensor (N,)}
           N co the = 0 (anh khong co object nao).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# 1. Cac ham hinh hoc dung chung
# ------------------------------------------------------------------------------
def bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """box1, box2: (..., 4). Ho tro broadcasting. Tra ve (...,) IoU/CIoU."""
    import math
    if xywh:
        (x1c, y1c, w1, h1), (x2c, y2c, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1c - w1_, x1c + w1_, y1c - h1_, y1c + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2c - w2_, x2c + w2_, y2c - h2_, y2c + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return (iou - (rho2 / c2 + v * alpha)).squeeze(-1)
            return (iou - rho2 / c2).squeeze(-1)
        c_area = cw * ch + eps
        return (iou - (c_area - union) / c_area).squeeze(-1)
    return iou.squeeze(-1)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """ltrb -> xyxy (hoac xywh)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    """xyxy -> ltrb, clamp trong [0, reg_max - eps] (reg_max o day la so bin - 1)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    return torch.cat((lt, rb), -1).clamp_(0, reg_max - 0.01)

# ------------------------------------------------------------------------------
# 2. Task-Aligned Assigner
# ------------------------------------------------------------------------------
class TaskAlignedAssigner(nn.Module):
    """
    Gan GT cho anchor dua tren align_metric = cls_score^alpha * IoU^beta.
    topk=N  -> kieu one-to-many (nhieu anchor duong / GT).
    topk=1  -> kieu one-to-one (dung cho nhanh NMS-free cua YOLOv10).
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores : (bs, A, nc)  da qua sigmoid
        pd_bboxes : (bs, A, 4)   xyxy, KHONG GIAN PIXEL (giong gt_bboxes)
        anc_points: (A, 2)       tam anchor, KHONG GIAN PIXEL (anchors_grid * stride)
        gt_labels : (bs, M, 1) long
        gt_bboxes : (bs, M, 4) xyxy pixel
        mask_gt   : (bs, M, 1) bool, True = GT that (khong phai padding)
        ->
        target_labels (bs,A) long, target_bboxes (bs,A,4), target_scores (bs,A,nc),
        fg_mask (bs,A) bool, target_gt_idx (bs,A) long
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        A = pd_scores.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full((self.bs, A), self.nc, dtype=torch.long, device=device),
                torch.zeros((self.bs, A, 4), device=device),
                torch.zeros((self.bs, A, self.nc), device=device),
                torch.zeros((self.bs, A), dtype=torch.bool, device=device),
                torch.zeros((self.bs, A), dtype=torch.long, device=device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        bs, M = self.bs, self.n_max_boxes
        A = pd_scores.shape[1]
        mask_gt = mask_gt.bool()

        overlaps = torch.zeros((bs, M, A), dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros((bs, M, A), dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros((2, bs, M), dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, M)
        ind[1] = gt_labels.squeeze(-1).clamp(0, self.nc - 1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes_exp = pd_bboxes.unsqueeze(1).expand(-1, M, -1, -1)[mask_gt]
        gt_boxes_exp = gt_bboxes.unsqueeze(2).expand(-1, -1, A, -1)[mask_gt]
        if pd_boxes_exp.numel():
            overlaps[mask_gt] = bbox_iou(gt_boxes_exp, pd_boxes_exp, xywh=False, CIoU=True).clamp(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.amax(-1, keepdim=True) > self.eps).expand_as(topk_idxs)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k: k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_candidates_in_gts(anc_points, gt_bboxes, eps=1e-9):
        n_anchors = anc_points.shape[0]
        bs, M, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat(
            (anc_points.unsqueeze(0) - lt, rb - anc_points.unsqueeze(0)), dim=2
        ).view(bs, M, n_anchors, -1)
        return deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes).permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        bs = gt_labels.shape[0]
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_labels.device).unsqueeze(-1)
        target_gt_idx_flat = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx_flat]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx_flat]

        target_labels = target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask.unsqueeze(-1).repeat(1, 1, self.nc).bool()
        target_scores = torch.where(fg_scores_mask, target_scores, torch.zeros_like(target_scores))

        return target_labels, target_bboxes, target_scores.float()


# ------------------------------------------------------------------------------
# 3. Bbox loss (CIoU + DFL)
# ------------------------------------------------------------------------------
class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """
        pred_dist  : (bs, A, 4*reg_max) logits truoc softmax, KHONG GIAN GRID (chua nhan stride)
        pred_bboxes: (bs, A, 4) xyxy, KHONG GIAN GRID
        anchor_points: (A, 2) KHONG GIAN GRID
        target_bboxes: (bs, A, 4) xyxy, KHONG GIAN GRID
        """
        if fg_mask.sum() == 0:
            return pred_bboxes.sum() * 0, pred_dist.sum() * 0

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum

        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (loss_l * wl + loss_r * wr).mean(-1, keepdim=True)


# ------------------------------------------------------------------------------
# 4. DetectionLoss: ghep TAL + BboxLoss + BCE cho ca 2 nhanh o2m / o2o
# ------------------------------------------------------------------------------
class DetectionLoss(nn.Module):
    def __init__(
        self, nc, reg_max=16,
        topk_o2m=10, topk_o2o=1,
        alpha=0.5, beta=6.0,
        box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
        o2m_weight=1.0, o2o_weight=1.0,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.box_gain, self.cls_gain, self.dfl_gain = box_gain, cls_gain, dfl_gain
        self.o2m_weight, self.o2o_weight = o2m_weight, o2o_weight

        self.assigner_o2m = TaskAlignedAssigner(topk=topk_o2m, num_classes=nc, alpha=alpha, beta=beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=topk_o2o, num_classes=nc, alpha=alpha, beta=beta)
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    # ---- tien xu ly GT: list[dict] -> tensor co padding ------------------------
    def preprocess_targets(self, targets, batch_size, device):
        n_max = max((t["boxes"].shape[0] for t in targets), default=0)
        n_max = max(n_max, 1)  # tranh tensor rong gay loi shape khi khong co GT nao trong ca batch
        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)

        for i, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = t["boxes"].to(device)
            gt_labels[i, :n, 0] = t["labels"].to(device)
            mask_gt[i, :n, 0] = True
        return gt_bboxes, gt_labels, mask_gt

    # ---- tinh loss cho 1 nhanh (o2m hoac o2o) ----------------------------------
    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, anchors, strides,
                      gt_bboxes, gt_labels, mask_gt):
        """
        cls_raw : (bs, A, nc) logit (CHUA sigmoid)
        box_pixel: (bs, A, 4) xyxy, khong gian PIXEL (dau ra decode san cua model)
        reg_raw : (bs, 4*reg_max, A) logit DFL truoc softmax
        anchors : (A, 2) grid units (offset 0.5, CHUA nhan stride)
        strides : (A, 1)
        """
        bs = cls_raw.shape[0]
        stride_b = strides.unsqueeze(0)              # (1, A, 1)
        anchors_pixel = anchors * strides             # (A, 2) - anchor o khong gian pixel

        pred_dist = reg_raw.transpose(1, 2).contiguous()   # (bs, A, 4*reg_max) grid units

        with torch.no_grad():
            pd_scores_sig = cls_raw.detach().sigmoid()

        target_labels, target_bboxes_pixel, target_scores, fg_mask, _ = assigner(
            pd_scores_sig, box_pixel.detach(), anchors_pixel,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum().item(), 1)

        # --- classification loss (tinh tren toan bo anchor, ca duong lan am) ---
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum

        # --- box + dfl loss (chi tinh tren anchor duong), quy ve khong gian grid ---
        pred_bboxes_grid = box_pixel / stride_b
        target_bboxes_grid = target_bboxes_pixel / stride_b
        loss_iou, loss_dfl = self.bbox_loss(
            pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid,
            target_scores, target_scores_sum, fg_mask,
        )

        n_pos = fg_mask.sum().item()
        return loss_iou, loss_cls, loss_dfl, n_pos

    def forward(self, preds, targets):
        """
        preds  : dict tra ve tu DetectHead o che do train:
                 {"o2m": {"cls","box","reg_raw"}, "o2o": {...}, "anchors", "strides"}
        targets: list[dict] do dai = batch_size, {"boxes": (N,4) xyxy pixel, "labels": (N,)}
        -> total_loss (scalar co grad), dict cac thanh phan loss (de log, .item() san)
        """
        device = preds["anchors"].device
        batch_size = preds["o2o"]["cls"].shape[0]
        anchors, strides = preds["anchors"], preds["strides"]

        gt_bboxes, gt_labels, mask_gt = self.preprocess_targets(targets, batch_size, device)

        iou_m, cls_m, dfl_m, npos_m = self._branch_loss(
            self.assigner_o2m, preds["o2m"]["cls"], preds["o2m"]["box"], preds["o2m"]["reg_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt,
        )
        iou_o, cls_o, dfl_o, npos_o = self._branch_loss(
            self.assigner_o2o, preds["o2o"]["cls"], preds["o2o"]["box"], preds["o2o"]["reg_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt,
        )

        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        items = {
            "loss": total.detach().item(),
            "loss_o2m": loss_o2m.detach().item(),
            "loss_o2o": loss_o2o.detach().item(),
            "o2m/iou": iou_m.detach().item(), "o2m/cls": cls_m.detach().item(), "o2m/dfl": dfl_m.detach().item(),
            "o2o/iou": iou_o.detach().item(), "o2o/cls": cls_o.detach().item(), "o2o/dfl": dfl_o.detach().item(),
            "o2m/n_pos": npos_m, "o2o/n_pos": npos_o,
        }
        return total, items