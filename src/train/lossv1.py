import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_iou_ciou(box1, box2, eps=1e-7):
    """box1: (N,4) box2: (N,4) xyxy, tra ve CIoU tung cap (N,)."""
    b1x1, b1y1, b1x2, b1y2 = box1.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2.unbind(-1)
    
    inter_x1 = torch.max(b1x1, b2x1)
    inter_y1 = torch.max(b1y1, b2y1)
    inter_x2 = torch.min(b1x2, b2x2)
    inter_y2 = torch.min(b1y2, b2y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    w1, h1 = (b1x2 - b1x1).clamp(0), (b1y2 - b1y1).clamp(0)
    w2, h2 = (b2x2 - b2x1).clamp(0), (b2y2 - b2y1).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cx1, cy1 = torch.min(b1x1, b2x1), torch.min(b1y1, b2y1)
    cx2, cy2 = torch.max(b1x2, b2x2), torch.max(b1y2, b2y2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    p1x, p1y = (b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2
    p2x, p2y = (b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2
    rho2 = (p1x - p2x) ** 2 + (p1y - p2y) ** 2

    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou.clamp(-1.0, 1.0)

def bbox_iou_plain(box1, box2, eps=1e-7):
    """IoU thuong, dung lam metric matching (khong can dao ham on dinh nhu CIoU)."""
    b1x1, b1y1, b1x2, b1y2 = box1.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2.unbind(-1)
    inter_x1 = torch.max(b1x1, b2x1)
    inter_y1 = torch.max(b1y1, b2y1)
    inter_x2 = torch.min(b1x2, b2x2)
    inter_y2 = torch.min(b1y2, b2y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    w1, h1 = (b1x2 - b1x1).clamp(0), (b1y2 - b1y1).clamp(0)
    w2, h2 = (b2x2 - b2x1).clamp(0), (b2y2 - b2y1).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union

class TaskAlignedAssigner:
    """
    Gan gt cho anchor point dua tren alignment metric = cls_score^alpha * iou^beta.
    topk=1  -> dung cho nhanh one-to-one (NMS-free).
    topk>1  -> dung cho nhanh one-to-many (giam sat phong phu, giong YOLOv8/v10).
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_boxes, anchor_points, gt_boxes, gt_labels, gt_mask):
        """
        pred_scores: (B, A, nc) sigmoid logits (dung raw logits -> sigmoid ben trong)
        pred_boxes : (B, A, 4) xyxy pixel
        anchor_points: (A, 2) toa do pixel tam anchor
        gt_boxes: (B, M, 4) xyxy pixel (co the pad 0)
        gt_labels: (B, M) long
        gt_mask: (B, M) bool, True = gt that (khong phai padding)
        Tra ve: target_labels (B,A), target_boxes (B,A,4), target_scores (B,A,nc), fg_mask (B,A)
        """
        B, A, nc = pred_scores.shape
        M = gt_boxes.shape[1]
        device = pred_scores.device

        if M == 0:
            return (torch.zeros(B, A, dtype=torch.long, device=device),
                    torch.zeros(B, A, 4, device=device),
                    torch.zeros(B, A, nc, device=device),
                    torch.zeros(B, A, dtype=torch.bool, device=device))

        # 1) anchor phai nam trong gt box
        ax, ay = anchor_points[:, 0], anchor_points[:, 1]
        lt = torch.stack([ax[None, None] - gt_boxes[..., 0:1], ay[None, None] - gt_boxes[..., 1:2]], dim=-1)
        rb = torch.stack([gt_boxes[..., 2:3] - ax[None, None], gt_boxes[..., 3:4] - ay[None, None]], dim=-1)
        deltas = torch.cat([lt, rb], dim=-1)  # (B,M,A,4)
        in_box = deltas.min(-1).values > 1e-3  # (B,M,A)

        # 2) cls score cho dung class cua gt
        pred_scores_sig = pred_scores.sigmoid()  # (B,A,nc)
        cls_for_gt = torch.gather(
            pred_scores_sig.unsqueeze(1).expand(-1, M, -1, -1), 3,
            gt_labels[..., None, None].expand(-1, -1, A, 1).clamp(0, nc - 1)
        ).squeeze(-1)  # (B,M,A)

        # 3) iou giua moi gt va moi pred box
        ious = bbox_iou_plain(
            gt_boxes.unsqueeze(2).expand(-1, -1, A, -1).reshape(-1, 4),
            pred_boxes.unsqueeze(1).expand(-1, M, -1, -1).reshape(-1, 4),
        ).view(B, M, A).clamp(0)

        align_metric = cls_for_gt.pow(self.alpha) * ious.pow(self.beta)
        align_metric = align_metric * in_box * gt_mask[..., None]

        # 4) chon topk anchor tot nhat cho moi gt
        topk = min(self.topk, A)
        topk_vals, topk_idx = align_metric.topk(topk, dim=-1)  # (B,M,topk)
        candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)
        valid = topk_vals > self.eps
        candidate_mask.scatter_(-1, topk_idx, valid)

        # 5) neu 1 anchor duoc >1 gt chon -> giu gt co align_metric cao nhat
        align_metric = align_metric * candidate_mask
        max_align, gt_idx_per_anchor = align_metric.max(dim=1)  # (B,A)
        fg_mask = max_align > self.eps

        target_labels = torch.gather(gt_labels, 1, gt_idx_per_anchor.clamp(0, M - 1))
        target_boxes = torch.gather(
            gt_boxes, 1, gt_idx_per_anchor.clamp(0, M - 1)[..., None].expand(-1, -1, 4)
        )
        target_labels = target_labels * fg_mask.long()

        # target_scores: soft label = normalized align metric (quality-aware, kieu varifocal)
        target_scores = torch.zeros(B, A, nc, device=device)
        norm_align = max_align.clone()
        # normalize theo gt: scale ve [0, max_iou_of_that_gt] cho on dinh (rut gon)
        target_scores.scatter_(-1, target_labels.unsqueeze(-1).clamp(0, nc - 1),
                                norm_align.unsqueeze(-1))
        target_scores = target_scores * fg_mask[..., None]

        return target_labels, target_boxes, target_scores, fg_mask

class DetectionLoss(nn.Module):
    """
    Tong loss = loss(o2m) + loss(o2o)
    Moi nhanh = w_cls * BCE(cls, soft target) + w_box * CIoU_loss + w_dfl * DFL_loss
    o2m dung TaskAlignedAssigner topk=10 (nhieu positive/gt -> hoc phong phu)
    o2o dung TaskAlignedAssigner topk=1  (1 positive/gt -> NMS-free khi infer)
    """

    def __init__(self, nc=80, reg_max=16,
                 w_cls=1.0, w_box=7.5, w_dfl=1.5,
                 w_o2o=1.0):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_dfl = w_dfl
        self.w_o2o = w_o2o
        self.assigner_o2m = TaskAlignedAssigner(topk=10, num_classes=nc)
        self.assigner_o2o = TaskAlignedAssigner(topk=1, num_classes=nc)

    def _dfl_loss(self, reg_raw, target_boxes, anchor_points, stride, fg_mask):
        """reg_raw: (B, 4*reg_max, A) logits truoc softmax; target ltrb (don vi grid cell)."""
        B, _, A = reg_raw.shape
        reg_max = self.reg_max
        anchors = anchor_points.unsqueeze(0)  # (1,A,2)
        stride = stride.unsqueeze(0)          # (1,A,1)

        tgt_lt = (anchors - target_boxes[..., :2]) / stride
        tgt_rb = (target_boxes[..., 2:] - anchors) / stride
        target_ltrb = torch.cat([tgt_lt, tgt_rb], -1).clamp(0, reg_max - 1 - 0.01)  # (B,A,4)

        reg = reg_raw.view(B, 4, reg_max, A).permute(0, 3, 1, 2)  # (B,A,4,reg_max)

        tl = target_ltrb.floor()
        tr = tl + 1
        wl = tr - target_ltrb
        wr = 1 - wl

        loss = (F.cross_entropy(reg.reshape(-1, reg_max), tl.reshape(-1).long(), reduction="none") * wl.reshape(-1)
                + F.cross_entropy(reg.reshape(-1, reg_max), tr.clamp(max=reg_max - 1).reshape(-1).long(), reduction="none") * wr.reshape(-1))
        loss = loss.view(B, A, 4).mean(-1)
        return (loss * fg_mask).sum() / fg_mask.sum().clamp(min=1)

    def _branch_loss(self, cls_logits, boxes, reg_raw, anchors, stride, gt_boxes, gt_labels, gt_mask, assigner):
        t_labels, t_boxes, t_scores, fg_mask = assigner(cls_logits.detach(), boxes.detach(),
                                                          anchors, gt_boxes, gt_labels, gt_mask)
        n_pos = fg_mask.sum().clamp(min=1)

        # cls loss: BCE voi soft target tren TOAN BO anchor (fg + bg)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, t_scores, reduction="none").sum() / n_pos

        # box loss: CIoU chi tren fg
        if fg_mask.sum() > 0:
            ciou = bbox_iou_ciou(boxes[fg_mask], t_boxes[fg_mask])
            box_loss = (1.0 - ciou).sum() / n_pos
        else:
            box_loss = boxes.sum() * 0.0

        dfl_loss = self._dfl_loss(reg_raw, t_boxes, anchors, stride, fg_mask.float())

        total = self.w_cls * cls_loss + self.w_box * box_loss + self.w_dfl * dfl_loss
        return total, {"cls": cls_loss.item(), "box": box_loss.item(), "dfl": dfl_loss.item(), "n_pos": int(fg_mask.sum())}

    def forward(self, preds, gt_boxes, gt_labels, gt_mask):
        """
        preds: dict tra ve tu DetectHead.forward()
        gt_boxes : (B, M, 4) xyxy pixel, da pad
        gt_labels: (B, M) long, da pad (gia tri padding tuy y, se bi mask)
        gt_mask  : (B, M) bool, True = box that
        """
        anchors, stride = preds["anchors"], preds["strides"]

        loss_o2m, log_o2m = self._branch_loss(
            preds["o2m"]["cls"], preds["o2m"]["box"], preds["o2m"]["reg_raw"],
            anchors, stride, gt_boxes, gt_labels, gt_mask, self.assigner_o2m)

        loss_o2o, log_o2o = self._branch_loss(
            preds["o2o"]["cls"], preds["o2o"]["box"], preds["o2o"]["reg_raw"],
            anchors, stride, gt_boxes, gt_labels, gt_mask, self.assigner_o2o)

        total = loss_o2m + self.w_o2o * loss_o2o
        logs = {"loss_total": total.item(), "o2m": log_o2m, "o2o": log_o2o}
        return total, logs