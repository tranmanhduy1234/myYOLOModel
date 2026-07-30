import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.loss import bbox_iou, dist2bbox, bbox2dist, TaskAlignedAssigner, BboxLoss
from face_lmk_config import FaceLmkConfig

def normalized_wing_loss(pred, target, w=0.1, epsilon=0.02):
    diff = (pred - target).abs()
    import math
    C = w - w * math.log(1 + w / epsilon)
    return torch.where(diff < w, w * torch.log(1 + diff / epsilon), diff - C)

def landmark_regression_loss(pred_norm, target_norm, loss_type='smooth_l1', beta=0.05):
    if loss_type == 'wing':
        return normalized_wing_loss(pred_norm, target_norm)
    return F.smooth_l1_loss(pred_norm, target_norm, beta=beta, reduction='none')

def geometric_consistency_loss(pred_norm, constraints, margin=0.02):
    if pred_norm.numel() == 0 or not constraints:
        return pred_norm.sum() * 0
    terms = []
    for (idx_a, idx_b, axis, sign) in constraints:
        a = pred_norm[:, idx_a, axis]
        b = pred_norm[:, idx_b, axis]
        viol = a - b + margin if sign > 0 else b - a + margin
        terms.append(F.relu(viol))
    return torch.stack(terms, dim=0).mean()

class FaceLandmarkDetectionLoss(nn.Module):

    def __init__(self, cfg: FaceLmkConfig):
        super().__init__()
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        (self.box_gain, self.cls_gain, self.dfl_gain, self.lmk_gain) = (cfg.box_gain, cfg.cls_gain, cfg.dfl_gain, cfg.lmk_gain)
        (self.o2m_weight, self.o2o_weight) = (cfg.o2m_weight, cfg.o2o_weight)
        self.lmk_margin = cfg.lmk_margin
        self.lmk_loss_type = cfg.lmk_loss_type
        self.geo_constraints = cfg.geo_constraints or []
        self.geo_gain = cfg.geo_gain
        self.geo_margin = cfg.geo_margin
        self.assigner_o2m = TaskAlignedAssigner(topk=cfg.topk_o2m, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=cfg.topk_o2o, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.bbox_loss = BboxLoss(cfg.reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def preprocess_targets(self, targets, batch_size, device):
        n_max = max((t['boxes'].shape[0] for t in targets), default=0)
        n_max = max(n_max, 1)
        K = self.num_landmarks
        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)
        gt_landmarks = torch.zeros(batch_size, n_max, K, 2, device=device)
        gt_lmk_valid = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
        for (i, t) in enumerate(targets):
            n = t['boxes'].shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = t['boxes'].to(device)
            gt_labels[i, :n, 0] = t['labels'].to(device)
            mask_gt[i, :n, 0] = True
            if 'landmarks' in t and t['landmarks'] is not None and (n > 0):
                lm = t['landmarks']
                if lm.shape[1] != K:
                    raise ValueError(f"target['landmarks'] có K={lm.shape[1]} điểm nhưng FaceLandmarkDetectionLoss được cấu hình num_landmarks={K} (qua cfg.sync_num_landmarks). Kiểm tra lại cfg dùng cho head/loss có được sync đúng với dataset.num_landmarks hay không.")
                gt_landmarks[i, :n] = lm.to(device)
                if 'landmarks_valid' in t and t['landmarks_valid'] is not None:
                    gt_lmk_valid[i, :n] = t['landmarks_valid'].to(device)
                else:
                    gt_lmk_valid[i, :n] = True
        return (gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid)

    @staticmethod
    def _gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        (bs, M) = (gt_landmarks.shape[0], gt_landmarks.shape[1])
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_landmarks.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * M
        target_landmarks = gt_landmarks.view(-1, gt_landmarks.shape[2], 2)[flat_idx]
        target_lmk_has_label = gt_lmk_valid.view(-1)[flat_idx]
        target_lmk_mask = fg_mask & target_lmk_has_label
        return (target_landmarks, target_lmk_mask)

    def _encode_landmark_targets(self, target_landmarks_pixel, target_bboxes_pixel):
        (x1, y1, x2, y2) = target_bboxes_pixel.unbind(-1)
        (w, h) = (x2 - x1, y2 - y1)
        m = self.lmk_margin
        x1e = (x1 - m * w).unsqueeze(-1)
        y1e = (y1 - m * h).unsqueeze(-1)
        we = (w * (1 + 2 * m)).unsqueeze(-1).clamp(min=0.001)
        he = (h * (1 + 2 * m)).unsqueeze(-1).clamp(min=0.001)
        tx = (target_landmarks_pixel[..., 0] - x1e) / we
        ty = (target_landmarks_pixel[..., 1] - y1e) / he
        target_norm = torch.stack([tx, ty], dim=-1).clamp(0.0, 1.0)
        return target_norm

    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, lmk_raw, anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid):
        bs = cls_raw.shape[0]
        A = cls_raw.shape[1]
        K = self.num_landmarks
        stride_b = strides.unsqueeze(0)
        anchors_pixel = anchors * strides
        pred_dist = reg_raw.transpose(1, 2).contiguous()
        pred_lmk_norm = torch.sigmoid(lmk_raw).transpose(1, 2).view(bs, A, K, 2)
        with torch.no_grad():
            pd_scores_sig = cls_raw.detach().sigmoid()
        (target_labels, target_bboxes_pixel, target_scores, fg_mask, target_gt_idx) = assigner(pd_scores_sig, box_pixel.detach(), anchors_pixel, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum().item(), 1)
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum
        pred_bboxes_grid = box_pixel / stride_b
        target_bboxes_grid = target_bboxes_pixel / stride_b
        (loss_iou, loss_dfl) = self.bbox_loss(pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid, target_scores, target_scores_sum, fg_mask)
        (target_landmarks_pixel, target_lmk_mask) = self._gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask)
        n_lmk_pos = target_lmk_mask.sum().item()
        if n_lmk_pos == 0:
            loss_lmk = pred_lmk_norm.sum() * 0
        else:
            target_norm = self._encode_landmark_targets(target_landmarks_pixel, target_bboxes_pixel)
            pred_sel = pred_lmk_norm[target_lmk_mask]
            target_sel = target_norm[target_lmk_mask]
            weight_sel = target_scores.sum(-1)[target_lmk_mask].unsqueeze(-1).unsqueeze(-1)
            per_point = landmark_regression_loss(pred_sel, target_sel, self.lmk_loss_type)
            loss_lmk = (per_point * weight_sel).sum() / (weight_sel.sum() * K * 2 + 1e-09)
        if self.geo_gain > 0 and self.geo_constraints and fg_mask.any():
            pred_fg = pred_lmk_norm[fg_mask]
            loss_geo = geometric_consistency_loss(pred_fg, self.geo_constraints, self.geo_margin)
        else:
            loss_geo = pred_lmk_norm.sum() * 0
        n_pos = fg_mask.sum().item()
        return (loss_iou, loss_cls, loss_dfl, loss_lmk, loss_geo, n_pos, n_lmk_pos)

    def forward(self, preds, targets):
        device = preds['anchors'].device
        batch_size = preds['o2o']['cls'].shape[0]
        (anchors, strides) = (preds['anchors'], preds['strides'])
        (gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid) = self.preprocess_targets(targets, batch_size, device)
        (iou_m, cls_m, dfl_m, lmk_m, geo_m, npos_m, nlmk_m) = self._branch_loss(self.assigner_o2m, preds['o2m']['cls'], preds['o2m']['box'], preds['o2m']['reg_raw'], preds['o2m']['lmk_raw'], anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid)
        (iou_o, cls_o, dfl_o, lmk_o, geo_o, npos_o, nlmk_o) = self._branch_loss(self.assigner_o2o, preds['o2o']['cls'], preds['o2o']['box'], preds['o2o']['reg_raw'], preds['o2o']['lmk_raw'], anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid)
        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m + self.lmk_gain * lmk_m + self.geo_gain * geo_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o + self.lmk_gain * lmk_o + self.geo_gain * geo_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o
        items = {'loss': total.detach().item(), 'loss_o2m': loss_o2m.detach().item(), 'loss_o2o': loss_o2o.detach().item(), 'o2m/iou': iou_m.detach().item(), 'o2m/cls': cls_m.detach().item(), 'o2m/dfl': dfl_m.detach().item(), 'o2m/lmk': lmk_m.detach().item(), 'o2m/geo': geo_m.detach().item(), 'o2o/iou': iou_o.detach().item(), 'o2o/cls': cls_o.detach().item(), 'o2o/dfl': dfl_o.detach().item(), 'o2o/lmk': lmk_o.detach().item(), 'o2o/geo': geo_o.detach().item(), 'o2m/n_pos': npos_m, 'o2o/n_pos': npos_o, 'o2m/n_lmk_pos': nlmk_m, 'o2o/n_lmk_pos': nlmk_o}
        return (total, items)
