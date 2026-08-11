import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.loss import BboxLoss, TaskAlignedAssigner
from src.transferLearning.config_lmk import FaceLmkConfig
#     pred = {
#         "o2m": {
#             "cls":     Tensor[B, A, nc], classification logits thô, chưa qua sigmoid.
#             "box":     Tensor[B, A, 4], bounding box đã decode từ reg_raw, xyxy=(x_min, y_min, x_max, y_max) theo đơn vị pixel.
#             "reg_raw": Tensor[B, 4 * reg_max, A], phân phối DFL thô cho bốn cạnh left, top, right, bottom.
#             "lmk_raw": Tensor[B, 2 * K, A], signed offsets của landmark so với tâm anchor, theo đơn vị feature-grid.
#         },
#         "o2o": {
#             "cls":     Tensor[B, A, nc],
#             "box":     Tensor[B, A, 4],
#             "reg_raw": Tensor[B, 4 * reg_max, A],
#             "lmk_raw": Tensor[B, 2 * K, A],
#         },
#         "anchors": Tensor[A, 2], tâm anchor (x, y) theo đơn vị feature-grid
#         "strides": Tensor[A, 1],
#     }

#     "targets": List[Dict],            # Danh sách dài B.
#                                       # Mỗi phần tử tương ứng với một ảnh và có cấu trúc:
#                                       # {
#                                       #   "boxes": Tensor[N, 4],           float32,
#                                       #            bbox dạng xyxy theo pixel trên ảnh H×W;
#                                       #
#                                       #   "labels": Tensor[N],             int64,
#                                       #             nhãn class của từng face;
#                                       #
#                                       #   "landmarks": Tensor[N, K, 2],    float32,
#                                       #                tọa độ (x, y) theo pixel trên ảnh H×W;
#                                       #
#                                       #   "landmarks_valid": Tensor[N],    bool,
#                                       #                      True nếu landmark của face đó
#                                       #                      được phép tham gia landmark loss.
#                                       # }

class FaceLandmarkDetectionLoss(nn.Module):
    def __init__(self, cfg: FaceLmkConfig):
        super().__init__()
        self.cfg, self.nc, self.reg_max = cfg, cfg.nc, cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        self.box_gain, self.cls_gain, self.dfl_gain, self.lmk_gain = cfg.box_gain, cfg.cls_gain, cfg.dfl_gain, cfg.lmk_gain
        self.o2m_weight, self.o2o_weight = cfg.o2m_weight, cfg.o2o_weight

        self.assigner_o2m = TaskAlignedAssigner(topk=cfg.topk_o2m, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=cfg.topk_o2o, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.bbox_loss = BboxLoss(cfg.reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        pw = torch.ones(self.num_landmarks, dtype=torch.float32)
        for idxs, w in ((cfg.eye_landmark_indices, cfg.eye_landmark_weight),
                        (cfg.mouth_landmark_indices, cfg.mouth_landmark_weight),
                        (cfg.nose_tip_landmark_indices, cfg.nose_tip_landmark_weight)):
            v_idxs = [i for i in idxs if 0 <= i < self.num_landmarks]
            if v_idxs:
                pw[v_idxs] = w
        self.register_buffer("landmark_point_weights", pw, persistent=True)

    def preprocess_targets(self, targets, batch_size, device):
        n_max = max(max((t["boxes"].shape[0] for t in targets), default=0), 1)
        K = self.num_landmarks

        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)
        gt_landmarks = torch.zeros(batch_size, n_max, K, 2, device=device)
        gt_lmk_valid = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)

        for b_idx, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            gt_bboxes[b_idx, :n] = t["boxes"].to(device)
            gt_labels[b_idx, :n, 0] = t["labels"].to(device)
            mask_gt[b_idx, :n, 0] = True

            lmk = t.get("landmarks")
            if lmk is not None:
                if lmk.ndim != 3 or lmk.shape[1:] != (K, 2):
                    raise ValueError(f"target['landmarks'] phải có shape [N, {K}, 2], nhận {tuple(lmk.shape)}.")
                if lmk.shape[0] != n:
                    raise ValueError(f"Số face của target['landmarks'] khác target['boxes']: {lmk.shape[0]} != {n}.")
                gt_landmarks[b_idx, :n] = lmk.to(device)

                valid = t.get("landmarks_valid")
                if valid is None:
                    gt_lmk_valid[b_idx, :n] = True
                else:
                    if valid.shape[0] != n:
                        raise ValueError(f"target['landmarks_valid'] phải có N phần tử, nhận {tuple(valid.shape)} với N={n}.")
                    gt_lmk_valid[b_idx, :n] = valid.to(device=device, dtype=torch.bool)

        return gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid

    @staticmethod
    def _gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        B, A = target_gt_idx.shape
        num_gt = gt_landmarks.shape[1]
        batch_grid = torch.arange(B, device=target_gt_idx.device, dtype=torch.long).view(B, 1).expand(B, A)
        safe_gt_idx = target_gt_idx.long().clamp(min=0, max=num_gt - 1)
        target_lmk_mask = fg_mask & gt_lmk_valid[batch_grid, safe_gt_idx]

        pos_b, pos_a = target_lmk_mask.nonzero(as_tuple=True)
        pos_gt = safe_gt_idx[pos_b, pos_a]
        return gt_landmarks[pos_b, pos_gt], target_lmk_mask, pos_b, pos_a

    def _landmark_loss(self, lmk_raw, anchors, strides, target_bboxes_pixel, target_scores, gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        B, C, A = lmk_raw.shape
        K = self.num_landmarks
        if C != K * 2: raise ValueError(f"lmk_raw có C={C}, cần C={K * 2} cho K={K}.")
        if anchors.shape != (A, 2): raise ValueError(f"anchors phải có shape {(A, 2)}, nhận {tuple(anchors.shape)}.")
        if strides.shape != (A, 1): raise ValueError(f"strides phải có shape {(A, 1)}, nhận {tuple(strides.shape)}.")

        tgt_lmk_px, tgt_lmk_mask, pos_b, pos_a = self._gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask)
        pred_offsets_grid = lmk_raw.transpose(1, 2)[pos_b, pos_a].reshape(-1, K, 2)

        sel_anchors = anchors[pos_a].to(dtype=pred_offsets_grid.dtype)
        sel_strides = strides[pos_a].to(dtype=pred_offsets_grid.dtype)
        pred_lmk_px = (sel_anchors.unsqueeze(1) + pred_offsets_grid) * sel_strides.unsqueeze(1)

        sel_boxes = target_bboxes_pixel[pos_b, pos_a]
        box_w = (sel_boxes[:, 2] - sel_boxes[:, 0]).clamp_min(self.cfg.lmk_scale_eps)
        box_h = (sel_boxes[:, 3] - sel_boxes[:, 1]).clamp_min(self.cfg.lmk_scale_eps)
        face_scale = torch.sqrt(box_w * box_h).view(-1, 1, 1)

        norm_err = (pred_lmk_px - tgt_lmk_px) / face_scale
        per_coord = F.smooth_l1_loss(norm_err, torch.zeros_like(norm_err), beta=self.cfg.lmk_smooth_l1_beta, reduction="none")

        assign_w = target_scores.sum(-1)[pos_b, pos_a].view(-1, 1, 1)
        pt_w = self.landmark_point_weights.to(dtype=per_coord.dtype).view(1, K, 1)

        weighted = per_coord * assign_w * pt_w
        normalizer = assign_w.sum() * pt_w.sum() * 2.0
        return (
            weighted.sum() / normalizer.clamp_min(self.cfg.loss_normalizer_eps),
            tgt_lmk_mask.sum(),
        )

    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, lmk_raw, anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid):
        stride_b = strides.unsqueeze(0)
        anchors_pixel = anchors * strides
        pred_dist = reg_raw.transpose(1, 2).contiguous()

        with torch.no_grad():
            pred_scores = cls_raw.detach().sigmoid()

        _, tgt_bboxes_px, tgt_scores, fg_mask, tgt_gt_idx = assigner(pred_scores, box_pixel.detach(), anchors_pixel, gt_labels, gt_bboxes, mask_gt)
        tgt_scores_sum = tgt_scores.sum().clamp_min(1.0)
        loss_cls = self.bce(cls_raw, tgt_scores).sum() / tgt_scores_sum

        loss_iou, loss_dfl = self.bbox_loss(pred_dist, box_pixel / stride_b, anchors, tgt_bboxes_px / stride_b, tgt_scores, tgt_scores_sum, fg_mask)
        loss_lmk, n_lmk_pos = self._landmark_loss(lmk_raw, anchors, strides, tgt_bboxes_px, tgt_scores, gt_landmarks, gt_lmk_valid, tgt_gt_idx, fg_mask)

        return loss_iou, loss_cls, loss_dfl, loss_lmk, fg_mask.sum(), n_lmk_pos

    def forward(self, preds, targets):
        device, B = preds["anchors"].device, preds["o2o"]["cls"].shape[0]
        anchors, strides = preds["anchors"], preds["strides"]

        gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid = self.preprocess_targets(targets, B, device)

        iou_m, cls_m, dfl_m, lmk_m, npos_m, nlmk_m = self._branch_loss(
            self.assigner_o2m, preds["o2m"]["cls"], preds["o2m"]["box"], preds["o2m"]["reg_raw"], preds["o2m"]["lmk_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid
        )
        iou_o, cls_o, dfl_o, lmk_o, npos_o, nlmk_o = self._branch_loss(
            self.assigner_o2o, preds["o2o"]["cls"], preds["o2o"]["box"], preds["o2o"]["reg_raw"], preds["o2o"]["lmk_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid
        )

        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m + self.lmk_gain * lmk_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o + self.lmk_gain * lmk_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        items = {
            "loss": total.detach().item(), "loss_o2m": loss_o2m.detach().item(), "loss_o2o": loss_o2o.detach().item(),
            "o2m/iou": iou_m.detach().item(), "o2m/cls": cls_m.detach().item(), "o2m/dfl": dfl_m.detach().item(), "o2m/lmk": lmk_m.detach().item(),
            "o2o/iou": iou_o.detach().item(), "o2o/cls": cls_o.detach().item(), "o2o/dfl": dfl_o.detach().item(), "o2o/lmk": lmk_o.detach().item(),
            "o2m/n_pos": npos_m.detach().item(), "o2o/n_pos": npos_o.detach().item(),
            "o2m/n_lmk_pos": nlmk_m.detach().item(), "o2o/n_lmk_pos": nlmk_o.detach().item()
        }
        return total, items
