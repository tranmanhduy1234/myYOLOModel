from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision.ops import box_iou


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    idx = np.clip(idx, 0, len(precision) - 1)
    return float(precision[idx].mean())


class MetricAccumulator:
    """Gom prediction/GT của nhánh o2o qua từng batch, compute() 1 lần cuối ra map_50_95/map_50/precision/recall/per_class_ap."""

    def __init__(
        self,
        nc: int,
        iou_thresholds: Optional[List[float]] = None,
        score_thres: float = 0.001,
        max_det: int = 300,
        pr_iou_thres: float = 0.5,
        pr_score_thres: float = 0.25,
    ) -> None:
        self.nc = nc
        self.iou_thresholds = iou_thresholds or list(np.round(np.arange(0.5, 1.0, 0.05), 2))
        self.score_thres = score_thres
        self.max_det = max_det
        self.pr_iou_thres = pr_iou_thres
        self.pr_score_thres = pr_score_thres

        self.preds_by_class = {c: [] for c in range(nc)}
        self.gts_by_class = {c: {} for c in range(nc)}
        self.n_gt_per_class = np.zeros(nc, dtype=np.int64)
        self.tp_pr = self.fp_pr = self.fn_pr = 0
        self.img_id = 0

    @torch.no_grad()
    def update(self, preds: Dict, targets: List[Dict]) -> None:
        cls_logits, boxes = preds["o2o"]["cls"], preds["o2o"]["box"]
        scores_all = torch.sigmoid(cls_logits)
        scores, labels = scores_all.max(dim=-1)

        for b in range(boxes.shape[0]):
            # === SỬA TẠI ĐÂY: Đưa tất cả tensor của sample về CPU ngay từ đầu ===
            gt_boxes = targets[b]["boxes"].detach().cpu()
            gt_labels = targets[b]["labels"].detach().cpu()
            b_boxes = boxes[b].detach().cpu()
            b_scores = scores[b].detach().cpu()
            b_labels = labels[b].detach().cpu()

            for c in gt_labels.unique().tolist():
                c = int(c)
                cls_gt = gt_boxes[gt_labels == c]
                self.gts_by_class[c][self.img_id] = cls_gt # Đã ở CPU, an toàn tuyệt đối
                self.n_gt_per_class[c] += len(cls_gt)

            keep = b_scores >= self.score_thres
            p_boxes, p_scores, p_labels = b_boxes[keep], b_scores[keep], b_labels[keep]
            if len(p_scores) > self.max_det:
                topk = p_scores.argsort(descending=True)[: self.max_det]
                p_boxes, p_scores, p_labels = p_boxes[topk], p_scores[topk], p_labels[topk]

            for box, score, c in zip(p_boxes, p_scores, p_labels.tolist()):
                self.preds_by_class[c].append((self.img_id, float(score), box))

            keep_pr = p_scores >= self.pr_score_thres
            pr_boxes, pr_labels = p_boxes[keep_pr], p_labels[keep_pr]
            
            # Đã cùng ở CPU nên matched_gt tạo ở CPU không gây văng lỗi Device mismatch nữa
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
            for box, c in zip(pr_boxes, pr_labels.tolist()):
                cand = (gt_labels == c) & (~matched_gt)
                if cand.any() and len(gt_boxes):
                    ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
                    ious[~cand] = 0.0
                    best_iou, best_j = ious.max(0)
                    if best_iou >= self.pr_iou_thres:
                        self.tp_pr += 1
                        matched_gt[best_j] = True
                        continue
                self.fp_pr += 1
            self.fn_pr += int((~matched_gt).sum())

            self.img_id += 1

    def compute(self) -> Dict:
        per_class_ap: List[float] = [0.0] * self.nc
        per_class_ap50: List[float] = [0.0] * self.nc

        for c in range(self.nc):
            if self.n_gt_per_class[c] == 0:
                continue
            preds_c = sorted(self.preds_by_class[c], key=lambda x: -x[1])
            ap_per_iou = []
            for iou_thr in self.iou_thresholds:
                matched = {img: torch.zeros(len(b), dtype=torch.bool) for img, b in self.gts_by_class[c].items()}
                tp = np.zeros(len(preds_c))
                fp = np.zeros(len(preds_c))
                for i, (img, score, box) in enumerate(preds_c):
                    gt_boxes = self.gts_by_class[c].get(img)
                    if gt_boxes is None or len(gt_boxes) == 0:
                        fp[i] = 1
                        continue
                    ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
                    best_iou, best_j = ious.max(0)
                    if best_iou >= iou_thr and not matched[img][best_j]:
                        tp[i] = 1
                        matched[img][best_j] = True
                    else:
                        fp[i] = 1
                tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
                recall = tp_cum / self.n_gt_per_class[c]
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                ap = _ap_101(recall, precision)
                ap_per_iou.append(ap)
                if iou_thr == 0.5:
                    per_class_ap50[c] = ap
            per_class_ap[c] = float(np.mean(ap_per_iou))

        valid = self.n_gt_per_class > 0
        map_50_95 = float(np.mean(np.array(per_class_ap)[valid])) if valid.any() else 0.0
        map_50 = float(np.mean(np.array(per_class_ap50)[valid])) if valid.any() else 0.0
        precision = self.tp_pr / max(self.tp_pr + self.fp_pr, 1)
        recall = self.tp_pr / max(self.tp_pr + self.fn_pr, 1)

        return {
            "map_50_95": map_50_95,
            "map_50": map_50,
            "precision": precision,
            "recall": recall,
            "per_class_ap": per_class_ap,
        }

@torch.no_grad()
def compute_map_metrics(model, loader, device, nc: int, move_batch, **kwargs) -> Dict:
    """Wrapper standalone (duyệt loader riêng) - giữ lại cho trường hợp cần chạy mAP độc lập với validate."""
    model.eval()
    acc = MetricAccumulator(nc=nc, **kwargs)
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        preds = model(images)
        acc.update(preds, targets)
    return acc.compute()