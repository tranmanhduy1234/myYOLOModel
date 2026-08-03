import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.loss import TaskAlignedAssigner, BboxLoss
try:
    from .config_lmk import FaceLmkConfig
except ImportError:
    from config_lmk import FaceLmkConfig

"""
Định dạng dữ liệu đầu vào của FaceLandmarkDetectionLoss.forward(preds, targets)
Ký hiệu: B = batch size, S = cfg.image_size (dataset.py), K = num_landmarks,
         nc = FaceLmkConfig.nc (mặc định 1), reg_max = FaceLmkConfig.reg_max,
         A = tổng số anchor cộng dồn trên mọi stride/scale (A = sum_s H_s * W_s).

1) targets: list dài B (= batch['targets'] từ face_landmark_collate trong dataset.py).
   Mỗi phần tử targets[i] là 1 dict ứng với 1 ảnh trong batch:
     - 'boxes'          : Tensor[Ni, 4] float32, (x1, y1, x2, y2) theo PIXEL của ảnh đã resize,
                           trong đoạn [0, S]. Ni = số mặt trong ảnh i, có thể = 0.
     - 'labels'         : Tensor[Ni]    long,    luôn = 0 (chỉ 1 class "face").
     - 'landmarks'      : Tensor[Ni, K, 2] float32, (x, y) theo PIXEL cùng hệ toạ độ với 'boxes'.
     - 'landmarks_valid': Tensor[Ni]    bool,    True nếu mặt đó có landmark annotation hợp lệ
                           (mặt có landmarks_valid=False vẫn tính loss box/cls, chỉ loss landmark bị bỏ).
   Không có chiều batch chung vì Ni khác nhau giữa các ảnh - đây là lý do preprocess_targets()
   phải pad về Tensor[B, n_max, ...] (n_max = Ni lớn nhất trong batch) trước khi dùng.

2) preds: dict trả về từ DetectHeadFaceLmk.forward(feats) khi model đang ở train() mode
   (model_face_lmk.py / model.py). Gồm 2 nhánh dự đoán song song trên CÙNG một tập anchor
   (one-to-many cho gán nhãn nhiều-anchor-một-GT, one-to-one cho suy luận không cần NMS):
     preds['o2m'] và preds['o2o'] - mỗi nhánh có cùng cấu trúc:
       - 'cls'    : Tensor[B, A, nc] float32, logit RAW (CHƯA sigmoid) - điểm phân loại face/không-face.
       - 'box'    : Tensor[B, A, 4] float32, (x1, y1, x2, y2) đã decode qua DFL (decode_box),
                    theo PIXEL cùng hệ toạ độ với targets['boxes'] (đã nhân stride).
       - 'reg_raw': Tensor[B, 4*reg_max, A] float32, logit RAW phân phối DFL cho (l, t, r, b)
                    theo đơn vị GRID (chưa nhân stride) - dùng để tính lại loss_dfl/loss_iou
                    trong BboxLoss (không dùng lại 'box' cho phần này).
       - 'lmk_raw': Tensor[B, K*2, A] float32, logit RAW landmark (CHƯA sigmoid). Muốn ra toạ độ
                    pixel thật phải sigmoid() rồi ánh xạ vào vùng bbox+lmk_margin (decode_landmarks),
                    đây cũng là phép biến đổi mà _encode_landmark_targets() làm ngược lại để tạo target.
     preds['anchors'] : Tensor[A, 2] float32, tâm anchor theo đơn vị GRID của từng stride
                        (CHƯA nhân stride - phải nhân với preds['strides'] để ra pixel).
     preds['strides']  : Tensor[A, 1] float32, stride (8/16/32...) tương ứng của từng anchor trong A.

   Lưu ý: A giống nhau giữa o2m/o2o vì cả 2 nhánh dùng chung lưới anchor (make_anchors chạy 1 lần
   trên feats, không phụ thuộc nhánh). B, A áp dụng cho mọi tensor trong preds['o2m']/preds['o2o'].
"""


class FaceLandmarkDetectionLoss(nn.Module):

    def __init__(self, cfg: FaceLmkConfig):
        super().__init__()
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        self.box_gain, self.cls_gain, self.dfl_gain, self.lmk_gain = cfg.box_gain, cfg.cls_gain, cfg.dfl_gain, cfg.lmk_gain
        self.o2m_weight, self.o2o_weight = cfg.o2m_weight, cfg.o2o_weight
        self.lmk_margin = cfg.lmk_margin
        self.assigner_o2m = TaskAlignedAssigner(topk=cfg.topk_o2m, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=cfg.topk_o2o, num_classes=cfg.nc, alpha=cfg.alpha, beta=cfg.beta)
        self.bbox_loss = BboxLoss(cfg.reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        point_weights = torch.ones(self.num_landmarks, dtype=torch.float32)
        for indices, weight in (
            (cfg.eye_landmark_indices, cfg.eye_landmark_weight),
            (cfg.mouth_landmark_indices, cfg.mouth_landmark_weight),
            (cfg.nose_tip_landmark_indices, cfg.nose_tip_landmark_weight),
        ):
            valid_indices = [idx for idx in indices if 0 <= idx < self.num_landmarks]
            if valid_indices:
                point_weights[valid_indices] = weight
        self.register_buffer('landmark_point_weights', point_weights, persistent=True)

    def preprocess_targets(self, targets, batch_size, device):
        n_max = max((t['boxes'].shape[0] for t in targets), default=0)
        n_max = max(n_max, 1)
        K = self.num_landmarks
        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)
        gt_landmarks = torch.zeros(batch_size, n_max, K, 2, device=device)
        gt_lmk_valid = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            n = t['boxes'].shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = t['boxes'].to(device)
            gt_labels[i, :n, 0] = t['labels'].to(device)
            mask_gt[i, :n, 0] = True
            if t.get('landmarks') is not None:
                lm = t['landmarks']
                if lm.shape[1] != K:
                    raise ValueError(f"target['landmarks'] có K={lm.shape[1]} nhưng loss cấu hình num_landmarks={K}.")
                gt_landmarks[i, :n] = lm.to(device)
                valid = t.get('landmarks_valid')
                gt_lmk_valid[i, :n] = valid.to(device) if valid is not None else True
        return gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid

    @staticmethod
    def _gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask):
        bs, M = gt_landmarks.shape[0], gt_landmarks.shape[1]
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_landmarks.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * M
        target_landmarks = gt_landmarks.view(-1, gt_landmarks.shape[2], 2)[flat_idx]
        target_lmk_mask = fg_mask & gt_lmk_valid.view(-1)[flat_idx]
        return target_landmarks, target_lmk_mask

    def _encode_landmark_targets(self, target_landmarks_pixel, target_bboxes_pixel):
        x1, y1, x2, y2 = target_bboxes_pixel.unbind(-1)
        w, h = x2 - x1, y2 - y1
        m = self.lmk_margin
        x1e = (x1 - m * w).unsqueeze(-1)
        y1e = (y1 - m * h).unsqueeze(-1)
        we = (w * (1 + 2 * m)).unsqueeze(-1).clamp(min=0.001)
        he = (h * (1 + 2 * m)).unsqueeze(-1).clamp(min=0.001)
        tx = (target_landmarks_pixel[..., 0] - x1e) / we
        ty = (target_landmarks_pixel[..., 1] - y1e) / he
        return torch.stack([tx, ty], dim=-1).clamp(0.0, 1.0)

    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, lmk_raw, anchors, strides,
                      gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid):
        bs, A, K = cls_raw.shape[0], cls_raw.shape[1], self.num_landmarks
        stride_b = strides.unsqueeze(0)
        anchors_pixel = anchors * strides
        pred_dist = reg_raw.transpose(1, 2).contiguous()
        pred_lmk_norm = torch.sigmoid(lmk_raw).transpose(1, 2).view(bs, A, K, 2)
        with torch.no_grad():
            pd_scores_sig = cls_raw.detach().sigmoid()
        target_labels, target_bboxes_pixel, target_scores, fg_mask, target_gt_idx = assigner(
            pd_scores_sig, box_pixel.detach(), anchors_pixel, gt_labels, gt_bboxes, mask_gt)
        # Giữ denominator trên device để không GPU->CPU sync giữa forward.
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum

        pred_bboxes_grid = box_pixel / stride_b
        target_bboxes_grid = target_bboxes_pixel / stride_b
        loss_iou, loss_dfl = self.bbox_loss(pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid,
                                             target_scores, target_scores_sum, fg_mask)

        target_landmarks_pixel, target_lmk_mask = self._gather_landmark_targets(gt_landmarks, gt_lmk_valid, target_gt_idx, fg_mask)
        n_lmk_pos = target_lmk_mask.sum()
        target_norm = self._encode_landmark_targets(target_landmarks_pixel, target_bboxes_pixel)
        pred_sel = pred_lmk_norm[target_lmk_mask]
        target_sel = target_norm[target_lmk_mask]
        assignment_weights = target_scores.sum(-1)[target_lmk_mask, None, None]
        point_weights = self.landmark_point_weights.view(1, K, 1)
        per_coordinate = F.smooth_l1_loss(
            pred_sel, target_sel, beta=self.cfg.lmk_smooth_l1_beta, reduction='none'
        )
        weighted = per_coordinate * assignment_weights * point_weights
        normalizer = assignment_weights.sum() * point_weights.sum() * 2
        loss_lmk = weighted.sum() / normalizer.clamp_min(1e-9)

        n_pos = fg_mask.sum()
        return loss_iou, loss_cls, loss_dfl, loss_lmk, n_pos, n_lmk_pos

    def forward(self, preds, targets):
        device = preds['anchors'].device
        batch_size = preds['o2o']['cls'].shape[0]
        anchors, strides = preds['anchors'], preds['strides']
        gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid = self.preprocess_targets(targets, batch_size, device)

        iou_m, cls_m, dfl_m, lmk_m, npos_m, nlmk_m = self._branch_loss(
            self.assigner_o2m, preds['o2m']['cls'], preds['o2m']['box'], preds['o2m']['reg_raw'], preds['o2m']['lmk_raw'],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid)
        iou_o, cls_o, dfl_o, lmk_o, npos_o, nlmk_o = self._branch_loss(
            self.assigner_o2o, preds['o2o']['cls'], preds['o2o']['box'], preds['o2o']['reg_raw'], preds['o2o']['lmk_raw'],
            anchors, strides, gt_bboxes, gt_labels, mask_gt, gt_landmarks, gt_lmk_valid)

        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m + self.lmk_gain * lmk_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o + self.lmk_gain * lmk_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        items = {
            'loss': total.detach().item(), 'loss_o2m': loss_o2m.detach().item(), 'loss_o2o': loss_o2o.detach().item(),
            'o2m/iou': iou_m.detach().item(), 'o2m/cls': cls_m.detach().item(), 'o2m/dfl': dfl_m.detach().item(), 'o2m/lmk': lmk_m.detach().item(),
            'o2o/iou': iou_o.detach().item(), 'o2o/cls': cls_o.detach().item(), 'o2o/dfl': dfl_o.detach().item(), 'o2o/lmk': lmk_o.detach().item(),
            'o2m/n_pos': npos_m.detach().item(), 'o2o/n_pos': npos_o.detach().item(),
            'o2m/n_lmk_pos': nlmk_m.detach().item(), 'o2o/n_lmk_pos': nlmk_o.detach().item(),
        }
        return total, items
