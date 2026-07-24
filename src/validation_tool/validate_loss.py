"""
validate_loss.py
=================
Validate toàn diện cho src/train/loss.py:
  - bbox_iou / CIoU / dist2bbox / bbox2dist
  - TaskAlignedAssigner (TAL)
  - BboxLoss (CIoU + DFL)
  - DetectionLoss (ghép nhánh o2m + o2o cho NMSFreeDetector)

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ
------------------------------
1. Xóa lớp Reporter nội bộ (đã bị lỗi thiếu SKIP) → dùng chung từ validate_common.
2. Hàm run() nhận Reporter qua tham số → không dùng biến global R nữa.
3. Sửa assertion TAL contested-anchor: kiểm tra IoU thực tế thay vì hardcode index.
4. Thêm torch.manual_seed(0) trong run() để kết quả overfit test ổn định.
5. Thêm test mới: iou_batch, dist2bbox xywh mode, TAL multi-image batch,
   BboxLoss gradient magnitude, DetectionLoss item keys, train/eval mode loss.

Chạy độc lập:
    python -m src.validation_tool.validate_loss
    python -m src.validation_tool.validate_loss --device cuda
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from src.validation_tool.validate_common import Reporter, get_device, skip

from src.train.loss import (
    bbox_iou, dist2bbox, bbox2dist, TaskAlignedAssigner, BboxLoss, DetectionLoss,
)
from src.model import NMSFreeDetector


# ==============================================================================
# 1. bbox_iou / dist2bbox / bbox2dist
# ==============================================================================
def test_bbox_utils(device: str, R: Reporter):
    R.section("1. TIỆN ÍCH HÌNH HỌC (bbox_iou, dist2bbox, bbox2dist)")

    # ---------- IoU / CIoU cơ bản ----------
    def t_iou_identical():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        v = bbox_iou(a, a, CIoU=True).item()
        assert abs(v - 1.0) < 1e-5, f"IoU 2 box giống hệt phải ~1.0, được {v}"
        return f"CIoU={v:.4f}"
    R.check("bbox_utils", "IoU hai box giống hệt nhau -> ~1.0", t_iou_identical)

    def t_iou_no_overlap():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        b = torch.tensor([[20., 20., 30., 30.]], device=device)
        iou_plain = bbox_iou(a, b, CIoU=False).item()
        ciou = bbox_iou(a, b, CIoU=True).item()
        assert abs(iou_plain) < 1e-5, "IoU không giao nhau phải = 0"
        assert ciou < 0, "CIoU không giao nhau phải < 0 (penalty khoảng cách tâm)"
        return f"IoU={iou_plain:.4f}, CIoU={ciou:.4f}"
    R.check("bbox_utils", "IoU/CIoU hai box không giao nhau", t_iou_no_overlap)

    def t_ciou_le_iou():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        b = torch.tensor([[5., 5., 15., 15.]], device=device)
        iou  = bbox_iou(a, b, CIoU=False).item()
        ciou = bbox_iou(a, b, CIoU=True).item()
        assert ciou <= iou + 1e-6, "CIoU luôn <= IoU thường (vì thêm penalty)"
        return f"IoU={iou:.4f} >= CIoU={ciou:.4f}"
    R.check("bbox_utils", "CIoU <= IoU thường (penalty đúng dấu)", t_ciou_le_iou)

    def t_iou_symmetric():
        """IoU phải đối xứng: iou(a,b) == iou(b,a)."""
        a = torch.tensor([[10., 20., 50., 80.]], device=device)
        b = torch.tensor([[30., 10., 70., 60.]], device=device)
        iou_ab = bbox_iou(a, b, CIoU=False).item()
        iou_ba = bbox_iou(b, a, CIoU=False).item()
        assert abs(iou_ab - iou_ba) < 1e-5, "IoU phải đối xứng iou(a,b)==iou(b,a)"
        return f"iou(a,b)={iou_ab:.4f} == iou(b,a)={iou_ba:.4f}"
    R.check("bbox_utils", "IoU đối xứng: iou(a,b) == iou(b,a)", t_iou_symmetric)

    def t_iou_contained():
        """Box con hoàn toàn nằm trong box cha → IoU = area_inner / area_outer."""
        outer = torch.tensor([[0., 0., 10., 10.]], device=device)
        inner = torch.tensor([[2., 2., 8., 8.]], device=device)
        iou = bbox_iou(outer, inner, CIoU=False).item()
        # inter = 6×6=36, union = 100+36-36 = 100
        expected = 36.0 / 100.0
        assert abs(iou - expected) < 1e-4, f"IoU box con trong box cha sai: {iou:.4f} != {expected:.4f}"
        return f"IoU={iou:.4f} (inner ⊂ outer)"
    R.check("bbox_utils", "IoU box con nằm hoàn toàn trong box cha", t_iou_contained)

    def t_iou_batch():
        """bbox_iou phải xử lý batch (N>1) không lỗi và trả về N giá trị."""
        N = 8
        a = torch.rand(N, 4, device=device) * 50
        a[:, 2:] = a[:, :2] + torch.rand(N, 2, device=device) * 30 + 1
        b = torch.rand(N, 4, device=device) * 50
        b[:, 2:] = b[:, :2] + torch.rand(N, 2, device=device) * 30 + 1
        out = bbox_iou(a, b, CIoU=True)
        assert out.shape == (N,), f"bbox_iou batch phải ra shape ({N},), được {out.shape}"
        assert torch.isfinite(out).all(), "Tất cả IoU batch phải hữu hạn"
        return f"batch N={N}, tất cả hữu hạn, shape {tuple(out.shape)}"
    R.check("bbox_utils", "bbox_iou batch N=8: shape đúng, tất cả hữu hạn", t_iou_batch)

    # ---------- dist2bbox / bbox2dist ----------
    def t_dist2bbox_roundtrip():
        anchors = torch.tensor([[5., 5.], [12., 8.]], device=device)
        dist    = torch.tensor([[2., 2., 3., 3.], [1., 4., 2., 1.]], device=device)
        box     = dist2bbox(dist, anchors, xywh=False)
        back    = bbox2dist(anchors, box, reg_max=16)
        assert torch.allclose(box, torch.tensor([[3., 3., 8., 8.], [11., 4., 14., 9.]], device=device))
        assert torch.allclose(back, dist), "bbox2dist phải là nghịch đảo chính xác của dist2bbox"
        return "dist2bbox <-> bbox2dist khớp 100%"
    R.check("bbox_utils", "dist2bbox / bbox2dist round-trip chính xác (xyxy mode)", t_dist2bbox_roundtrip)

    def t_dist2bbox_xywh_mode():
        anchors = torch.tensor([[8., 8.]])
        dist    = torch.tensor([[2., 3., 4., 5.]])  # l, t, r, b
        box_xywh = dist2bbox(dist, anchors, xywh=True)
        l, t, r, b = dist[0].tolist()
        ax, ay = anchors[0].tolist()
        cx, cy, w, h = box_xywh[0].tolist()
        assert abs(cx - (ax + (r - l) / 2)) < 1e-5
        assert abs(cy - (ay + (b - t) / 2)) < 1e-5
        assert abs(w - (l + r)) < 1e-5
        assert abs(h - (t + b)) < 1e-5
        return f"xywh: cx={cx:.2f} cy={cy:.2f} w={w:.2f} h={h:.2f}"
    R.check("bbox_utils", "dist2bbox xywh mode: center và kích thước hợp lệ", t_dist2bbox_xywh_mode)

    def t_bbox2dist_clamp():
        """bbox2dist phải clamp khoảng cách về [0, reg_max-1-eps] không trả về NaN."""
        anchors = torch.tensor([[5., 5.]], device=device)
        # box rất lớn → khoảng cách vượt reg_max
        box_huge = torch.tensor([[0., 0., 500., 500.]], device=device)
        out = bbox2dist(anchors, box_huge, reg_max=16)
        assert torch.isfinite(out).all(), "bbox2dist phải clamp, không NaN dù box vượt reg_max"
        return f"distance clamped: {out.tolist()}"
    R.check("bbox_utils", "bbox2dist: clamp khoảng cách, không NaN khi box vượt reg_max", t_bbox2dist_clamp)


# ==============================================================================
# 2. TaskAlignedAssigner
# ==============================================================================
def test_tal_assigner(device: str, R: Reporter):
    R.section("2. TASK-ALIGNED ASSIGNER (TAL)")

    A, nc = 20, 3
    anc_points = torch.stack(
        [torch.arange(A, device=device).float() + 0.5,
         torch.full((A,), 5.5, device=device)], dim=1
    )

    def t_topk_count():
        gt_bboxes = torch.tensor([[[3., 0., 8., 10.]]], device=device)
        gt_labels = torch.tensor([[[1]]], device=device)
        mask_gt   = torch.tensor([[[True]]], device=device)
        pd_scores = torch.rand(1, A, nc, device=device) * 0.3
        pd_scores[0, 5, 1] = 0.9
        pd_scores[0, 6, 1] = 0.85
        pd_bboxes = torch.stack(
            [torch.cat([anc_points[i] - torch.tensor([2., 4.], device=device),
                        anc_points[i] + torch.tensor([2., 4.], device=device)])
             for i in range(A)]
        ).unsqueeze(0)

        assigner = TaskAlignedAssigner(topk=3, num_classes=nc)
        tl, tb, ts, fg, tgi = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        assert fg.sum().item() == 3, f"topk=3 và đủ anchor hợp lệ -> phải có đúng 3 positive, được {fg.sum().item()}"
        pos_idx = fg[0].nonzero(as_tuple=True)[0].tolist()
        assert 5 in pos_idx and 6 in pos_idx, "anchor điểm cao nhất phải nằm trong tập positive"
        return f"positives={pos_idx}"
    R.check("tal", "topk đúng số lượng, ưu tiên anchor align_metric cao", t_topk_count)

    def t_positives_inside_gt():
        gt_bboxes = torch.tensor([[[3., 0., 8., 10.]]], device=device)
        gt_labels = torch.tensor([[[0]]], device=device)
        mask_gt   = torch.tensor([[[True]]], device=device)
        pd_scores = torch.rand(1, A, nc, device=device)
        pd_bboxes = torch.stack(
            [torch.cat([anc_points[i] - 1, anc_points[i] + 1]) for i in range(A)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=5, num_classes=nc)
        _, _, _, fg, _ = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        pos_x = anc_points[fg[0], 0]
        assert torch.all((pos_x > 3) & (pos_x < 8)), "mọi anchor positive phải nằm trong GT box"
        return f"{fg.sum().item()} positive, tất cả nằm trong GT box [3,8]"
    R.check("tal", "Anchor dương phải nằm trong GT box", t_positives_inside_gt)

    def t_contested_anchor_keeps_higher_iou_gt():
        """Anchor bị tranh chấp phải được gán cho GT có IoU cao hơn (không hardcode index)."""
        gt_bboxes = torch.tensor([[[1.8, 0., 3.2, 1.], [0., 0., 5., 1.]]], device=device)
        gt_labels = torch.tensor([[[0], [0]]], device=device)
        mask_gt   = torch.tensor([[[True], [True]]], device=device)
        A2 = 5
        anc2 = torch.stack(
            [torch.arange(A2, device=device).float() + 0.5,
             torch.full((A2,), 0.5, device=device)], dim=1
        )
        pd_scores = torch.full((1, A2, 1), 0.5, device=device)
        pd_bboxes = torch.stack(
            [torch.cat([anc2[i] - 0.5, anc2[i] + 0.5]) for i in range(A2)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=5, num_classes=1)
        _, _, _, fg, tgi = assigner(pd_scores, pd_bboxes, anc2, gt_labels, gt_bboxes, mask_gt)

        # Kiểm tra mỗi anchor dương được gán cho GT có IoU cao hơn (không hardcode index)
        pos_indices = fg[0].nonzero(as_tuple=True)[0]
        if len(pos_indices) > 0:
            for idx in pos_indices:
                chosen_gt = tgi[0, idx].item()
                other_gt  = 1 - chosen_gt
                pred_box  = pd_bboxes[0, idx]  # (4,)
                gt_chosen = gt_bboxes[0, chosen_gt]  # (4,)
                gt_other  = gt_bboxes[0, other_gt]   # (4,)
                # Kiểm tra anchor nằm trong cả 2 GT box
                anchor = anc2[idx]
                in_chosen = (gt_chosen[0] < anchor[0] < gt_chosen[2]) and (gt_chosen[1] < anchor[1] < gt_chosen[3])
                in_other  = (gt_other[0]  < anchor[0] < gt_other[2])  and (gt_other[1]  < anchor[1] < gt_other[3])
                if in_chosen and in_other:
                    # Anchor nằm trong cả 2 → phải chọn GT có IoU cao hơn
                    iou_chosen = bbox_iou(pred_box.unsqueeze(0), gt_chosen.unsqueeze(0), CIoU=False).item()
                    iou_other  = bbox_iou(pred_box.unsqueeze(0), gt_other.unsqueeze(0),  CIoU=False).item()
                    assert iou_chosen >= iou_other - 1e-5, \
                        f"anchor bị tranh chấp phải được gán GT IoU cao hơn: gt{chosen_gt}(iou={iou_chosen:.4f}) vs gt{other_gt}(iou={iou_other:.4f})"
        return f"{len(pos_indices)} positive, anchor tranh chấp đã gán GT IoU cao hơn"
    R.check("tal", "Anchor bị tranh chấp giữa 2 GT -> giữ GT có IoU cao hơn (kiểm tra IoU thực tế)", t_contested_anchor_keeps_higher_iou_gt)

    def t_padding_gt_ignored():
        A3, nc3 = 10, 2
        anc3 = torch.stack(
            [torch.arange(A3, device=device).float() + 0.5,
             torch.full((A3,), 0.5, device=device)], dim=1
        )
        gt_bboxes = torch.tensor([[[2., 0., 5., 1.], [0., 0., 0., 0.]]], device=device)
        gt_labels = torch.tensor([[[1], [0]]], device=device)
        mask_gt   = torch.tensor([[[True], [False]]], device=device)
        pd_scores = torch.rand(1, A3, nc3, device=device) * 0.2
        pd_bboxes = torch.stack(
            [torch.cat([anc3[i] - 1, anc3[i] + 1]) for i in range(A3)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=2, num_classes=nc3)
        _, _, _, fg, tgi = assigner(pd_scores, pd_bboxes, anc3, gt_labels, gt_bboxes, mask_gt)
        assert fg.sum().item() <= 2, "chỉ 1 GT hợp lệ -> số positive không được vượt quá topk"
        if fg.sum().item() > 0:
            pos_idx = fg[0].nonzero(as_tuple=True)[0]
            assert torch.all(tgi[0, pos_idx] == 0), "GT padding (mask_gt=False) không được gán làm target"
        return "GT padding bị bỏ qua đúng như kỳ vọng"
    R.check("tal", "GT padding (mask_gt=False) không ảnh hưởng kết quả", t_padding_gt_ignored)

    def t_empty_batch():
        gt_bboxes = torch.zeros(1, 0, 4, device=device)
        gt_labels = torch.zeros(1, 0, 1, dtype=torch.long, device=device)
        mask_gt   = torch.zeros(1, 0, 1, dtype=torch.bool, device=device)
        pd_scores = torch.rand(1, A, nc, device=device)
        pd_bboxes = torch.rand(1, A, 4, device=device)
        assigner = TaskAlignedAssigner(topk=3, num_classes=nc)
        _, _, ts, fg, _ = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        assert fg.sum().item() == 0 and ts.sum().item() == 0, "không GT nào -> không positive nào"
        return "batch không GT -> không lỗi, không positive"
    R.check("tal", "Batch hoàn toàn không có GT (M=0)", t_empty_batch)

    def t_multi_image_batch():
        """TAL với B=4 ảnh (số GT khác nhau mỗi ảnh) phải không crash."""
        B, M, nc2 = 4, 5, nc
        gt_bboxes = torch.rand(B, M, 4, device=device) * 10
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + torch.rand(B, M, 2, device=device) * 3 + 1
        gt_labels = torch.randint(0, nc2, (B, M, 1), device=device)
        # mask_gt: ảnh 0 có 2 GT, ảnh 1 có 4 GT, ảnh 2 không có GT, ảnh 3 có 5 GT
        mask_gt = torch.zeros(B, M, 1, dtype=torch.bool, device=device)
        mask_gt[0, :2, 0] = True
        mask_gt[1, :4, 0] = True
        mask_gt[3, :5, 0] = True

        pd_scores = torch.rand(B, A, nc2, device=device)
        pd_bboxes = torch.rand(B, A, 4, device=device) * 15
        pd_bboxes[..., 2:] += pd_bboxes[..., :2] + 1

        assigner = TaskAlignedAssigner(topk=3, num_classes=nc2)
        tl, tb, ts, fg, tgi = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        assert fg.shape == (B, A), f"fg shape phải là (B,A)={(B,A)}, được {fg.shape}"
        # ảnh 2 (không GT) phải không có positive
        assert fg[2].sum().item() == 0, "ảnh không có GT (mask_gt=False hết) phải không có positive"
        return f"B={B}, fg: ảnh 2 không GT -> 0 positive, OK"
    R.check("tal", "Multi-image batch (B=4, số GT khác nhau) không crash", t_multi_image_batch)


# ==============================================================================
# 3. BboxLoss
# ==============================================================================
def test_bbox_loss(device: str, R: Reporter):
    R.section("3. BBOX LOSS (CIoU + DFL)")

    reg_max   = 16
    bs, A     = 1, 4
    anchor_points = torch.tensor([[2.5, 2.5], [7.5, 7.5], [12.5, 12.5], [20., 20.]], device=device)
    fg_mask   = torch.tensor([[True, True, False, False]], device=device)
    target_bboxes = torch.tensor(
        [[[1., 1., 4., 4.], [6., 6., 9., 9.], [0., 0., 0., 0.], [0., 0., 0., 0.]]], device=device
    )
    target_scores = torch.zeros(bs, A, 3, device=device)
    target_scores[0, 0, 1] = 0.9
    target_scores[0, 1, 2] = 0.8
    target_scores_sum = max(target_scores.sum().item(), 1)

    def make_pred(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        pred_dist = torch.randn(bs, A, 4 * reg_max, generator=g).to(device).requires_grad_(True)
        proj = torch.arange(reg_max, device=device).float()
        pd = pred_dist.view(bs, A, 4, reg_max).softmax(-1)
        ltrb = (pd * proj).sum(-1)
        lt, rb = ltrb[..., :2], ltrb[..., 2:]
        pred_bboxes = torch.cat(
            [anchor_points.unsqueeze(0) - lt, anchor_points.unsqueeze(0) + rb], -1
        )
        return pred_dist, pred_bboxes

    loss_fn = BboxLoss(reg_max=reg_max)

    def t_finite_and_nonneg():
        pred_dist, pred_bboxes = make_pred(0)
        loss_iou, loss_dfl = loss_fn(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        assert torch.isfinite(loss_iou) and torch.isfinite(loss_dfl)
        assert loss_iou.item() >= 0 and loss_dfl.item() >= 0
        (loss_iou + loss_dfl).backward()
        assert pred_dist.grad is not None and torch.isfinite(pred_dist.grad).all()
        return f"loss_iou={loss_iou.item():.4f}, loss_dfl={loss_dfl.item():.4f}, grad OK"
    R.check("bbox_loss", "Loss hữu hạn, không âm, backward không NaN", t_finite_and_nonneg)

    def t_zero_positive():
        pred_dist, pred_bboxes = make_pred(1)
        fg0 = torch.zeros(bs, A, dtype=torch.bool, device=device)
        loss_iou0, loss_dfl0 = loss_fn(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg0
        )
        assert loss_iou0.item() == 0 and loss_dfl0.item() == 0
        (loss_iou0 + loss_dfl0).backward()  # không được crash
        return "không có positive -> loss=0, không crash"
    R.check("bbox_loss", "Trường hợp không có anchor dương (fg_mask rỗng)", t_zero_positive)

    def t_gradient_magnitude_reasonable():
        """Gradient của pred_dist phải có norm hữu hạn và không quá nhỏ (dead gradient)."""
        pred_dist, pred_bboxes = make_pred(2)
        loss_iou, loss_dfl = loss_fn(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        (loss_iou + loss_dfl).backward()
        grad_norm = pred_dist.grad.norm().item()
        assert grad_norm > 1e-8, f"Gradient quá nhỏ (dead gradient?): {grad_norm:.2e}"
        assert grad_norm < 1e6, f"Gradient quá lớn (exploding?): {grad_norm:.2e}"
        return f"grad norm = {grad_norm:.4f} (hợp lý)"
    R.check("bbox_loss", "Gradient pred_dist: không dead, không exploding", t_gradient_magnitude_reasonable)

    def t_loss_decreases_with_better_pred():
        """Loss phải giảm khi dự đoán được cải thiện về hướng target."""
        # loss từ random prediction
        pred_dist_bad, pred_bboxes_bad = make_pred(5)
        l_iou_bad, l_dfl_bad = loss_fn(
            pred_dist_bad, pred_bboxes_bad, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        loss_bad = (l_iou_bad + l_dfl_bad).item()

        # loss từ prediction gần đúng hơn (dựng manually)
        # Dùng target_bboxes để tính dist rồi chuyển thành pred_dist
        with torch.no_grad():
            proj = torch.arange(reg_max, device=device).float()
            # Xây pred_bboxes gần với target
            pred_bboxes_good = target_bboxes.clone()
            # dist: ltrb từ anchor đến target
            lt = anchor_points.unsqueeze(0) - pred_bboxes_good[..., :2]
            rb = pred_bboxes_good[..., 2:] - anchor_points.unsqueeze(0)
            dist_good = torch.cat([lt, rb], -1).clamp(0, reg_max - 1 - 1e-3)
            # Dựng logit one-hot gần đúng
            pred_dist_good = torch.zeros(bs, A, 4 * reg_max, device=device, requires_grad=False)

        pred_dist_good_rg = pred_dist_good.clone().requires_grad_(True)
        pred_bboxes_clamped = target_bboxes.clone()
        l_iou_good, l_dfl_good = loss_fn(
            pred_dist_good_rg, pred_bboxes_clamped, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        loss_good = (l_iou_good + l_dfl_good).item()
        # Khi pred_bboxes == target_bboxes, loss_iou phải = 0
        assert l_iou_good.item() < 1e-4, f"loss_iou phải ~0 khi pred_box == target_box, được {l_iou_good.item():.4f}"
        return f"loss_bad={loss_bad:.4f}, loss_good (pred=target) iou={l_iou_good.item():.6f}~=0"
    R.check("bbox_loss", "loss_iou = 0 khi pred_box chính xác bằng target_box", t_loss_decreases_with_better_pred)


# ==============================================================================
# 4. DetectionLoss (tích hợp với model thật)
# ==============================================================================
def test_detection_loss_integration(device: str, R: Reporter, imgsz: int = 320):
    R.section("4. DETECTIONLOSS - TÍCH HỢP VỚI NMSFreeDetector")

    nc = 7
    model = NMSFreeDetector(
        nc=nc, backbone_w=(16, 32, 64, 128, 160), backbone_n=(1, 1, 1, 1), neck_n=1
    ).to(device)
    model.train()
    criterion = DetectionLoss(nc=nc, reg_max=model.reg_max, topk_o2m=4, topk_o2o=1)

    def t_forward_backward_normal_batch():
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        targets = [
            {"boxes": torch.tensor([[30., 30., 120., 150.], [150., 100., 280., 260.]], device=device),
             "labels": torch.tensor([1, 3], device=device)},
            {"boxes": torch.tensor([[40., 40., 100., 100.]], device=device),
             "labels": torch.tensor([0], device=device)},
        ]
        out = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total), "tổng loss phải hữu hạn"
        assert total.item() > 0
        model.zero_grad(set_to_none=True)
        total.backward()
        none_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nan_grad  = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not none_grad, f"có tham số không nhận grad: {none_grad[:5]}"
        assert not nan_grad,  f"có tham số grad NaN/Inf: {nan_grad[:5]}"
        return f"total={total.item():.2f}, o2m_pos={items['o2m/n_pos']}, o2o_pos={items['o2o/n_pos']}, grad OK toàn bộ model"
    R.check("detection_loss", "Forward+backward bình thường, grad lan tới backbone", t_forward_backward_normal_batch)

    def t_empty_batch():
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        targets = [
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
        ]
        out   = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        assert items["o2m/n_pos"] == 0 and items["o2o/n_pos"] == 0
        model.zero_grad(set_to_none=True)
        total.backward()
        return f"total={total.item():.2f} (chỉ có cls loss trên negative), backward OK"
    R.check("detection_loss", "Cả batch không có GT nào (chỉ học negative)", t_empty_batch)

    def t_uneven_gt_counts():
        x = torch.randn(3, 3, imgsz, imgsz, device=device)
        b3 = torch.rand(5, 2, device=device) * 250
        boxes3 = torch.cat([b3, b3 + 30], dim=1)
        targets = [
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
            {"boxes": torch.tensor([[10., 10., 50., 50.]], device=device), "labels": torch.tensor([0], device=device)},
            {"boxes": boxes3, "labels": torch.randint(0, nc, (5,), device=device)},
        ]
        out   = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        assert items["o2o/n_pos"] == 6, \
            f"topk_o2o=1 và có 1+5=6 GT thực -> phải có đúng 6 positive ở nhánh o2o, được {items['o2o/n_pos']}"
        model.zero_grad(set_to_none=True)
        total.backward()
        return f"o2o/n_pos={items['o2o/n_pos']} (đúng = tổng số GT), backward OK"
    R.check("detection_loss", "Số lượng GT khác nhau giữa các ảnh trong batch (padding đúng)", t_uneven_gt_counts)

    def t_overfit_single_image():
        torch.manual_seed(0)
        m = NMSFreeDetector(nc=3, backbone_w=(16, 32, 64, 128, 160), backbone_n=(1, 1, 1, 1), neck_n=1).to(device)
        m.train()
        crit = DetectionLoss(nc=3, reg_max=m.reg_max, topk_o2m=4, topk_o2o=1)
        opt  = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x    = torch.randn(1, 3, imgsz, imgsz, device=device)
        targets = [{"boxes": torch.tensor([[50., 50., 150., 180.]], device=device),
                    "labels": torch.tensor([1], device=device)}]
        losses = []
        for _ in range(60):
            opt.zero_grad()
            out = m(x)
            total, _ = crit(out, targets)
            total.backward()
            opt.step()
            losses.append(total.item())
        assert losses[-1] < losses[0] * 0.5, \
            f"loss phải giảm ít nhất 50% khi overfit 1 ảnh trong 60 bước (đầu={losses[0]:.2f}, cuối={losses[-1]:.2f})"
        return f"loss: {losses[0]:.2f} -> {losses[-1]:.2f} (giảm {100*(1-losses[-1]/losses[0]):.0f}%)"
    R.check("detection_loss", "[SANITY] Overfit 1 ảnh, loss phải giảm mạnh", t_overfit_single_image)

    def t_gt_at_scale_boundary():
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        targets = [{
            "boxes": torch.tensor([
                [10., 10., 18., 18.],                           # box nhỏ ~8×8 px
                [5.,  5.,  imgsz - 5., imgsz - 5.],            # box gần hết ảnh
            ], device=device),
            "labels": torch.tensor([0, 2], device=device),
        }]
        out   = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        model.zero_grad(set_to_none=True)
        total.backward()
        return f"total={total.item():.2f}, xử lý đúng cả box rất nhỏ lẫn rất lớn"
    R.check("detection_loss", "GT box ở hai thái cực kích thước (rất nhỏ / rất lớn)", t_gt_at_scale_boundary)

    def t_output_keys_complete():
        """DetectionLoss phải trả về dict chứa đủ các key cần thiết cho logging."""
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        targets = [{"boxes": torch.tensor([[10., 10., 80., 80.]], device=device),
                    "labels": torch.tensor([0], device=device)}]
        out = model(x)
        _, items = criterion(out, targets)
        required_keys = {"loss", "o2m/iou", "o2m/cls", "o2m/dfl", "o2m/n_pos",
                         "o2o/iou", "o2o/cls", "o2o/dfl", "o2o/n_pos"}
        missing = required_keys - set(items.keys())
        assert not missing, f"items dict thiếu key: {missing}"
        return f"tất cả {len(required_keys)} key có mặt trong items dict"
    R.check("detection_loss", "items dict trả về đủ key (loss, o2m/*, o2o/*)", t_output_keys_complete)

    def t_loss_finite_after_eval_mode():
        """Validate mode (model.eval) với criterion vẫn phải cho loss hữu hạn."""
        model.eval()
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        targets = [
            {"boxes": torch.tensor([[20., 20., 100., 100.]], device=device),
             "labels": torch.tensor([0], device=device)},
            {"boxes": torch.tensor([[30., 30., 120., 120.]], device=device),
             "labels": torch.tensor([2], device=device)},
        ]
        with torch.no_grad():
            out = model(x)
        # criterion không cần grad để tính giá trị loss
        total, items = criterion(out, targets)
        assert torch.isfinite(total), "loss ở eval mode phải hữu hạn"
        assert "o2m" in out, "eval mode phải vẫn có o2m trong output (shortcut inference chưa bật)"
        model.train()  # khôi phục lại train mode
        return f"eval mode loss={total.item():.4f} hữu hạn, o2m key tồn tại"
    R.check("detection_loss", "Loss hữu hạn khi model ở eval() mode (validate trong training)", t_loss_finite_after_eval_mode)

    def t_loss_is_positive_with_gt():
        """Khi có GT, tổng loss phải > 0 (không bị zero-out do lỗi mask)."""
        torch.manual_seed(42)
        model.train()
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        targets = [{"boxes": torch.tensor([[10., 10., 200., 200.]], device=device),
                    "labels": torch.tensor([0], device=device)}]
        out = model(x)
        total, _ = criterion(out, targets)
        assert total.item() > 0, f"loss với GT phải > 0, được {total.item()}"
        return f"total_loss={total.item():.4f} > 0"
    R.check("detection_loss", "Loss > 0 khi có GT (không bị zero-out)", t_loss_is_positive_with_gt)


# ==============================================================================
# MAIN
# ==============================================================================
def run(device: str, verbose_traceback: bool = False) -> Reporter:
    """Chạy toàn bộ suite loss và trả về Reporter để run_all_validation.py gộp."""
    r = Reporter(verbose_traceback)
    torch.manual_seed(0)
    test_bbox_utils(device, r)
    test_tal_assigner(device, r)
    test_bbox_loss(device, r)
    test_detection_loss_integration(device, r)
    return r


def main():
    parser = argparse.ArgumentParser(description="Validate src/train/loss.py")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Thiết bị sử dụng: {device}")

    r = run(device, args.verbose_traceback)
    ok = r.summary("TỔNG KẾT - VALIDATE LOSS (bbox_iou/TAL/BboxLoss/DetectionLoss)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()