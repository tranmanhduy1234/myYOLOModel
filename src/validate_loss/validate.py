"""
validate_loss.py
=================
Validate toan dien cho src/loss.py: bbox_iou/CIoU, dist2bbox/bbox2dist,
TaskAlignedAssigner (TAL), BboxLoss (CIoU+DFL), va DetectionLoss (ghep
nhanh o2m + o2o cho NMSFreeDetector).

Chay:
    python validate_loss.py
    python validate_loss.py --device cuda
"""

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

from src.validate_loss.loss import (
    bbox_iou, dist2bbox, bbox2dist, TaskAlignedAssigner, BboxLoss, DetectionLoss,
)
from src.model import NMSFreeDetector


# ==============================================================================
# Reporter (giong validate_model.py de dong bo phong cach)
# ==============================================================================
class Reporter:
    def __init__(self):
        self.results = []

    def check(self, section, name, fn):
        try:
            detail = fn()
            self.results.append((section, name, True, detail or ""))
            print(f"  [PASS] {name}" + (f" -> {detail}" if detail else ""))
        except AssertionError as e:
            self.results.append((section, name, False, str(e)))
            print(f"  [FAIL] {name} -> AssertionError: {e}")
        except Exception as e:
            self.results.append((section, name, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {name} -> {type(e).__name__}: {e}")
            if VERBOSE_TRACEBACK:
                traceback.print_exc()

    def section(self, title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r[2])
        failed = total - passed
        print(f"\n{'=' * 78}\nTONG KET\n{'=' * 78}")
        print(f"Tong so kiem tra : {total}")
        print(f"  Dat (PASS)      : {passed}")
        print(f"  Khong dat (FAIL): {failed}")
        if failed:
            print("\nDanh sach KHONG DAT:")
            for section, name, ok, detail in self.results:
                if not ok:
                    print(f"  - [{section}] {name}: {detail}")
        print()
        return failed == 0

VERBOSE_TRACEBACK = False
R = Reporter()

# ==============================================================================
# 1. bbox_iou / dist2bbox / bbox2dist
# ==============================================================================
def test_bbox_utils(device):
    R.section("1. TIEN ICH HINH HOC (bbox_iou, dist2bbox, bbox2dist)")

    def t_iou_identical():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        v = bbox_iou(a, a, CIoU=True).item()
        assert abs(v - 1.0) < 1e-5, f"IoU 2 box giong het phai ~1.0, duoc {v}"
        return f"CIoU={v:.4f}"
    R.check("bbox_utils", "IoU hai box giong het nhau -> ~1.0", t_iou_identical)

    def t_iou_no_overlap():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        b = torch.tensor([[20., 20., 30., 30.]], device=device)
        iou_plain = bbox_iou(a, b, CIoU=False).item()
        ciou = bbox_iou(a, b, CIoU=True).item()
        assert abs(iou_plain) < 1e-5, "IoU khong giao nhau phai = 0"
        assert ciou < 0, "CIoU khong giao nhau phai < 0 (penalty khoang cach tam)"
        return f"IoU={iou_plain:.4f}, CIoU={ciou:.4f}"
    R.check("bbox_utils", "IoU/CIoU hai box khong giao nhau", t_iou_no_overlap)

    def t_ciou_le_iou():
        a = torch.tensor([[0., 0., 10., 10.]], device=device)
        b = torch.tensor([[5., 5., 15., 15.]], device=device)
        iou = bbox_iou(a, b, CIoU=False).item()
        ciou = bbox_iou(a, b, CIoU=True).item()
        assert ciou <= iou + 1e-6, "CIoU luon <= IoU thuong (vi tru them penalty)"
        return f"IoU={iou:.4f} >= CIoU={ciou:.4f}"
    R.check("bbox_utils", "CIoU <= IoU thuong (penalty dung dau)", t_ciou_le_iou)

    def t_dist2bbox_roundtrip():
        anchors = torch.tensor([[5., 5.], [12., 8.]], device=device)
        dist = torch.tensor([[2., 2., 3., 3.], [1., 4., 2., 1.]], device=device)
        box = dist2bbox(dist, anchors, xywh=False)
        back = bbox2dist(anchors, box, reg_max=16)
        assert torch.allclose(box, torch.tensor([[3., 3., 8., 8.], [11., 4., 14., 9.]], device=device))
        assert torch.allclose(back, dist), "bbox2dist phai la nghich dao chinh xac cua dist2bbox"
        return "dist2bbox <-> bbox2dist khop 100%"
    R.check("bbox_utils", "dist2bbox / bbox2dist round-trip chinh xac", t_dist2bbox_roundtrip)


# ==============================================================================
# 2. TaskAlignedAssigner
# ==============================================================================
def test_tal_assigner(device):
    R.section("2. TASK-ALIGNED ASSIGNER (TAL)")

    A, nc = 20, 3
    anc_points = torch.stack(
        [torch.arange(A, device=device).float() + 0.5, torch.full((A,), 5.5, device=device)], dim=1
    )

    def t_topk_count():
        gt_bboxes = torch.tensor([[[3., 0., 8., 10.]]], device=device)
        gt_labels = torch.tensor([[[1]]], device=device)
        mask_gt = torch.tensor([[[True]]], device=device)
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
        assert fg.sum().item() == 3, f"topk=3 va du anchor hop le -> phai co dung 3 positive, duoc {fg.sum().item()}"
        pos_idx = fg[0].nonzero(as_tuple=True)[0].tolist()
        assert 5 in pos_idx and 6 in pos_idx, "anchor diem cao nhat phai nam trong tap positive"
        return f"positives={pos_idx}"
    R.check("tal", "topk dung so luong, uu tien anchor align_metric cao", t_topk_count)

    def t_positives_inside_gt():
        gt_bboxes = torch.tensor([[[3., 0., 8., 10.]]], device=device)
        gt_labels = torch.tensor([[[0]]], device=device)
        mask_gt = torch.tensor([[[True]]], device=device)
        pd_scores = torch.rand(1, A, nc, device=device)
        pd_bboxes = torch.stack(
            [torch.cat([anc_points[i] - 1, anc_points[i] + 1]) for i in range(A)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=5, num_classes=nc)
        _, _, _, fg, _ = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        pos_x = anc_points[fg[0], 0]
        assert torch.all((pos_x > 3) & (pos_x < 8)), "moi anchor positive phai nam trong GT box"
        return f"{fg.sum().item()} positive, tat ca nam trong GT box [3,8]"
    R.check("tal", "Anchor duong phai nam trong GT box", t_positives_inside_gt)

    def t_contested_anchor_keeps_higher_iou_gt():
        gt_bboxes = torch.tensor([[[1.8, 0., 3.2, 1.], [0., 0., 5., 1.]]], device=device)
        gt_labels = torch.tensor([[[0], [0]]], device=device)
        mask_gt = torch.tensor([[[True], [True]]], device=device)
        A2 = 5
        anc2 = torch.stack(
            [torch.arange(A2, device=device).float() + 0.5, torch.full((A2,), 0.5, device=device)], dim=1
        )
        pd_scores = torch.full((1, A2, 1), 0.5, device=device)
        pd_bboxes = torch.stack(
            [torch.cat([anc2[i] - 0.5, anc2[i] + 0.5]) for i in range(A2)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=5, num_classes=1)
        _, _, _, fg, tgi = assigner(pd_scores, pd_bboxes, anc2, gt_labels, gt_bboxes, mask_gt)
        assert tgi[0, 2].item() == 0, "anchor nam trong ca 2 GT phai duoc gan cho GT co IoU cao hon (GT nho, khop hon)"
        return f"anchor tranh chap gan dung cho GT idx={tgi[0,2].item()}"
    R.check("tal", "Anchor bi tranh chap giua 2 GT -> giu GT co IoU cao hon", t_contested_anchor_keeps_higher_iou_gt)

    def t_padding_gt_ignored():
        A3, nc3 = 10, 2
        anc3 = torch.stack(
            [torch.arange(A3, device=device).float() + 0.5, torch.full((A3,), 0.5, device=device)], dim=1
        )
        gt_bboxes = torch.tensor([[[2., 0., 5., 1.], [0., 0., 0., 0.]]], device=device)  # box 2 la padding
        gt_labels = torch.tensor([[[1], [0]]], device=device)
        mask_gt = torch.tensor([[[True], [False]]], device=device)
        pd_scores = torch.rand(1, A3, nc3, device=device) * 0.2
        pd_bboxes = torch.stack(
            [torch.cat([anc3[i] - 1, anc3[i] + 1]) for i in range(A3)]
        ).unsqueeze(0)
        assigner = TaskAlignedAssigner(topk=2, num_classes=nc3)
        _, _, _, fg, tgi = assigner(pd_scores, pd_bboxes, anc3, gt_labels, gt_bboxes, mask_gt)
        assert fg.sum().item() <= 2, "chi 1 GT hop le -> so positive khong duoc vuot qua topk"
        if fg.sum().item() > 0:
            pos_idx = fg[0].nonzero(as_tuple=True)[0]
            assert torch.all(tgi[0, pos_idx] == 0), "GT padding (mask_gt=False) khong duoc gan lam target"
        return "GT padding bi bo qua dung nhu ky vong"
    R.check("tal", "GT padding (mask_gt=False) khong anh huong ket qua", t_padding_gt_ignored)

    def t_empty_batch():
        gt_bboxes = torch.zeros(1, 0, 4, device=device)
        gt_labels = torch.zeros(1, 0, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(1, 0, 1, dtype=torch.bool, device=device)
        pd_scores = torch.rand(1, A, nc, device=device)
        pd_bboxes = torch.rand(1, A, 4, device=device)
        assigner = TaskAlignedAssigner(topk=3, num_classes=nc)
        _, _, ts, fg, _ = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        assert fg.sum().item() == 0 and ts.sum().item() == 0, "khong GT nao -> khong positive nao"
        return "batch khong GT -> khong loi, khong positive"
    R.check("tal", "Batch hoan toan khong co GT (M=0)", t_empty_batch)


# ==============================================================================
# 3. BboxLoss
# ==============================================================================
def test_bbox_loss(device):
    R.section("3. BBOX LOSS (CIoU + DFL)")

    reg_max = 16
    bs, A = 1, 4
    anchor_points = torch.tensor([[2.5, 2.5], [7.5, 7.5], [12.5, 12.5], [20., 20.]], device=device)
    fg_mask = torch.tensor([[True, True, False, False]], device=device)
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
        pred_bboxes = torch.cat([anchor_points.unsqueeze(0) - lt, anchor_points.unsqueeze(0) + rb], -1)
        return pred_dist, pred_bboxes

    loss_fn = BboxLoss(reg_max=reg_max)

    def t_finite_and_nonneg():
        pred_dist, pred_bboxes = make_pred(0)
        loss_iou, loss_dfl = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                      target_scores, target_scores_sum, fg_mask)
        assert torch.isfinite(loss_iou) and torch.isfinite(loss_dfl)
        assert loss_iou.item() >= 0 and loss_dfl.item() >= 0
        (loss_iou + loss_dfl).backward()
        assert pred_dist.grad is not None and torch.isfinite(pred_dist.grad).all()
        return f"loss_iou={loss_iou.item():.4f}, loss_dfl={loss_dfl.item():.4f}, grad OK"
    R.check("bbox_loss", "Loss huu han, khong am, backward khong NaN", t_finite_and_nonneg)

    def t_zero_positive():
        pred_dist, pred_bboxes = make_pred(1)
        fg0 = torch.zeros(bs, A, dtype=torch.bool, device=device)
        loss_iou0, loss_dfl0 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                        target_scores, target_scores_sum, fg0)
        assert loss_iou0.item() == 0 and loss_dfl0.item() == 0
        (loss_iou0 + loss_dfl0).backward()  # khong duoc crash
        return "khong co positive -> loss=0, khong crash"
    R.check("bbox_loss", "Truong hop khong co anchor duong (fg_mask rong)", t_zero_positive)


# ==============================================================================
# 4. DetectionLoss (tich hop voi model that)
# ==============================================================================
def test_detection_loss_integration(device, imgsz=320):
    R.section("4. DETECTIONLOSS - TICH HOP VOI NMSFreeDetector")

    nc = 4
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
        assert torch.isfinite(total), "tong loss phai huu han"
        assert total.item() > 0
        model.zero_grad(set_to_none=True)
        total.backward()
        none_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nan_grad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not none_grad, f"co tham so khong nhan grad: {none_grad[:5]}"
        assert not nan_grad, f"co tham so grad NaN/Inf: {nan_grad[:5]}"
        return f"total={total.item():.2f}, o2m_pos={items['o2m/n_pos']}, o2o_pos={items['o2o/n_pos']}, grad OK toan bo model"
    R.check("detection_loss", "Forward+backward binh thuong, grad lan toi backbone", t_forward_backward_normal_batch)

    def t_empty_batch():
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        targets = [
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
        ]
        out = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        assert items["o2m/n_pos"] == 0 and items["o2o/n_pos"] == 0
        model.zero_grad(set_to_none=True)
        total.backward()  # cls loss van phai co grad (toan bo la negative)
        return f"total={total.item():.2f} (chi co cls loss tren negative), backward OK"
    R.check("detection_loss", "Ca batch khong co GT nao (chi hoc negative)", t_empty_batch)

    def t_uneven_gt_counts():
        x = torch.randn(3, 3, imgsz, imgsz, device=device)
        b3 = torch.rand(5, 2, device=device) * 250
        boxes3 = torch.cat([b3, b3 + 30], dim=1)
        targets = [
            {"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)},
            {"boxes": torch.tensor([[10., 10., 50., 50.]], device=device), "labels": torch.tensor([0], device=device)},
            {"boxes": boxes3, "labels": torch.randint(0, nc, (5,), device=device)},
        ]
        out = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        assert items["o2o/n_pos"] == 6, f"topk_o2o=1 va co 1+5=6 GT thuc -> phai co dung 6 positive o nhanh o2o, duoc {items['o2o/n_pos']}"
        model.zero_grad(set_to_none=True)
        total.backward()
        return f"o2o/n_pos={items['o2o/n_pos']} (dung = tong so GT, vi topk_o2o=1), backward OK"
    R.check("detection_loss", "So luong GT khac nhau giua cac anh trong batch (padding dung)", t_uneven_gt_counts)

    def t_overfit_single_image():
        torch.manual_seed(0)
        m = NMSFreeDetector(nc=3, backbone_w=(16, 32, 64, 128, 160), backbone_n=(1, 1, 1, 1), neck_n=1).to(device)
        m.train()
        crit = DetectionLoss(nc=3, reg_max=m.reg_max, topk_o2m=4, topk_o2o=1)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
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
            f"loss phai giam it nhat 50% khi overfit 1 anh trong 60 buoc (dau={losses[0]:.2f}, cuoi={losses[-1]:.2f})"
        return f"loss: {losses[0]:.2f} -> {losses[-1]:.2f} (giam {100*(1-losses[-1]/losses[0]):.0f}%)"
    R.check("detection_loss", "[SANITY] Overfit 1 anh, loss phai giam manh", t_overfit_single_image)

    def t_gt_at_scale_boundary():
        # GT box rat nho (co the roi vao P3) va rat lon (co the roi vao P5) cung luc
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        targets = [{
            "boxes": torch.tensor([
                [10., 10., 18., 18.],           # box nho ~8x8 px
                [5., 5., imgsz - 5., imgsz - 5.],  # box gan het anh
            ], device=device),
            "labels": torch.tensor([0, 2], device=device),
        }]
        out = model(x)
        total, items = criterion(out, targets)
        assert torch.isfinite(total)
        model.zero_grad(set_to_none=True)
        total.backward()
        return f"total={total.item():.2f}, xu ly dung ca box rat nho lan rat lon"
    R.check("detection_loss", "GT box o hai thai cuc kich thuoc (rat nho / rat lon)", t_gt_at_scale_boundary)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    global VERBOSE_TRACEBACK
    VERBOSE_TRACEBACK = args.verbose_traceback

    device = torch.device(args.device)
    print(f"Thiet bi su dung: {device}")
    torch.manual_seed(0)

    test_bbox_utils(device)
    test_tal_assigner(device)
    test_bbox_loss(device)
    test_detection_loss_integration(device)

    ok = R.summary()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()