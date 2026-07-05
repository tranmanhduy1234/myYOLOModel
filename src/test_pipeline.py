import torch
from model import NMSFreeDetector
from train.loss import DetectionLoss


def make_fake_gt(B, M, nc, img_size=480, device="cpu"):
    gt_boxes = torch.zeros(B, M, 4, device=device)
    gt_labels = torch.zeros(B, M, dtype=torch.long, device=device)
    gt_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    for b in range(B):
        n = torch.randint(1, M + 1, (1,)).item()
        for i in range(n):
            cx, cy = torch.rand(2) * img_size
            w, h = torch.rand(2) * 100 + 20
            x1, y1 = (cx - w / 2).clamp(0), (cy - h / 2).clamp(0)
            x2, y2 = (cx + w / 2).clamp(max=img_size), (cy + h / 2).clamp(max=img_size)
            gt_boxes[b, i] = torch.tensor([x1, y1, x2, y2])
            gt_labels[b, i] = torch.randint(0, nc, (1,))
            gt_mask[b, i] = True
    return gt_boxes, gt_labels, gt_mask


@torch.no_grad()
def inference_nms_free(preds, conf_thres=0.3, max_det=100):
    """Suy luan KHONG NMS: chi loc theo confidence + top-k (nho da train one-to-one)."""
    cls = preds["o2o"]["cls"].sigmoid()          # (B,A,nc)
    box = preds["o2o"]["box"]                    # (B,A,4)
    B = cls.shape[0]
    results = []
    for b in range(B):
        scores, labels = cls[b].max(-1)           # (A,)
        keep = scores > conf_thres
        s, l, bx = scores[keep], labels[keep], box[b][keep]
        if s.numel() > max_det:
            topv, topi = s.topk(max_det)
            s, l, bx = topv, l[topi], bx[topi]
        out = torch.cat([bx, l.unsqueeze(-1).float(), s.unsqueeze(-1)], -1)  # [x1,y1,x2,y2,cls,score]
        results.append(out)
    return results


def main():
    torch.manual_seed(0)
    device = "cpu"
    nc = 80
    model = NMSFreeDetector(nc=nc).to(device)
    criterion = DetectionLoss(nc=nc)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 480, 480, device=device)
    gt_boxes, gt_labels, gt_mask = make_fake_gt(2, 5, nc, device=device)

    print("=== Forward + Loss + Backward (vai buoc train thu) ===")
    for step in range(3):
        preds = model(x)
        loss, logs = criterion(preds, gt_boxes, gt_labels, gt_mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"step {step} | loss_total={logs['loss_total']:.4f} "
              f"| o2m(cls={logs['o2m']['cls']:.3f},box={logs['o2m']['box']:.3f},dfl={logs['o2m']['dfl']:.3f},npos={logs['o2m']['n_pos']}) "
              f"| o2o(cls={logs['o2o']['cls']:.3f},box={logs['o2o']['box']:.3f},dfl={logs['o2o']['dfl']:.3f},npos={logs['o2o']['n_pos']})")

    print("\n=== Inference NMS-free ===")
    model.eval()
    with torch.no_grad():
        preds = model(x)
    dets = inference_nms_free(preds, conf_thres=0.0, max_det=5)  # conf=0 de chac chan co output demo
    for i, d in enumerate(dets):
        print(f"Anh {i}: {d.shape[0]} box, vi du dong dau: {d[0].tolist() if d.shape[0]>0 else None}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTong tham so model: {n_params:,} ({n_params/1e6:.2f}M)")

if __name__ == "__main__":
    main()