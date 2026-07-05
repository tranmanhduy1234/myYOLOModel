"""
CNN-LSTM Driver State Recognition
==================================
- CNN: dùng trực tiếp NMSFreeDetector (import nguyên class, KHÔNG viết lại) -
  chỉ lấy phần "trunk" (backbone + neck) làm feature extractor, bỏ DetectHead
  vì bài toán này là phân loại trạng thái, không phải detect.
- LSTM: nn.LSTM (pytorch) 3 layer, học đặc trưng thời gian theo chuỗi khung hình.
- Benchmark: tốc độ inference với sliding-window 150 frame trên video giả
  10s @ 30fps (300 frame), và đếm tham số từng thành phần.

Lưu ý: đổi tên module bên dưới cho đúng tên file chứa class NMSFreeDetector
trong project của bạn (ví dụ model.py, detector.py, nms_free_detector.py...).
"""

import time
import torch
import torch.nn as nn

from src.model import NMSFreeDetector  # <-- đổi thành đúng tên file của bạn


# ----------------------------------------------------------------------
# 1. CNN Feature Extractor = trunk (backbone + neck) của NMSFreeDetector
# ----------------------------------------------------------------------
class CNNFeatureExtractor(nn.Module):
    """Tận dụng backbone+neck của NMSFreeDetector đã pretrain để trích đặc
    trưng không gian. Head detect bị bỏ (nn.Identity) vì không dùng tới."""

    def __init__(self, detector: NMSFreeDetector, feat_dim=256):
        super().__init__()
        self.detector = detector
        self.detector.head = nn.Identity()  # bỏ DetectHead, không tốn tham số/compute
        c5 = detector.neck_chs[-1]          # kênh sâu nhất của neck (p5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c5, feat_dim)

    def forward(self, x):
        # x: (N, 3, H, W)
        p3, p4, p5 = self.detector.backbone(x)
        p3, p4, p5 = self.detector.neck(p3, p4, p5)
        v = self.pool(p5).flatten(1)     # (N, c5)
        return self.proj(v)              # (N, feat_dim)


# ----------------------------------------------------------------------
# 2. CNN-LSTM cho nhận dạng trạng thái tài xế
# ----------------------------------------------------------------------
class CNNLSTMDriverState(nn.Module):
    def __init__(self, num_classes=5, feat_dim=256,
                 hidden_size=256, num_lstm_layers=3,
                 nc=80, reg_max=16,
                 backbone_w=(38, 76, 152, 304, 608),
                 backbone_n=(1, 3, 3, 1),
                 neck_n=1, strides=(8, 16, 32),
                 trunk_ckpt=None, freeze_trunk=False,
                 cnn_batch_size=16):
        super().__init__()
        # Số frame đưa qua CNN cùng lúc. GPU nhỏ (VD 3-4GB) -> để 8-16.
        # Không ảnh hưởng kết quả, chỉ đánh đổi tốc độ lấy bộ nhớ.
        self.cnn_batch_size = cnn_batch_size

        # Khởi tạo nguyên bản NMSFreeDetector (nc/reg_max/strides chỉ để dựng
        # kiến trúc backbone+neck cho đúng, không dùng head của nó)
        detector = NMSFreeDetector(nc=nc, reg_max=reg_max,
                                    backbone_w=backbone_w, backbone_n=backbone_n,
                                    neck_n=neck_n, strides=strides)

        if trunk_ckpt is not None:
            detector.load_trunk(trunk_ckpt)   # nạp backbone+neck đã pretrain

        self.cnn = CNNFeatureExtractor(detector, feat_dim=feat_dim)

        if freeze_trunk:
            self.cnn.detector.freeze_trunk(True)  # đóng băng khi finetune

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, 3, H, W)  -  B=batch, T=số frame trong cửa sổ
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Chạy CNN theo từng chunk nhỏ để tránh OOM (activation memory là
        # nguyên nhân chính, không phải số tham số) thay vì đẩy cả B*T=150
        # ảnh qua backbone+neck trong 1 lần.
        feats_list = []
        for i in range(0, x.size(0), self.cnn_batch_size):
            chunk = x[i:i + self.cnn_batch_size]
            feats_list.append(self.cnn(chunk))
        feats = torch.cat(feats_list, dim=0).view(B, T, -1)  # (B, T, feat_dim)

        out, (h_n, c_n) = self.lstm(feats)  # out: (B, T, hidden)
        last = out[:, -1, :]                # trạng thái tại frame cuối cửa sổ
        logits = self.classifier(last)
        return logits


# ----------------------------------------------------------------------
# 3. Đếm tham số từng thành phần
# ----------------------------------------------------------------------
def count_params(module):
    return sum(p.numel() for p in module.parameters())


def benchmark_params(model: CNNLSTMDriverState):
    backbone_p = count_params(model.cnn.detector.backbone)
    neck_p = count_params(model.cnn.detector.neck)
    proj_p = count_params(model.cnn.proj)
    lstm_p = count_params(model.lstm)
    fc_p = count_params(model.classifier)
    total = backbone_p + neck_p + proj_p + lstm_p + fc_p

    print("===== Số tham số từng thành phần =====")
    print(f"Backbone (NMSFreeDetector) : {backbone_p:,} ({backbone_p/1e6:.2f} M)")
    print(f"Neck/PAFPN                 : {neck_p:,} ({neck_p/1e6:.2f} M)")
    print(f"Projection (CNN -> feat)   : {proj_p:,} ({proj_p/1e6:.2f} M)")
    print(f"LSTM (3 layer)             : {lstm_p:,} ({lstm_p/1e6:.2f} M)")
    print(f"Classifier (FC)            : {fc_p:,} ({fc_p/1e6:.2f} M)")
    print(f"TỔNG                       : {total:,} ({total/1e6:.2f} M)")
    return {"backbone": backbone_p, "neck": neck_p, "proj": proj_p,
            "lstm": lstm_p, "fc": fc_p, "total": total}


# ----------------------------------------------------------------------
# 4. Benchmark tốc độ inference: sliding-window 150 frame trên video giả
# ----------------------------------------------------------------------
def benchmark_inference(model, device, img_size=480,
                         video_seconds=10, fps=30,
                         window=150, stride=15):
    model.eval().to(device)
    n_frames = video_seconds * fps  # 300 frame
    video = torch.randn(n_frames, 3, img_size, img_size, device=device)

    starts = list(range(0, n_frames - window + 1, stride))
    if not starts:
        starts = [0]
        window = n_frames

    print(f"\n===== Benchmark inference (device={device}) =====")
    print(f"Video giả: {n_frames} frame ({video_seconds}s @ {fps}fps)")
    print(f"Cửa sổ trượt: window={window}, stride={stride} -> {len(starts)} cửa sổ")

    with torch.no_grad():
        for _ in range(3):  # warm-up
            clip = video[0:window].unsqueeze(0)
            _ = model(clip)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for s in starts:
            clip = video[s:s + window].unsqueeze(0)  # (1, window, 3, H, W)
            t0 = time.perf_counter()
            _ = model(clip)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

    latencies = torch.tensor(latencies)
    total_time = latencies.sum().item()
    avg_ms = latencies.mean().item() * 1000
    fps_effective = len(starts) / total_time

    print(f"Tổng thời gian xử lý {len(starts)} cửa sổ: {total_time:.3f}s")
    print(f"Latency trung bình / cửa sổ: {avg_ms:.2f} ms")
    print(f"Thông lượng: {fps_effective:.2f} cửa sổ/giây")
    print(f"Tương đương ~{fps_effective * window:.1f} frame/giây nếu chạy pipeline liên tục")
    return {"total_time": total_time, "avg_latency_ms": avg_ms, "windows_per_sec": fps_effective}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5 trạng thái tài xế: tỉnh táo, buồn ngủ, mất tập trung, dùng điện thoại, ngáp
    model = CNNLSTMDriverState(num_classes=5, feat_dim=256,
                                hidden_size=256, num_lstm_layers=10)

    benchmark_params(model)
    # img_size=224: đủ dùng cho driver monitoring (mặt/tay tài xế), nhẹ hơn
    # nhiều so với 480 (detection full-scene) -> tránh OOM trên GPU nhỏ.
    # Nếu vẫn OOM: giảm tiếp img_size hoặc giảm cnn_batch_size khi tạo model.
    benchmark_inference(model, device, img_size=224,
                         video_seconds=10, fps=30,
                         window=150, stride=15)