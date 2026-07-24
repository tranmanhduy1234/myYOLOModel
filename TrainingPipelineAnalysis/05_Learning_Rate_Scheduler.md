# CHƯƠNG 5: CHIẾN LƯỢC SUY GIẢM TỐC ĐỘ HỌC (LEARNING RATE SCHEDULER)

## 1. GIỚI THIỆU CHƯƠNG

Trong quá trình huấn luyện mô hình Học sâu, **Tốc độ Học (Learning Rate - LR)** được coi là siêu tham số nhạy cảm và quan trọng nhất. Một Tốc độ Học quá lớn ở giai đoạn đầu có thể khiến mô hình bị bùng nổ loss và phân kỳ; ngược lại, một Tốc độ Học không giảm ở giai đoạn cuối sẽ làm cho các tham số bị dao động liên tục quanh điểm cực tiểu mà không thể hội tụ (Overshooting).

Dự án cài đặt một chiến lược điều phối Tốc độ Học mượt mà kết hợp giữa **Khởi động Tuyến tính (Linear Warmup)** và **Suy giảm Cosine (Cosine Annealing Decay)** theo từng step lặp (Per-step Decay), được định nghĩa trong hàm `lr_lambda_factory` tại tệp [`src/train/engine.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L102-L114).

Chương này trình bày chi tiết cơ sở toán học, công thức tính toán bước lặp, sự thay đổi tốc độ học theo thời gian, và tương tác giữa LR Scheduler với cơ chế an toàn AMP.

---

## 2. NỘI DUNG PHÂN TÍCH LÝ THUYẾT VÀ CÔNG THỨC TOÁN HỌC

### 2.1. Phân Tích Hàm Tạo Lambda LR (`lr_lambda_factory`)

Tốc độ học tại bước lặp $k$ (global step) được tính toán thông qua hàm `torch.optim.lr_scheduler.LambdaLR` với hệ số tỉ lệ $\lambda(k)$:

$$\text{lr}(k) = \text{lr}_0 \times \lambda(k)$$

với $\text{lr}_0 = 10^{-3}$ (`cfg.lr0`).

Hàm lambda được cài đặt trong mã nguồn:

```python
def lr_lambda_factory(cfg: TrainConfig, steps_per_epoch):
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, cfg.epochs * steps_per_epoch)

    def _lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * cosine

    return _lambda
```

---

### 2.2. Giai Đoạn 1: Khởi Động Tuyến Tính (Linear Warmup Phase)

#### 1. Công thức toán học:
Với $k < k_{\text{warmup}}$ (trong đó $k_{\text{warmup}} = \text{warmup\_epochs} \times N_{\text{steps\_per\_epoch}}$):

$$\lambda(k) = \frac{k}{k_{\text{warmup}}}$$

Tốc độ học thực tế tăng dần một cách tuyến tính từ $0$ đến $\text{lr}_0$:

$$\text{lr}(k) = \text{lr}_0 \cdot \frac{k}{k_{\text{warmup}}}$$

- Với cấu hình `warmup_epochs = 3.0`, nếu một epoch có $N_{\text{steps}} = 1000$ steps $\implies k_{\text{warmup}} = 3000$ steps.
- Tại $k = 0 \implies \text{lr} = 0.0$.
- Tại $k = 1500 \implies \text{lr} = 0.5 \times 10^{-3} = 5 \times 10^{-4}$.
- Tại $k = 3000 \implies \text{lr} = 1.0 \times 10^{-3}$.

#### 2. Mục đích và Ý nghĩa Kỹ thuật:
Ở những iteration đầu tiên, các trọng số của Detection Head vừa được khởi tạo ngẫu nhiên hoặc khởi tạo theo prior, đồng thời các tham số thống kê `running_mean` và `running_var` của BatchNorm2d chưa phản ánh đúng phân phối dữ liệu thực tế. Việc ép một tốc độ học lớn $\text{lr}_0 = 10^{-3}$ ngay từ step 1 sẽ tạo ra các vector gradient cực đại, làm phá hủy các cấu trúc đặc trưng tiền huấn luyện (pretrained features) của Backbone.

Giai đoạn Linear Warmup giúp mô hình "làm nóng" nhẹ nhàng:
- Giúp BatchNorm2d tích lũy đủ thông tin thống kê mượt mà.
- Cho phép Task-Aligned Assigner (TAL) ổn định việc gán nhãn soft-target.
- Tránh hiện tượng bùng nổ Loss ở 3 epoch đầu tiên.

---

### 2.3. Giai Đoạn 2: Suy Giảm Cosine (Cosine Annealing Decay Phase)

#### 1. Công thức toán học:
Với $k \ge k_{\text{warmup}}$:

Đầu tiên, tính tiến trình hoàn thành $\phi(k) \in [0, 1]$:

$$\phi(k) = \min\left(1.0, \frac{k - k_{\text{warmup}}}{k_{\text{total}} - k_{\text{warmup}}}\right)$$

Hệ số $\lambda(k)$ được tính theo hàm Cosine:

$$\lambda(k) = \alpha_{\text{min}} + (1 - \alpha_{\text{min}}) \times \frac{1}{2} \left[ 1 + \cos\left( \pi \cdot \phi(k) \right) \right]$$

trong đó $\alpha_{\text{min}} = \text{lr\_min\_factor} = 0.01$.

Tốc độ học suy giảm mượt mà từ $\text{lr}_0$ về $\text{lr}_{\text{final}} = \text{lr}_0 \times \alpha_{\text{min}} = 10^{-3} \times 0.01 = 10^{-5}$:

$$\text{lr}(k) = 10^{-5} + \left(10^{-3} - 10^{-5}\right) \times \frac{1}{2} \left[ 1 + \cos\left( \pi \cdot \phi(k) \right) \right]$$

```text
Learning Rate Schedule Profile over 100 Epochs

  LR (1e-3)
   1.0 |      /----\
       |     /      \   <-- Warmup Phase (Epoch 0 -> 3)
   0.8 |    /        \
       |   /          \  <-- Cosine Annealing Decay (Epoch 3 -> 100)
   0.5 |  /            \
       | /              \
   0.2 |/                \
   0.0 +-------------------\-------------------> Step / Epoch
       0     3             50               100
```

#### 2. Ưu điểm của Cosine Decay so với Step Decay (Giảm theo nấc):
- **Không có điểm gãy đột ngột (Smooth Transition)**: Giảm bớt sự biến động lớn của gradient khi chuyển giao giữa các giai đoạn huấn luyện.
- **Tốc độ suy giảm chậm ở giữa và cuối**: Ở giữa quá trình ($epoch \sim 50$), hàm Cosine giảm với tốc độ vừa phải cho phép mô hình tiếp tục thăm dò không gian tham số. Khi tiến về những epoch cuối ($epoch > 80$), tốc độ suy giảm chậm lại và tiệm cận về $\text{lr}_{\text{min}} = 10^{-5}$, giúp các trọng số "hạ cánh mượt mà" vào đáy của cực tiểu cục bộ (Local Minimum).

---

### 2.4. Tương Tác Giữa LR Scheduler Và Cơ Chế AMP Scaler

Một chi tiết thiết kế cực kỳ tinh tế trong hàm [`train_one_epoch`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L186-L210):

```python
scale_before = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
scale_after = scaler.get_scale()
skip_lr_sched = (scale_after < scale_before)  # scale giảm => step vừa rồi đã bị skip

if not skip_lr_sched:
    scheduler.step()
```

- **Nguyên lý**: Khi AMP GradScaler phát hiện gradient chứa giá trị NaN hoặc Inf, bước `scaler.step(optimizer)` sẽ **bỏ qua (skip)** việc cập nhật trọng số `optimizer.step()`, đồng thời hạ hệ số scale $S$.
- **Xử lý LR Scheduler**: Nếu bước cập nhật trọng số bị skip, cờ `skip_lr_sched` trở thành `True`. Hệ thống sẽ **KHÔNG gọi `scheduler.step()`**.
- **Ý nghĩa**: Điều này đảm bảo rằng lịch trình tốc độ học chỉ tiến lên khi mô hình thực sự thực hiện một bước cập nhật trọng số hợp lệ. Nếu bỏ qua cờ này, scheduler vẫn giảm LR ngay cả khi trọng số không được cập nhật, làm lệch lịch trình Cosine decay so với số bước cập nhật thực tế.

---

## 3. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 3.1. Bảng Thống Kê Thay Đổi Tốc Độ Học Theo Epoch (Mô phỏng 100 Epochs)

| Epoch | Tiến Trình | Hệ Số $\lambda(k)$ | Tốc Độ Học $\text{lr}(k)$ | Trạng Thái Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **0.0** | Start Warmup | $0.000$ | $0.000000$ | Khởi tạo mô hình, ổn định BN statistics |
| **1.5** | Mid Warmup | $0.500$ | $0.000500$ | Học các đường nét đặc trưng cơ bản |
| **3.0** | End Warmup | $1.000$ | $0.001000$ | Đạt đỉnh LR, bắt đầu pha Cosine decay |
| **25.0** | $22.7\%$ Decay | $0.874$ | $0.000874$ | Huấn luyện tốc độ cao, hội tụ nhanh |
| **50.0** | $48.5\%$ Decay | $0.505$ | $0.000505$ | Tinh chỉnh đặc trưng đa quy mô |
| **75.0** | $74.2\%$ Decay | $0.136$ | $0.000136$ | Tiến gần cực tiểu cục bộ |
| **99.0** | $100\%$ Finish | $0.010$ | $0.000010$ | Tối ưu hóa cực hạn, đóng băng trọng số |

### 3.2. Đánh Giá Ảnh Hưởng Tới Độ Hội Tụ

Việc tính toán LR Decay **theo từng iteration (Per-step)** thay vì **theo từng epoch** giúp cho đường cong suy giảm tốc độ học liên tục tuyệt đối mà không có các bước nhảy bậc thang. Điều này triệt tiêu hoàn toàn tình trạng mất ổn định loss thường thấy ở đầu các epoch khi sử dụng StepLR truyền thống.

---

## 4. KẾT LUẬN CHƯƠNG

Chiến lược điều phối Tốc độ Học trong dự án thể hiện một thiết kế học thuật mượt mà và an toàn. Bằng cách kết hợp **3 epoch Linear Warmup**, **Cosine Annealing Decay cập nhật per-step**, đưa tốc độ học về ngưỡng $\text{lr}_{\text{min}} = 1\% \times \text{lr}_0$, kết hợp với **cơ chế bỏ qua step thông minh khi AMP Scaler gặp Inf/NaN**, hệ thống vừa bảo vệ được các đặc trưng khởi tạo ban đầu, vừa giúp mô hình hội tụ sâu và mượt mà vào đáy của hàm mất mát.
