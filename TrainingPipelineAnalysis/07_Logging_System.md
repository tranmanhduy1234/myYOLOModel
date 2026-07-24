# CHƯƠNG 7: HỆ THỐNG LOGGING MULTI-TIER VÀ REAL-TIME MONITORING

## 1. GIỚI THIỆU CHƯƠNG

Trong quá trình huấn luyện các mô hình Học sâu lớn, **Hệ thống Quan sát và Ghi nhật ký (Logging System)** đóng vai trò là "bảng điều khiển trung tâm" giúp các kỹ sư và nhà nghiên cứu theo dõi trạng thái sức khỏe của mô hình (Model Health), phát hiện sớm các bất thường toán học (như hiện tượng bùng nổ gradient, triệt tiêu gradient, bão hòa trọng số, hay rò rỉ bộ nhớ GPU), và đánh giá động học hội tụ theo thời gian.

Một hệ thống logging kém chất lượng có thể gây ra hai tác hại:
1. **Làm chậm Tốc độ Huấn luyện (Performance Bottleneck)**: Đọc/ghi đĩa I/O quá dày hoặc tính toán histogram trên CPU/GPU ở mọi iteration sẽ làm GPU rơi vào trạng thái nhàn rỗi (GPU Idle).
2. **Thiếu Tín hiệu Debug (Blind Spot)**: Không trích xuất đủ chỉ số RMSNorm hay Update-to-Weight Ratio sẽ khiến kỹ sư không thể tìm ra nguyên nhân khi mô hình bị đứng loss hoặc nổ NaN.

Chương này phân tích kiến trúc **Hệ thống Logging Kép (Dual-Tier Logging System)** được cài đặt trong [`src/utils/logging_setup.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/logging_setup.py) (Text File Logging) và [`src/utils/tb_logger.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/tb_logger.py) (TensorBoard Logging), đồng thời làm rõ chiến lược phân tầng tần suất ghi nhật ký.

---

## 2. NỘI DUNG PHÂN TÍCH HỆ THỐNG LOGGING MULTI-TIER

Hệ thống logging được thiết kế theo 2 tầng độc lập nhưng phối hợp mượt mà:

```text
+-----------------------------------------------------------------------------------+
|                            NMSFreeDetector Training Loop                          |
+-----------------------------------------------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     | TẦNG 1: TEXT LOGGING              | TẦNG 2: TENSORBOARD LOGGING       |
     | (src/utils/logging_setup.py)      | (src/utils/tb_logger.py)          |
     +-----------------------------------+-----------------------------------+
     | - Output: ./logs/train_*.log      | - Output: ./runs/                 |
     | - Format: Timed Text Stream       | - Writer: SummaryWriter           |
     | - Interval: Every log_interval    | - Managed by: TrainingLogger      |
     | - Purpose: Audit & Text History   | - Purpose: Multi-chart Visuals    |
     +-----------------------------------+-----------------------------------+
```

---

### 2.1. Tầng 1: Hệ Thống Ghi Văn Bản (Text File Logging)

Được cấu hình thông qua hàm [`setup_logging`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/logging_setup.py#L15-L56).

#### Đặc điểm kỹ thuật:
- **Tệp đầu ra**: Tự động tạo tệp nhật ký dạng `./logs/train_{YYYYMMDD_HHMMSS}.log`.
- **Cấu hình Singleton Logger**: Sử dụng tên logger chung `"train"`. Hàm `_ensure_text_logging` trong `engine.py` đảm bảo chỉ có 1 handler duy nhất được gắn vào logger, tránh việc mở 2 tệp log trùng lặp khi chạy.
- **Định dạng Log (Log Formatter)**:
  `[YYYY-MM-DD HH:MM:SS][LEVEL][train] Message`
- **Tần suất**: Ghi thông tin chi tiết mỗi `log_interval` step (mặc định 20 step):

```text
[2026-07-24 11:38:31][INFO][train] [epoch 0] step 20/2125 loss=4.1234 (o2m iou=0.345 cls=1.234 dfl=0.567 npos=120) (o2o iou=0.312 cls=1.102 dfl=0.512 npos=12) lr=0.000007 t=12.4s
```

- **Thông tin theo dõi**: Trích xuất chi tiết từng thành phần iou/cls/dfl và số lượng positive anchor (`npos`) cho **cả hai nhánh o2m và o2o**, giúp kỹ sư theo dõi sự lệch pha giữa hai nhánh ngay lập tức trên file log.

---

### 2.2. Tầng 2: Hệ Thống Giám Sát Trực Quan TensorBoard (`TrainingLogger`)

Được cài đặt trong lớp [`TrainingLogger`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/tb_logger.py#L55-L328).

Lớp `TrainingLogger` đóng vai trò là bộ quản lý tập trung toàn bộ dữ liệu TensorBoard, tự động phân bổ và kiểm soát tần suất ghi dữ liệu dựa trên hai ngưỡng:
- `log_interval = 20`: Tần suất ghi cho các đại lượng Scalar nhẹ (Loss, LR, Memory).
- `histogram_interval = 100`: Tần suất ghi cho các đại lượng Histogram và Matrix nặng (Gradients, Weights, Update Ratios, BN Statistics).

---

## 3. PHÂN TÍCH NỘI DUNG CÁC MỤC LOG TENSORBOARD

### 3.1. Theo Dõi Loss và Phân Tích Tỷ Lệ (`log_losses` & `log_loss_ratios`)

- **Bảng điều khiển Multi-Scalar**: Đóng gói các đường cong loss vào chung 1 biểu đồ để so sánh trực quan (`add_scalars`):
  - `phase/loss_total`: Total Loss vs o2m Loss vs o2o Loss.
  - `phase/loss_o2m_parts`: IoU vs Cls vs DFL of o2m.
  - `phase/loss_o2o_parts`: IoU vs Cls vs DFL of o2o.
  - `phase/n_pos`: Số lượng anchor dương của o2m vs o2o (Tách riêng do n_pos là số nguyên lớn $50-200$, gộp chung với loss sẽ làm biểu đồ loss bị bẹt phẳng).

---

### 3.2. Theo Dõi Gradient và Chống Phân Kỳ (`log_gradients`)

Được gọi **SAU `scaler.unscale_()` và TRƯỚC `clip_grad_norm_`**:

```python
# Đoạn mã trong train_one_epoch
if do_grad_log:
    total_norm = tb_logger.log_gradients(model, global_step)
    if not math.isfinite(total_norm):
        logger.warning(f"[epoch {epoch}] step {step}: gradient NaN/Inf truoc khi clip")
```

#### Công thức tính RMSNorm của Gradient cho từng tham số:

$$\text{RMS}(g_\theta) = \frac{\|g_\theta\|_2}{\sqrt{N_\theta}}$$

với $N_\theta$ là số lượng phần tử trong tham số $\theta$.

#### Ý nghĩa Kỹ thuật:
- **Thời điểm ghi**: Việc ghi log gradient SAU khi `unscale_` giúp trích xuất đúng giá trị thật của gradient (không bị nhân với hệ số $2^{16}$ của AMP). Việc ghi TRƯỚC khi `clip_grad_norm_` cho phép kỹ sư thấy được độ lớn thực sự của gradient trước khi nó bị gọt giũa, từ đó phát hiện sớm hiện tượng gradient bị phình to (Gradient Explosion).
- **Xử lý An toàn (Histogram Crash Prevention)**: PyTorch `add_histogram` sẽ gây ra lỗi `ValueError: histogram is empty` và crash chương trình nếu tensor chứa NaN/Inf. Hàm `log_gradients` trong `tb_logger.py` kiểm tra cờ `torch.isfinite(param.grad).all()`. Nếu phát hiện NaN/Inf, hệ thống chủ động bỏ qua histogram của param đó, phát cảnh báo ra log và tiếp tục huấn luyện thay vì ngắt chương trình.

---

### 3.3. Theo Dõi Trọng Số và Tỷ Lệ Cập Nhật (`log_weights` & `log_weight_updates`)

#### 1. Snapshot Tham Số:
Trước khi bước vào `optimizer.zero_grad()`, hàm tĩnh [`snapshot_params`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/tb_logger.py#L75-L78) chụp lại toàn bộ giá trị trọng số của mô hình theo TÊN (`name`):
```python
prev_params = TrainingLogger.snapshot_params(model)  # Chụp W_t
```

#### 2. Tính toán Update Ratio (Tỷ lệ Cập nhật Trọng số):
Sau khi gọi `optimizer.step()`, hàm `log_weight_updates` tính toán tỷ lệ cập nhật tương đối cho từng layer:

$$\text{Update\_Ratio}(\theta) = \mathbb{E}\left[ \frac{|\theta_{t+1} - \theta_t|}{|\theta_t| + \epsilon} \right]$$

```text
Rule of Thumb for Update-to-Weight Ratio in Deep Learning

Update_Ratio ~ 1e-3 (0.001)  --> IDEAL! Parameter updates are healthy.
Update_Ratio < 1e-5          --> TOO SMALL! Learning rate is too low or grads vanishing.
Update_Ratio > 1e-1          --> TOO LARGE! Learning rate is too high, risks instability.
```

Chỉ số này được ghi lên mục `Update_Ratio/{name}` trên TensorBoard, cung cấp một chỉ báo định lượng cực kỳ chính xác về tốc độ học thích hợp cho từng lớp.

---

### 3.4. Theo Dõi Bộ Nhớ GPU Phần Cứng (`log_gpu_memory`)

Hàm `log_gpu_memory` truy vấn trực tiếp driver PyTorch CUDA để trích xuất 4 chỉ số VRAM:
1. `System/GPU_memory_allocated_GB`: Bộ nhớ VRAM thực tế đang chứa Tensor.
2. `System/GPU_memory_reserved_GB`: Bộ nhớ VRAM đang được PyTorch Caching Allocator giữ chỗ.
3. `System/GPU_max_memory_allocated_GB`: Đỉnh bộ nhớ VRAM cao nhất từ trước tới nay (Peak Memory).
4. `System/GPU_memory_utilization`: Tỷ lệ hiệu dụng bộ nhớ $\frac{\text{Allocated}}{\text{Reserved}}$.

Chỉ số này giúp kỹ sư phát hiện lập tức các sự cố rò rỉ bộ nhớ VRAM (Memory Leaks) nếu thấy đường `allocated` tăng tiến dần theo thời gian mà không đi ngang.

---

## 4. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 4.1. Bảng Thống Kê Phân Tầng Tần Suất Ghi Logging

| Loại Chỉ Số Log | Mục TensorBoard | Tần Suất Ghi | Chi Phí Tính Toán | Mục Đích Giám Sát |
| :--- | :--- | :--- | :--- | :--- |
| **Loss Total & Parts** | `train/loss_*`, `val/loss_*` | Every Step ($1$) | Rất thấp (Scalar) | Động học hội tụ hàm loss |
| **Learning Rate** | `Learning_Rate/group_*` | `log_interval` ($20$) | Rất thấp (Scalar) | Kiểm tra Cosine Warmup decay |
| **GPU Memory** | `System/GPU_*` | `log_interval` ($20$) | Rất thấp (CUDA call) | Đánh giá VRAM & Memory Leaks |
| **Gradient RMSNorm** | `Gradients_RMS/*` | `log_interval` ($20$) | Thấp (Norm calculation) | Kiểm soát độ lớn gradient |
| **Gradient Histograms** | `Gradients/*` | `hist_interval` ($100$)| Trung bình | Quan sát phân phối gradient |
| **Weight Stats & RMS** | `Weights_Stats/*` | `log_interval` ($20$) | Thấp | Kiểm tra phương sai trọng số |
| **Update Ratio** | `Update_Ratio/*` | `log_interval` ($20$) | Trung bình (Difference) | Đánh giá tốc độ cập nhật trọng số |
| **BN Statistics** | `BN/*/running_mean` | `hist_interval` ($100$)| Trung bình (Module Scan)| Kiểm tra sự ổn định BatchNorm |

---

## 5. KẾT LUẬN CHƯƠNG

Hệ thống Logging Kép Multi-Tier trong dự án là một mô hình thiết kế giám sát hiện đại, toàn diện và tối ưu hóa cao. Bằng cách phân chia thông minh giữa **File Text Logging mượt mà cho việc kiểm toán** và **TensorBoard Dynamic Logging trực quan cho việc phân tích**, đồng thời áp dụng **Chiến lược Phân tầng Tần suất (Scalar 20 steps vs Histogram 100 steps)**, hệ thống đảm bảo trích xuất đầy đủ mọi chỉ số sức khỏe của mô hình (từ Loss, Gradient RMSNorm, Weight Update Ratio đến GPU Memory Profiling) mà không tạo ra bất kỳ điểm nghẽn hiệu năng I/O nào cho GPU.
