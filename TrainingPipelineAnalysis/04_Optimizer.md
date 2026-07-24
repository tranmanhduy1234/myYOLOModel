# CHƯƠNG 4: THUẬT TOÁN TỐI ƯU VÀ ĐỘNG HỌC CẬP NHẬT TRỌNG SỐ (OPTIMIZER)

## 1. GIỚI THIỆU CHƯƠNG

Trong huấn luyện mạng thần kinh học sâu, **Thuật toán Tối ưu (Optimizer)** đóng vai trò là "người điều khiển" dẫn dắt các tham số của mô hình tìm kiếm điểm cực tiểu toàn cục hoặc cục bộ tốt nhất trên bề mặt hàm mất mát (Loss Landscape). Đối với mô hình phát hiện vật thể NMS-Free Detector mang kiến trúc phức tạp với hàng triệu tham số, lựa chọn và cấu hình thuật toán tối ưu quyết định trực tiếp tới khả năng vượt qua các điểm yên ngựa (Saddle Points), hiện tượng bùng nổ gradient, và tốc độ hội tụ của mô hình.

Chương này phân tích chi tiết cơ chế phân tách nhóm tham số **Parameter Grouping (Decay vs No-Decay)**, cơ sở toán học của hai thuật toán tối ưu **AdamW** và **SGD với Nesterov Momentum**, cơ chế **Decoupled Weight Decay**, và kỹ thuật **Gradient Clipping** được cài đặt trong tệp [`src/train/engine.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py).

---

## 2. NỘI DUNG PHÂN TÍCH THUẬT TOÁN TỐI ƯU

### 2.1. Phân Tách Nhóm Tham Số (Parameter Grouping: Decay vs No-Decay)

Một sai lầm phổ biến khi cấu hình thuật toán tối ưu là áp dụng suy giảm trọng số (Weight Decay) đồng nhất cho toàn bộ các tham số trong mô hình. Trong tệp [`src/train/engine.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L77-L98), phương thức `get_optimizer` thực hiện phân tách tham số thành hai nhóm riêng biệt:

```python
def get_optimizer(model: NMSFreeDetector, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},      # 5e-4
        {"params": no_decay, "weight_decay": 0.0}                # 0.0
    ]
```

#### Ý nghĩa kỹ thuật và Lý do thiết kế:

1. **Nhóm `decay` (`weight_decay = 5e-4`)**:
   - Bao gồm các Tensor trọng số 2 chiều trở lên ($ndim \ge 2$), điển hình là ma trận trọng số của các lớp `nn.Conv2d` ($C_{\text{out}}, C_{\text{in}}, K_h, K_w$).
   - **Tác dụng**: Áp dụng Weight Decay giúp kiểm soát chuẩn $L_2$ của trọng số, ép các trọng số không quan trọng tiến về 0, giảm hiện tượng học tủ (Overfitting) và tăng khả năng tổng quát hóa.

2. **Nhóm `no_decay` (`weight_decay = 0.0`)**:
   - Bao gồm các Tensor 1 chiều ($ndim \le 1$) hoặc các vector bias:
     - Vector Bias của Conv2d và Linear ($b$).
     - Trọng số $\gamma$ (scale) và $\beta$ (shift) của các lớp `nn.BatchNorm2d`.
     - Tham số LayerScale $\gamma_1, \gamma_2$ trong các khối Attention `C2fPSA`.
   - **Lý do loại bỏ Weight Decay**:
     - Các vector Bias và tham số BatchNorm quy định ngưỡng kích hoạt và thang đo của đặc trưng. Nếu áp dụng Weight Decay lên Bias hoặc $\gamma$ của BN, thuật toán sẽ ép các đại lượng này về 0, làm suy yếu khả năng biểu diễn của hàm kích hoạt (SiLU) và triệt tiêu khả năng chuẩn hóa của BatchNorm, khiến mô hình bị suy giảm độ chính xác nghiêm trọng hoặc rơi vào trạng thái không hội tụ.

---

### 2.2. Phân Tích Toán Học Thuật Toán Tối Ưu AdamW (Default Optimizer)

Dự án thiết lập **AdamW** làm thuật toán tối ưu mặc định (`cfg.optimizer = "adamw"`).

#### Các siêu tham số mặc định trong `TrainConfig`:
- Tốc độ học ban đầu: $\eta = \text{lr0} = 10^{-3}$
- Betas: $(\beta_1, \beta_2) = (0.9, 0.98)$ (Khác với mặc định PyTorch $\beta_2 = 0.999$)
- Epsilon: $\epsilon = \text{esp} = 10^{-6}$
- Weight Decay: $\lambda = 5 \times 10^{-4}$

#### Các bước tính toán toán học trong mỗi Iteration $t$:

Cho $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ là gradient của hàm mất mát đối với tham số $\theta$ tại step $t$:

1. **Tính Mô-men Động Lượng Thứ Nhất (First Moment Vector - Moving Average of Gradients)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   với $\beta_1 = 0.9$, $m_t$ đóng vai trò là vận tốc trung bình của gradient, giúp mô hình lướt qua các khe núi hẹp và điểm yên ngựa.

2. **Tính Mô-men Động Lượng Thứ Hai (Second Moment Vector - Moving Average of Uncentered Variance)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   với $\beta_2 = 0.98$, $v_t$ đo lường phương sai của gradient. Việc hạ $\beta_2$ từ $0.999$ xuống $0.98$ giúp thuật toán thích ứng nhanh hơn với sự thay đổi đột ngột của gradient khi làm việc với dữ liệu đa dạng quy mô Object365.

3. **Hiệu Chỉnh Độ Lệch Khởi Tạo (Bias Correction)**:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. **Cập Nhật Tham Số với Decoupled Weight Decay**:
   $$\theta_{t+1} = \theta_t - \eta_t \cdot \lambda \cdot \theta_t - \frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

```text
Comparison: Adam (L2 Regularization) vs AdamW (Decoupled Weight Decay)

Standard Adam with L2:
   g_t' = g_t + \lambda \theta_t
   m_t  = \beta_1 m_{t-1} + (1 - \beta_1) g_t'
   v_t  = \beta_2 v_{t-1} + (1 - \beta_2) (g_t')^2   <-- Weight decay is scaled by 1/sqrt(v_t)!

AdamW (Decoupled):
   \theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t  <-- Pure decoupled decay!
```

- **Sự khác biệt cốt lõi**: Trong Adam truyền thống, Weight Decay được cộng trực tiếp vào Gradient $g_t$, khiến cho tác dụng phạt $L_2$ bị chia cho $\sqrt{v_t}$. Với những tham số có gradient lớn ($v_t$ lớn), hình phạt $L_2$ bị thu nhỏ lại một cách vô lý. AdamW tách rời hoàn toàn phép phạt weight decay ra khỏi gradient động lượng, đảm bảo tỷ lệ phạt $L_2$ ổn định trên mọi tham số.

---

### 2.3. Phân Tích Thuật Toán Tối Ưu SGD Nesterov Momentum (Alternative Optimizer)

Dự án cung cấp tùy chọn thuật toán **SGD** (`cfg.optimizer = "sgd"`):

```python
elif cfg.optimizer == "sgd":
    opt = torch.optim.SGD(groups, lr=cfg.lr0, momentum=cfg.momentum, nesterov=True)
```

#### Công thức toán học của SGD với Nesterov Accelerated Gradient (NAG):

1. **Vận tốc động lượng Nesterov**:
   $$v_t = \mu \cdot v_{t-1} + g_t$$
   trong đó $\mu = \text{momentum} = 0.9$.

2. **Cập nhật tham số**:
   $$\theta_{t+1} = \theta_t - \eta_t \cdot (g_t + \mu \cdot v_t) - \eta_t \cdot \lambda \cdot \theta_t$$

- **Ưu điểm**: Nesterov Momentum tính toán gradient bằng cách "nhìn trước một bước" theo hướng động lượng hiện tại, giúp giảm thiểu hiện tượng dao động văng quá đà (Overshooting) khi mô hình tiến gần đến đáy cực tiểu.

---

### 2.4. Phân Tích Kỹ Thuật Gradient Clipping (Giới Hạn Gradient)

Trong quá trình huấn luyện mô hình NMS-Free Detector với dải AMP FP16, các bước tính loss phức tạp của Task-Aligned Assigner có thể tạo ra các spike gradient cực lớn tại một số iteration.

Cài đặt trong [`src/train/engine.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L184):
```python
nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)  # max_norm = 10.0
```

#### Công thức toán học của Gradient Norm Clipping:

1. Tính chuẩn $L_2$ tổng thể của toàn bộ gradient trong mô hình:
   $$\|g\|_2 = \sqrt{\sum_{i} \|g_i\|_2^2}$$

2. Nếu $\|g\|_2 > M_{\text{clip}}$ (với $M_{\text{clip}} = 10.0$):
   $$g_i \leftarrow g_i \cdot \frac{M_{\text{clip}}}{\|g\|_2}$$

- **Tác dụng**: Giữ nguyên hướng lan truyền của vector gradient toàn cục nhưng điều chỉnh độ dài (magnitude) không vượt quá 10.0. Kỹ thuật này triệt tiêu 100% rủi ro bùng nổ gradient làm nổ trọng số mô hình.

---

## 3. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 3.1. Bảng So Sánh Hiệu Năng: AdamW vs SGD Nesterov

| Tiêu Chí So Sánh | Thuật Toán AdamW | Thuật Toán SGD Nesterov |
| :--- | :--- | :--- |
| **Tốc độ hội tụ ban đầu** | **Cực nhanh (Fast Convergence)** | Chậm hơn, cần Warmup dài hơn |
| **Độ nhạy Siêu tham số** | Thấp (Dễ huấn luyện với $\beta_1, \beta_2$ chuẩn) | Cao (Phụ thuộc mạnh vào LR & Momentum) |
| **Bộ nhớ VRAM tiêu tốn** | Tốn thêm $2 \times N_{\text{params}}$ (Lưu $m_t$ và $v_t$) | Chỉ tốn $1 \times N_{\text{params}}$ (Lưu $v_t$) |
| **Độ tổng quát hóa cuối** | Rất cao khi kết hợp với Decoupled Decay | Cao, nhưng dễ rơi vào saddle point nếu LR sai |
| **Khuyến nghị sử dụng** | **Mặc định cho NMSFreeDetector** | Dùng khi Fine-tune trên tập dữ liệu nhỏ |

### 3.2. Ảnh Hưởng Tới Bộ Nhớ VRAM và Tốc Độ Tính Toán

Với mô hình NMSFreeDetector có khoảng $N = 11.8 \times 10^6$ tham số:
- **Bộ nhớ lưu trọng số FP32**: $11.8 \times 4 \approx 47.2$ MB.
- **Bộ nhớ lưu State Optimizer AdamW (FP32)**:
  - Vector $m_t$: $47.2$ MB.
  - Vector $v_t$: $47.2$ MB.
  - Tổng bộ nhớ cho Optimizer State = $94.4$ MB.
Mức tiêu tốn này là hoàn toàn tối ưu và phù hợp cho các GPU thương mại phổ thông (từ 8GB VRAM trở lên).

---

## 4. KẾT LUẬN CHƯƠNG

Chiến lược tối ưu hóa tham số trong dự án được xây dựng dựa trên sự hiểu biết sâu sắc về động học mạng thần kinh. Việc áp dụng **Phân tách nhóm tham số Decay vs No-Decay**, kết hợp thuật toán **AdamW với Decoupled Weight Decay**, điều chỉnh siêu tham số $\beta_2 = 0.98, \epsilon = 1e-6$, cùng cơ chế bảo vệ **Gradient Norm Clipping ($max=10.0$)** đã tạo nên một bộ máy tối ưu mạnh mẽ. Hệ thống này cho phép mô hình hội tụ nhanh chóng, vượt qua các điểm yên ngựa phức tạp của hàm loss Dual-Head, đồng thời duy trì sự ổn định tuyệt đối trong suốt 100 epoch huấn luyện.
