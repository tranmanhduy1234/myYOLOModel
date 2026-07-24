# CHƯƠNG 3: LÝ THUYẾT TOÁN HỌC VÀ CƠ CHẾ KHỞI TẠO TRỌNG SỐ

## 1. GIỚI THIỆU CHƯƠNG

Trong mạng thần kinh sâu (Deep Neural Networks), **Khởi tạo Trọng số (Weight Initialization)** là điểm xuất phát toán học quyết định toàn bộ động học truyền tín hiệu (Signal Propagation Dynamics) trong cả hai chiều: lan truyền tiến (Forward Pass) và lan truyền ngược (Backward Pass). Một chiến lược khởi tạo sai lầm có thể dẫn đến hai thảm họa toán học điển hình:

1. **Bùng nổ Tín hiệu / Gradient (Exploding Signal & Gradient)**: Phương sai của activations hoặc gradients tăng theo cấp số nhân qua từng lớp, làm các giá trị tham số tiến đến $\pm \infty$ (NaN/Inf) khiến mô hình lập tức phân kỳ.
2. **Triệt tiêu Tín hiệu / Gradient (Vanishing Signal & Gradient)**: Phương sai suy giảm về 0 qua các lớp sâu, khiến các lớp Backbone ở đầu mạng không nhận được tín hiệu học, mô hình rơi vào trạng thái đứng yên (Stagnation).

Đặc biệt đối với kiến trúc mô hình phát hiện vật thể không sử dụng NMS (**NMS-Free Detector**), quá trình khởi tạo càng trở nên nhạy cảm hơn do mạng phải gánh vác hai nhiệm vụ phân loại và định vị phức tạp đồng thời trên nhiều quy mô (multi-scale pyramid) mà không có bước lọc hậu xử lý NMS.

Chương này đi sâu phân tích toàn bộ lý thuyết toán học, các chứng minh phương sai, và cơ chế **Khởi tạo Trọng số 3 Pha (3-Phase Weight Initialization)** được cài đặt trong tệp [`src/utils/init_weights.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py) và [`src/head.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py).

---

## 2. QUY TRÌNH KHỞI TẠO TRỌNG SỐ 3 PHA (3-PHASE INITIALIZATION)

Tệp [`src/utils/init_weights.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py) định nghĩa hàm trung tâm `initialize_weights(model)` thực thi quy trình 3 pha nghiêm ngặt:

```text
               +-------------------------------------------------------+
               |              Mô hình NMSFreeDetector                  |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |  PHA 1: Quét toàn bộ Module (Recursive Scan)          |
               |  - Conv2d: Kaiming Normal (fan_out, relu), bias=0      |
               |  - BatchNorm2d: gamma=1.0, beta=0.0, eps=1e-3         |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |  PHA 2: Bỏ qua các Lớp Trọng số Cố định (Frozen)      |
               |  - DFL Module: weight = [0, 1, ..., reg_max-1]        |
               |  - DFL weight.requires_grad = False (Frozen)          |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |  PHA 3: Khôi phục & Cài đặt Bias Đặc thù Đầu ra        |
               |  - Cls Bias: Focal Prior (-4.5951)                    |
               |  - Stride-Aware Scaling: log(5 / nc / (img_size/S)^2)  |
               |  - Reg Bias: Constant (1.0)                           |
               +-------------------------------------------------------+
```

---

## 3. PHÂN TÍCH TOÁN HỌC CÁC PHƯƠNG PHÁP KHỞI TẠO

Các công thức toán học chi tiết được tổng hợp trong tệp [`formulas/weight_init_mathematics.md`](formulas/weight_init_mathematics.md).

### 3.1. Phân Tích Khởi Tạo Kaiming Normal (He Normal) Cho Conv2d

#### 1. Đặt vấn đề và Giả định Toán học:
Xét một lớp Convolution thứ $l$ với đầu vào $x_l$, trọng số $W_l$, và đầu ra trước kích hoạt $z_l = W_l x_l$. 
Giả sử $x_l$ và $W_l$ là các biến ngẫu nhiên độc lập lập lại (i.i.d), có trung bình bằng 0 ($\mathbb{E}[x_l] = 0, \mathbb{E}[W_l] = 0$).

Phương sai của một phần tử $z_l$ được tính bằng:
$$\text{Var}(z_l) = n_l \cdot \text{Var}(W_l \cdot x_l) = n_l \cdot \text{Var}(W_l) \cdot \mathbb{E}[x_l^2]$$
trong đó $n_l = C_{\text{in}} \cdot K_h \cdot K_w$ là số lượng kết nối đầu vào của một neuron (**fan_in**).

Khi đưa qua hàm kích hoạt phi tuyến (như ReLU hoặc SiLU), một nửa số giá trị âm bị triệt tiêu về 0. Do đó:
$$\mathbb{E}[x_l^2] = \frac{1}{2} \text{Var}(z_{l-1})$$

Suy ra phương sai tín hiệu truyền qua lớp $l$:
$$\text{Var}(z_l) = \left( \frac{1}{2} n_l \text{Var}(W_l) \right) \text{Var}(z_{l-1})$$

Để phương sai không bị bùng nổ hay suy giảm qua $L$ lớp sâu ($\text{Var}(z_L) = \text{Var}(z_0)$), ta phải bắt buộc hệ số nhân bằng 1:
$$\frac{1}{2} n_l \text{Var}(W_l) = 1 \implies \text{Var}(W_l) = \frac{2}{n_l}$$

#### 2. Lý do chọn `mode='fan_out'` thay vì `fan_in`:
Trong cài đặt mã nguồn dự án ([`_init_conv2d`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L27-L38)):
```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

- Nếu dùng `fan_in`: $n_l = C_{\text{in}} \cdot K_h \cdot K_w$.
- Nếu dùng `fan_out`: $n_l = C_{\text{out}} \cdot K_h \cdot K_w$.

Trong các mạng CNN hiện đại chứa nhiều kết nối tắt (Residual Connections, C2f, C2fCIB), số lượng kênh đầu ra $C_{\text{out}}$ thường tăng gấp đôi ở các công đoạn Downsampling (như qua khối `SCDown`). Việc chọn `mode='fan_out'` đảm bảo phương sai của **Gradient ở chiều lan truyền ngược (Backward Pass)** được bảo toàn ổn định tuyệt đối:
$$\text{Var}\left(\frac{\partial \mathcal{L}}{\partial x_l}\right) = \left( \frac{1}{2} n_l^{\text{out}} \text{Var}(W_l) \right) \text{Var}\left(\frac{\partial \mathcal{L}}{\partial z_l}\right) = 1 \cdot \text{Var}\left(\frac{\partial \mathcal{L}}{\partial z_l}\right)$$

#### 3. Sự tương thích giữa `nonlinearity='relu'` và hàm kích hoạt SiLU:
PyTorch chưa cung cấp tham số `nonlinearity='silu'` trong hàm `kaiming_normal_`. Hàm SiLU (Swish) $f(x) = x \cdot \sigma(x)$ có dạng đường cong mượt gần tiệm cận với ReLU ở vùng dương ($x > 0$) và tiệm cận về 0 ở vùng âm ($x < -3$). Việc sử dụng `nonlinearity='relu'` chính là xấp xỉ toán học tốt nhất và chính xác nhất cho SiLU, giữ hệ số căn $\sqrt{2}$ chuẩn xác.

---

### 3.2. Phân Tích Khởi Tạo BatchNorm2d (Constant Init)

Cài đặt trong [`_init_batchnorm2d`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L40-L51):
```python
m.eps = 1e-3
m.momentum = 0.03
nn.init.constant_(m.weight, 1.0)
nn.init.constant_(m.bias, 0.0)
```

#### 1. Ý nghĩa toán học của $\gamma = 1.0, \beta = 0.0$:
Biểu thức chuẩn hóa BatchNorm2d đối với feature map $x$:
$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
Bằng cách đặt trọng số học được $\gamma = 1.0$ (scale) và $\beta = 0.0$ (bias/shift), ở thời điểm bắt đầu huấn luyện $t=0$, BatchNorm2d hoạt động đúng nghĩa là một phép biến đổi chuẩn hóa dữ liệu mượt mà (Identity scale transform).

#### 2. Ý nghĩa của `eps = 1e-3` và `momentum = 0.03`:
- **PyTorch mặc định**: `eps = 1e-5`, `momentum = 0.1`.
- **Cài đặt YOLO chuẩn**: `eps = 1e-3`, `momentum = 0.03`.

*Tác dụng*:
- `eps = 1e-3` lớn hơn giúp ngăn ngừa triệt để lỗi chia cho 0 hoặc căn bậc hai của số gần bằng 0 khi huấn luyện bằng độ chính xác nửa AMP FP16 (nơi dải số bé nhất dễ bị underflow).
- `momentum = 0.03` nhỏ hơn làm cho trung bình động `running_mean` và `running_var` được cập nhật mượt mà hơn qua hàng nghìn iteration (Exponential Moving Average với hệ số $0.97$), tránh việc các batch nhỏ lẻ cục bộ làm biến động quá mạnh thông số thống kê của BN.

---

### 3.3. Phân Tích Khởi Tạo Trọng Số Cố Định DFL (Distribution Focal Loss Module)

Trong tệp [`src/blocks.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/blocks.py#L89-L103), lớp `DFL` biểu diễn phân phối khoảng cách bounding box rời rạc:

```python
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1
```

- Trọng số của lớp Conv2d 1x1 này không được lấy mẫu ngẫu nhiên mà được gán cố định bằng dãy số thực ngón đại số: $W = [0.0, 1.0, 2.0, \dots, 15.0]$.
- Cờ `.requires_grad_(False)` đóng đóng băng toàn bộ gradient của lớp này.
- **Pha 1 của `initialize_weights` chủ động bỏ qua `DFL`**:
  ```python
  if isinstance(m, DFL):
      continue
  ```
  Nếu Pha 1 không bỏ qua `DFL`, hàm `_init_conv2d` sẽ ghi đè dãy số $[0, 1, \dots, 15]$ bằng các giá trị ngẫu nhiên Kaiming Normal, phá hủy hoàn toàn khả năng giải mã khoảng cách hình học của mô hình.

---

### 3.4. Khôi Phục Bias Đặc Thù Cho Head (Focal Prior & Stride-Aware Bias)

Pha 1 đã ghi đè tất cả bias của Conv2d bằng $0.0$. Tuy nhiên, các lớp đầu ra của Detection Head cần giá trị bias đặc thù để đảm bảo sự ổn định ban đầu.

#### 1. Focal Class Prior Bias ($\text{prior} \approx -4.5951$):
Cài đặt trong [`ScaleHead._init_bias`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py#L41-L46):
```python
prior = -math.log((1 - 0.01) / 0.01)  # -ln(99) = -4.59512
for m in (self.cls_o2m, self.cls_o2o):
    nn.init.constant_(m.bias, prior)
```
- **Bản chất toán học**: Ở đầu quá trình huấn luyện, trên tổng số $A = 8400$ anchor, chỉ có khoảng 10-20 anchor chứa vật thể (Foreground), còn lại 99%+ là nền (Background).
- Nếu khởi tạo $bias = 0 \implies \text{Sigmoid}(0) = 0.5$. Mô hình sẽ dự đoán xác suất 50% cho tất cả 80 lớp đối tượng ở mọi anchor. Khi tính loss Binary Cross-Entropy (BCE), giá trị loss sẽ bùng nổ lên cực lớn:
  $$\mathcal{L}_{\text{BCE}} = -\log(0.5) \approx 0.693 \quad \text{cho mỗi class} \implies \text{Total Loss} \approx 80 \times 0.693 \approx 55.4$$
- Bằng cách gán bias bằng $-\ln(99) \approx -4.5951 \implies \text{Sigmoid}(-4.5951) = 0.01$. Xác suất dự đoán ban đầu cho mọi class được hạ xuống 1%. Loss BCE ban đầu cho background (chiếm 99% data) giảm xuống gần bằng 0:
  $$\mathcal{L}_{\text{BCE\_bg}} = -\log(1 - 0.01) = -\log(0.99) \approx 0.01005$$
  giúp Loss phân loại ban đầu không bị bùng nổ, đưa tổng loss về dải an toàn $\sim 1.5 - 3.0$.

#### 2. Stride-Aware Classification Bias Scaling:
Cài đặt nâng cao trong [`ScaleHead.init_stride_bias`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py#L48-L55) và được gọi tự động trong `DetectHead.__init__`:

```python
def init_stride_bias(self, stride, img_size=640):
    value = math.log(5 / self.nc / (img_size / stride) ** 2)
    for m in (self.cls_o2m, self.cls_o2o):
        nn.init.constant_(m.bias, value)
```

- **Công thức toán học**:
  $$b_{\text{cls}}(S) = \ln\left( \frac{5}{N_{\text{cls}} \cdot \left(\frac{H_{\text{img}}}{S}\right)^2} \right)$$
- **Ý nghĩa**: Mật độ ô lưới (grid cells) ở các cấp độ feature map là hoàn toàn khác nhau:
  - Cấp P3 ($S=8$): Có $60 \times 60 = 3600$ ô lưới (hoặc $80 \times 80 = 6400$). Mật độ ô lưới rất dày nên xác suất một ô lưới chứa vật thể nhỏ hơn nhiều.
  - Cấp P5 ($S=32$): Chỉ có $15 \times 15 = 225$ ô lưới (hoặc $20 \times 20 = 400$). Mật độ thưa hơn nên xác suất một ô lưới chứa vật thể lớn hơn.
- Việc điều chỉnh bias phân loại tỉ lệ nghịch với diện tích lưới $(H_{\text{img}}/S)^2$ giúp phản ánh chính xác mật độ vật thể kỳ vọng ở từng quy mô, triệt tiêu hoàn toàn sự lệch pha gradient giữa 3 đầu ra P3, P4, P5 ngay từ epoch đầu tiên.

---

## 4. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 4.1. Bảng Thống Kê So Sánh Tác Động Khởi Tạo Trọng Số

| Thành Phần Module | Phương Pháp Khởi Tạo | Tác Động Tới Gradient | Tác Động Tới Hội Tụ |
| :--- | :--- | :--- | :--- |
| **Conv2d (Backbone/Neck)** | Kaiming Normal (fan_out) | Cân bằng phương sai lan truyền ngược | Triệt tiêu Vanishing/Exploding Grad |
| **BatchNorm2d** | Constant $\gamma=1, \beta=0, \epsilon=1e-3$ | Giữ ổn định thang đo dải AMP FP16 | Bảo vệ không bị nổ số chia ở $t=0$ |
| **DFL Module** | Constant Weight $[0..15]$ Frozen | Ngắt lan truyền gradient về DFL | Đảm bảo tính chính xác giải mã box |
| **Cls Head Bias** | Focal Prior (-4.5951) | Giảm 99% nhiễu gradient background | Tránh bùng nổ BCE Loss ở Epoch 1 |
| **Stride-Aware Bias** | Scale-Density Prior Logit | Cân bằng tín hiệu giữa P3, P4, P5 | Giúp 3 Scale Head học đồng đều |

---

## 5. KẾT LUẬN CHƯƠNG

Khởi tạo trọng số trong mô hình NMS-Free Detector không phải là các câu lệnh ngẫu nhiên vô thức, mà là một công trình toán học được tính toán tỉ mỉ. Việc áp dụng **Quy trình Khởi tạo 3 Pha** - kết hợp giữa **Kaiming Normal fan_out cho Conv2d**, **Constant Init cho BatchNorm2d**, **Bảo toàn trọng số cố định cho DFL**, và **Điều chỉnh bias phân loại theo Focal Prior & Stride Scaling** - đã tạo ra một trạng thái xuất phát lý tưởng. Sự chuẩn bị kỹ lưỡng này giúp mô hình hoàn toàn miễn dịch với các sự cố bùng nổ/triệt tiêu gradient, duy trì loss ban đầu ở mức tối ưu và đảm bảo tốc độ hội tụ nhanh vượt trội ngay từ những bước lặp huấn luyện đầu tiên.
