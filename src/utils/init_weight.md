# 📘 TÀI LIỆU TOÁN HỌC & LÝ DO KHỞI TẠO TRỌNG SỐ (WEIGHT INITIALIZATION GUIDE)
## MÔ HÌNH NMS-FREE DETECTOR (YOLOV10 ARCHITECTURE)

---

## 📋 I. TỔNG QUAN QUY TRÌNH KHỞI TẠO 3 PHA (3-PHASE WORKFLOW)

Quá trình khởi tạo trọng số trong [`src/utils/init_weights.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py) được thực hiện theo cơ chế **3 pha tuần tự (3-Phase Pipeline)** nhằm đảm bảo mọi lớp tích chập, chuẩn hóa, chú ý (attention) và đầu ra dự đoán đều đạt trạng thái cân bằng số học trước khi bước vào huấn luyện:

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ PHA 1: Khởi tạo toàn cục (Global Architecture Initialization)                │
  │ • Conv2d       → Kaiming Normal (mode='fan_out', nonlinearity='relu')        │
  │ • BatchNorm2d  → gamma=1.0, beta=0.0, eps=1e-3, momentum=0.03                │
  └──────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                                         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ PHA 2: Bảo tồn & Khóa các tầng chức năng đặc thù                             │
  │ • DFL Block    → Khóa bất biến: vector cố định [0, 1, 2, ..., reg_max-1]     │
  │ • LayerScale   → gamma1, gamma2 = 1e-2 trong khối C2fPSA (Attention)         │
  └──────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                                         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ PHA 3: Khôi phục Bias đầu ra đặc thù cho Detection Head                      │
  │ • Cls Bias     → Stride-Aware Prior: log(5 / nc / (img_size/stride)²)        │
  │ • Reg Bias     → Constant = 1.0                                              │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 II. CHI TIẾT TỪNG THÀNH PHẦN, CƠ SỞ TOÁN HỌC & LÝ DO THIẾT KẾ

---

### 1. Lớp Tích chập (`nn.Conv2d`): Kaiming Normal (`fan_out`)

* **Vị trí cài đặt:** [`src/utils/init_weights.py:25-36`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L25-L36)
* **Code thực thi:**
  ```python
  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
  if m.bias is not None:
      nn.init.constant_(m.bias, 0.0)
  ```

#### 📐 Cơ sở toán học
Trọng số ma trận $W$ được lấy mẫu ngẫu nhiên từ phân phối chuẩn Gaussian:
$$W \sim \mathcal{N}\left(0, \, \sigma^2\right) \quad \text{với} \quad \sigma = \sqrt{\frac{2}{\text{fan\_out}}}$$
Trong đó:
$$\text{fan\_out} = c_{\text{out}} \times k_h \times k_w$$
($c_{\text{out}}$: số lượng channel đầu ra, $k_h, k_w$: chiều cao và chiều rộng của kernel tích chập).

#### 💡 Lý do thiết kế & Ý nghĩa kỹ thuật
1. **Khắc phục sự triệt tiêu phương sai của hàm kích hoạt phi tuyến:**
   * Trong mạng tích chập sâu sử dụng hàm kích hoạt phi tuyến như ReLU ($f(x) = \max(0, x)$) hoặc SiLU ($f(x) = x \cdot \sigma(x)$), khoảng một nửa số giá trị đầu vào ở vùng âm bị triệt tiêu về 0.
   * Khởi tạo Xavier (Glorot) vốn giả định hàm kích hoạt tuyến tính quanh điểm 0 sẽ làm phương sai của tín hiệu giảm đi một nửa qua mỗi tầng:
     $$\text{Var}(y) = \frac{1}{2} \cdot \text{Var}(x)$$
   * Hệ số $\sqrt{2}$ trong công thức Kaiming He (2015) giúp nhân đôi phương sai ban đầu, bù đắp chính xác phần năng lượng bị tiêu hao bởi hàm kích hoạt.
2. **Tại sao chọn `mode='fan_out'` thay vì `fan_in`?**
   * Trong mạng CNN sâu có nhiều nhánh phân tách và cộng gộp Residual (như `C2f`, `C2fCIB`), nguy cơ mất ổn định gradient ở chiều lan truyền ngược (Backward pass) cao hơn chiều truyền xuôi (Forward pass).
   * Khởi tạo theo `fan_out` giúp bảo toàn phương sai của gradient lan truyền ngược từ tầng sâu nhất về tận các tầng đầu tiên:
     $$\text{Var}\left(\frac{\partial L}{\partial x_l}\right) = \text{Var}\left(\frac{\partial L}{\partial x_L}\right)$$
3. **Tại sao chọn `nonlinearity='relu'` cho mạng dùng SiLU?**
   * PyTorch hiện chưa hỗ trợ `mode='silu'` riêng trong `nn.init.kaiming_normal_`. 
   * Hàm SiLU có độ dốc trung bình ở miền dương xấp xỉ bằng $1.0$ (tương đương ReLU), do đó hệ số gain $\sqrt{2}$ của ReLU là xấp xỉ toán học tối ưu nhất cho SiLU.

---

### 2. Lớp Chuẩn hóa Batch (`nn.BatchNorm2d`): Tham số chuẩn YOLO

* **Vị trí cài đặt:** [`src/utils/init_weights.py:38-49`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L38-L49)
* **Code thực thi:**
  ```python
  m.eps = 1e-3
  m.momentum = 0.03
  nn.init.constant_(m.weight, 1.0)
  nn.init.constant_(m.bias, 0.0)
  ```

#### 📐 Cơ sở toán học
Công thức biến đổi của BatchNorm:
$$y = \gamma \cdot \hat{x} + \beta = \gamma \cdot \left(\frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\right) + \beta$$
Trong đó:
* $\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$ (trung bình của batch)
* $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$ (phương sai của batch)
* Giá trị thống kê toàn cục cập nhật qua từng bước: $\mu_{\text{running}} = (1 - m)\mu_{\text{running}} + m\mu_B$

#### 💡 Lý do thiết kế & Ý nghĩa kỹ thuật
1. **$\gamma = 1.0, \beta = 0.0$ (Identity Mapping ban đầu):**
   * Đảm bảo ở những bước huấn luyện đầu tiên, BatchNorm chỉ làm nhiệm vụ chuẩn hóa phân phối (đưa về mean 0, std 1) mà không thực hiện phép co giãn hay dịch chuyển làm biến dạng luồng thông tin của mạng.
2. **$\epsilon = 10^{-3}$ (So với mặc định PyTorch $10^{-5}$):**
   * Khi huấn luyện với chế độ Tự động trộn độ chính xác (**AMP - Mixed Precision FP16**), dải động của số thực 16-bit bị hạn chế (số dương nhỏ nhất có thể biểu diễn là $\approx 6 \times 10^{-5}$).
   * Nếu $\sigma_B^2 \approx 0$ trong các vùng ảnh đồng màu, $\epsilon = 10^{-5}$ rất dễ gây ra lỗi tràn số dưới (**Underflow**) hoặc lỗi chia cho 0. Giá trị $\epsilon = 10^{-3}$ đảm bảo an toàn tuyệt đối về mặt số học.
3. **$\text{momentum} = 0.03$ (So với mặc định PyTorch $0.1$):**
   * Trong bài toán Object Detection, kích thước batch trên mỗi GPU thường nhỏ (4 - 16). Kích thước batch nhỏ khiến $\mu_B$ và $\sigma_B^2$ bị dao động mạnh giữa các bước.
   * Hệ số momentum $0.03$ tương đương việc tính trung bình động trượt trên một cửa sổ khoảng $\frac{1}{0.03} \approx 33$ batches, giúp làm mịn các giá trị thống kê `running_mean` và `running_var`, tránh rung giật mạng.

---

### 3. Khối Distribution Focal Loss (`DFL`): Toán tử Kỳ vọng Bất biến

* **Vị trí cài đặt:** [`src/blocks.py:89-102`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/blocks.py#L89-L102) & [`src/utils/init_weights.py:54-55`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L54-L55)
* **Code thực thi:**
  ```python
  self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
  x = torch.arange(c1, dtype=torch.float)
  self.conv.weight.data[:] = x.view(1, c1, 1, 1)
  ```

#### 📐 Cơ sở toán học
DFL không dự đoán trực tiếp một số thực duy nhất cho khoảng cách bounding box mà mô hình hóa khoảng cách dưới dạng phân phối xác suất rời rạc trên $\text{reg\_max} = 16$ bin (tương ứng với các khoảng cách $i \in \{0, 1, 2, \dots, 15\}$ tính theo đơn vị grid cell). 

Tọa độ khoảng cách giải mã $\hat{y}$ là **Kỳ vọng toán học (Mathematical Expectation)** của phân phối:
$$\hat{y} = \mathbb{E}[x] = \sum_{i=0}^{\text{reg\_max}-1} i \cdot P(y = i) = \sum_{i=0}^{\text{reg\_max}-1} i \cdot \text{Softmax}(z_i)$$

#### 💡 Lý do thiết kế & Ý nghĩa kỹ thuật
* Lớp `nn.Conv2d(16, 1, 1)` ở đây thực chất là một **phép nhân vô hướng cố định (Fixed Linear Combination)** giữa vector trọng số $[0, 1, 2, \dots, 15]$ với vector xác suất sau Softmax.
* Đây là một phép biến đổi toán học giải mã, **không phải là tham số học**. Do đó, lớp này bắt buộc phải đặt `requires_grad=False` và phải được bỏ qua trong hàm `_initialize_trainable_layers` để không bị Kaiming Normal ghi đè ngẫu nhiên.

---

### 4. Khối Chú ý PSA (`Attention / C2fPSA`): LayerScale Stabilization

* **Vị trí cài đặt:** [`src/blocks.py:137-138`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/blocks.py#L137-L138)
* **Code thực thi:**
  ```python
  self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))  # layer_scale = 1e-2
  self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
  ```

#### 📐 Cơ sở toán học
Công thức khối Self-Attention kết hợp LayerScale:
$$x_{l+1} = x_l + \gamma_1 \cdot \text{SelfAttention}(x_l) + \gamma_2 \cdot \text{FFN}(x_l) \quad \text{với} \quad \gamma_1, \gamma_2 = 10^{-2}$$

#### 💡 Lý do thiết kế & Ý nghĩa kỹ thuật
* Trong các epoch đầu tiên, ma trận tương quan không gian $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)$ chưa được học, các giá trị attention phân bổ hỗn loạn và có thể sinh ra xung gradient đột biến.
* Việc nhân với hệ số co giãn $\gamma = 0.01$ giúp nhánh Attention ban đầu chỉ đóng góp $1\%$ năng lượng tín hiệu. Mạng hoạt động ổn định như một kiến trúc CNN thuần túy (Identity shortcut chiếm $99\%$), sau đó gradient sẽ dần dần mở rộng $\gamma$ để tiếp nhận đặc trưng toàn cục từ Self-Attention.

---

### 5. Bias Phân loại Đầu ra (`ScaleHead.cls`): Stride-Aware Focal Prior

* **Vị trí cài đặt:** [`src/head.py:48-56`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py#L48-L56) & [`src/utils/init_weights.py:107-112`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py#L107-L112)
* **Code thực thi:**
  ```python
  def init_stride_bias(self, stride, img_size=480):
      value = math.log(5 / self.nc / (img_size / stride) ** 2)
      for m in (self.cls_o2m, self.cls_o2o):
          nn.init.constant_(m.bias, value)
  ```

#### 📐 Cơ sở toán học & Dẫn xuất xác suất
1. **Nguyên lý Focal Loss Prior (Lin et al., RetinaNet):**
   Đầu ra phân loại của mỗi anchor là logit $z$. Sau khi đi qua hàm Sigmoid, xác suất dự đoán là:
   $$p = \sigma(z) = \frac{1}{1 + e^{-z}} \implies z = \log\left(\frac{p}{1 - p}\right)$$
   Khi $p \ll 1$ (xác suất xuất hiện vật thể rất nhỏ), ta có $1 - p \approx 1$, do đó:
   $$z \approx \log(p)$$

2. **Dẫn xuất xác suất kỳ vọng theo Stride ($p_{\text{stride}}$):**
   * Trong một bức ảnh tự nhiên thông thường, số lượng đối tượng trung bình xuất hiện là $\approx 5$ vật thể.
   * Số lượng class mục tiêu là $N_c$.
   * Tại tầng feature map có stride $s$, tổng số anchor (grid cells) trên ảnh là:
     $$N_{\text{anchors}} = \left(\frac{\text{img\_size}}{s}\right)^2$$
   * Xác suất tiền nghiệm (Prior probability) để một grid cell bất kỳ chứa đúng đối tượng của class $c$ là:
     $$p = \frac{5}{N_c \times \left(\frac{\text{img\_size}}{s}\right)^2}$$
   * Do đó, giá trị bias tối ưu được thiết lập:
     $$b_{\text{cls}} = \log\left(\frac{5}{N_c \cdot (\text{img\_size} / s)^2}\right)$$

#### 📊 Bảng đối soát số học cụ thể ($N_c = 80, \text{img\_size} = 480$):

| Feature Map | Stride ($s$) | Kích thước Grid | Tổng số Grid Cells | Xác suất tiền nghiệm ($p$) | Giá trị Bias khởi tạo ($b_{\text{cls}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P3** (Small objects) | 8 | $60 \times 60$ | 3,600 | $\frac{5}{80 \times 3600} \approx 1.736 \times 10^{-5}$ | **$-10.96$** |
| **P4** (Medium objects)| 16 | $30 \times 30$ | 900 | $\frac{5}{80 \times 900} \approx 6.944 \times 10^{-5}$ | **$-9.57$** |
| **P5** (Large objects) | 32 | $15 \times 15$ | 225 | $\frac{5}{80 \times 225} \approx 2.777 \times 10^{-4}$ | **$-8.19$** |

#### 💡 Lý do thiết kế & Ý nghĩa kỹ thuật
* **Chống hiện tượng "Bùng nổ Gradient" (Gradient Explosion) ở Step đầu:**
  Nếu khởi tạo bias mặc định $b = 0 \implies p = 0.5$, toàn bộ hàng nghìn anchor nền (background) sẽ đồng loạt sinh ra Binary Cross-Entropy Loss cực lớn ($-\log(1 - 0.5) = 0.693$ trên mỗi anchor). Xung gradient khổng lồ này sẽ ngay lập tức phá vỡ toàn bộ các đặc trưng có sẵn của mạng.
* **Thích ứng động theo mật độ không gian:**
  Tầng P3 có số lượng cells lớn gấp 16 lần tầng P5 nhưng số lượng vật thể không tăng tương ứng $\rightarrow$ bias của P3 cần âm sâu hơn ($-10.96$) so với P5 ($-8.19$).

---

### 6. Bias Hồi quy Đầu ra (`ScaleHead.reg`): Bounding Box Distance Prior

* **Vị trí cài đặt:** [`src/head.py:46`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py#L46)
* **Code thực thi:**
  ```python
  for m in (self.reg_o2m, self.reg_o2o):
      nn.init.constant_(m.bias, 1.0)
  ```

#### 📐 Cơ sở toán học & Ý nghĩa hình học
* Đầu ra của nhánh hồi quy biểu diễn khoảng cách $(l, t, r, b)$ từ tâm anchor đến 4 cạnh của bounding box (Left, Top, Right, Bottom), tính theo đơn vị grid cell.
* Khởi tạo $b_{\text{reg}} = 1.0$ tương đương với việc định hướng cho mô hình bắt đầu dự đoán các hộp bao có kích thước ban đầu:
  $$\text{Width} = l + r = 1.0 + 1.0 = 2.0 \text{ grid cells}$$
  $$\text{Height} = t + b = 1.0 + 1.0 = 2.0 \text{ grid cells}$$
* **Tác dụng:** Tránh trường hợp mạng dự đoán các khoảng cách âm hoặc các hộp bao khổng lồ phủ kín toàn bộ bức ảnh ở bước đầu, giúp phân phối xác suất DFL tập trung học từ vùng không gian lân cận của từng anchor point.

---

## 📊 III. BẢNG TỔNG HỢP MA TRẬN KHỞI TẠO (SUMMARY MATRIX)

| Thành phần mô hình | Loại Layer | Phương pháp khởi tạo | Tham số chi tiết | Mục đích số học & Tác dụng thực tế |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone & Neck** | `nn.Conv2d` | Kaiming Normal | `mode='fan_out'`, `nonlinearity='relu'`, `bias=0.0` | Bảo toàn phương sai gradient truyền ngược trong mạng residual sâu |
| **Toàn bộ mạng** | `nn.BatchNorm2d`| Constant Init | `weight=1.0`, `bias=0.0`, `eps=1e-3`, `momentum=0.03` | Giữ nguyên tín hiệu ban đầu; chống underflow FP16/AMP; làm mịn thống kê running |
| **Head (DFL)** | `nn.Conv2d (1x1)`| Constant Fixed | `weight=[0, ..., 15]`, `requires_grad=False` | Đóng vai trò toán tử kỳ vọng toán học $\mathbb{E}[x]$, không được cập nhật |
| **PSA Block** | `LayerScale` | Constant Scale | `gamma1=1e-2`, `gamma2=1e-2` | Khống chế nhiễu Attention ban đầu, đảm bảo hội tụ ổn định |
| **Head (Classification)**| `nn.Conv2d (1x1)`| Stride-Aware Prior| $b = \log\left(\frac{5}{N_c \cdot (\text{img\_size}/s)^2}\right)$ | Cân bằng tỷ lệ nền/vật thể; thích ứng mật độ anchor theo từng tầng P3-P5 |
| **Head (Regression)** | `nn.Conv2d (1x1)`| Constant Bias | `bias=1.0` | Khởi tạo kích thước bounding box chuẩn quanh anchor ($\approx 2 \times 2$ cells) |

---
*Tài liệu được trích xuất và đối soát trực tiếp từ mã nguồn hệ thống [`src/utils/init_weights.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/init_weights.py), [`src/head.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/head.py) và [`src/blocks.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/blocks.py).*
