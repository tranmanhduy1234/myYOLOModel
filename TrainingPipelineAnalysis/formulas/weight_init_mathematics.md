# TỔNG HỢP CÔNG THỨC TOÁN HỌC KHỞI TẠO TRỌNG SỐ (WEIGHT INITIALIZATION)

## 1. Kaiming Normal (He Normal) cho Lớp Conv2d

Để bảo toàn phương sai của tín hiệu truyền thẳng (forward signal variance) qua các lớp Convolution kèm kích hoạt SiLU/ReLU:

$$\text{Var}(y_l) = \text{Var}(x_l)$$

Trọng số $W_l \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$ của lớp `nn.Conv2d` được lấy mẫu từ phân phối chuẩn:

$$W_{l, i, j, k, m} \sim \mathcal{N}\left(0, \sigma_l^2\right)$$

với độ lệch chuẩn $\sigma_l$:

$$\sigma_l = \sqrt{\frac{2}{\text{fan\_out}}}$$

trong đó:
$$\text{fan\_out} = C_{\text{out}} \times K_h \times K_w$$

- Mode chọn `fan_out`: Giữ phương sai gradient ổn định trong chiều backward và chiều forward qua các đường nhánh residual.
- Nonlinearity chọn `relu`: Hệ số nhân $\sqrt{2}$ bù đắp cho việc triệt tiêu 50% tín hiệu âm khi đi qua hàm ReLU/SiLU.

---

## 2. BatchNorm2d Initialization

Với mỗi lớp `nn.BatchNorm2d`:

- Trọng số tỉ lệ (Scale / Gamma): $W_{\text{BN}} = 1.0$ (Hằng số)
- Độ lệch (Shift / Beta): $B_{\text{BN}} = 0.0$ (Hằng số)
- Epsilon: $\epsilon = 10^{-3}$ (Khác chuẩn mặc định PyTorch $10^{-5}$)
- Momentum: $\eta = 0.03$ (Khác chuẩn mặc định PyTorch $0.1$)

Ý nghĩa: Khởi tạo BatchNorm2d thành hàm đồng nhất (Identity transformation) ở thời điểm $t=0$:

$$\text{BN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \approx 1.0 \cdot x + 0.0 = x$$

---

## 3. Focal Prior Logit Initialization cho Classification Head Bias

Để giải quyết vấn đề mất cân bằng nghiêm trọng giữa Foreground và Background ở thời điểm bắt đầu huấn luyện (khi 99%+ anchor là background):

Đặt xác suất ưu tiên ban đầu $p = 0.01$:

$$\sigma(b_{\text{cls}}) = p \implies \frac{1}{1 + e^{-b_{\text{cls}}}} = p$$

Giải phương trình tìm bias logit $b_{\text{cls}}$:

$$b_{\text{cls}} = -\ln\left(\frac{1 - p}{p}\right) = -\ln\left(\frac{1 - 0.01}{0.01}\right) = -\ln(99) \approx -4.59512$$

---

## 4. Stride-Aware Classification Bias Scaling

Cài đặt nâng cao trong `ScaleHead.init_stride_bias(stride, img_size)` ghi đè prior hằng số bằng prior biến đổi theo mật độ anchor kỳ vọng tại từng cấp độ stride $S \in \{8, 16, 32\}$:

$$b_{\text{cls}}(S) = \ln\left( \frac{5}{N_{\text{cls}} \times \left(\frac{H_{\text{img}}}{S}\right)^2} \right)$$

với:
- $N_{\text{cls}} = 80$: Số lượng lớp đối tượng.
- $H_{\text{img}} = 640$ (hoặc $480$): Kích thước ảnh đầu vào.
- $S \in \{8, 16, 32\}$: Stride của feature map (P3, P4, P5).

Ví dụ với $H_{\text{img}} = 640, N_{\text{cls}} = 80$:
- P3 ($S=8 \implies \text{grid} = 80 \times 80 = 6400$):
  $$b_{\text{cls}}(8) = \ln\left(\frac{5}{80 \times 6400}\right) = \ln\left(\frac{5}{512000}\right) \approx -11.5366$$
- P5 ($S=32 \implies \text{grid} = 20 \times 20 = 400$):
  $$b_{\text{cls}}(32) = \ln\left(\frac{5}{80 \times 400}\right) = \ln\left(\frac{5}{32000}\right) \approx -8.7641$$

Tác dụng: Phản ánh chính xác mật độ đối tượng trên ô lưới (P3 chứa ít đối tượng trên mỗi cell hơn so với P5), triệt tiêu hoàn toàn hiện tượng bùng nổ Loss BCE ở vài iteration đầu tiên.
