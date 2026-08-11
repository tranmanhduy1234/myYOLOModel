# Nền tảng toán học chuyên sâu của dự án CNNModel

## 0. Phạm vi và cách đọc tài liệu

Tài liệu này được xây dựng trực tiếp từ mã nguồn hiện có của dự án, chủ yếu từ các khối sau:

- detector lõi: `src/blocks.py`, `src/backbone_neck.py`, `src/head.py`, `src/model.py`;
- huấn luyện và hàm mục tiêu: `src/train/loss.py`, `src/train/engine.py`, `src/train/ema.py`;
- dữ liệu và đánh giá: `src/train/dataloader1_obj365.py`, `src/evaluation/mAPEvaluation.py`;
- nhánh landmark khuôn mặt hoàn chỉnh: `src/transferLearning1/`;
- nhánh landmark đang được tổ chức lại: `src/transferLearning/`.

Về bản chất, đây không phải một CNN phân loại đơn thuần. Hệ thống là một **dense, anchor-point-based, anchor-box-free detector** theo tinh thần YOLOv10, có:

1. backbone tích chập đa tỉ lệ;
2. PAFPN hợp nhất đặc trưng hai chiều;
3. hai đầu dự đoán one-to-many (O2M) và one-to-one (O2O);
4. hồi quy hộp bằng phân phối rời rạc DFL;
5. gán nhãn động Task-Aligned Assignment;
6. suy luận O2O hướng tới NMS-free;
7. một biến thể transfer learning dự đoán đồng thời hộp mặt và 478 landmark.

Các ký hiệu được dùng xuyên suốt:

| Ký hiệu | Ý nghĩa |
|---|---|
| $B$ | batch size |
| $C$ | số kênh đặc trưng |
| $H,W$ | chiều cao, chiều rộng feature map |
| $N=HW$ | số vị trí không gian trong một feature map |
| $A$ | tổng số anchor point của mọi mức P3/P4/P5 |
| $M$ | số ground-truth tối đa trong một batch sau padding |
| $n_c$ | số lớp |
| $R$ | `reg_max`, mặc định $16$ |
| $K$ | số landmark trên một khuôn mặt, thường $478$ |
| $s_l$ | stride tại pyramid level $l$, trong dự án là $8,16,32$ |

Một nguyên tắc bắt buộc khi đọc code là không được trộn lẫn **định dạng tọa độ** và **không gian tọa độ**:

- `xyxy`: $(x_1,y_1,x_2,y_2)$;
- `cxcywh`: $(c_x,c_y,w,h)$;
- `ltrb`: khoảng cách từ anchor point đến bốn cạnh $(l,t,r,b)$;
- pixel-space: đơn vị pixel của ảnh đầu vào;
- grid-space: đơn vị ô lưới feature map, với $x_{pixel}=s_lx_{grid}$.

---

## 1. Mô hình hóa bài toán

### 1.1 Phát hiện đối tượng như một ánh xạ có cấu trúc

Ảnh đầu vào là tensor

$$
X\in[0,1]^{B\times3\times H_0\times W_0}.
$$

Detector thực hiện ánh xạ tham số hóa bởi $\theta$:

$$
f_\theta:X\mapsto
\left\{Z^{cls},Z^{reg},\hat B\right\}_{O2M,O2O},
$$

trong đó:

$$
Z^{cls}\in\mathbb R^{B\times A\times n_c},\qquad
Z^{reg}\in\mathbb R^{B\times A\times4R},\qquad
\hat B\in\mathbb R^{B\times A\times4}.
$$

Mỗi vị trí trên P3, P4, P5 sinh ra một anchor **point**, không phải một anchor box có kích thước/aspect ratio định trước. Vì vậy kích thước hộp được học thông qua bốn phân phối khoảng cách đến bốn cạnh.

Với ảnh vuông $S\times S$ và strides $(8,16,32)$:

$$
A=\left(\frac S8\right)^2+\left(\frac S{16}\right)^2+\left(\frac S{32}\right)^2.
$$

Ở $S=480$:

$$
A=60^2+30^2+15^2=4725.
$$

Ở $S=640$, giá trị quen thuộc là $A=8400$.

### 1.2 Bài toán landmark

Đầu landmark mở rộng ánh xạ trên thành

$$
Z^{lmk}\in\mathbb R^{B\times A\times K\times2}.
$$

Code không ép landmark nằm trong bounding box dự đoán. Mỗi điểm được biểu diễn bằng một offset có dấu, tính từ anchor point:

$$
\hat{\mathbf p}_{baj}
=s_a\left(\mathbf a_a+\Delta\mathbf p_{baj}\right),
\qquad j=1,\ldots,K,
$$

với $\mathbf a_a$ ở grid-space, $\Delta\mathbf p$ cũng ở grid-space và kết quả $\hat{\mathbf p}$ ở pixel-space. Thiết kế này tách sai số landmark khỏi sai số kích thước hộp dự đoán.

---

## 2. Đại số tensor của tích chập

### 2.1 Tích chập 2D

Với đầu vào $X\in\mathbb R^{C_{in}\times H\times W}$ và kernel
$W\in\mathbb R^{C_{out}\times(C_{in}/g)\times k_h\times k_w}$, grouped convolution được viết:

$$
Y_{c_o,i,j}
=b_{c_o}+\sum_{c_i\in\mathcal G(c_o)}
\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
W_{c_o,c_i,u,v}X_{c_i,is_h+u d_h-p_h,js_w+v d_w-p_w}.
$$

Kích thước đầu ra theo một chiều là

$$
H_{out}=\left\lfloor
\frac{H+2p-d(k-1)-1}{s}+1
\right\rfloor.
$$

Hàm `autopad` chọn

$$
p=\left\lfloor\frac{k_{eff}}2\right\rfloor,
\qquad k_{eff}=d(k-1)+1,
$$

nên với stride $1$ và kernel lẻ, kích thước không gian được bảo toàn.

Số tham số convolution có bias là

$$
P=C_{out}\left(\frac{C_{in}}gk_hk_w+1\right),
$$

và bỏ hạng $C_{out}$ khi `bias=False` như các lớp `Conv` trước BatchNorm.

### 2.2 Depthwise và pointwise convolution

Depthwise convolution đặt $g=C_{in}=C_{out}$, nên chi phí xấp xỉ

$$
\operatorname{MAC}_{DW}=HWCk^2,
$$

thay vì

$$
\operatorname{MAC}_{dense}=HWC_{in}C_{out}k^2.
$$

Một cặp depthwise $k\times k$ và pointwise $1\times1$ có chi phí

$$
HW(C_{in}k^2+C_{in}C_{out}),
$$

giúp nhánh classification trong `ScaleHead` tiết kiệm đáng kể. `DWConv` của dự án dùng $g=\gcd(C_{in},C_{out})$; nó đúng là depthwise khi hai số kênh bằng nhau và trở thành grouped convolution tổng quát khi khác nhau.

### 2.3 Batch Normalization

Với mỗi kênh $c$, trên mini-batch và các vị trí không gian:

$$
\mu_c=\frac1m\sum_{i=1}^m x_{ic},\qquad
\sigma_c^2=\frac1m\sum_{i=1}^m(x_{ic}-\mu_c)^2,
$$

$$
\operatorname{BN}(x_{ic})
=\gamma_c\frac{x_{ic}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}+\beta_c.
$$

Code đặt $\varepsilon=10^{-3}$, momentum PyTorch $m_{BN}=0.03$, do đó running mean được cập nhật theo quy ước của PyTorch:

$$
\mu_{run}^{(t+1)}=(1-m_{BN})\mu_{run}^{(t)}+m_{BN}\mu_{batch}^{(t)}.
$$

Khi freeze trunk ở transfer learning, BatchNorm phải chuyển sang `eval()`. Chỉ đặt `requires_grad=False` là chưa đủ, vì running statistics không phải parameter và vẫn có thể thay đổi trong `train()`.

### 2.4 SiLU/Swish

Activation của dự án là

$$
\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}.
$$

Đạo hàm:

$$
\frac{d}{dx}\operatorname{SiLU}(x)
=\sigma(x)+x\sigma(x)(1-\sigma(x)).
$$

SiLU trơn, không triệt gradient hoàn toàn ở miền âm như ReLU, nhưng không đơn điệu trên toàn trục số. Điều này thường cho tối ưu mượt hơn trong detector sâu.

---

## 3. Các khối biểu diễn đặc trưng

### 3.1 Residual learning và Bottleneck

`Bottleneck` tính

$$
Y=\begin{cases}
X+F(X;\theta),&C_{in}=C_{out}\text{ và bật shortcut},\\
F(X;\theta),&\text{ngược lại}.
\end{cases}
$$

Jacobian của nhánh residual là

$$
\frac{\partial Y}{\partial X}=I+\frac{\partial F}{\partial X}.
$$

Hạng identity tạo đường truyền gradient trực tiếp. Ngay cả khi $\partial F/\partial X$ nhỏ, gradient vẫn có thành phần không bị nhân liên tiếp bởi nhiều ma trận trọng số.

### 3.2 C2f như partial dense aggregation

Đặt phép chiếu đầu vào thành hai phần:

$$
[U_0,V_0]=\operatorname{split}(\phi_{1\times1}(X)).
$$

Sau đó

$$
V_i=B_i(V_{i-1}),\qquad i=1,\ldots,n,
$$

và

$$
Y=\phi_{1\times1}\left(
[U_0,V_0,V_1,\ldots,V_n]
\right).
$$

Cấu trúc này có hai tác dụng: một phần đặc trưng đi theo đường ngắn, phần còn lại được biến đổi sâu; phép concatenate bảo tồn thông tin ở nhiều độ sâu thay vì cộng chúng và có nguy cơ triệt tiêu lẫn nhau.

### 3.3 CIB và C2fCIB

Compact Inverted Block trong code là chuỗi:

$$
DW_{3\times3}	o PW_{1\times1}\to DW_{3\times3}	o PW_{1\times1}\to DW_{3\times3},
$$

kèm residual khi số kênh phù hợp. Hai pointwise convolution chịu trách nhiệm trộn kênh; depthwise convolution trộn không gian với chi phí thấp. `C2fCIB` thay Bottleneck thường trong C2f bằng CIB, đặc biệt ở tầng sâu có nhiều kênh.

### 3.4 SPPF và receptive field

`SPPF` áp dụng MaxPool $5\times5$ ba lần liên tiếp, stride $1$. Receptive field hiệu dụng của $n$ phép pooling nối tiếp là

$$
k_{eff}=1+n(k-1).
$$

Do đó ba đầu ra tương ứng receptive field $5$, $9$, $13$. Kết quả


$$
Y=\phi_{1\times1}([X,M(X),M^2(X),M^3(X)])
$$

tổng hợp ngữ cảnh đa tỉ lệ mà không cần ba kernel lớn độc lập.

### 3.5 SCDown

SCDown thực hiện pointwise convolution đổi số kênh, sau đó depthwise convolution stride $2$:

$$
Y=DW_{3\times3,s=2}(PW_{1\times1}(X)).
$$

So với convolution đặc stride $2$, phép phân rã này giảm FLOPs, trong khi pointwise layer vẫn học phép chiếu giữa các kênh trước khi lấy mẫu xuống.

---

## 4. Partial self-attention ở tầng sâu

### 4.1 Scaled dot-product attention

Với tensor $X\in\mathbb R^{B\times C\times H\times W}$, đặt $N=HW$, $h$ head và $d=C/h$. Phép chiếu $1\times1$ sinh

$$
Q,K,V\in\mathbb R^{B\times h\times N\times d}.
$$

Attention mỗi head:

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right),
\qquad O=AV.
$$

Hệ số $1/\sqrt d$ xuất phát từ giả thiết các phần tử $q_i,k_i$ độc lập, phương sai $1$:

$$
\operatorname{Var}(q^\top k)
=\operatorname{Var}\left(\sum_{i=1}^dq_ik_i\right)\approx d.
$$

Không scale, logit tăng theo $\sqrt d$, softmax dễ bão hòa và gradient nhỏ.

### 4.2 Positional encoding cục bộ

Attention thuần túy không tự biết cấu trúc lân cận 2D. Code bổ sung

$$
O'=O+DWConv_{3\times3}(V),
$$

tức một positional encoding cục bộ có tính tịnh tiến. Nó đưa inductive bias của CNN vào cơ chế tương tác toàn cục.

### 4.3 LayerScale

Hai residual update là

$$
X'=X+\gamma_1\odot\operatorname{Proj}(O'),
$$

$$
Y=X'+\gamma_2\odot\operatorname{FFN}(X'),
$$

với $\gamma_1,\gamma_2$ khởi tạo $10^{-2}$. Ở đầu quá trình học, block gần identity, làm giảm nguy cơ attention ngẫu nhiên phá hỏng đặc trưng CNN đã ổn định.

### 4.4 Độ phức tạp

Ma trận attention có kích thước $N\times N$, nên chi phí chính là

$$
O(BN^2C).
$$

Đặt PSA ở P5, nơi $N$ nhỏ, là lựa chọn toán học có chủ đích. Nếu đặt ở P3 của ảnh $480$, $N=3600$ và attention matrix mỗi head có gần 13 triệu phần tử; ở P5, $N=225$, chỉ khoảng 50 nghìn phần tử.

`C2fPSA` chỉ đưa một nửa số kênh qua attention:

$$
[X_a,X_b]=\operatorname{split}(X),\quad
Y=\phi([X_a,\operatorname{Attn}(X_b)]),
$$

giảm thêm chi phí trong khi giữ nhánh convolutional trực tiếp.

---

## 5. Backbone và PAFPN đa tỉ lệ

### 5.1 Pyramid feature

Backbone tạo:

$$
P_3\in\mathbb R^{B\times C_3\times H_0/8\times W_0/8},
$$

$$
P_4\in\mathbb R^{B\times C_4\times H_0/16\times W_0/16},
\qquad
P_5\in\mathbb R^{B\times C_5\times H_0/32\times W_0/32}.
$$

### 5.2 Top-down path

PAFPN lan truyền ngữ nghĩa tầng sâu xuống:

$$
P_4^{td}=F_4([\operatorname{Up}(P_5),P_4]),
$$

$$
P_3^{out}=F_3([\operatorname{Up}(P_4^{td}),P_3]).
$$

Nearest-neighbor upsampling không tạo tham số mới. Concatenation giữ riêng từng nguồn trước khi block C2f học cách trộn chúng.

### 5.3 Bottom-up path

Đường bottom-up đưa thông tin định vị trở lại tầng sâu:

$$
P_4^{out}=G_4([D(P_3^{out}),P_4^{td}]),
$$

$$
P_5^{out}=G_5([D(P_4^{out}),P_5]).
$$

FPN thuần chủ yếu truyền ngữ nghĩa xuống; PAN bổ sung đường định vị đi lên. Vì gradient từ head chảy qua cả hai hướng, mỗi scale nhận supervision gián tiếp từ các scale khác.

---

## 6. Detection head và anchor point

### 6.1 Decoupled head

Mỗi scale có hai stem độc lập:

- classification stem học tính bất biến nội lớp;
- regression stem giữ thông tin hình học chính xác.

Tách hai stem giảm xung đột gradient giữa hai mục tiêu. Nếu $g_{cls}$ và $g_{box}$ là gradient trên cùng tham số, xung đột xảy ra khi

$$
g_{cls}^{\top}g_{box}<0.
$$

Decoupling không loại bỏ hoàn toàn xung đột tại trunk chung, nhưng tránh xung đột ở các layer cuối chuyên biệt.

### 6.2 Sinh anchor point

Tại ô $(i,j)$, anchor grid là

$$
\mathbf a_{ij}=(j+0.5,i+0.5).
$$

Tọa độ pixel:

$$
\mathbf a_{ij}^{px}=s_l\mathbf a_{ij}.
$$

Offset $0.5$ đặt anchor tại tâm cell. Tất cả anchor từ ba scale được flatten và concatenate theo thứ tự P3, P4, P5.

### 6.3 Dự đoán phân phối khoảng cách

Với mỗi cạnh $e\in\{l,t,r,b\}$, head sinh logits

$$
\mathbf z_e=(z_{e0},\ldots,z_{e,R-1}).
$$

Xác suất bin:

$$
p_{er}=\frac{e^{z_{er}}}{\sum_{q=0}^{R-1}e^{z_{eq}}}.
$$

DFL layer cố định vector trọng số $(0,1,\ldots,R-1)$ và lấy kỳ vọng:

$$
\hat d_e=\mathbb E[r]=\sum_{r=0}^{R-1}r p_{er}.
$$

Do đó $\hat d_e\in[0,R-1]$ theo đơn vị grid. Box decode:

$$
\hat B^{grid}
=(a_x-\hat l,a_y-\hat t,a_x+\hat r,a_y+\hat b),
$$

$$
\hat B^{px}=s_l\hat B^{grid}.
$$

DFL biểu diễn được bất định đa mode tốt hơn hồi quy một scalar. Ví dụ kỳ vọng $3.7$ có thể đến từ phân phối tập trung ở bin $3$ và $4$, thay vì yêu cầu một neuron tuyến tính phát trực tiếp $3.7$.

### 6.4 Giới hạn biểu diễn và đa tỉ lệ

Khoảng cách cực đại một cạnh tại scale $l$ xấp xỉ

$$
d_{max}^{px}=(R-1)s_l.
$$

Với $R=16$, các giá trị là $120$, $240$, $480$ pixel cho strides $8,16,32$. Chính pyramid đa tỉ lệ giúp cùng một số bin bao phủ nhiều kích cỡ đối tượng.

---

## 7. Dual assignment: O2M và O2O

### 7.1 Vì sao cần hai nhánh

Nhánh O2M dùng nhiều positive anchor cho mỗi GT, cung cấp gradient dày và học ổn định. Nhánh O2O chỉ chọn một positive anchor cho mỗi GT, ép mô hình học một ánh xạ gần một-một giữa vật thể và dự đoán.

Nếu chỉ có O2M, nhiều dự đoán tốt có thể cùng biểu diễn một vật thể và cần NMS. Nếu chỉ có O2O từ đầu, supervision quá thưa và tối ưu khó hơn. Dự án tối ưu đồng thời:

$$
\mathcal L=\omega_m\mathcal L_{O2M}+\omega_o\mathcal L_{O2O}.
$$

Hai head không dùng chung layer cuối, nhưng dùng chung backbone/neck nên O2M đóng vai trò auxiliary dense supervision cho biểu diễn chung.

### 7.2 Task-Aligned metric

Với GT $i$, anchor $a$, class thật $c_i$, đặt

$$
s_{ia}=\sigma(z^{cls}_{a,c_i}),
$$

$$
u_{ia}=\max(\operatorname{CIoU}(B_i,\hat B_a),0).
$$

Code clamp CIoU về không âm trước khi lũy thừa. Alignment metric là

$$
m_{ia}=s_{ia}^{\alpha}u_{ia}^{\beta},
$$

với mặc định $\alpha=0.5,\beta=6$. Vì $\beta$ lớn, localization là điều kiện rất mạnh; classification chỉ có thể nâng hạng các box đã đủ đúng về hình học.

Trong log-domain:

$$
\log m_{ia}=\alpha\log s_{ia}+\beta\log u_{ia}.
$$

Biểu diễn này cho thấy $\alpha,\beta$ chính là hệ số tuyến tính điều khiển độ nhạy tương đối trong không gian log.

### 7.3 Miền ứng viên

Anchor chỉ là ứng viên nếu tâm của nó nằm nghiêm ngặt trong GT:

$$
a_x-x_1>\epsilon,\quad a_y-y_1>\epsilon,
\quad x_2-a_x>\epsilon,\quad y_2-a_y>\epsilon.
$$

Sau đó chọn top-$k$ theo $m_{ia}$:

- O2M: $k=10$;
- O2O: $k=1$.

Nếu một anchor được nhiều GT chọn, code giữ GT có overlap lớn nhất:

$$
i^*(a)=\arg\max_i u_{ia}.
$$

Do đó mỗi anchor có tối đa một target, dù một GT có thể sở hữu nhiều anchor trong O2M.

### 7.4 Soft classification target

Với tập positive $P_i$ của GT $i$:

$$
m_i^{max}=\max_{a\in P_i}m_{ia},\qquad
u_i^{max}=\max_{a\in P_i}u_{ia}.
$$

Chất lượng chuẩn hóa của anchor là

$$
q_{ia}=\frac{m_{ia}u_i^{max}}{m_i^{max}+\epsilon}.
$$

Target classification:

$$
\mathbf t_a=q_{ia}\operatorname{onehot}(c_i)
$$

cho positive, và vector $0$ cho background. Đây là **quality-aware target**: độ tin cậy classification được buộc phản ánh cả khả năng phân lớp và định vị.

### 7.5 Điều kiện NMS-free

O2O làm giảm duplicate bằng supervision, không phải bằng một định lý bảo đảm tuyệt đối. Suy luận NMS-free hợp lý khi:

1. mỗi GT chỉ tạo một positive;
2. head O2O được dùng lúc inference;
3. score được hiệu chỉnh đủ tốt để duplicate còn lại nằm dưới ngưỡng hoặc ngoài top-$k$.

Trong nhánh landmark inference, code chỉ lấy O2O khi `eval()` và `return_o2m=False`. Ở head detector lõi hiện tại, đường `return None, out_o2o` nằm sau một `return` không thể tới, nên implementation vẫn tính O2M trong forward; ý tưởng toán học vẫn là dùng O2O cho suy luận nhưng tối ưu tính toán này chưa thực sự có hiệu lực trong file `src/head.py`.

---

## 8. Hình học IoU, GIoU, DIoU và CIoU

### 8.1 IoU

Với hai box $B,B^*$:

$$
\operatorname{IoU}(B,B^*)=
\frac{|B\cap B^*|}{|B\cup B^*|}.
$$

IoU bất biến với phép scale đồng nhất. Nếu nhân mọi tọa độ với $s$ thì cả giao và hợp nhân $s^2$, tỷ số không đổi. Vì vậy CIoU có thể tính trong grid-space hay pixel-space nếu hai box cùng hệ tọa độ.

Nhược điểm: khi hai box không giao nhau, IoU bằng $0$ và không mang thông tin về khoảng cách.

### 8.2 GIoU

Gọi $C$ là box nhỏ nhất bao cả hai box:

$$
\operatorname{GIoU}
=\operatorname{IoU}-\frac{|C\setminus(B\cup B^*)|}{|C|}.
$$

GIoU tạo gradient cả khi không overlap, nhưng không trực tiếp tối ưu khoảng cách tâm.

### 8.3 DIoU

Gọi $\rho^2$ là bình phương khoảng cách tâm và $c^2$ là bình phương đường chéo $C$:

$$
\operatorname{DIoU}=\operatorname{IoU}-\frac{\rho^2(\mathbf b,\mathbf b^*)}{c^2}.
$$

### 8.4 CIoU

Độ lệch aspect ratio:

$$
v=\frac4{\pi^2}
\left[
\arctan\left(\frac{w^*}{h^*}\right)
-\arctan\left(\frac wh\right)
\right]^2,
$$

$$
\eta=\frac{v}{1-\operatorname{IoU}+v+\epsilon}.
$$

Khi đó

$$
\operatorname{CIoU}
=\operatorname{IoU}
-\frac{\rho^2}{c^2}
-\eta v.
$$

Loss hình học là

$$
\ell_{CIoU}=1-\operatorname{CIoU}.
$$

Ba thành phần lần lượt ép overlap, tâm và aspect ratio. Trong code, $\eta$ được tính dưới `no_grad`; hệ số cân bằng không được backpropagate, nhưng $v$ vẫn có gradient.

### 8.5 Weighted CIoU loss

Đặt quality weight của positive anchor

$$
w_a=\sum_{c=1}^{n_c}t_{ac}=q_a,
\qquad Q=\max\left(\sum_aw_a,1\right).
$$

Loss box là

$$
\mathcal L_{iou}=
\frac1Q\sum_{a\in FG}w_a(1-\operatorname{CIoU}_a).
$$

Anchor có assignment quality thấp đóng góp ít hơn. Điều này liên kết assignment, classification và regression thành một hệ mục tiêu nhất quán.

---

## 9. Distribution Focal Loss

### 9.1 Encode target

Với anchor $\mathbf a=(a_x,a_y)$ và GT box $B^*=(x_1,y_1,x_2,y_2)$ trong grid-space:

$$
\mathbf d^*=(a_x-x_1,a_y-y_1,x_2-a_x,y_2-a_y).
$$

Mỗi phần tử được clamp vào

$$
[0,R-1-0.01].
$$

Clamping bảo đảm bin phải $r+1$ luôn nằm trong miền $0,\ldots,R-1$.

### 9.2 Soft label hai bin

Với target thực $y$:

$$
y_l=\lfloor y\rfloor,\qquad y_r=y_l+1,
$$

$$
w_l=y_r-y,\qquad w_r=y-y_l,
\qquad w_l+w_r=1.
$$

DFL một cạnh:

$$
\ell_{DFL}(\mathbf z,y)
=w_l\operatorname{CE}(\mathbf z,y_l)
+w_r\operatorname{CE}(\mathbf z,y_r).
$$

Nhãn này có kỳ vọng đúng bằng $y$:

$$
w_ly_l+w_ry_r=y.
$$

Đây là lý do nội suy hai bin không chỉ là heuristic mà là một mã hóa phân phối bảo toàn giá trị mục tiêu.

### 9.3 Loss tổng

Trung bình bốn cạnh rồi weighting:

$$
\mathcal L_{dfl}
=\frac1Q\sum_{a\in FG}w_a
\frac14\sum_{e\in\{l,t,r,b\}}
\ell_{DFL}(\mathbf z_{ae},d^*_{ae}).
$$

Lưu ý quan trọng: logits không “có đơn vị”, nhưng index bin mang nghĩa khoảng cách theo **grid cell**. Đưa target pixel trực tiếp vào `bbox2dist` sẽ làm target bị clamp sai nghiêm trọng.

---

## 10. Classification loss

Với logit $z$ và soft target $t\in[0,1]$:

$$
\operatorname{BCEWithLogits}(z,t)
=\max(z,0)-zt+\log(1+e^{-|z|}).
$$

Dạng trên ổn định số hơn công thức trực tiếp

$$
-t\log\sigma(z)-(1-t)\log(1-\sigma(z)).
$$

Loss classification trên toàn bộ anchor và class:

$$
\mathcal L_{cls}
=\frac1Q\sum_{a=1}^A\sum_{c=1}^{n_c}
\operatorname{BCEWithLogits}(z_{ac},t_{ac}).
$$

Background không bị bỏ qua; nó có target $0$ và đóng góp vào BCE. Vì $A$ lớn trong khi positive thưa, bias prior âm ở head là thiết yếu để loss ban đầu không bị background áp đảo.

Đạo hàm theo logit rất gọn:

$$
\frac{\partial\ell}{\partial z}=\sigma(z)-t.
$$

Soft target làm gradient phản ánh quality: một positive kém có $t$ nhỏ, nên mô hình không bị ép cho score gần $1$ dù box chưa tốt.

---

## 11. Hàm mục tiêu detection hoàn chỉnh

Với nhánh $r\in\{m,o\}$:

$$
\mathcal L_r
=\lambda_{box}\mathcal L_{iou}^{(r)}
+\lambda_{cls}\mathcal L_{cls}^{(r)}
+\lambda_{dfl}\mathcal L_{dfl}^{(r)}.
$$

Mặc định trong `TrainConfig`:

$$
\lambda_{box}=7.5,\qquad
\lambda_{cls}=0.5,\qquad
\lambda_{dfl}=1.5.
$$

Loss cuối:

$$
\boxed{
\mathcal L_{det}
=\omega_m\mathcal L_m+\omega_o\mathcal L_o
}
$$

với $\omega_m=\omega_o=1$ mặc định.

Các hệ số không thể so sánh chỉ bằng trị số, vì ba loss có scale và số hạng khác nhau. Cách đánh giá đúng là quan sát **gradient norm** hoặc đóng góp đã nhân gain, không chỉ loss thô.

---

## 12. Pipeline tọa độ: bất biến và chỗ dễ sai

### 12.1 Letterbox

Với ảnh gốc $W\times H$, ảnh đích $S\times S$:

$$
r=\min\left(\frac SW,\frac SH\right),
$$

$$
W'=\operatorname{round}(rW),\qquad
H'=\operatorname{round}(rH),
$$

$$
p_x=\left\lfloor\frac{S-W'}2\right\rfloor,
\qquad
p_y=\left\lfloor\frac{S-H'}2\right\rfloor.
$$

Điểm và box biến đổi affine:

$$
x'=rx+p_x,\qquad y'=ry+p_y.
$$

Inverse ở inference:

$$
x=\frac{x'-p_x}{r},\qquad
y=\frac{y'-p_y}{r}.
$$

Letterbox bảo toàn aspect ratio, khác resize độc lập theo hai trục. Padding phải được cộng cho cả hai góc box và mọi landmark.

### 12.2 Pixel-space trong assigner

Các đại lượng sau cùng ở pixel-space:

$$
\hat B^{px},\quad B^{*px},\quad
\mathbf a^{px}=s\mathbf a^{grid}.
$$

TAL so sánh trực tiếp chúng. Dù IoU không đổi theo scale, điều kiện “anchor nằm trong GT” dùng khoảng cách có đơn vị nên bắt buộc cùng hệ tọa độ.

### 12.3 Grid-space trong DFL

Trước `BboxLoss`, code chia box theo stride từng anchor:

$$
\hat B^{grid}_a=\frac{\hat B^{px}_a}{s_a},\qquad
B^{*grid}_a=\frac{B^{*px}_a}{s_a}.
$$

Sau đó `bbox2dist(anchor_grid, target_box_grid)` mới cho target đúng với các bin DFL.

### 12.4 Bảng kiểm tra đơn vị

| Tensor | Shape | Space | Format |
|---|---:|---|---|
| `anchors` | $A\times2$ | grid | $xy$ |
| `strides` | $A\times1$ | pixel/grid | scalar |
| `anchors * strides` | $A\times2$ | pixel | $xy$ |
| `reg_raw` | $B\times4R\times A$ | bin-grid | logits |
| `box` từ head | $B\times A\times4$ | pixel | `xyxy` |
| GT từ dataloader | $N\times4$ | pixel sau letterbox | `xyxy` |
| DFL target | $B\times A\times4$ | grid | `ltrb` |

Một invariant hữu ích để debug:

$$
\operatorname{decode}(\operatorname{bbox2dist}(a,B),a)=B
$$

trước khi có clamping và sai số số học.

---

## 13. Landmark 478: biểu diễn và loss

### 13.1 Decode anchor-relative

Với raw output $L\in\mathbb R^{B\times2K\times A}$, sau transpose và reshape:

$$
\Delta P\in\mathbb R^{B\times A\times K\times2}.
$$

Decode:

$$
\hat P_{baj}^{px}
=s_a(A_a^{grid}+\Delta P_{baj}^{grid}).
$$

Offset không qua sigmoid hay tanh, do đó có thể âm và không bị chặn trong cell hoặc bbox. Điều này cần thiết vì một anchor positive có thể nằm gần một cạnh khuôn mặt, còn landmark nằm về phía đối diện.

### 13.2 Tại sao không bbox-relative

Một mã hóa bbox-relative thường có dạng

$$
\hat x=x_1+\sigma(u)w,\qquad
\hat y=y_1+\sigma(v)h.
$$

Khi đó gradient landmark phụ thuộc trực tiếp vào box dự đoán:

$$
\frac{\partial\hat x}{\partial x_1}=1,
\qquad
\frac{\partial\hat x}{\partial w}=\sigma(u).
$$

Sai số box sẽ kéo theo toàn bộ landmark. Thiết kế anchor-relative hiện tại loại liên kết decode này. Bbox GT chỉ được dùng làm thước đo scale trong loss, không tham gia tọa độ dự đoán.

### 13.3 Scale normalization

Với GT face box có chiều rộng $w_i$ và chiều cao $h_i$, code dùng

$$
s_i^{face}=\sqrt{\max(w_i,\epsilon)\max(h_i,\epsilon)}.
$$

Sai số chuẩn hóa:

$$
\mathbf e_{iaj}=
\frac{\hat{\mathbf p}_{iaj}-\mathbf p^*_{ij}}
{s_i^{face}}.
$$

$\sqrt{wh}$ là căn bậc hai diện tích, có tính đồng bậc một theo phép scale ảnh. Nếu toàn bộ ảnh và annotation nhân $c$:

$$
\frac{c(\hat p-p^*)}{\sqrt{(cw)(ch)}}
=\frac{\hat p-p^*}{\sqrt{wh}},
$$

nên loss landmark bất biến scale.

### 13.4 Smooth L1/Huber dạng beta

Với một tọa độ lỗi $e$:

$$
\rho_\beta(e)=
\begin{cases}
\dfrac{e^2}{2\beta},&|e|<\beta,\\[4pt]
|e|-\dfrac\beta2,&|e|\ge\beta.
\end{cases}
$$

Đạo hàm:

$$
\rho_\beta'(e)=
\begin{cases}
e/\beta,&|e|<\beta,\\
\operatorname{sign}(e),&|e|\ge\beta.
\end{cases}
$$

Gần $0$, loss bậc hai giúp hội tụ chính xác; xa $0$, gradient bị chặn ở độ lớn $1$, bền vững hơn MSE trước annotation nhiễu hoặc augmentation mạnh.

### 13.5 Trọng số vùng giải phẫu

Mỗi landmark $j$ có trọng số $v_j$. Mặc định:

- điểm thường: $1$;
- mắt và iris: $3$;
- môi: $3$;
- chóp mũi: $4$.

Với quality assignment $q_{ia}$, loss landmark là

$$
\mathcal L_{lmk}
=\frac{
\sum_{(i,a)\in\mathcal V}
q_{ia}\sum_{j=1}^Kv_j
\sum_{d\in\{x,y\}}\rho_\beta(e_{iajd})
}{
2\left(\sum_{(i,a)\in\mathcal V}q_{ia}\right)
\left(\sum_{j=1}^Kv_j\right)+\epsilon
}.
$$

$\mathcal V$ chỉ gồm positive anchor được gán cho GT có annotation landmark hợp lệ. Cùng quality weight với box loss làm anchor kém tin cậy đóng góp ít hơn.

### 13.6 Loss face-landmark hoàn chỉnh

Mỗi nhánh:

$$
\mathcal L_r^{face}
=\lambda_{box}\mathcal L_{iou}^{(r)}
+\lambda_{cls}\mathcal L_{cls}^{(r)}
+\lambda_{dfl}\mathcal L_{dfl}^{(r)}
+\lambda_{lmk}\mathcal L_{lmk}^{(r)}.
$$

Mặc định nhánh hoàn chỉnh trong `transferLearning1` dùng

$$
(\lambda_{box},\lambda_{cls},\lambda_{dfl},\lambda_{lmk})
=(7.5,0.5,1.5,2.0).
$$

Tổng hai nhánh vẫn là

$$
\mathcal L_{face}=\omega_m\mathcal L_m^{face}
+\omega_o\mathcal L_o^{face}.
$$

### 13.7 Tối ưu bộ nhớ khi $K=478$

Không tạo target đầy đủ $B\times A\times K\times2$. Code chỉ gather các hàng positive hợp lệ. Nếu $B=4,A=4725,K=478$, tensor đầy đủ float32 cần riêng:

$$
4\cdot4725\cdot478\cdot2\cdot4\approx72.3\text{ MB},
$$

chưa kể gradient và tensor trung gian. Gather theo $N_{pos}\ll BA$ giảm bộ nhớ theo tỷ lệ xấp xỉ $N_{pos}/(BA)$.

---

## 14. Hình học augmentation cho landmark

### 14.1 Horizontal flip có ngữ nghĩa

Lật hình học quanh trục dọc ảnh kích thước $S$:

$$
x'=S-x.
$$

Box `xyxy`:

$$
x_1'=S-x_2,\qquad x_2'=S-x_1.
$$

Với landmark MediaPipe, chỉ lật tọa độ là chưa đủ: chỉ số “mắt trái” sau lật phải đổi thành “mắt phải”. Do đó

$$
P'=F(P)_\pi,
$$

với $\pi$ là permutation 478 phần tử. Đây là phép biến đổi vừa hình học vừa ngữ nghĩa.

### 14.2 Affine transform bằng tọa độ thuần nhất

Điểm $\tilde p=(x,y,1)^\top$. Ma trận affine tổng hợp trong code:

$$
H=T_{back}RShS T_{origin},
$$

trong đó $S$ là scale bất đẳng hướng, $Sh$ là shear, $R$ là rotation và $T$ là translation.

$$
\tilde p'=H\tilde p.
$$

Thứ tự nhân ma trận quan trọng vì nhìn chung

$$
AB\ne BA.
$$

### 14.3 Projective transform

Với homography tổng quát

$$
H=\begin{bmatrix}
h_{11}&h_{12}&h_{13}\\
h_{21}&h_{22}&h_{23}\\
h_{31}&h_{32}&h_{33}
\end{bmatrix},
$$

tọa độ biến đổi:

$$
x'=\frac{h_{11}x+h_{12}y+h_{13}}
{h_{31}x+h_{32}y+h_{33}},
$$

$$
y'=\frac{h_{21}x+h_{22}y+h_{23}}
{h_{31}x+h_{32}y+h_{33}}.
$$

Mẫu số được chặn tránh gần $0$. Box mới không thể chỉ biến đổi hai góc trong mọi trường hợp; code lấy mẫu chu vi rồi lấy min/max, đặc biệt cần thiết trước/sau distortion phi tuyến.

### 14.4 Radial distortion

Đưa pixel về normalized camera coordinates:

$$
x_n=\frac{x-c_x}{f_x},\qquad y_n=\frac{y-c_y}{f_y},
\qquad r^2=x_n^2+y_n^2.
$$

Với chỉ hệ số radial $k_1,k_2$:

$$
g(r)=1+k_1r^2+k_2r^4,
$$

$$
x_d=x_ng(r),\qquad y_d=y_ng(r).
$$

Trở về pixel:

$$
u_d=f_xx_d+c_x,\qquad v_d=f_yy_d+c_y.
$$

Ảnh được dựng bằng inverse mapping để tránh lỗ pixel, còn landmark được forward-project. Biên box vốn thẳng có thể thành cong, nên code lấy 9 mẫu mỗi cạnh thay vì chỉ bốn góc.

### 14.5 Visibility filtering

Sau biến đổi, box được clip vào ảnh. Tỷ lệ nhìn thấy:

$$
r_{vis}=\frac{A_{clipped}}{\max(A_{unclipped},\epsilon)}.
$$

Sample được giữ nếu box hữu hạn, đủ kích thước và $r_{vis}$ vượt ngưỡng. Landmark validity yêu cầu tất cả điểm trong ảnh, hoặc tỷ lệ điểm trong ảnh vượt `min_face_visibility`, tùy cấu hình.

---

## 15. Tối ưu hóa

### 15.1 SGD với momentum và Nesterov

Dạng cơ bản:

$$
v_t=\mu v_{t-1}+g_t,
\qquad
\theta_{t+1}=\theta_t-\eta v_t.
$$

Nesterov đánh giá hướng gradient tại điểm nhìn trước; trong cách diễn giải phổ biến:

$$
g_t=\nabla\mathcal L(\theta_t-\eta\mu v_{t-1}).
$$

Momentum lọc nhiễu stochastic theo thời gian và tăng tốc theo các hướng gradient nhất quán.

### 15.2 AdamW

Moment bậc một và hai:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

Bias correction:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

AdamW decouple weight decay:

$$
\theta_{t+1}
=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Khác với cộng $\lambda\theta$ vào gradient của Adam, decay tách rời không bị rescale bởi $1/\sqrt{\hat v_t}$.

Trong detector chính, bias và parameter một chiều (đặc biệt BN $\gamma,\beta$) nằm ở nhóm `weight_decay=0`. Các kernel nhiều chiều mới bị decay.

### 15.3 Warmup và cosine decay

Với $T_w$ warmup step:

$$
\eta_t=\eta_0\frac{t}{T_w},\qquad t<T_w.
$$

Sau warmup, đặt tiến độ

$$
p=\frac{t-T_w}{T-T_w}\in[0,1],
$$

$$
\eta_t=\eta_0\left[
\eta_{min}^{factor}
+\left(1-\eta_{min}^{factor}\right)
\frac{1+\cos(\pi p)}2
\right].
$$

Warmup hạn chế update lớn khi BatchNorm statistics, DFL distribution và classification logits chưa ổn định. Cosine decay có đạo hàm bằng $0$ ở cuối, giúp hạ learning rate mượt.

### 15.4 Gradient clipping

Với gradient toàn cục $g$ và ngưỡng $c$:

$$
g' = g\min\left(1,\frac c{\|g\|_2+\epsilon}\right).
$$

Clipping giữ hướng gradient và chỉ co độ lớn. Nó là cơ chế bảo vệ trước batch bất thường, không sửa được nguyên nhân gốc như label lỗi hoặc learning rate quá cao.

### 15.5 Automatic Mixed Precision

Loss scaling dùng hệ số $S$:

$$
\tilde{\mathcal L}=S\mathcal L,
\qquad
\nabla\tilde{\mathcal L}=S\nabla\mathcal L.
$$

Trước optimizer step, gradient được chia lại cho $S$. Nếu phát hiện Inf/NaN, scaler bỏ step và giảm $S$. Code chỉ cập nhật scheduler/EMA khi optimizer step thành công, tránh làm “thời gian học” tiến lên trong khi parameter không đổi.

### 15.6 Exponential Moving Average

EMA parameter:

$$
\bar\theta_t=d_t\bar\theta_{t-1}+(1-d_t)\theta_t,
$$

với warmup decay

$$
d_t=d_{max}\left(1-e^{-t/T_{ema}}\right).
$$

Khi $d$ cố định, trọng số của parameter cách hiện tại $k$ bước là $(1-d)d^k$. Effective window xấp xỉ

$$
N_{eff}\approx\frac1{1-d}.
$$

Với $d=0.9998$, $N_{eff}\approx5000$ update. EMA giảm variance của nghiệm SGD và thường tổng quát hóa tốt hơn checkpoint tức thời.

---

## 16. Transfer learning hai giai đoạn

### 16.1 Stage 1: head-only

Backbone và neck được freeze:

$$
\nabla_{\theta_{trunk}}\mathcal L=0,
\qquad
\theta_{trunk}^{t+1}=\theta_{trunk}^{t}.
$$

Head landmark mới học ánh xạ từ không gian đặc trưng pretrained sang detection/landmark. Learning rate head cao hơn vì tham số mới chưa được tối ưu.

### 16.2 Stage 2: discriminative fine-tuning

Toàn model được mở, nhưng

$$
0<\eta_{trunk}<\eta_{head}.
$$

Đây là regularization theo không gian tham số: trunk chỉ được di chuyển chậm khỏi nghiệm pretrained, trong khi head tiếp tục thích nghi nhanh.

### 16.3 Góc nhìn Bayesian

Có thể xem pretrained weights $\theta_0$ là một prior. Fine-tuning với decay hoặc learning rate nhỏ ngầm ưu tiên nghiệm gần $\theta_0$:

$$
\theta^*=\arg\min_\theta
\mathcal L_{target}(\theta)+\lambda\|\theta-\theta_0\|^2.
$$

Freeze hoàn toàn tương ứng prior có độ chính xác vô hạn trên trunk; unfreeze với LR nhỏ là prior mềm hơn.

---

## 17. Khởi tạo tham số

### 17.1 Kaiming normal

Với convolution, code dùng

$$
W_{ijkl}\sim\mathcal N\left(0,\frac{2}{fan_{out}}\right),
$$

$$
fan_{out}=C_{out}k_hk_w.
$$

`nonlinearity='relu'` là xấp xỉ cho SiLU vì PyTorch không có gain SiLU riêng trong API này. Chọn `fan_out` ưu tiên ổn định gradient backward.

### 17.2 Bias phân lớp theo mật độ đối tượng

Từ prior xác suất $p$, logit bias là

$$
b=\log\frac p{1-p}.
$$

Khởi tạo sơ bộ $p=0.01$ cho $b\approx-4.595$. Sau đó code ghi đè bằng stride-aware bias:

$$
b_l=\log\left[
\frac{5}{n_c(S/s_l)^2}
\right].
$$

Ý nghĩa: giả sử khoảng 5 object trên một ảnh và phân bố đều theo class/grid, prior ở cell độ phân giải cao phải nhỏ hơn.

Một chi tiết implementation cần lưu ý: detector chính train ảnh `480`, nhưng `DetectHead.init_stride_bias` mặc định dùng `img_size=640` và constructor không truyền `cfg.img_size`. Vì vậy bias hiện tại của detector chính được tính theo $640$, không phải $480$. Nhánh face-landmark truyền rõ `img_size=480` nên nhất quán.

### 17.3 Regression và landmark bias

Regression logits có bias $1$, ban đầu cho phân phối gần đều vì mọi bin được cộng cùng hằng số; softmax bất biến với dịch chuyển đồng nhất:

$$
\operatorname{softmax}(z+c\mathbf1)=\operatorname{softmax}(z).
$$

Landmark output có weight $\mathcal N(0,10^{-6})$ và bias $0$, nên offset ban đầu gần $0$ và landmark ban đầu gần anchor point. Điều này tốt hơn offset ngẫu nhiên lớn.

DFL convolution là toán tử kỳ vọng cố định với weight $(0,1,\ldots,R-1)$ và `requires_grad=False`; hàm initialize phải bỏ qua nó.

---

## 18. Đánh giá: Precision, Recall và COCO-style mAP

### 18.1 Matching

Prediction được sắp giảm dần theo score. Tại ngưỡng IoU $\tau$, một prediction là TP nếu:

1. class đúng;
2. IoU với một GT chưa được match đạt $\tau$;
3. mỗi GT chỉ được match một lần.

Nếu không, prediction là FP. GT không được match là FN.

### 18.2 Precision và Recall

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN}.
$$

Precision đo độ “sạch” của prediction; recall đo độ phủ GT. Thay score threshold tạo một đường precision-recall.

### 18.3 Precision envelope

Code dựng precision đơn điệu:

$$
\tilde P(r)=\max_{r'\ge r}P(r').
$$

AP 101 điểm:

$$
AP_\tau=\frac1{101}\sum_{k=0}^{100}
\tilde P\left(\frac{k}{100}\right).
$$

### 18.4 mAP

Với tập class có GT là $\mathcal C$:

$$
mAP_{50}=\frac1{|\mathcal C|}\sum_{c\in\mathcal C}AP_{c,0.50},
$$

$$
mAP_{50:95}
=\frac1{10|\mathcal C|}
\sum_{c\in\mathcal C}
\sum_{\tau\in\{0.50,0.55,\ldots,0.95\}}
AP_{c,\tau}.
$$

$mAP_{50:95}$ nhạy với độ chính xác localization hơn $mAP_{50}$. Khoảng cách lớn giữa hai chỉ số thường có nghĩa detector tìm đúng vật thể nhưng box chưa khít.

### 18.5 Đặc điểm evaluator trong dự án

Evaluator dùng nhánh O2O, lấy class có sigmoid score lớn nhất tại mỗi anchor, lọc score và giữ tối đa `max_det`. Nó không gọi NMS, phù hợp mục tiêu O2O. Tuy nhiên “không NMS” làm metric rất nhạy với duplicate còn sót; duplicate thứ hai của cùng GT được tính FP.

---

## 19. Dòng gradient và tương tác đa nhiệm

Backbone/neck nhận tổng gradient:

$$
g_{shared}
=\omega_m g_m+\omega_o g_o
+\lambda_{lmk}g_{lmk}\quad\text{(ở mô hình face)}.
$$

Độ lớn tổng không nói hết tương tác. Cosine similarity giữa hai task:

$$
\cos(g_i,g_j)=
\frac{g_i^\top g_j}{\|g_i\|\|g_j\|}.
$$

- dương: hai mục tiêu hỗ trợ nhau;
- gần $0$: tương đối độc lập;
- âm: xung đột gradient.

Đây là cơ sở toán học để điều chỉnh `o2m_weight`, `o2o_weight`, `lmk_gain`, thay vì chỉ nhìn loss scalar. Một loss có trị số nhỏ vẫn có thể tạo gradient lớn nếu độ dốc lớn.

Quality weighting cũng làm các task liên kết. Cùng $q_a$ xuất hiện trong classification target, CIoU/DFL weight và landmark weight. Vì vậy assigner là một phần của hàm mục tiêu, dù chạy trong `no_grad` và không khả vi.

Toàn pipeline là tối ưu luân phiên kiểu “hard latent assignment”:

1. với $\theta_t$, tính assignment $Z_t=\mathcal A(f_{\theta_t}(X),Y)$;
2. coi $Z_t$ là hằng số;
3. cập nhật $\theta$ để giảm $\mathcal L(\theta;Z_t)$.

Assignment thay đổi qua các step khi prediction tốt dần, tương tự một generalized EM nhưng không tối ưu likelihood tường minh.

---

## 20. Các invariant và sanity check toán học

### 20.1 Shape invariant

Với ảnh $S$ chia hết cho $32$:

$$
A=(S/8)^2+(S/16)^2+(S/32)^2.
$$

Các tensor phải thỏa:

$$
Z^{cls}.shape=(B,A,n_c),
$$

$$
Z^{reg}.shape=(B,4R,A),
$$

$$
\hat B.shape=(B,A,4).
$$

### 20.2 Xác suất DFL

Với mỗi cạnh:

$$
\sum_{r=0}^{R-1}p_r=1,
\qquad 0\le\mathbb E[r]\le R-1.
$$

Nếu mọi logit bằng nhau:

$$
p_r=1/R,\qquad
\mathbb E[r]=\frac{R-1}{2}.
$$

### 20.3 Scale equivariance của box decode

Nếu đổi stride từ $s$ thành $cs$ và giữ grid prediction:

$$
\hat B^{px}_{new}=c\hat B^{px}_{old}.
$$

Đây là tính đồng biến scale cần có giữa các pyramid level.

### 20.4 Landmark zero-offset

Nếu $\Delta p=0$:

$$
\hat p=s a,
$$

tức mọi landmark tại anchor pixel. Đây là hành vi khởi tạo mong đợi và kiểm tra được trực tiếp.

### 20.5 Empty-target batch

Khi không có positive:

- box và DFL loss phải là zero tensor còn nối computation graph;
- classification vẫn học background;
- mẫu số được clamp tối thiểu $1$;
- không được tạo NaN do chia $0$.

### 20.6 Flip permutation

Permutation landmark đúng phải là involution:

$$
\pi(\pi(j))=j.
$$

Lật hai lần phải khôi phục landmark ban đầu, sai số chỉ do floating point:

$$
F_\pi(F_\pi(P))\approx P.
$$

---

## 21. Phân tích ổn định số

### 21.1 Epsilon không chỉ để “tránh lỗi”

Các $\epsilon$ trong IoU, BN, normalization và optimizer quyết định conditioning khi mẫu số nhỏ. Quá nhỏ trong float16 có thể underflow; quá lớn tạo bias đáng kể. AMP thường giữ một số toán tử nhạy như reduction/softmax ở precision phù hợp thông qua autocast policy.

### 21.2 Softmax và log-sum-exp

Softmax ổn định được tính bằng

$$
p_i=\frac{e^{z_i-z_{max}}}{\sum_je^{z_j-z_{max}}}.
$$

Trừ $z_{max}$ không đổi kết quả nhưng tránh overflow. `cross_entropy` và `BCEWithLogitsLoss` của PyTorch dùng các dạng ổn định tương tự; không nên tự sigmoid rồi log thủ công.

### 21.3 CIoU degeneracy

Box có $w\le0$ hoặc $h\le0$ làm aspect ratio và diện tích mất nghĩa. Dataloader lọc box suy biến, nhưng prediction ban đầu vẫn có thể rộng/hẹp bất thường. DFL với khoảng cách không âm bảo đảm box decode có $x_2\ge x_1$, $y_2\ge y_1$.

### 21.4 TAL với lũy thừa lớn

Với $u\in(0,1)$ và $\beta=6$, $u^6$ giảm rất nhanh:

$$
0.5^6=0.015625,\qquad 0.8^6\approx0.262.
$$

Điều này tạo phân biệt localization mạnh nhưng có thể làm metric cực nhỏ ở đầu training. Top-k vẫn hoạt động theo thứ hạng, còn normalization bằng $\epsilon$ tránh chia zero.

---

## 22. Các lưu ý kiến trúc rút ra trực tiếp từ code

1. **Main detector là anchor-point-based, không phải anchor-box-based.** Không có tập aspect ratio prior; `anchors` chỉ là tâm cell.

2. **O2O là cơ sở NMS-free, O2M là supervision phụ trợ.** Hai nhánh có parameter head riêng nhưng trunk chung.

3. **Assigner không khả vi.** Prediction dùng để chọn target được `detach`; gradient không đi xuyên qua top-k, argmax hay assignment.

4. **CIoU trong TAL bị clamp về $[0,\infty)$.** CIoU âm không được đưa vào $u^\beta$.

5. **DFL phụ thuộc grid-space.** Đây là điều kiện đơn vị nghiêm ngặt nhất của loss.

6. **Landmark độc lập hình học khỏi predicted bbox.** GT bbox chỉ chuẩn hóa scale; phát biểu ngược lại không đúng với `transferLearning1/loss_lmk.py` hiện tại.

7. **Nhánh `src/transferLearning/` hiện chưa chứa pipeline train/loss hoàn chỉnh**, trong khi `src/transferLearning1/` có model, loss, trainer và inference đầy đủ. Các công thức landmark loss trong tài liệu bám vào nhánh hoàn chỉnh đó; phần dataloader mới trong `transferLearning/` vẫn dùng cùng nền tảng homography/radial/flip.

8. **Input detection mặc định là 480 nhưng classification bias lõi đang mặc định theo 640.** Đây là sai khác cấu hình/implementation cần nhớ khi diễn giải prior ban đầu.

9. **Main `ScaleHead.forward` hiện có unreachable inference return.** Vì vậy `eval()` vẫn tính O2M ở detector lõi, dù kết quả đánh giá chỉ dùng O2O.

10. **mAP evaluator là một implementation nội bộ kiểu COCO 101-point**, hữu ích và nhất quán cho dự án nhưng không nên mặc nhiên xem là bit-for-bit giống COCO API trong mọi edge case.

11. **`src/transferLearning/config_lmk.py` hiện có dấu phẩy sau `min_face_visibility: float = 0.6,`.** Trong Python, giá trị mặc định này trở thành tuple `(0.6,)`, không phải scalar `0.6`, và có thể làm phép kiểm tra xác suất trong `DatasetConfig.__post_init__` phát sinh `TypeError`. Bản `transferLearning1/config_lmk.py` dùng `0.60` đúng kiểu số thực.

---

## 23. Tóm tắt công thức lõi

Chuỗi toán học quan trọng nhất của detector:

$$
X\xrightarrow{Backbone+PAFPN}\{F_3,F_4,F_5\}
\xrightarrow{DualHead}
\{Z^{cls},Z^{reg}\}_{m,o}.
$$

$$
p_{er}=\operatorname{softmax}(z_{er}),\qquad
\hat d_e=\sum_{r=0}^{R-1}rp_{er}.
$$

$$
\hat B^{px}=s(a_x-\hat l,a_y-\hat t,a_x+\hat r,a_y+\hat b).
$$

$$
m_{ia}=\sigma(z_{a,c_i})^\alpha
\max(\operatorname{CIoU}(B_i,\hat B_a),0)^\beta.
$$

$$
q_{ia}=\frac{m_{ia}\max_{a'\in P_i}u_{ia'}}
{\max_{a'\in P_i}m_{ia'}+\epsilon}.
$$

$$
\mathcal L_r
=\lambda_{box}\mathcal L_{CIoU}
+\lambda_{cls}\mathcal L_{BCE}
+\lambda_{dfl}\mathcal L_{DFL}.
$$

$$
\mathcal L_{det}=\omega_m\mathcal L_m+\omega_o\mathcal L_o.
$$

Với landmark:

$$
\hat p_{aj}^{px}=s_a(a_a^{grid}+\Delta p_{aj}^{grid}),
$$

$$
e_{iaj}=\frac{\hat p_{aj}^{px}-p_{ij}^{*px}}{\sqrt{w_ih_i}},
$$

$$
\mathcal L_r^{face}=\mathcal L_r+\lambda_{lmk}\mathcal L_{SmoothL1}(e).
$$

Toàn bộ thiết kế có thể hiểu là sự phối hợp của ba loại inductive bias:

- **CNN bias**: locality, translation equivariance, weight sharing;
- **pyramid bias**: scale decomposition bằng P3/P4/P5;
- **assignment bias**: classification confidence phải đồng thuận với localization quality.

O2M giúp tối ưu dễ, O2O giúp suy luận ít hậu xử lý, DFL biến hồi quy box thành ước lượng phân phối, còn landmark anchor-relative tách hình học điểm khỏi sai số hộp. Đó là lõi toán học chi phối toàn bộ dự án.
