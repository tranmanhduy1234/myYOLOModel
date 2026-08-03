# Nền tảng toán học của Face Detection & 478-Point Landmark Transfer Learning

Tài liệu này hệ thống hóa phần toán học đang được hiện thực trong pipeline tại
`src/transferLearning`, từ dữ liệu đầu vào đến suy luận. Công thức được đối chiếu
với source hiện tại, chủ yếu tại:

- [`config_lmk.py`](./config_lmk.py): siêu tham số và trọng số loss;
- [`dataset_lmk.py`](./dataset_lmk.py): letterbox, đổi hệ tọa độ và horizontal flip;
- [`mediapipe_478.py`](./mediapipe_478.py): hoán vị landmark trái/phải;
- [`model_lmk.py`](./model_lmk.py): head phát hiện mặt và landmark;
- [`loss_lmk.py`](./loss_lmk.py): gán nhãn và hàm mất mát đa nhiệm;
- [`train_lmk.py`](./train_lmk.py): transfer learning hai giai đoạn, LR, EMA;
- [`inference.py`](./inference.py): giải mã, lọc và khôi phục tọa độ.

Backbone, neck và một số toán tử loss được tái sử dụng từ `src/blocks.py`,
`src/backbone_neck.py` và `src/train/loss.py`, nên chúng cũng được trình bày ở đây.

---

## 1. Bài toán và ký hiệu

Mô hình giải đồng thời hai nhiệm vụ:

1. phát hiện bounding box của khuôn mặt;
2. hồi quy $K=478$ landmark MediaPipe cho mỗi khuôn mặt.

Các ký hiệu chính:

| Ký hiệu | Ý nghĩa | Giá trị hiện tại |
|---|---|---:|
| $B$ | batch size | 8 mặc định |
| $S$ | cạnh ảnh vuông sau letterbox | 480 |
| $K$ | số landmark mỗi mặt | 478 |
| $C$ | số lớp | 1, lớp `face` |
| $R$ | số bin DFL (`reg_max`) | 16 |
| $s_l$ | stride tại feature level $l$ | 8, 16, 32 |
| $A$ | tổng số anchor point | 4725 |
| $N_i$ | số mặt trong ảnh thứ $i$ | thay đổi theo ảnh |
| $M$ | số GT tối đa trong batch sau padding | $\max_i N_i$, tối thiểu 1 |

Quy ước bounding box:

$$
\mathbf b=(x_1,y_1,x_2,y_2),\qquad
w=x_2-x_1,\quad h=y_2-y_1.
$$

Pipeline dùng ba loại không gian cần phân biệt:

- **normalized**: tọa độ annotation trong $[0,1]$;
- **pixel**: tọa độ trên ảnh letterbox $480\times480$;
- **grid**: tọa độ pixel chia cho stride của feature level.

Quan hệ giữa pixel và grid là:

$$
\mathbf x_{pixel}=s_l\mathbf x_{grid},\qquad
\mathbf x_{grid}=\frac{\mathbf x_{pixel}}{s_l}.
$$

IoU và CIoU là các đại lượng không đơn vị nên không thay đổi nếu hai box cùng
được scale từ grid sang pixel. DFL thì phụ thuộc grid vì mỗi bin biểu diễn một
đơn vị grid cell.

---

## 2. Hình học dữ liệu

### 2.1. Từ normalized sang pixel

Với ảnh gốc có kích thước $W\times H$, một điểm normalized
$(u,v)\in[0,1]^2$ tương ứng với điểm ảnh gốc:

$$
x=uW,\qquad y=vH.
$$

Tương tự, bounding box normalized được đổi sang pixel bằng cách nhân các tọa
độ $x$ với $W$, các tọa độ $y$ với $H$.

Source kiểm tra toàn bộ annotation trước khi train:

- kích thước ảnh hữu hạn và dương;
- bbox/landmark không chứa `NaN` hoặc `Inf`;
- tọa độ nằm trong miền normalized cho phép;
- mọi face có cùng $K$;
- bbox vẫn đủ lớn sau phép letterbox.

### 2.2. Letterbox về 480 × 480

Letterbox giữ nguyên tỉ lệ hình học thay vì kéo méo ảnh. Với kích thước đích
$S\times S$:

$$
r=\min\left(\frac{S}{W},\frac{S}{H}\right),
$$

$$
W'=\operatorname{round}(rW),\qquad
H'=\operatorname{round}(rH).
$$

Phần padding trái và trên là:

$$
p_x=\left\lfloor\frac{S-W'}{2}\right\rfloor,\qquad
p_y=\left\lfloor\frac{S-H'}{2}\right\rfloor.
$$

Một điểm ảnh gốc được ánh xạ sang ảnh model bởi phép affine:

$$
x'=rx+p_x,\qquad y'=ry+p_y.
$$

Do đó box sau letterbox là:

$$
\mathbf b'=\big(rx_1+p_x,\;ry_1+p_y,\;rx_2+p_x,\;ry_2+p_y\big).
$$

Tất cả tọa độ cuối cùng được clip về ([0,S]). Train và inference cùng dùng
chính sách letterbox này, vì vậy không có sai lệch hình học giữa hai pha.

### 2.3. Khôi phục tọa độ về ảnh gốc

Inference dùng phép biến đổi ngược:

$$
x=\frac{x'-p_x}{r},\qquad
y=\frac{y'-p_y}{r}.
$$

Sau đó (x) được clip về ([0,W]), (y) về ([0,H]). Công thức áp dụng giống
nhau cho bốn cạnh bbox và mọi landmark.

### 2.4. Horizontal flip bảo toàn semantic

Nếu chỉ thay $x\leftarrow S-x$ mà giữ nguyên index, điểm “mắt trái” sẽ mang
vị trí của “mắt phải”, làm sai nhãn semantic. Pipeline dùng một hoán vị
$\pi:\{0,\ldots,477\}\to\{0,\ldots,477\}$ lấy từ topology MediaPipe.

Với output landmark index (i):

$$
x'_i=S-x_{\pi(i)},\qquad y'_i=y_{\pi(i)}.
$$

Hoán vị thỏa hai tính chất:

$$
\{\pi(i)\mid 0\le i<K\}=\{0,\ldots,K-1\},
$$

$$
\pi(\pi(i))=i.
$$

Tính chất thứ hai nói rằng flip hai lần trả lại đúng semantic ban đầu. Bbox
được flip theo:

$$
x'_1=S-x_2,\qquad x'_2=S-x_1,
$$

trong khi (y_1,y_2) giữ nguyên.

Ở chế độ `paired`, mỗi record (z) sinh hai mẫu ((z,F(z))). Nếu phân phối góc
yaw ban đầu là (p(y)), phân phối sau augmentation tỉ lệ với:

$$
p_{paired}(y)=\frac{p(y)+p(-y)}{2}.
$$

Suy ra (p_{paired}(y)=p_{paired}(-y)): lệch trái/phải được cân bằng theo cặp,
không chỉ cân bằng theo kỳ vọng ngẫu nhiên.

### 2.5. Photometric augmentation

Brightness, contrast, saturation và hue chỉ thay đổi giá trị màu, không thay
đổi tọa độ. Có thể hình dung brightness và contrast đơn giản dưới dạng:

$$
I_b=\operatorname{clip}(a_bI),
$$

$$
I_c=\operatorname{clip}\big(a_c(I-\mu)+\mu\big),
$$

với (mu) là độ sáng trung bình. Triển khai thực tế dùng `ColorJitter` của
Torchvision và có thể thay đổi thứ tự các phép màu. Vì không có phép affine
không gian nào ở đây, bbox và landmark không cần biến đổi thêm.

---

## 3. Các phép toán nền trong mạng

### 3.1. Convolution

Với kernel $K\times K$, convolution 2D tại output channel $c_o$ là:

$$
Y_{c_o,i,j}=b_{c_o}+\sum_{c_i}\sum_{m,n}
W_{c_o,c_i,m,n}X_{c_i,i+m,j+n}.
$$

Kích thước không gian output tổng quát:

$$
H_{out}=\left\lfloor
\frac{H_{in}+2p-d(K-1)-1}{s}+1
\right\rfloor,
$$

với padding (p), dilation (d) và stride (s). `autopad` chọn padding để
giữ kích thước khi (s=1) và kernel lẻ.

Một block `Conv` trong dự án là:

$$
Y=\operatorname{SiLU}(\operatorname{BN}(W*X)).
$$

Batch Normalization ở chế độ train:

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y=\gamma\hat x+\beta.
$$

SiLU:

$$
\operatorname{SiLU}(x)=x\sigma(x),\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

### 3.2. Depthwise convolution

Depthwise convolution xử lý từng channel độc lập. Với (C_{in}=C_{out}=C),
chi phí xấp xỉ:

$$
K^2CHW
$$

thay vì (K^2C^2HW) của convolution dense. `DWConv` trong classification
stem giúp giảm phép tính trong khi vẫn học đặc trưng không gian.

### 3.3. Residual, Bottleneck, C2f và CIB

Residual block có dạng:

$$
Y=X+F(X),
$$

khi số channel đầu vào và đầu ra bằng nhau. Đường tắt làm gradient có thêm
đường truyền trực tiếp:

$$
\frac{\partial Y}{\partial X}=I+\frac{\partial F}{\partial X},
$$

giúp mạng sâu dễ tối ưu hơn.

`C2f` chia feature thành hai nhánh, cho một nhánh đi qua chuỗi bottleneck rồi
concatenate toàn bộ feature trung gian:

$$
Y=\operatorname{Conv}\big([X_1,X_2,F_1(X_2),\ldots,F_n(\cdot)]\big).
$$

`CIB` kết hợp pointwise $1\times1$ và depthwise $3\times3$, nhằm trao đổi
thông tin giữa channel với chi phí không gian thấp. `C2fCIB` dùng CIB thay cho
bottleneck thường ở các stage sâu.

### 3.4. SPPF

SPPF áp dụng max pooling $5\times5$ ba lần liên tiếp rồi concatenate:

$$
Y=\operatorname{Conv}\big([X,P(X),P^2(X),P^3(X)]\big).
$$

Nó tổng hợp ngữ cảnh ở nhiều receptive field mà không cần nhiều kernel lớn.

### 3.5. Attention trong C2fPSA

Feature $X\in\mathbb R^{B\times C\times H\times W}$ được reshape với
$N=HW$ token và chiếu thành query, key, value. Với $d=C/h$ là chiều mỗi
attention head:

$$
Q,K,V=W_{qkv}X,
$$

$$
P=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right),
\qquad O=PV.
$$

Source cộng positional encoding cục bộ bằng depthwise convolution trên (V),
sau đó dùng hai residual có LayerScale:

$$
X_1=X+\gamma_1\odot\operatorname{Proj}(O+PE(V)),
$$

$$
Y=X_1+\gamma_2\odot\operatorname{FFN}(X_1).
$$

Giá trị khởi tạo $\gamma_1=\gamma_2=10^{-2}$ giữ contribution của attention
nhỏ ở đầu quá trình học.

---

## 4. Backbone và PAFPN neck

### 4.1. Kích thước feature

Với input $B\times3\times480\times480$, cấu hình channel
((56,112,224,448,640)) tạo:

| Feature | Stride | Tensor shape | Vai trò |
|---|---:|---|---|
| stem | 2 | (B\times56\times240\times240) | đặc trưng mức thấp |
| stage 1 | 4 | (B\times112\times120\times120) | cạnh/texture |
| (P_3) | 8 | (B\times224\times60\times60) | mặt nhỏ, chi tiết landmark |
| (P_4) | 16 | (B\times448\times30\times30) | mặt trung bình |
| (P_5) | 32 | (B\times640\times15\times15) | mặt lớn, ngữ cảnh rộng |

### 4.2. Feature pyramid hai chiều

PAFPN thực hiện top-down:

$$
P_4^{td}=F_4([\operatorname{Up}(P_5),P_4]),
$$

$$
P_3^{out}=F_3([\operatorname{Up}(P_4^{td}),P_3]),
$$

rồi bottom-up:

$$
P_4^{out}=G_4([\operatorname{Down}(P_3^{out}),P_4^{td}]),
$$

$$
P_5^{out}=G_5([\operatorname{Down}(P_4^{out}),P_5]).
$$

Top-down truyền semantic mạnh từ tầng sâu xuống độ phân giải cao; bottom-up đưa
thông tin định vị chi tiết trở lại tầng sâu. Ba output vẫn có stride 8, 16, 32
và channel 224, 448, 640.

---

## 5. Detection head đa nhiệm

### 5.1. Hai nhánh assignment

Mỗi scale có hai đường dự đoán độc lập:

- **o2m (one-to-many)**: mỗi GT có thể gán tối đa 10 anchor, tạo supervision dày;
- **o2o (one-to-one)**: mỗi GT chọn tối đa 1 anchor, dùng trực tiếp khi inference.

Mỗi đường lại có ba stem/output:

| Output | Số channel mỗi vị trí | Ý nghĩa |
|---|---:|---|
| classification | (C=1) | logit xác suất mặt |
| box regression | (4R=64) | 16 logit DFL cho mỗi cạnh l/t/r/b |
| landmark | (2K=956) | raw logit x/y cho 478 điểm |

Tại (S=480), số vị trí dự đoán là:

$$
A=60^2+30^2+15^2=3600+900+225=4725.
$$

Shape sau khi ghép ba level:

$$
\text{cls}\in\mathbb R^{B\times4725\times1},
$$

$$
\text{reg\_raw}\in\mathbb R^{B\times64\times4725},
$$

$$
\text{lmk\_raw}\in\mathbb R^{B\times956\times4725}.
$$

### 5.2. Anchor point

Với grid cell hàng (i), cột (j), anchor trong grid space đặt ở tâm cell:

$$
\mathbf a_{ij}^{grid}=(j+0.5,i+0.5).
$$

Anchor trong pixel space:

$$
\mathbf a_{ij}^{pixel}=s_l\mathbf a_{ij}^{grid}.
$$

Đây là anchor **point**, không phải anchor box có width/height cố định.

### 5.3. Classification prior

Classification logit (z) được đổi thành xác suất bởi sigmoid:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Bias theo stride được khởi tạo:

$$
b_s=\log\left(\frac{5}{C(S/s)^2}\right).
$$

Với (C=1,S=480), bias xấp xỉ (-6.579,-5.193,-3.807) lần lượt cho
stride 8, 16, 32. Prior thấp hạn chế số lượng positive giả ở lúc khởi tạo.

Landmark weight được khởi tạo gần 0 và bias bằng 0, nên sigmoid ban đầu gần
(0.5): điểm ban đầu nằm gần tâm vùng bbox mở rộng.

---

## 6. Distribution Focal Loss và giải mã bounding box

### 6.1. Khoảng cách rời rạc đến bốn cạnh

Thay vì hồi quy trực tiếp bốn số thực, mỗi cạnh $e\in\{l,t,r,b\}$ có $R=16$
logit $z_{e,k}$, $k=0,\ldots,15$. Xác suất bin:

$$
p_{e,k}=\frac{e^{z_{e,k}}}{\sum_{j=0}^{R-1}e^{z_{e,j}}}.
$$

Module DFL lấy kỳ vọng phân phối:

$$
\hat d_e=\sum_{k=0}^{R-1}k\,p_{e,k}.
$$

Với anchor (mathbf a=(a_x,a_y)) trong grid space:

$$
\hat x_1=a_x-\hat l,\qquad
\hat y_1=a_y-\hat t,
$$

$$
\hat x_2=a_x+\hat r,\qquad
\hat y_2=a_y+\hat b.
$$

Sau đó nhân stride để nhận box pixel:

$$
\hat{\mathbf b}^{pixel}=s_l\hat{\mathbf b}^{grid}.
$$

### 6.2. Soft label DFL

Với target distance thực (y), đặt:

$$
y_l=\lfloor y\rfloor,\qquad y_r=y_l+1,
$$

$$
w_l=y_r-y,\qquad w_r=y-y_l.
$$

Loss cho một cạnh là nội suy hai cross-entropy lân cận:

$$
L_{DFL}(y)=w_l\,CE(\mathbf z,y_l)+w_r\,CE(\mathbf z,y_r).
$$

Source lấy trung bình trên bốn cạnh. Target được clamp tối đa 14.99 khi
$R=16$, bảo đảm hai bin $y_l,y_r$ đều thuộc tập $\{0,\ldots,15\}$.

DFL giữ lại mức bất định của tọa độ và cho phép dự đoán số thực bằng kỳ vọng,
thay vì ép khoảng cách vào duy nhất một bin.

---

## 7. IoU và CIoU

Với hai box (A,B):

$$
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|+\epsilon}.
$$

CIoU bổ sung khoảng cách tâm và độ lệch tỉ lệ:

$$
CIoU=IoU-\frac{\rho^2(\mathbf c_A,\mathbf c_B)}{c^2}-\alpha_v v,
$$

trong đó (c) là đường chéo box nhỏ nhất bao cả hai box,

$$
v=\frac{4}{\pi^2}
\left(
\arctan\frac{w_B}{h_B+\epsilon}
-
\arctan\frac{w_A}{h_A+\epsilon}
\right)^2,
$$

$$
\alpha_v=\frac{v}{1-IoU+v+\epsilon}.
$$

Loss định vị cơ bản là (1-CIoU). Nhờ hai thành phần bổ sung, box không giao
nhau vẫn nhận được tín hiệu về khoảng cách tâm, đồng thời mô hình học đúng tỉ
lệ width/height.

---

## 8. Task-Aligned Assigner

### 8.1. Alignment metric

Với GT (g), anchor prediction (a), classification probability đúng lớp
(p_{a,c_g}) và overlap CIoU đã clamp không âm (u_{g,a}), metric là:

$$
t_{g,a}=p_{a,c_g}^{\alpha}u_{g,a}^{\beta}.
$$

Cấu hình hiện tại:

$$
\alpha=0.5,\qquad\beta=6.
$$

Số mũ overlap lớn làm chất lượng định vị ảnh hưởng mạnh hơn classification
trong lựa chọn positive.

### 8.2. Điều kiện positive

Anchor chỉ là ứng viên của GT nếu tâm anchor nằm **bên trong** GT:

$$
\min(a_x-x_1,\;a_y-y_1,\;x_2-a_x,\;y_2-a_y)>\epsilon.
$$

Trong tập ứng viên, assigner chọn top-(k) theo (t_{g,a}):

$$
k_{o2m}=10,\qquad k_{o2o}=1.
$$

Nếu một anchor được nhiều GT chọn, source giữ GT có CIoU cao nhất. Vì vậy một
anchor chỉ thuộc tối đa một GT, nhưng ở o2m một GT có thể sở hữu nhiều anchor.

### 8.3. Soft target chất lượng

Với các positive của GT (g), đặt:

$$
t_g^{max}=\max_a t_{g,a},\qquad
u_g^{max}=\max_a u_{g,a}.
$$

Chất lượng chuẩn hóa của anchor là:

$$
q_{g,a}=\frac{t_{g,a}u_g^{max}}{t_g^{max}+\epsilon}.
$$

(q_{g,a}) nhân vào one-hot target classification. Anchor khớp cả lớp lẫn vị
trí tốt sẽ đóng góp lớn hơn trong classification, CIoU, DFL và landmark loss.

Assignment chạy trong `torch.no_grad()` trên prediction đã `detach`, nên phép
chọn top-k không tham gia đồ thị gradient.

---

## 9. Landmark parameterization

### 9.1. Vùng bbox mở rộng

Landmark không được decode trên toàn ảnh mà tương đối theo bbox dự đoán. Với
margin (m=0.05):

$$
x_{1e}=x_1-mw,\qquad y_{1e}=y_1-mh,
$$

$$
w_e=(1+2m)w,\qquad h_e=(1+2m)h.
$$

Vùng landmark rộng hơn bbox 5% mỗi phía, tổng width/height bằng 110% box.

### 9.2. Decode prediction

Với raw logits ((z^x_{a,k},z^y_{a,k})), tọa độ tương đối là:

$$
\hat t^x_{a,k}=\sigma(z^x_{a,k}),\qquad
\hat t^y_{a,k}=\sigma(z^y_{a,k}).
$$

Tọa độ pixel:

$$
\hat x_{a,k}=x_{1e}+\hat t^x_{a,k}w_e,
$$

$$
\hat y_{a,k}=y_{1e}+\hat t^y_{a,k}h_e.
$$

### 9.3. Encode target

Loss thực hiện phép ngược dựa trên target bbox đã được assign:

$$
t^x_{a,k}=\operatorname{clip}
\left(\frac{x_{a,k}-x_{1e}}{w_e},0,1\right),
$$

$$
t^y_{a,k}=\operatorname{clip}
\left(\frac{y_{a,k}-y_{1e}}{h_e},0,1\right).
$$

Điểm quan trọng: landmark branch học tọa độ **tương đối với box target**, trong
khi inference decode theo **box dự đoán**. Chất lượng bbox vì vậy tác động trực
tiếp đến landmark pixel cuối cùng.

---

## 10. Hàm mất mát

### 10.1. BCE classification

Với logit $z$ và soft target $q\in[0,1]$:

$$
BCE(z,q)=-q\log\sigma(z)-(1-q)\log(1-\sigma(z)).
$$

Source cộng BCE trên toàn bộ anchor và class, rồi chuẩn hóa bằng:

$$
Q=\max\left(1,\sum_{a,c}q_{a,c}\right),
$$

$$
L_{cls}=\frac{\sum_{a,c}BCE(z_{a,c},q_{a,c})}{Q}.
$$

### 10.2. CIoU loss có trọng số assignment

Với positive anchor $a$, đặt $q_a=\sum_c q_{a,c}$:

$$
L_{IoU}=\frac{\sum_{a\in\mathcal P}q_a
\left(1-CIoU(\hat{\mathbf b}_a,\mathbf b_a)\right)}{Q}.
$$

### 10.3. DFL loss có trọng số assignment

$$
L_{DFL}=\frac{\sum_{a\in\mathcal P}q_a
\left(\frac14\sum_{e\in\{l,t,r,b\}}L_{DFL}^{a,e}\right)}{Q}.
$$

### 10.4. Smooth L1 landmark loss

Với sai số tọa độ $e=\hat t-t$, Smooth L1 có tham số
$\delta=0.05$:

$$
\ell_{smooth}(e)=
\begin{cases}
\dfrac{e^2}{2\delta}, & |e|<\delta,\\[4pt]
|e|-\dfrac{\delta}{2}, & |e|\ge\delta.
\end{cases}
$$

Hàm bậc hai gần 0 tạo gradient mượt; nhánh tuyến tính cho outlier hạn chế
gradient quá lớn.

### 10.5. Trọng số theo vùng landmark

Mỗi landmark có trọng số (omega_k):

$$
\omega_k=
\begin{cases}
3, & k\text{ thuộc mắt hoặc iris},\\
3, & k\text{ thuộc miệng},\\
4, & k=1\text{ (đỉnh mũi)},\\
1, & \text{các điểm còn lại}.
\end{cases}
$$

Với (mathcal P_{lmk}) là positive anchor có landmark hợp lệ, source tính:

$$
L_{lmk}=
\frac{
\sum_{a\in\mathcal P_{lmk}}q_a
\sum_{k=1}^{K}\omega_k
\sum_{d\in\{x,y\}}\ell_{smooth}(\hat t_{a,k,d}-t_{a,k,d})
}{
\left(\sum_{a\in\mathcal P_{lmk}}q_a\right)
\left(\sum_{k=1}^{K}\omega_k\right)\cdot2+\epsilon
}.
$$

Phép chuẩn hóa theo tổng trọng số giúp tăng sự chú ý tương đối cho mắt, miệng,
mũi mà không làm độ lớn loss tăng đơn thuần theo số landmark.

### 10.6. Loss của mỗi branch và loss tổng

Cho $r\in\{o2m,o2o\}$:

$$
L_r=lambda_{box}L_{IoU}^{r}
+\lambda_{cls}L_{cls}^{r}
+\lambda_{dfl}L_{DFL}^{r}
+\lambda_{lmk}L_{lmk}^{r}.
$$

Giá trị hiện tại:

$$
\lambda_{box}=7.5,\quad
\lambda_{cls}=0.5,\quad
\lambda_{dfl}=1.5,\quad
\lambda_{lmk}=2.0.
$$

Loss toàn mô hình:

$$
L=\lambda_{o2m}L_{o2m}+\lambda_{o2o}L_{o2o},
$$

với $\lambda_{o2m}=\lambda_{o2o}=1$. Khi batch không có positive, bbox,
DFL và landmark loss trả về 0 nhưng vẫn giữ computational graph; classification
vẫn học background.

---

## 11. Transfer learning hai giai đoạn

Gọi tham số toàn mạng:

$$
\theta=(\theta_B,\theta_N,\theta_H),
$$

với (B,N,H) lần lượt là backbone, neck và head.

### 11.1. Giai đoạn 1: chỉ học head

Trong 5 epoch đầu:

$$
\theta_B\leftarrow\theta_B^{pretrained},\qquad
\theta_N\leftarrow\theta_N^{pretrained},
$$

$$
\nabla_{\theta_B}L=0,\qquad
\nabla_{\theta_N}L=0,
$$

$$
\theta_H\leftarrow\theta_H-\eta_H\nabla_{\theta_H}L.
$$

Backbone và neck có learning rate bằng 0, `requires_grad=False`; BatchNorm của
trunk cũng ở eval mode nên running mean/variance không đổi. Head có base LR
(10^{-3}). Mục đích là cho head mới thích nghi với face/landmark mà không phá
biểu diễn pretrained.

### 11.2. Giai đoạn 2: fine-tune toàn mạng

Trong 45 epoch tiếp theo:

$$
\eta_H=3\times10^{-4},\qquad
\eta_{B,N}=3\times10^{-5}=0.1\eta_H.
$$

Toàn trunk được mở khóa, nhưng update nhỏ hơn head một bậc độ lớn:

$$
\theta_{B,N}\leftarrow\theta_{B,N}
-0.1\eta_H\nabla_{\theta_{B,N}}L.
$$

Optimizer được giữ nguyên giữa hai stage, vì vậy momentum/moment của head không
bị xóa tại ranh giới stage.

---

## 12. Warmup và cosine learning-rate schedule

Mỗi stage có scheduler độc lập. Gọi:

- (u): local step trong stage, bắt đầu từ 0;
- (T): tổng số step của stage;
- (W): số warmup step;
- (f_{min}): hệ số LR tối thiểu;
- (eta_g^{base}): base LR của parameter group (g).

Trong warmup:

$$
f(u)=\frac{u+1}{W},\qquad 0\le u<W.
$$

Sau warmup:

$$
q(u)=\operatorname{clip}
\left(\frac{u-W}{\max(T-W-1,1)},0,1\right),
$$

$$
f(u)=f_{min}+(1-f_{min})\frac{1+\cos(\pi q(u))}{2}.
$$

LR thực tế:

$$
\eta_g(u)=\eta_g^{base}f(u).
$$

Stage 1 dùng warmup 0.5 epoch, (f_{min}=0.2). Stage 2 dùng warmup 1 epoch,
(f_{min}=0.01). Vì base LR trunk ở stage 1 bằng 0, mọi hệ số scheduler vẫn
cho (eta_{trunk}=0).

---

## 13. Tối ưu hóa và ổn định số học

### 13.1. AdamW

Mặc định pipeline dùng AdamW. Với gradient (g_t):

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Cấu hình: $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$, weight decay
$\lambda=5\times10^{-4}$. AdamW tách weight decay khỏi moment gradient.

Nếu chọn SGD, source dùng momentum 0.937 và Nesterov.

### 13.2. Gradient clipping

Với vector gradient toàn mạng (g) và ngưỡng (C=10):

$$
g\leftarrow g\cdot\min\left(1,\frac{C}{\lVert g\rVert_2}\right).
$$

Clipping chỉ thay gradient khi norm vượt ngưỡng, giúp hạn chế exploding
gradient, đặc biệt lúc head landmark mới bắt đầu học.

### 13.3. Automatic Mixed Precision

AMP dùng số học precision thấp cho các phép phù hợp và giữ các phép nhạy cảm ở
precision cao. Loss scaling nhân loss với (s):

$$
\tilde L=sL,\qquad
\nabla\tilde L=s\nabla L.
$$

Trước optimizer step, gradient được chia lại cho (s). Nếu phát hiện overflow,
step bị bỏ qua và EMA không được cập nhật.

### 13.4. Exponential Moving Average

Sau optimizer step thành công thứ (t), decay động là:

$$
d_t=d_{max}\left(1-e^{-t/\tau}\right),
$$

với $d_{max}=0.9998,\tau=2000$. EMA update:

$$
\theta_t^{EMA}=d_t\theta_{t-1}^{EMA}+(1-d_t)\theta_t.
$$

Ở các bước đầu (d_t) nhỏ nên EMA nhanh theo kịp model; về sau (d_t) tiến
dần đến 0.9998 để làm trơn nhiễu update. Validation và inference ưu tiên trọng
số EMA khi checkpoint có sẵn.

---

## 14. Inference

Inference chỉ dùng o2o. Với logit classification (z_a):

$$
s_a=\sigma(z_a).
$$

Giữ candidate nếu:

$$
s_a>\tau_{conf},
$$

với $\tau_{conf}=0.25$ mặc định. Nếu còn quá nhiều candidate, lấy top
`max_det=100` theo score.

Mặc định $\tau_{IoU}=0$, tức không chạy NMS vì o2o được huấn luyện theo thiết
kế NMS-free. Nếu người dùng đặt $\tau_{IoU}>0$, NMS tùy chọn sắp box theo score,
giữ box tốt nhất và loại box còn lại nếu:

$$
IoU(b_i,b_j)>\tau_{IoU}.
$$

Cuối cùng bbox và landmark được ánh xạ ngược qua letterbox theo Mục 2.3.

---

## 15. Luồng toán học end-to-end

```text
Ảnh + annotation normalized
    ↓ normalized → pixel, letterbox 480, paired semantic flip
X ∈ R^(B×3×480×480)
    ↓ Backbone
P3, P4, P5 tại stride 8/16/32
    ↓ PAFPN
feature đa tỉ lệ 224/448/640 channel
    ↓ dual head o2m + o2o
cls logits + DFL logits + landmark logits
    ↓ DFL expectation + anchor decode
bbox pixel
    ↓ Task-Aligned Assigner
positive anchors + soft quality targets
    ↓ encode landmark theo target bbox mở rộng
BCE + CIoU + DFL + weighted Smooth L1
    ↓ optimizer hai parameter group
stage 1 freeze trunk → stage 2 differential LR
    ↓ EMA
checkpoint model/EMA
    ↓ inference o2o
sigmoid → threshold/top-k → optional NMS → inverse letterbox
```

---

## 16. Các metric nên dùng khi có train/validation split

Phần này là **khuyến nghị đánh giá**, chưa được triển khai trong validation
hiện tại; source hiện chỉ lấy trung bình total validation loss.

### 16.1. Normalized Mean Error cho landmark

Sai số Euclid trung bình:

$$
E=\frac1K\sum_{k=1}^{K}
\left\lVert\hat{\mathbf p}_k-\mathbf p_k\right\rVert_2.
$$

Để so sánh giữa các kích thước mặt, chuẩn hóa theo đường chéo bbox:

$$
NME_{bbox}=\frac{E}{\sqrt{w^2+h^2}}.
$$

Cũng có thể chuẩn hóa bằng inter-ocular distance, nhưng phải định nghĩa cố định
hai điểm/tâm mắt và cần thận trọng với ảnh profile khi một mắt bị che.

### 16.2. PCK

Percentage of Correct Keypoints tại ngưỡng $\tau$:

$$
PCK(\tau)=\frac{1}{K}
\sum_{k=1}^{K}
\mathbf 1\left[
\frac{\lVert\hat{\mathbf p}_k-\mathbf p_k\rVert_2}{d_{norm}}<\tau
\right].
$$

Nên báo riêng PCK/NME cho mắt, iris, miệng, đỉnh mũi và toàn bộ 478 điểm để
xác nhận landmark weighting thực sự cải thiện đúng vùng ưu tiên.

### 16.3. Detection metrics

Khi có split validation độc lập, nên báo precision, recall, AP tại IoU 0.5 và
mAP trên dải 0.5:0.95. Một prediction được tính true positive khi đúng lớp và:

$$
IoU(\hat b,b^{GT})\ge\tau.
$$

### 16.4. Đánh giá theo pose

Do dữ liệu ban đầu lệch phân phối yaw, metric nên được phân tầng:

- quay trái;
- gần chính diện;
- quay phải;
- các khoảng (|yaw|) tăng dần.

Paired flip cân bằng số mẫu train về mặt toán học, nhưng metric theo pose vẫn
cần thiết để phát hiện chênh lệch do occlusion, chất lượng ảnh hoặc quy ước yaw.

---

## 17. Các bất biến cần giữ khi thay đổi pipeline

1. `image_size` của dataset, model bias và inference phải cùng bằng 480.
2. Số landmark (K) phải giống giữa train, val, head, loss và checkpoint.
3. Tọa độ truyền vào assigner phải cùng ở pixel space.
4. Tọa độ truyền vào BboxLoss/DFL phải cùng ở grid space.
5. Encode và decode landmark phải dùng cùng `lmk_margin`.
6. Horizontal flip phải biến đổi cả tọa độ lẫn index semantic bằng cùng (pi).
7. Stage 1 phải freeze tham số và BatchNorm của cả backbone lẫn neck.
8. Stage 2 phải giữ $0<\eta_{trunk}<\eta_{head}$.
9. Resume chỉ an toàn khi training plan trong checkpoint trùng config hiện tại.
10. Validation/test không được dùng paired augmentation của train.

Các bất biến trên quan trọng hơn việc thay đổi riêng lẻ một hyperparameter: chỉ
cần phá một quan hệ không gian hoặc semantic, loss vẫn có thể chạy nhưng sẽ tối
ưu một mục tiêu sai.
