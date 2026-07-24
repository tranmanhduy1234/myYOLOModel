# TỔNG HỢP CÔNG THỨC TOÁN HỌC HÀM MẤT MÁT (LOSS FUNCTIONS)

## 1. Công Thức Tổng Thể Dual-Head Detection Loss

$$\mathcal{L}_{\text{total}} = w_{o2m} \cdot \mathcal{L}_{o2m} + w_{o2o} \cdot \mathcal{L}_{o2o}$$

Trong đó, loss của từng nhánh $x \in \{o2m, o2o\}$ được xác định bằng:

$$\mathcal{L}_{o2x} = \lambda_{\text{box}} \cdot \mathcal{L}_{\text{iou}}^{o2x} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}^{o2x} + \lambda_{\text{dfl}} \cdot \mathcal{L}_{\text{dfl}}^{o2x}$$

- $w_{o2m} = 1.0, w_{o2o} = 1.0$ (Cấu hình `w_o2m`, `w_o2o` trong `TrainConfig`).
- $\lambda_{\text{box}} = 7.5, \lambda_{\text{cls}} = 1.0, \lambda_{\text{dfl}} = 1.5$ (Cấu hình `box_gain`, `cls_gain`, `dfl_gain`).

---

## 2. Task-Aligned Assigner (TAL) Alignment Metric

Alignment Metric $m$ dùng để đánh giá độ tương thích giữa dự đoán của anchor $a$ và Ground Truth $i$:

$$m_{i, a} = s_{i, a}^{\alpha} \times \text{CIoU}(B_{\text{pred}, a}, B_{\text{gt}, i})^{\beta}$$

- $s_{i, a}$: Xác suất phân loại của anchor $a$ đối với lớp của GT $i$ (sau hàm Sigmoid).
- $\text{CIoU}(B_{\text{pred}, a}, B_{\text{gt}, i})$: Chỉ số Complete IoU tính ở không gian `[PIXEL]`.
- $\alpha = 0.5, \beta = 6.0$: Các siêu tham số kiểm soát trọng số tương đối giữa khả năng phân loại và khả năng định vị.

### Chuẩn hóa Trọng số Soft Target (Soft Score Normalization):

$$\hat{m}_{i, a} = m_{i, a} \times \frac{\max_{a' \in \text{Pos}(i)} \text{CIoU}(B_{\text{pred}, a'}, B_{\text{gt}, i})}{\max_{a'' \in \text{Pos}(i)} m_{i, a''} + \epsilon}$$

Vector nhãn phân loại của anchor $a$ được làm mềm theo chỉ số chuẩn hóa:

$$T_{\text{cls}, a} = \text{OneHot}(y_{i}) \times \hat{m}_{i, a}$$

---

## 3. Complete Intersection over Union (CIoU) Loss

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{CIoU} = 1 - \left( \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v \right)$$

Trong đó:
- $\text{IoU} = \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|}$
- $\rho(b, b^{gt})$: Khoảng cách Euclid giữa tâm hai bounding box $b = (x_c, y_c)$ và $b^{gt} = (x_c^{gt}, y_c^{gt})$.
- $c$: Độ dài đường chéo của bounding box nhỏ nhất bao quanh cả $B$ và $B^{gt}$.
- $v$: Đo lường sự đồng dạng về tỷ lệ khung hình (aspect ratio similarity):
  $$v = \frac{4}{\pi^2} \left( \arctan\left(\frac{w^{gt}}{h^{gt}}\right) - \arctan\left(\frac{w}{h}\right) \right)^2$$
- $\alpha$: Trọng số cân bằng non-tradeoff parameter:
  $$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

Trọng số chuẩn hóa CIoU loss trong batch:

$$\mathcal{L}_{\text{iou}}^{\text{branch}} = \frac{\sum_{a \in \text{Pos}} (1 - \text{CIoU}_a) \cdot w_a}{\sum_{a \in \text{Pos}} w_a}$$

với $w_a = \sum_{c} T_{\text{cls}, a, c}$.

---

## 4. Distribution Focal Loss (DFL)

DFL rời rạc hóa khoảng cách liên tục $y \in [0, \text{reg\_max}-1]$ (tính bằng đơn vị grid cell) thành phân phối xác suất trên $\text{reg\_max} = 16$ bin:

$$y_l = \lfloor y \rfloor, \quad y_r = y_l + 1, \quad w_l = y_r - y, \quad w_r = y - y_l$$

Giá trị DFL loss cho một cạnh (Left/Top/Right/Bottom) được tính theo nội suy Cross-Entropy:

$$\mathcal{L}_{\text{dfl}}(P, y) = w_l \cdot \text{CE}(P, y_l) + w_r \cdot \text{CE}(P, y_r)$$

với $\text{CE}(P, k) = -\log P(k) = -\log \left( \frac{\exp(S_k)}{\sum_{j=0}^{\text{reg\_max}-1} \exp(S_j)} \right)$.

DFL loss tổng hợp trung bình trên 4 cạnh và chuẩn hóa theo batch:

$$\mathcal{L}_{\text{dfl}}^{\text{branch}} = \frac{\sum_{a \in \text{Pos}} \left( \frac{1}{4} \sum_{e \in \{l, t, r, b\}} \mathcal{L}_{\text{dfl}}(P_{a, e}, y_{a, e}) \right) \cdot w_a}{\sum_{a \in \text{Pos}} w_a}$$

---

## 5. Binary Cross-Entropy (BCE) Classification Loss

$$\mathcal{L}_{\text{cls}}^{\text{branch}} = \frac{1}{\sum_{a \in \text{Pos}} w_a} \sum_{a=1}^{A} \sum_{c=1}^{N_{cls}} \text{BCE}(\hat{z}_{a, c}, T_{\text{cls}, a, c})$$

với:
$$\text{BCE}(\hat{z}, t) = -t \cdot \log(\sigma(\hat{z})) - (1 - t) \cdot \log(1 - \sigma(\hat{z}))$$

trong đó $\hat{z}$ là logit chưa qua Sigmoid $\sigma(\hat{z}) = \frac{1}{1 + e^{-\hat{z}}}$.
