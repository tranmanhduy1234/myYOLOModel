# CHƯƠNG 6: CẤU TRÚC HÀM MẤT MÁT VÀ ĐỘNG HỌC KHÔNG GIANG TỌA ĐỘ (LOSS FUNCTION)

## 1. GIỚI THIỆU CHƯƠNG

Trong mô hình phát hiện vật thể không sử dụng NMS (**NMS-Free Detector**), **Hàm Mất Mát (Loss Function)** đóng vai trò là "la bàn hướng dẫn" phức tạp nhất. Khác với các mô hình phát hiện vật thể truyền thống vốn dựa vào thuật toán lọc trùng lặp NMS (Non-Maximum Suppression) ở bước suy luận, NMS-Free Detector đòi hỏi mạng thần kinh phải tự học cách dự đoán **duy nhất một bounding box chính xác nhất cho mỗi vật thể** ở đầu ra One-to-One (`o2o`), trong khi vẫn phải duy trì khả năng hội tụ nhanh nhờ sự hỗ trợ của đầu ra One-to-Many (`o2m`).

Chương này trình bày phân tích chuyên sâu toàn bộ cơ sở toán học, các phương trình vi phân gradient, thuật toán gán nhãn động **Task-Aligned Assigner (TAL)**, cơ chế chuyển đổi hai không gian tọa độ **[PIXEL] $\leftrightarrow$ [GRID]**, cùng ba thành phần mất mát cốt lõi (**CIoU Loss**, **Distribution Focal Loss - DFL**, và **BCE Classification Loss**) được cài đặt trong tập tin [`src/train/loss.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py).

---

## 2. KIẾN TRÚC DUAL-HEAD VÀ TỔNG HỢP LOSS

Sơ đồ tổng thể cấu trúc hàm loss Dual-Head được biểu diễn trong tệp sơ đồ [`diagrams/loss_structure.mermaid`](diagrams/loss_structure.mermaid).

```text
                     DetectionLoss.forward(preds, targets)
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
         Nhánh One-to-Many (o2m)                 Nhánh One-to-One (o2o)
         (assigner_o2m: topk=10)                 (assigner_o2o: topk=1)
                   |                                       |
        +----------+----------+                 +----------+----------+
        |          |          |                 |          |          |
        v          v          v                 v          v          v
     BCE Cls   CIoU Box    DFL Reg           BCE Cls   CIoU Box    DFL Reg
     (L_cls)   (L_iou)     (L_dfl)           (L_cls)   (L_iou)     (L_dfl)
        |          |          |                 |          |          |
        +----------+----------+                 +----------+----------+
                   |                                       |
                   v                                       v
        L_o2m = 7.5*iou + 1.0*cls + 1.5*dfl     L_o2o = 7.5*iou + 1.0*cls + 1.5*dfl
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                        L_total = 1.0*L_o2m + 1.0*L_o2o
```

### Công Thức Toán Học Tổng Thể:

$$\mathcal{L}_{\text{total}} = w_{o2m} \cdot \mathcal{L}_{o2m} + w_{o2o} \cdot \mathcal{L}_{o2o}$$

trong đó loss của mỗi nhánh $x \in \{o2m, o2o\}$ được cấu thành từ 3 thành phần:

$$\mathcal{L}_{o2x} = \lambda_{\text{box}} \cdot \mathcal{L}_{\text{iou}}^{o2x} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}^{o2x} + \lambda_{\text{dfl}} \cdot \mathcal{L}_{\text{dfl}}^{o2x}$$

Cấu hình tham số mặc định trong `TrainConfig`:
- Trọng số nhánh: $w_{o2m} = 1.0$, $w_{o2o} = 1.0$.
- Hệ số thành phần: $\lambda_{\text{box}} = 7.5$ (`box_gain`), $\lambda_{\text{cls}} = 1.0$ (`cls_gain`), $\lambda_{\text{dfl}} = 1.5$ (`dfl_gain`).

---

## 3. RANH GIỚI KHÔNG GIAN TỌA ĐỘ: [PIXEL] VS [GRID]

Một đóng góp kỹ thuật cực kỳ quan trọng được ghi chú chi tiết trong docstring của [`src/train/loss.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L27-L62) là việc phân định rõ ranh giới giữa hai không gian tọa độ. Sự nhầm lẫn giữa hai không gian này sẽ lập tức làm hỏng phép tính IoU và khiến gradient của DFL bị sai lệch.

Sơ đồ chi tiết chuyển đổi không gian tọa độ được thể hiện trong [`diagrams/coordinate_transformation.mermaid`](diagrams/coordinate_transformation.mermaid).

```text
+------------------------------------------------------------------------------------+
| 1. KHÔNG GIAN [PIXEL] (Pixel Coordinate Space)                                      |
+------------------------------------------------------------------------------------+
| - Đơn vị: Thang đo pixel thực tế của ảnh đầu vào (từ 0 đến imgsz, ví dụ: 0->480).  |
| - Đối tượng sử dụng:                                                                |
|   + Ground Truth Bounding Boxes (targets["boxes"]).                                 |
|   + Decoded Predicted Boxes (box_pixel từ Head).                                    |
|   + Anchor Points Pixel (anchors_pixel = anchors_grid * stride).                    |
| - Module áp dụng: TaskAlignedAssigner (TAL).                                        |
| - Lý do: So sánh trực tiếp vị trí dự đoán với nhãn thực tế của ảnh gốc.             |
+------------------------------------------------------------------------------------+
                                       |
                                       | Phép Quy Đổi Kích Thước:
                                       | GRID = PIXEL / stride
                                       v
+------------------------------------------------------------------------------------+
| 2. KHÔNG GIAN [GRID] (Feature Map Grid Space)                                      |
+------------------------------------------------------------------------------------+
| - Đơn vị: Số lượng ô lưới trên feature map (từ 0 đến H_feat/W_feat).                |
| - Đối tượng sử dụng:                                                                |
|   + Grid Anchor Points (anchors: gx+0.5, gy+0.5).                                   |
|   + Logits phân phối DFL (pred_dist: 4 * reg_max).                                  |
|   + Grid Decoded Boxes (pred_bboxes_grid = box_pixel / stride).                     |
|   + Grid Target Boxes (target_bboxes_grid = target_pixel / stride).                 |
| - Module áp dụng: BboxLoss (CIoU Loss và Distribution Focal Loss - DFL).            |
| - Lý do: DFL biểu diễn khoảng cách ltrb dưới dạng các bin rời rạc (0..15 grid cell).|
+------------------------------------------------------------------------------------+
```

---

## 4. ALGORITHM: TASK-ALIGNED ASSIGNER (TAL)

Lớp [`TaskAlignedAssigner`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L218-L584) thực hiện chiến lược gán nhãn động giữa Ground Truth (GT) và các Anchor Points.

### 4.1. Công Thức Chỉ Số Alignment Metric

Độ tương thích $m_{i, a}$ giữa Ground Truth $i$ và Anchor $a$ được tính bằng kết hợp giữa điểm phân loại (Classification Score) và chất lượng định vị (CIoU):

$$m_{i, a} = s_{i, a}^{\alpha} \times \text{CIoU}(B_{\text{pred}, a}, B_{\text{gt}, i})^{\beta}$$

- $s_{i, a}$: Xác suất phân loại của anchor $a$ đối với lớp nhãn của GT $i$ (sau hàm Sigmoid).
- $\text{CIoU}$: Chỉ số Complete IoU giữa bounding box dự đoán của anchor $a$ và GT $i$ (tính ở không gian `[PIXEL]`).
- Siêu tham số: $\alpha = 0.5$, $\beta = 6.0$. 
- **Ý nghĩa siêu tham số**: Việc đặt $\beta = 6.0 \gg \alpha = 0.5$ khiến cho chỉ số Alignment bị chi phối rất mạnh bởi độ chính xác của Bounding Box. Chỉ những Anchor nào định vị cực kỳ chính xác vật thể mới đạt được điểm metric $m$ cao.

### 4.2. Quy Trình 4 Bước Gán Nhãn TAL

#### Bước 1: Xác định Tập Anchor Ứng Viên Dương (Candidate Positive Selection)
Một anchor $a$ chỉ được xét làm ứng viên dương cho GT $i$ nếu thỏa mãn đồng thời 3 điều kiện:
1. Tâm của Anchor $(x_a, y_a)$ nằm bên trong khung hình của GT $i$ (`select_candidates_in_gts`).
2. GT $i$ thực sự tồn tại (không phải nhãn padding).
3. Anchor $a$ nằm trong Top-$k$ có điểm $m_{i, a}$ cao nhất của GT $i$ (`select_topk_candidates`).
   - Nhánh `o2m`: $topk = 10 \implies$ chọn 10 anchor tốt nhất cho mỗi GT.
   - Nhánh `o2o`: $topk = 1 \implies$ chọn duy nhất 1 anchor tốt nhất cho mỗi GT.

#### Bước 2: Giải Quyết Xung Đột (Assignment Conflict Resolution)
Nếu một Anchor $a$ vô tình lọt vào Top-$k$ của nhiều GT khác nhau (do các vật thể nằm gần hoặc đè lên nhau), hàm `select_highest_overlaps` giải quyết xung đột bằng cách:
**Gán Anchor $a$ cho GT nào có chỉ số CIoU (`overlaps`) cao nhất với nó**.
$$\text{GT}_{\text{assigned}}(a) = \arg\max_{i \in \text{Candidates}(a)} \text{CIoU}(B_{\text{pred}, a}, B_{\text{gt}, i})$$
- *Nguyên tắc*: Một Anchor chỉ được đại diện cho duy nhất 1 GT; nhưng một GT có thể sở hữu nhiều Anchor (ở nhánh `o2m`).

#### Bước 3: Ánh Xạ Nhãn (Target Mapping)
Từ kết quả gán nhãn, tập dữ liệu nhãn được gom từ chiều GT ($M$) sang chiều Anchor ($A$):
- `target_labels`: Vector $(B, A)$ chứa nhãn lớp đối tượng.
- `target_bboxes_pixel`: Tensor $(B, A, 4)$ `[PIXEL]` chứa vị trí bbox mục tiêu.
- `target_scores`: Matrix One-Hot $(B, A, N_{cls})$.

#### Bước 4: Chuẩn Hóa Soft Target (Soft Score Normalization)
Để khuyến khích các anchor có chất lượng định vị tốt hơn đóng góp nhiều hơn vào gradient phân loại, vector one-hot `target_scores` được nhân làm mềm với hệ số metric đã chuẩn hóa:

$$\hat{m}_{i, a} = m_{i, a} \times \frac{\max_{a' \in \text{Pos}(i)} \text{CIoU}(B_{\text{pred}, a'}, B_{\text{gt}, i})}{\max_{a'' \in \text{Pos}(i)} m_{i, a''} + \epsilon}$$

$$T_{\text{cls}, a} = \text{one\_hot}(y_i) \cdot \hat{m}_{i, a}$$

---

## 5. PHÂN TÍCH CHI TIẾT CÁC HÀM MẤT MÁT THÀNH PHẦN

### 5.1. Complete IoU (CIoU) Loss

Được cài đặt trong hàm [`bbox_iou`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L77-L153) và gọi trong [`BboxLoss.forward`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L592-L669).

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{CIoU} = 1 - \left( \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v \right)$$

#### Chi tiết toán học từng thành phần:
1. **Khoảng cách Tâm $\rho^2$**:
   $$\rho^2(b, b^{gt}) = (x_c - x_c^{gt})^2 + (y_c - y_c^{gt})^2$$
   Ép tâm của bounding box dự đoán tiệm cận về tâm của GT.

2. **Đường chéo Box Bao $c^2$**:
   $$c^2 = C_w^2 + C_h^2$$
   với $C_w, C_h$ là chiều rộng và chiều cao của bounding box nhỏ nhất bao phủ cả box dự đoán và GT.

3. **Độ tương đồng Tỷ lệ Khung hình $v$ và Trọng số $\alpha$**:
   $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

#### Chuẩn hóa theo Batch:
$$\mathcal{L}_{\text{iou}}^{\text{branch}} = \frac{\sum_{a \in \text{Pos}} (1 - \text{CIoU}_a) \cdot w_a}{\sum_{a \in \text{Pos}} w_a}$$
với $w_a = \sum_{c} T_{\text{cls}, a, c}$ là tổng trọng số Soft Target của anchor dương $a$.

---

### 5.2. Distribution Focal Loss (DFL)

Được cài đặt trong [`BboxLoss._df_loss`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L671-L699).

#### 1. Đặt Vấn Đề:
Các mô hình hồi quy khoảng cách truyền thống dự đoán một giá trị duy nhất $y \in \mathbb{R}$ đại diện cho khoảng cách từ anchor đến cạnh box. Tuy nhiên, trong thực tế, biên của vật thể thường bị mờ hoặc bị che khuất, dẫn đến sự không chắc chắn (Uncertainty). DFL giải quyết vấn đề này bằng cách dự đoán một **Phân phối Xác suất Rời rạc** trên $\text{reg\_max} = 16$ bin ($0, 1, 2, \dots, 15$ grid cells).

#### 2. Toán Học của DFL:
Cho khoảng cách nhãn thực tế $y \in [0, 15]$ dạng số thực (đã đổi sang không gian `[GRID]`).

1. **Xác định 2 bin ngầm kề cận**:
   $$y_l = \lfloor y \rfloor, \quad y_r = y_l + 1$$
2. **Tính trọng số nội suy (Linear Interpolation Weights)**:
   $$w_l = y_r - y, \quad w_r = y - y_l \quad (w_l + w_r = 1)$$
3. **Tính Loss DFL qua Nội suy Cross-Entropy**:
   $$\mathcal{L}_{\text{dfl}}(P, y) = w_l \cdot \text{CE}(P, y_l) + w_r \cdot \text{CE}(P, y_r)$$
   trong đó $\text{CE}(P, k) = -\log P(k) = -\log \left( \frac{\exp(S_k)}{\sum_{j=0}^{15} \exp(S_j)} \right)$ là Cross-Entropy của logit phân phối $S$ tại bin $k$.

```text
DFL Soft-Label Discretization Concept

Continuous GT Distance: y = 3.7 grid cells
   y_l = 3  (left bin)   --> Weight w_l = 4.0 - 3.7 = 0.3
   y_r = 4  (right bin)  --> Weight w_r = 3.7 - 3.0 = 0.7

DFL Loss = 0.3 * CrossEntropy(Logits, 3) + 0.7 * CrossEntropy(Logits, 4)
```

---

### 5.3. Binary Cross-Entropy (BCE) Classification Loss

Được cài đặt bằng `nn.BCEWithLogitsLoss(reduction="none")` trong [`DetectionLoss`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L729).

$$\mathcal{L}_{\text{cls}}^{\text{branch}} = \frac{1}{\sum_{a \in \text{Pos}} w_a} \sum_{a=1}^{A} \sum_{c=1}^{N_{\text{cls}}} \text{BCE}(\hat{z}_{a, c}, T_{\text{cls}, a, c})$$

với $\text{BCE}(\hat{z}, t) = -t \cdot \log(\sigma(\hat{z})) - (1 - t) \cdot \log(1 - \sigma(\hat{z}))$.

- Loss phân loại được tính trên **toàn bộ $A = 8400$ anchor (cả Foreground lẫn Background)**.
- Với Foreground Anchor: $t = T_{\text{cls}, a, c} \in (0, 1]$ (Soft target).
- Với Background Anchor: $t = 0$.

---

## 6. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 6.1. Bảng Thống Kê Các Thành Phần Loss Và Trọng Số

| Thành Phần Loss | Thuộc Nhánh | Siêu Tham Số Trọng Số | Không Gian Tọa Độ | Mục Tiêu Tối Ưu |
| :--- | :--- | :--- | :--- | :--- |
| **BCE Cls Loss** | `o2m` & `o2o` | $\lambda_{\text{cls}} = 1.0$ | Không phụ thuộc | Phân loại chính xác 80 class |
| **CIoU Box Loss** | `o2m` & `o2o` | $\lambda_{\text{box}} = 7.5$ | `[GRID]` (sau quy đổi) | Tối ưu độ phủ khung hình & tỷ lệ |
| **DFL Reg Loss** | `o2m` & `o2o` | $\lambda_{\text{dfl}} = 1.5$ | `[GRID]` | Học phân phối góc cạnh mượt mà |
| **Branch o2m** | Main Training | $w_{o2m} = 1.0$ | Multiple Pos ($topk=10$) | Tăng tốc độ hội tụ đặc trưng |
| **Branch o2o** | NMS-Free Head | $w_{o2o} = 1.0$ | Single Pos ($topk=1$) | Học loại bỏ trùng lặp không NMS |

---

## 7. KẾT LUẬN CHƯƠNG

Cấu trúc Hàm Mất Mát của NMS-Free Detector là một đỉnh cao trong thiết kế thuật toán học sâu hiện đại. Bằng việc kết hợp sáng tạo giữa **Kiến trúc Dual-Head (o2m dense supervision + o2o sparse NMS-free)**, **Thuật toán gán nhãn động Task-Aligned Assigner (TAL)**, **Sự phân định ranh giới chuẩn xác giữa hai không gian [PIXEL] và [GRID]**, cùng bộ ba loss **CIoU + DFL + BCE**, mô hình không chỉ hội tụ với tốc độ kinh ngạc mà còn đạt được khả năng phát hiện vật thể sắc nét, loại bỏ hoàn toàn sự phụ thuộc vào thuật toán hậu xử lý NMS truyền thống.
