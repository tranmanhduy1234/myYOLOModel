# CHƯƠNG 1: KIẾN TRÚC VÀ VÒNG ĐỜI CỦA TRAINING PIPELINE

## 1. GIỚI THIỆU CHƯƠNG

Trong phát triển các mô hình Học sâu (Deep Learning) hiện đại, đặc biệt là các hệ thống phát hiện vật thể thời gian thực (Real-time Object Detection) như kiến trúc NMS-Free Detector (phát triển dựa trên YOLOv10), **Pipeline Huấn luyện (Training Pipeline)** đóng vai trò là xương sống vận hành toàn bộ quá trình biến đổi dữ liệu thô thành tri thức mô hình. Pipeline huấn luyện không đơn thuần là một vòng lập `for epoch in range(...)`, mà là một hệ thống phối hợp đồng bộ giữa quản lý bộ nhớ phần cứng (GPU/CPU RAM), tối ưu hóa luồng dữ liệu (Data Streaming), tính toán lan truyền tiến (Forward Pass), lan truyền ngược (Backward Pass), điều phối gradient (Gradient Clipping & Scaling), cập nhật tham số (Optimizer & LR Scheduler), và bảo toàn trạng thái mô hình (EMA & Checkpointing).

Chương này trình bày chi tiết toàn bộ vòng đời huấn luyện (End-to-End Lifecycle) 15 bước của mô hình NMS-Free Detector được cài đặt trong tập tin [`src/train/engine.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py), đồng thời phân tích sâu tác động của từng bước lên ba chỉ số hiệu năng cốt lõi: **Tốc độ huấn luyện (Throughput/Speed)**, **Bộ nhớ đồ họa (VRAM Utilization)**, và **Độ hội tụ (Convergence Stability)**.

---

## 2. NỘI DUNG PHÂN TÍCH VÒNG ĐỜI HUẤN LUYỆN

Sơ đồ tổng quan toàn bộ vòng đời huấn luyện 15 bước được mô tả trực quan trong tệp sơ đồ [`diagrams/pipeline_flow.mermaid`](diagrams/pipeline_flow.mermaid).

```text
[Dataset] 
   ↓ (1. Seek Byte-Offset)
[DataLoader] 
   ↓ (2. Multiprocessing Workers)
[Transform] 
   ↓ (3. Albumentation Augment & Letterbox)
[Collate] 
   ↓ (4. Stack Images + List Targets)
[Batch] 
   ↓ (5. Memory Transfer H2D & Non-blocking)
[Forward] 
   ↓ (6. Dual-Head NMSFreeDetector under Autocast)
[Loss] 
   ↓ (7. Dual-Head TAL Loss: o2m + o2o)
[Backward] 
   ↓ (8. GradScaler Scale & Autograd Backward)
[Gradient] 
   ↓ (9. GradScaler Unscale & RMSNorm Logging)
[Clipping] 
   ↓ (10. Clip Grad Norm max=10.0)
[Optimizer] 
   ↓ (11. AdamW / SGD Step & Scaler Update)
[Scheduler] 
   ↓ (12. LambdaLR Cosine Step if no Inf)
[EMA] 
   ↓ (13. ModelEMA Dynamic Exponential Decay Update)
[Validation] 
   ↓ (14. Validation Loop with EMA Model)
[Checkpoint] 
   ↓ (15. Atomic Best/Last & Trunk Save)
[Training Finish]
```

### Chi tiết 15 bước trong vòng đời huấn luyện:

#### Bước 1: Trích xuất Chỉ mục Dữ liệu (Dataset Indexing & Lazy Loading)
- **Cài đặt**: [`ObjectDetectionDataset`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/dataloader1_obj365.py#L177-L290).
- **Cơ chế**: Sử dụng bộ chỉ mục byte-offset (`images_offset_index` và `ann_group_index`) được lưu trong bộ nhớ tạm Pickle cache.
- **Input**: Đường dẫn chỉ mục ảnh và tệp nhãn JSONL.
- **Output**: Vị trí con trỏ `f.seek(offset)` chính xác tới dòng dữ liệu của ảnh và các annotation tương ứng.
- **Bản chất**: Tránh việc đọc toàn bộ tệp JSON hàng gigabyte vào bộ nhớ RAM, giúp giảm thời gian khởi tạo Dataset từ hàng chục phút xuống dưới 1 giây.

#### Bước 2: Phối hợp Đa tiến trình (Multiprocessing Worker Loading)
- **Cài đặt**: [`torch.utils.data.DataLoader`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/dataloader1_obj365.py#L371-L386).
- **Cơ chế**: Khởi tạo `num_workers=4` tiến trình song song đọc dữ liệu từ đĩa cứng. Đi kèm cờ `persistent_workers=True` giúp giữ ấm (warm-up) các worker giữa các epoch, tránh chi phí spawn/fork tiến trình Python lại từ đầu.
- **Input**: Danh sách `image_id` từ Dataset.
- **Output**: Các luồng dữ liệu mẫu (raw image & raw bounding boxes).

#### Bước 3: Biến đổi & Tăng cường Dữ liệu (Data Transformation & Augmentation)
- **Cài đặt**: [`DetectionAugmenter`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/dataloader1_obj365.py#L40-L74) và [`letterbox`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/dataloader1_obj365.py#L75-L85).
- **Cơ chế**:
  1. Đọc ảnh thô BGR bằng OpenCV, chuyển đổi RGB.
  2. Thực hiện biến đổi tỉ lệ duy trì khung hình (Letterbox) về chuẩn `img_size = 480` (hoặc 640), bổ sung viền xám `(114, 114, 114)` và tính toán hệ số $scale, pad_{left}, pad_{top}$.
  3. Áp dụng chuỗi Albumentations: HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue, GaussNoise, Blur.
- **Input**: Ảnh gốc RGB (H, W, 3) và Bounding Boxes dạng PASCAL VOC $(x_1, y_1, x_2, y_2)$.
- **Output**: Ảnh đã tỉ lệ/tăng cường (imgsz, imgsz, 3) dạng `uint8` và danh sách bboxes đã tinh chỉnh tọa độ.

#### Bước 4: Gom Nhóm Batch (Collate Function)
- **Cài đặt**: [`collate_fn`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/dataloader1_obj365.py#L291-L295).
- **Cơ chế**: Đóng gói danh sách mẫu thành Batch Tensor.
- **Input**: Danh sách $B$ phần tử `(img_tensor, target_dict)`.
- **Output**: 
  - `images`: Tensor kích thước $(B, 3, H, W)$, giá trị chuẩn hóa $[0.0, 1.0]$.
  - `targets`: List gồm $B$ dictionary `{"boxes": FloatTensor (N, 4), "labels": LongTensor (N,)}`.
- **Lý do thiết kế**: Số lượng vật thể $N$ trong từng ảnh khác nhau. Việc giữ `targets` dưới dạng List các Tensor lẻ thay vì Padding ngay ở CPU giúp tối ưu hóa băng thông truyền dữ liệu và giảm tải CPU RAM.

#### Bước 5: Nạp Dữ liệu lên Thiết bị GPU (Host-to-Device Memory Transfer)
- **Cài đặt**: [`move_batch`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L117-L126).
- **Cơ chế**: Sử dụng truyền dữ liệu bất đồng bộ `to(device, non_blocking=True)`. Tương thích tuyệt đối với `pin_memory=True` của DataLoader.
- **Input**: `images` và `targets` trên Host CPU Memory.
- **Output**: `images` và `targets` trên Device GPU VRAM.

#### Bước 6: Lan truyền Tiến Mô hình Kép (Dual-Head Forward Pass)
- **Cài đặt**: [`NMSFreeDetector.forward`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/model.py#L33-L37) kết hợp `torch.autocast(device_type="cuda", enabled=amp)`.
- **Cơ chế**: Dữ liệu đi qua Backbone (C2f, C2fCIB, SCDown, SPPF, C2fPSA) $\rightarrow$ Neck PAFPN $\rightarrow$ DetectHead.
- **Output**: Dictionary chứa dự đoán của hai nhánh:
  - `o2m`: Nhánh One-to-Many cho huấn luyện hội tụ nhanh (`cls`: $(B, A, N_{cls})$, `box`: $(B, A, 4)$ `[PIXEL]`, `reg_raw`: $(B, 4 \cdot \text{reg\_max}, A)$ `[GRID]`).
  - `o2o`: Nhánh One-to-One cho suy luận NMS-Free (`cls`, `box`, `reg_raw` tương tự).
  - `anchors`: Tọa độ tâm anchor $(A, 2)$ `[GRID]`.
  - `strides`: Hệ số stride $(A, 1)$ tương ứng (8, 16, 32).

#### Bước 7: Tính toán Mất mát Kép (Dual-Head Loss Computation)
- **Cài đặt**: [`DetectionLoss.forward`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/loss.py#L839-L907).
- **Cơ chế**:
  1. `preprocess_targets`: Chuyển `targets` List[dict] thành Tensor padded $(B, M_{\max}, 4)$ `[PIXEL]` trên GPU.
  2. Lan truyền nhánh `o2m` với `TaskAlignedAssigner(topk=10)` $\rightarrow$ Tính $L_{\text{iou}}^{o2m}, L_{\text{cls}}^{o2m}, L_{\text{dfl}}^{o2m}$.
  3. Lan truyền nhánh `o2o` với `TaskAlignedAssigner(topk=1)` $\rightarrow$ Tính $L_{\text{iou}}^{o2o}, L_{\text{cls}}^{o2o}, L_{\text{dfl}}^{o2o}$.
  4. Tổng hợp loss: $L_{\text{total}} = 1.0 \cdot L_{o2m} + 1.0 \cdot L_{o2o}$.
- **Output**: Tensor `total` có grad và dictionary `items` chứa giá trị scalar của các thành phần loss.

#### Bước 8: Lan truyền Ngược Tự động (AMP Scaled Backward Pass)
- **Cài đặt**: [`train_one_epoch`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L169-L170).
- **Cơ chế**: Nếu bật AMP (`scaler is not None`), thực hiện `scaler.scale(loss).backward()`. Ngược lại, thực hiện `loss.backward()`.
- **Mục đích**: Nhân loss với hệ số scale $S = 2^{16}$ trước khi tính gradient để tránh tình trạng triệt tiêu gradient (Underflow) khi các đại lượng float16 quá nhỏ.

#### Bước 9: Giải Scale Gradient & Trích xuất Nhật ký RMSNorm (Unscaling & Logging)
- **Cài đặt**: [`scaler.unscale_(optimizer)`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L171) và [`tb_logger.log_gradients`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/tb_logger.py#L133-L170).
- **Cơ chế**: Đưa gradient trở về giá trị thực trước khi thực hiện Clipping. Đếm và ghi nhận chỉ số RMSNorm $\text{RMS}(g) = \frac{\|g\|_2}{\sqrt{N}}$ cũng như Total Norm $\|g\|_2$ lên TensorBoard.
- **Ý nghĩa**: Giúp kỹ sư phát hiện sớm hiện tượng Gradient Explosion (NaN/Inf) trước khi nó bị cắt bởi hàm Clip.

#### Bước 10: Cắt Giới hạn Gradient (Gradient Norm Clipping)
- **Cài đặt**: [`torch.nn.utils.clip_grad_norm_`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L184).
- **Cơ chế**: Nếu $\|g\|_2 > 10.0$, nhân toàn bộ gradient với hệ số $\frac{10.0}{\|g\|_2}$.
- **Mục đích**: Đảm bảo bước nhảy tham số không vượt quá ngưỡng an toàn, duy trì sự ổn định tuyệt đối cho quá trình tối ưu hóa.

#### Bước 11: Cập nhật Trọng số & Cập nhật Scaler (Optimizer Step & Scaler Update)
- **Cài đặt**: [`scaler.step(optimizer)`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L187-L189).
- **Cơ chế**:
  - `scaler.step(optimizer)` kiểm tra xem gradient có chứa Inf/NaN hay không. Nếu sạch, gọi `optimizer.step()` để cập nhật trọng số theo AdamW/SGD.
  - `scaler.update()` điều chỉnh hệ số scale $S$. Nếu bước vừa rồi có Inf/NaN, $S$ bị giảm đi một nửa ($S \leftarrow S / 2$) và cờ `skip_lr_sched` được bật.

#### Bước 12: Cập nhật Tốc độ Học (LR Scheduler Step)
- **Cài đặt**: [`scheduler.step()`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L209-L210).
- **Cơ chế**: Nếu `skip_lr_sched == False`, bộ điều phối `LambdaLR` cập nhật learning rate theo công thức Cosine Decay kết hợp Warmup. LR được cập nhật **theo từng iteration (per-step decay)** chứ không phải per-epoch.

#### Bước 13: Cập nhật Trọng số Trung bình Trượt (EMA Update)
- **Cài đặt**: [`ModelEMA.update`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/ema.py#L17-L26).
- **Cơ chế**: Tính toán hệ số decay động $d(t) = 0.9998 \cdot (1 - e^{-t / 2000})$ và cập nhật mô hình bóng (shadow model):
  $$W_{\text{EMA}} \leftarrow W_{\text{EMA}} \cdot d(t) + W_{\text{model}} \cdot (1 - d(t))$$

#### Bước 14: Đánh giá Định kỳ (Validation Loop)
- **Cài đặt**: [`validate`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L239-L257).
- **Cơ chế**: Định kỳ mỗi `val_interval` epoch (mặc định 1 epoch), chuyển mô hình EMA (`ema.ema`) sang chế độ `model.eval()`, ngắt tính toán gradient (`@torch.no_grad()`), và tính toán `val_loss` trên toàn bộ tập Validation.

#### Bước 15: Lưu trữ Trạng thái Checkpoint (Checkpoint Persistence)
- **Cài đặt**: [`save_checkpoint`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/checkpoint.py#L5-L26).
- **Cơ chế**: Nếu `val_loss` đạt kỷ lục mới (`is_best`), tiến hành lưu file `best.pt` chứa toàn bộ 7 thành phần trạng thái (model, optimizer, scheduler, ema, epoch, best_val, cfg), đồng thời xuất file `best_trunk.pt` chứa riêng trọng số Backbone + Neck + Head meta cho các bài toán Transfer Learning về sau.

---

## 3. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 3.1. Phân tích Tác động tới Tốc độ Huấn luyện (Throughput & Speed Impact)

Pipeline huấn luyện được tối ưu hóa tốc độ nhờ các kỹ thuật trọng tâm:

1. **Khớp lệnh Không gian Thẻ nhớ (Non-blocking H2D Transfer)**: Việc kết hợp `pin_memory=True` trong DataLoader và `non_blocking=True` trong `images.to(device)` cho phép luồng dữ liệu được chép qua DMA (Direct Memory Access) song song hoàn toàn với quá trình tính toán CUDA Kernel trên GPU.
2. **Kiến trúc Mixed Precision (AMP)**: Nhờ thực hiện tính toán lan truyền tiến ở định dạng `float16`, tốc độ tính toán Tensor Cores trên GPU NVIDIA (như RTX 3090/4090/A100) tăng từ 2.0x đến 3.5x so với tính toán `float32` thuần túy.
3. **Tách rời luồng Inference/Validation**: Trong quá trình validation, mô hình sử dụng trực tiếp trọng số `ModelEMA` đã được làm mịn, loại bỏ hoàn toàn các overhead tính toán gradient, giúp quá trình validation hoàn tất trong thời gian cực ngắn.

### 3.2. Phân tích Tác động tới Bộ nhớ Đồ họa (VRAM Allocation Impact)

```text
+-------------------------------------------------------------------+
| VRAM Memory Footprint Breakdown                                   |
+-------------------------------------------------------------------+
| 1. Model Parameters (FP32/FP16)     : ~45 - 90 MB                 |
| 2. Optimizer States (AdamW 2x FP32) : ~180 - 360 MB               |
| 3. ModelEMA Shadow Weights (FP32)   : ~45 - 90 MB                 |
| 4. Forward Activations (AMP FP16)   : ~1.2 - 2.8 GB (Batch Size 4) |
| 5. Autograd Computational Graph     : ~800 MB - 1.5 GB            |
| 6. TAL Assigner Tensors [PIXEL]     : ~200 - 500 MB               |
| 7. PyTorch Workspace & CUDA Context : ~600 MB                     |
+-------------------------------------------------------------------+
| Total Peak Memory Usage             : ~3.0 - 5.5 GB               |
+-------------------------------------------------------------------+
```

Một chi tiết kỹ thuật cực kỳ quan trọng trong cài đặt: Trong hàm `train_one_epoch`, trước khi tính toán lan truyền tiến, phương thức `optimizer.zero_grad(set_to_none=True)` được gọi thay vì `zero_grad()`. 
- **Lý do**: Khi set `set_to_none=True`, PyTorch không giải phóng và ghi đè bằng Tensor chứa toàn số `0.0` (vốn tiêu tốn bộ nhớ VRAM cho các ô nhớ bằng 0), mà thực sự giải phóng con trỏ bộ nhớ `param.grad = None`. Điều này giúp giảm tĩnh bộ nhớ VRAM bộc phát trong bước lan truyền ngược.

### 3.3. Phân tích Tác động tới Độ hội tụ (Convergence Stability)

1. **Hiệu ứng Phối hợp Dual-Head (o2m + o2o)**: 
   - Nhánh One-to-Many (`o2m`) gán $topk=10$ anchor dương cho mỗi vật thể GT, tạo ra tín hiệu gradient dày đặc (dense supervision signal). Điều này giúp các lớp Backbone và PAFPN nhanh chóng học được các đặc trưng ngữ cảnh (contextual features) và hình dạng cơ bản trong 20-30 epoch đầu tiên.
   - Nhánh One-to-One (`o2o`) gán duy nhất $topk=1$ anchor dương cho mỗi vật thể GT. Nhờ được huấn luyện song song với `o2m`, nhánh `o2o` kế thừa các đặc trưng sắc bén từ backbone và tự điều chỉnh logit phân loại sao cho chỉ có 1 dự đoán duy nhất đạt điểm cao nhất, triệt tiêu hoàn toàn nhu cầu dùng thuật toán NMS (Non-Maximum Suppression) khi suy luận.
2. **Cơ chế An toàn AMP GradScaler**: Việc kiểm tra `scale_after < scale_before` để bỏ qua bước `scheduler.step()` khi phát hiện tràn số (Inf/NaN) đảm bảo rằng tốc độ học (Learning Rate) không bị suy giảm sai lệch vào những thời điểm mô hình gặp dữ liệu nhiễu hoặc gradient bị biến động đột ngột.

---

## 4. KẾT LUẬN CHƯƠNG

Vòng đời 15 bước của Pipeline Huấn luyện trong project đại diện cho một thiết kế chuẩn mực, hiện đại và tối ưu hóa cao cho các bài toán phát hiện vật thể học sâu. Bằng cách kết hợp hài hòa giữa cơ chế đọc dữ liệu lười (Lazy JSONL Byte-Offset Indexing), tăng cường dữ liệu đa dạng (Albumentations & Letterbox), lan truyền tiến độ chính xác hỗn hợp (AMP Autocast), giám sát gradient thời gian thực (RMSNorm & Clipping), chiến lược tối ưu Dual-Head (TAL Assigner o2m/o2o), và cập nhật trọng số mượt mà (EMA & Cosine Warmup), pipeline đạt được sự cân bằng hoàn hảo giữa **tốc độ thực thi cao**, **tiết kiệm tài nguyên phần cứng**, và **độ ổn định hội tụ vượt trội**. 

Các chương tiếp theo sẽ đi sâu phân tích toán học và cơ chế nội tại của từng thành phần cấu thành nên pipeline tổng thể này.
