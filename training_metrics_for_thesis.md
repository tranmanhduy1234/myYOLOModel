# Phân tích chỉ số huấn luyện mô hình NMSFreeDetector

## 1. Tổng quan

Trong quá trình huấn luyện mô hình phát hiện đối tượng NMSFreeDetector, việc giám sát và phân tích các chỉ số huấn luyện đóng vai trò quan trọng trong việc đánh giá hiệu quả học tập của mô hình. Tài liệu này trình bày hệ thống các chỉ số được ghi nhận bởi `TrainingLogger`, bao gồm ý nghĩa, khoảng giá trị tham chiếu và phương pháp kết hợp nhiều chỉ số để chẩn đoán hành vi học của mô hình.

Các chỉ số được ghi nhận với tần suất khác nhau phụ thuộc vào chi phí tính toán: các chỉ số loss được ghi mỗi step, các chỉ số gradient và trọng số được ghi theo `log_interval`, và các histogram phân phối được ghi theo `histogram_interval`.

## 2. Chỉ số Loss

### 2.1 Các thành phần Loss

Hệ thống ghi nhận các thành phần loss sau:

| Chỉ số | Ý nghĩa | Đặc điểm quan sát |
|--------|---------|-------------------|
| `loss`, `loss_o2m`, `loss_o2o` | Tổng loss và loss theo từng nhánh | Xu hướng hội tụ tổng thể; nhánh o2o thường học chậm hơn o2m ở giai đoạn đầu |
| `o2m or o2o /iou` | CIoU loss (định vị bounding box) | Giảm dần đều là tích cực; biến động đột ngột cho thấy box đang bất ổn |
| `o2m or o2o /cls` | BCE classification loss | Có thể duy trì ở mức cao do mất cân bằng lớp, không nhất thiết là lỗi |
| `o2m or o2o /dfl` | Distribution Focal Loss (độ sắc nét biên box) | Hội tụ nhanh hơn iou vì cấu trúc tương tự bài toán phân loại |
| `o2m or o2o /n_pos` | Số anchor dương sau assignment | Giá trị 0 kéo dài cho thấy lỗi dữ liệu hoặc assigner, không phải lỗi mô hình |

### 2.2 Tỷ trọng đóng góp Loss

Hệ thống ghi nhận tỷ lệ đóng góp của mỗi nhánh vào tổng loss (`o2m_ratio`, `o2o_ratio`). Với trọng số mặc định `o2m_weight = o2o_weight = 1.0`, tỷ lệ này phản ánh độ khó tương đối giữa hai nhánh tại thời điểm đó.

Nếu tỷ lệ o2o duy trì ở mức thấp (ví dụ < 15%) trong khi số anchor dương vẫn bình thường, nhánh o2o có thể đang bị "lu mờ" bởi gradient từ nhánh o2m thông qua phần trunk dùng chung. Đây là tín hiệu để cân nhắc điều chỉnh `o2o_weight` hoặc `topk_o2o`.

Tương tự, nếu một thành phần loss (iou, cls, dfl) chiếm áp đảo (> 70-80%) tổng loss của nhánh trong suốt nhiều epoch, các hệ số gain tương ứng (`box_gain`, `cls_gain`, `dfl_gain`) có thể đang lệch, khiến optimizer ưu tiên giảm thành phần đó mà bỏ bê các thành phần còn lại.

## 3. Chỉ số Gradient

### 3.1 Chuẩn Gradient toàn cục

Chỉ số `Gradients/total_norm` đo chuẩn L2 của toàn bộ gradient mô hình tại mỗi step, được tính trước khi áp dụng `clip_grad_norm_`. Đây là chỉ số quan trọng vì nó phản ánh giá trị gradient thực sự mà không bị che giấu bởi clipping. Chỉ số `Gradients/avg_norm` là trung bình trượt 100 step gần nhất, giúp quan sát xu hướng mượt mà hơn.

Giải thích các tình huống:
- `total_norm` liên tục vượt xa `grad_clip_norm` (mặc định 10.0): Clipping đang hoạt động thường xuyên, learning rate hiệu dụng bị cắt bớt liên tục. Nếu diễn ra suốt training, cần xem xét lại `lr0` hoặc khởi tạo trọng số.
- `total_norm` giảm dần về gần 0 nhưng `loss` không giảm tương ứng: Dấu hiệu vanishing gradient hoặc mô hình rơi vào điểm bão hòa cục bộ.
- `total_norm` là NaN/Inf: Loss đã phân kỳ, cần dừng và kiểm tra dữ liệu batch.

### 3.2 RMS Gradient theo Layer

Chỉ số `Gradients_RMS/{name}` đo RMS gradient = `||grad||_2 / sqrt(numel)`, chuẩn hóa theo số phần tử nên có thể so sánh giữa các layer có kích thước khác nhau.

Phân tích theo layer:
- So sánh RMS gradient giữa các stage của backbone (stem → stage1 → ... → stage4): Nếu giảm dần rõ rệt từ layer sâu về layer nông qua nhiều epoch, đây là dấu hiệu vanishing gradient ở phần đầu mạng. Với kiến trúc dùng nhiều residual (Bottleneck, CIB có shortcut), hiện tượng này nên nhẹ hơn nhiều so với plain CNN.
- So sánh nhánh `cls_stem_o2o`/`cls_stem_o2m` với `reg_stem_o2o`/`reg_stem_o2m` trong `ScaleHead`: Nếu RMS của nhánh classification luôn thấp hơn hẳn nhánh regression suốt training, có thể `cls_gain` đang set thấp hơn mức cần thiết so với `box_gain`/`dfl_gain`.

## 4. Chỉ số Trọng số

### 4.1 Thống kê Trọng số

Chỉ số `Weights_Stats/{name}/{mean,std,rms,max,min}` ghi nhận cho mọi tham số (weight và bias) sau mỗi `optimizer.step()`.

Giải thích các tình huống:
- `std` của weight một layer tăng dần không giới hạn qua nhiều epoch, đặc biệt ở layer cuối (`cls_o2o`, `reg_o2o`): Dấu hiệu sớm của weight đang "trôi" mất kiểm soát. Cần đối chiếu ngay với `Update_Ratio` cùng layer đó.
- `std` collapse về gần 0 ở nhiều layer cùng lúc: Có thể mô hình đang rơi vào **collapse mode** (mọi anchor dự đoán gần giống nhau bất kể input). Kết hợp xem `o2m/cls`, `o2o/cls` có đứng yên ở một mức cố định hay không để xác nhận.
- `max`/`min` của bias các layer classification (`cls_o2m`, `cls_o2o`): Các bias này được khởi tạo theo công thức focal-prior riêng cho từng stride (`init_stride_bias`). Nên theo dõi xem sau vài epoch đầu chúng có đang dịch chuyển hợp lý (thường tăng dần lên khi mô hình tự tin hơn về việc có object) chứ không đứng yên tuyệt đối.
- BatchNorm `weight` (gamma) tiến gần 0 ở nhiều kênh cùng lúc trong 1 layer: Cảnh báo **kênh đó đang "chết"** (channel bị BN triệt tiêu gần như hoàn toàn).

### 4.2 Tỷ lệ Cập nhật Trọng số

Chỉ số `Update_Ratio/{name} = mean(|Δw| / (|w| + eps))` và `Update_Magnitude/{name} = ||Δw||_2` được khuyến nghị rộng rãi để theo dõi mức độ học thực sự của từng layer, độc lập với scale tuyệt đối của gradient hay weight.

Quy tắc tham chiếu: `update_ratio` nên nằm trong khoảng **~1e-3 đến 1e-2** (tức mỗi step, trọng số thay đổi khoảng 0.1%-1% giá trị hiện tại). Đây là kinh nghiệm chung cho các optimizer dạng SGD/Adam-family. So sánh giữa các layer trong cùng lần train đáng tin cậy hơn so với so sánh với giá trị tuyệt đối.

Phân tích các tình huống:
- `update_ratio` liên tục **> 1e-1** ở một layer: Layer đó đang bị update quá mạnh so với phần còn lại của mạng, thường là dấu hiệu `lr0` quá cao cho riêng nhóm tham số này. Dễ dẫn tới layer đó bị "quá khớp cục bộ" nhanh hơn phần còn lại.
- `update_ratio` liên tục **< 1e-5** ở một layer trong khi các layer khác vẫn ở mức bình thường: Layer gần như **không học** (learning rate hiệu dụng quá thấp cho nó, hoặc gradient qua nó gần như bằng 0).
- So sánh `update_ratio` giữa **backbone/neck** và **head**: Nếu đang fine-tune với trunk chưa freeze, backbone thường có `update_ratio` nhỏ hơn head nhiều lần là hợp lý (trunk đã có pretrain, head khởi tạo mới học nhanh hơn).

## 5. Các chỉ số Hệ thống

### 5.1 Learning Rate và Weight Decay

Hệ thống theo dõi `Learning_Rate/group_{i}` với 2 nhóm: có weight decay và không weight decay (bias và tham số 1-D như BN weight/bias). Chỉ số `Weight_Decay/group_{i}` và `Training/epoch` cũng được ghi nhận.

Phân tích lịch trình học:
- Xác nhận đúng hình dạng lịch trình: tăng tuyến tính trong `warmup_epochs` đầu, sau đó giảm theo cosine tới `lr0 * lr_min_factor`.
- `Weight_Decay/group_1` (nhóm `no_decay`) phải luôn bằng 0.0 theo thiết kế trong `get_optimizer`.

### 5.2 GPU Memory

Chỉ số `System/GPU_memory_allocated_GB`, `_reserved_GB`, `_max_memory_allocated_GB`, và `_utilization` (= allocated / reserved) phục vụ mục đích vận hành.

Phân tích:
- `max_memory_allocated` tăng dần đều qua nhiều step rồi ổn định ở epoch 2 trở đi là bình thường (cache warm-up của CUDA allocator). Nếu tăng liên tục không dừng qua nhiều epoch có thể chỉ ra rò rỉ bộ nhớ.
- `GPU_memory_utilization` thấp kéo dài (< 0.5) trong khi `reserved` cao cho thấy bộ nhớ đã cấp phát nhưng không dùng hết, có thể xem xét tăng `batch_size`.

## 6. Exponential Moving Average (EMA)

Chỉ số EMA bao gồm `EMA/current_decay`, `EMA/updates`, `EMA/warmup_progress`, `EMA/param_norm`, và `EMA/param_count`.

Phân tích hành vi EMA:
- `current_decay` tăng dần từ 0 tới gần `ema_decay` (0.9998) theo công thức warmup (`decay * (1 - exp(-updates/warmup_updates))`).
- Khi `warmup_progress` đạt 1.0, EMA gần như hoạt động ở decay tối đa suốt quá trình. Điều này giải thích **tại sao `val_loss` (tính trên EMA model) có thể "trễ" hơn nhiều so với `train_loss`**: EMA với decay cao nghĩa là model dùng để validate là trung bình trượt rất dài của hàng chục nghìn step gần nhất, nên luôn chậm phản ứng hơn train_loss. Đây là hành vi kỳ vọng, không phải lỗi.
- `EMA/param_norm` giảm dần liên tục qua nhiều epoch trong khi `Weights_Stats/*/rms` của model gốc lại tăng: Do decay cao, EMA "chưa bắt kịp" xu hướng thật của model gốc. Đây là dấu hiệu bình thường ở early training, nhưng nếu khoảng cách này không thu hẹp lại về cuối training (khi model gốc đã ổn định), nên xem xét `ema_decay` có đang quá cao so với tổng số step huấn luyện hay không.

## 7. Batch Normalization

Chỉ số BatchNorm bao gồm `BN/{layer}/running_mean`, `running_var`, `gamma_mean`, `gamma_std`, `beta_mean`, `beta_std`, được ghi mỗi `histogram_interval` step do chi phí duyệt toàn bộ module.

Phân tích:
- `running_var` tiến rất gần 0 ở một layer: Activation đi qua layer đó gần như là hằng số (không còn phân biệt được giữa các input khác nhau) — dấu hiệu **"BN collapse"**, thường là hệ quả của learning rate quá cao ở giai đoạn đầu làm activation bão hòa. Kết hợp xem `gamma_std` cùng layer: nếu `gamma_std` cũng rất nhỏ, gần như chắc chắn layer đó đang không đóng góp gì cho forward pass.
- `running_mean`/`running_var` dao động mạnh giữa các lần ghi (không hội tụ ổn định qua các epoch) ở layer nào đó trong khi các layer khác đã ổn định: Có thể layer đó đang nhận input có phân phối thay đổi nhiều (ví dụ do vị trí trong neck nơi feature map bị concat từ nhiều nguồn khác scale nhau — đáng chú ý với `c2f_p4`/`c2f_n5` trong `PAFPN` vì đây là nơi nồng độ nhất của việc concat 2 nguồn feature khác tầng).

## 8. Hàm kích hoạt (Activation)

Chỉ số `Activations/{layer}/{mean,std,max,min}` được ghi nhận qua `ActivationTracker` sử dụng forward hook. Chỉ số này không tự động bật trong vòng lặp chính (`train_one_epoch`), cần chủ động gọi `register_hooks(model)` nếu muốn sử dụng.

Phân tích:
- SiLU activation `mean` tiến về 0 và `std` rất nhỏ ở một layer: **Dead unit** (layer gần như luôn output ~0 bất kể input, khác với ReLU nhưng hiện tượng tương tự vẫn có thể xảy ra ở vùng input rất âm của SiLU).
- `max` activation tăng không kiểm soát qua các layer sâu dần (không có dấu hiệu bão hòa): Có thể là tiền đề của gradient/weight explosion xuất hiện ở bước sau.

## 9. Phương pháp chẩn đoán chéo

Bảng sau tổng hợp các phương pháp kết hợp nhiều chỉ số để chẩn đoán các hiện tượng bất thường trong quá trình huấn luyện:

| Hiện tượng quan sát | Nhóm chỉ số cần đối chiếu | Kết luận khả dĩ |
|---------------------|---------------------------|-----------------|
| `loss` không giảm dù train nhiều epoch | `n_pos`, `total_norm` gradient, `update_ratio` | `n_pos≈0` → lỗi dữ liệu/assigner; `total_norm≈0` + `update_ratio≈0` → learning rate quá thấp hoặc optimizer không cập nhật đúng param group |
| `loss` giảm rồi tăng đột ngột (spike/NaN) | `total_norm` trước clip, LR schedule, warning NaN/Inf trong log | Thường là LR đỉnh warmup quá cao hoặc 1 batch chứa box/label bất thường |
| `val_loss` (EMA) tệ hơn hẳn `train_loss` | `EMA/current_decay`, `warmup_progress` | Có thể chỉ là EMA chưa "bắt kịp" do decay cao — không vội kết luận overfit |
| Một vài layer dường như "không học" | RMS gradient theo layer, `update_ratio` theo layer | Xác nhận cả 2 đều gần 0 → layer chết thật; nếu chỉ update_ratio thấp nhưng RMS gradient bình thường → có thể do weight_decay/lr riêng nhóm đó |
| Nghi ngờ overfit sớm | `Update_Ratio` các layer head so với backbone, so `train_loss` vs `val_loss` theo epoch | update_ratio head cao bất thường kéo dài + val_loss tách xa train_loss dần |
| Nghi ngờ 1 kênh/layer "chết" | BN `gamma_std`/`running_var` + Activation `std` cùng layer | Cả 2 cùng gần 0 → gần như chắc chắn kênh/layer đó không đóng góp |

## 10. Kết luận

Việc phân tích các chỉ số huấn luyện của mô hình NMSFreeDetector đòi hỏi kết hợp đồng thời nhiều nguồn thông tin khác nhau. Không có chỉ số nào nên được đọc độc lập; hầu hết các kết luận đáng tin cậy đều cần ít nhất hai nhóm chỉ số xác nhận lẫn nhau.

Các chỉ số cần ưu tiên theo dõi bao gồm:
- **Loss và các thành phần**: Đánh giá tổng thể quá trình học
- **Gradient (total_norm và RMS theo layer)**: Phát hiện vanishing/exploding gradient
- **Update Ratio**: Đánh giá mức độ học thực sự của từng layer
- **EMA**: Hiểu sự khác biệt giữa train và validation loss
- **BatchNorm và Activation**: Phát hiện dead unit và channel

Khi debug các sự cố xảy ra nhanh (vài chục step), có thể cần giảm `histogram_interval` để có đủ điểm dữ liệu quan sát. So sánh theo epoch đáng tin cậy hơn cho các kết luận dài hạn; so sánh theo step phù hợp hơn cho việc bắt các sự cố tức thời.
