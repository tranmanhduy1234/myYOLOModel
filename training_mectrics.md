# Hướng dẫn đọc chỉ số training của NMSFreeDetector

Tài liệu này mô tả **toàn bộ** chỉ số được `TrainingLogger` (`tb_logger.py`) ghi
ra, kèm ý nghĩa, khoảng giá trị lành mạnh, dấu hiệu bất thường, và — quan
trọng nhất — cách **kết hợp nhiều chỉ số với nhau** để chẩn đoán hành vi học
thực sự của mô hình thay vì chỉ nhìn từng con số rời rạc. Đây là bản mở rộng
sâu hơn của phần loss (mục 1), cộng thêm gradient, weight, EMA, hệ thống,
activation, BatchNorm — tức toàn bộ những gì `TrainingLogger` cung cấp.

Quy ước tần suất ghi trong code: các chỉ số "rẻ" (loss) ghi **mỗi step**; các
chỉ số "đắt" hơn (gradient scalar, weight stats, LR, update ratio) ghi mỗi
`log_interval` step; histogram/BN (đắt nhất) ghi mỗi `histogram_interval`
step. Khi đọc biểu đồ, nhớ trục x của các nhóm khác nhau **không cùng mật độ
điểm** — đừng nhầm "ít điểm hơn" là "ít biến động hơn".

---

## 1. Loss (`log_losses`, `log_loss_ratios`)

### 1.1 Các thành phần (đã mô tả chi tiết ở bản trước, tóm tắt lại để tiện tra cứu)

| Key | Ý nghĩa | Đọc thế nào |
|---|---|---|
| `loss`, `loss_o2m`, `loss_o2o` | Tổng loss / theo nhánh | Xu hướng hội tụ tổng thể; `o2o` học chậm hơn `o2m` ở đầu training là bình thường |
| `o2m or o2o /iou` | CIoU loss (định vị box) | Giảm dần đều là tốt; spike đột ngột = box "nổ" |
| `o2m or o2o /cls` | BCE classification | Có thể chững ở mức > 0 do mất cân bằng lớp, không hẳn là lỗi |
| `o2m or o2o /dfl` | Distribution Focal Loss (độ sắc nét biên box) | Hội tụ nhanh hơn iou vì gần giống bài toán phân loại |
| `o2m or o2o /n_pos` | Số anchor dương sau assignment | `n_pos=0` kéo dài = lỗi dữ liệu/assigner, không phải lỗi mô hình |

### 1.2 `log_loss_ratios` — tỷ trọng đóng góp (chỉ số hay bị bỏ qua nhưng rất hữu ích)

- `o2m_ratio`, `o2o_ratio` = tỷ lệ đóng góp của mỗi nhánh vào `loss` tổng.
  Vì `o2m_weight = o2o_weight = 1.0` mặc định, tỷ lệ này phản ánh **độ khó
  tương đối** giữa hai nhánh tại thời điểm đó chứ không phải do trọng số cấu
  hình. Nếu `o2o_ratio` gần như không đổi ở mức rất thấp (ví dụ luôn < 15%)
  trong khi `o2o/n_pos` vẫn dương bình thường — nhánh o2o có thể đang bị
  "lu mờ" bởi gradient từ o2m thông qua trunk dùng chung, học chậm hơn nhiều
  so với khả năng thực tế. Đây là tín hiệu để cân nhắc tăng `o2o_weight` hoặc
  `topk_o2o` một chút.
- `iou`/`cls`/`dfl` ratio trong `o2m_component_ratios`: nếu một thành phần
  chiếm áp đảo (> 70-80%) tổng loss của nhánh trong suốt nhiều epoch, gain
  tương ứng (`box_gain`/`cls_gain`/`dfl_gain`) có thể đang lệch, khiến
  optimizer "ưu tiên" giảm thành phần đó mà bỏ bê các thành phần còn lại.

---

## 2. Gradient (`log_gradients`)

### 2.1 `Gradients/total_norm`, `Gradients/avg_norm`
- `total_norm`: chuẩn L2 của **toàn bộ** gradient mô hình tại step đó, tính
  **trước** `clip_grad_norm_` — đây là lý do quan trọng nhất khiến chỉ số này
  hữu ích: nó cho biết gradient thật sự lớn tới đâu, không bị clip che giấu.
- `avg_norm`: trung bình trượt 100 step gần nhất của `total_norm` — dùng để
  nhìn xu hướng mượt hơn, tách khỏi nhiễu từng step.

**Cách đọc:**
- `total_norm` liên tục vượt xa `grad_clip_norm` (mặc định 10.0) → clipping
  đang hoạt động thường xuyên, nghĩa là learning rate hiệu dụng bị "cắt bớt"
  liên tục; nếu diễn ra suốt training (không chỉ vài step đầu warmup) nên xem
  lại `lr0` hoặc khởi tạo trọng số.
- `total_norm` giảm dần về gần 0 nhưng `loss` **không** giảm tương ứng → dấu
  hiệu kinh điển của **vanishing gradient** hoặc mô hình đã rơi vào một điểm
  bão hoà cục bộ (plateau) — kết hợp xem `Weights_Stats/*/std` (mục 3) có
  đang co lại không.
- `total_norm` là `NaN`/`Inf` (code đã bắt và log warning riêng qua text
  logger, không phải TensorBoard) — dấu hiệu loss đã phân kỳ, cần dừng và
  kiểm tra dữ liệu batch đó (thường do box/label bất thường lọt qua, hoặc
  `lr` quá cao ngay sau warmup).

### 2.2 `Gradients_RMS/{name}` (theo từng layer)
RMS gradient = `||grad||_2 / sqrt(numel)` — chuẩn hoá theo số phần tử nên **so
sánh được giữa các layer có kích thước khác nhau** (khác với `total_norm` vốn
là tổng toàn cục).

**Cách đọc theo layer:**
- So sánh RMS gradient giữa các stage của `backbone` (stem → stage1 → ... →
  stage4): nếu **giảm dần rõ rệt** khi đi từ layer sâu (gần output) về layer
  nông (gần input) qua nhiều epoch — đây là dấu hiệu vanishing gradient cổ
  điển ở phần đầu mạng; các layer đầu backbone gần như không còn học được gì.
  Vì kiến trúc dùng nhiều residual (`Bottleneck`, `CIB` có shortcut), hiện
  tượng này nên nhẹ hơn nhiều so với plain CNN, nếu vẫn thấy rõ rệt là điểm
  đáng chú ý.
- Nhánh `cls_stem_o2o`/`cls_stem_o2m` so với `reg_stem_o2o`/`reg_stem_o2m` (2
  nhánh trong `ScaleHead`): nếu RMS của nhánh classification luôn thấp hơn
  hẳn nhánh regression suốt training → có thể `cls_gain` đang set thấp hơn
  mức cần thiết so với `box_gain`/`dfl_gain` cho tốc độ học 2 nhánh cân bằng.
- Layer histogram `Gradients/{name}` (khi `do_hist=True`): nhìn hình dạng
  phân phối — gradient phân phối gần Gaussian quanh 0 là bình thường; phân
  phối có 1 vài giá trị outlier cực lớn tách khỏi phần còn lại (long tail) là
  dấu hiệu instability cục bộ, dù `total_norm` trung bình vẫn ổn.

---

## 3. Weight / Bias (`log_weights`)

`Weights_Stats/{name}/{mean,std,rms,max,min}` ghi cho **mọi tham số** (weight
lẫn bias) sau mỗi `optimizer.step()`.

**Cách đọc:**
- `std` của weight một layer **tăng dần không giới hạn** qua nhiều epoch, đặc
  biệt ở layer cuối (`cls_o2o`, `reg_o2o`) → dấu hiệu sớm của weight đang
  "trôi" mất kiểm soát, dù `weight_decay=5e-4` đang áp dụng; nên đối chiếu
  ngay với `Update_Ratio` (mục 4) cùng layer đó.
- `std` collapse về gần 0 (đặc biệt ở nhiều layer cùng lúc) → có thể mô hình
  đang rơi vào **collapse mode** (mọi anchor dự đoán gần giống nhau bất kể
  input) — kết hợp xem `o2m/cls`, `o2o/cls` có đứng yên ở một mức cố định hay
  không để xác nhận.
- `max`/`min` của bias các layer classification (`cls_o2m`, `cls_o2o`): các
  bias này được khởi tạo theo công thức focal-prior riêng cho từng stride
  (`init_stride_bias`) — nên **theo dõi xem sau vài epoch đầu chúng có đang
  dịch chuyển hợp lý** (thường tăng dần lên khi mô hình tự tin hơn về việc có
  object) chứ không đứng yên tuyệt đối (đứng yên tuyệt đối trong hàng trăm
  step đầu có thể là dấu hiệu gradient chưa chảy tới layer này).
- BatchNorm `weight` (gamma) tiến gần 0 ở nhiều kênh cùng lúc trong 1 layer —
  cảnh báo **kênh đó đang "chết"** (channel bị BN triệt tiêu gần như hoàn
  toàn) — nên đối chiếu thêm mục BatchNorm (6) và Activation (8) cùng tên
  layer.

---

## 4. Weight Update Ratio (`log_weight_updates`)

`Update_Ratio/{name} = mean(|Δw| / (|w| + eps))`, `Update_Magnitude/{name} =
||Δw||_2`. Đây là chỉ số **được khuyến nghị rộng rãi (Karpathy et al.)** để
theo dõi "mức độ học thực sự" của từng layer, độc lập với scale tuyệt đối của
gradient hay weight.

**Quy tắc kinh nghiệm phổ biến:** `update_ratio` nên nằm trong khoảng
**~1e-3 đến 1e-2** (tức mỗi step, trọng số thay đổi khoảng 0.1%-1% giá trị
hiện tại). Đây không phải ngưỡng cứng của kiến trúc này mà là kinh nghiệm
chung cho SGD/Adam-family; vẫn nên coi là "khoảng tham chiếu ban đầu" hơn là
chân lý tuyệt đối, và **so layer này với layer khác trong cùng lần train**
đáng tin cậy hơn là so với con số tuyệt đối:

- `update_ratio` liên tục **> 1e-1** ở một layer → layer đó đang bị update
  quá mạnh so với phần còn lại của mạng, thường là dấu hiệu `lr0` quá cao cho
  riêng nhóm tham số này (kể cả khi `total_norm` tổng thể vẫn "bình thường")
  — dễ dẫn tới layer đó bị "quá khớp cục bộ" nhanh hơn phần còn lại.
- `update_ratio` liên tục **< 1e-5** ở một layer trong khi các layer khác vẫn
  ở mức bình thường → layer gần như **không học** (learning rate hiệu dụng
  quá thấp cho nó, hoặc gradient qua nó gần như bằng 0 — xem lại RMS gradient
  layer này ở mục 2.2 để xác nhận nguyên nhân).
- So sánh `update_ratio` giữa **backbone/neck** (`freeze_trunk()` có thể bật
  tắt) và **head**: nếu đang fine-tune với trunk chưa freeze, backbone
  thường có `update_ratio` nhỏ hơn head nhiều lần là hợp lý (trunk đã có
  pretrain, head khởi tạo mới học nhanh hơn); nếu backbone lại update mạnh
  hơn head thì đáng xem lại có đang vô tình fine-tune "quá lực" lên phần đã
  học tốt hay không.

---

## 5. Learning Rate & Weight Decay (`log_learning_rate`)

`Learning_Rate/group_{i}` (2 group: có/không weight decay, xem `get_optimizer`
trong `engine.py`), `Weight_Decay/group_{i}`, `Training/epoch`.

**Cách đọc:**
- Xác nhận đúng hình dạng lịch trình: tăng tuyến tính trong `warmup_epochs`
  đầu, sau đó giảm theo cosine tới `lr0 * lr_min_factor`. Nếu đường LR trên
  TensorBoard không khớp hình dạng này (ví dụ bị "gãy khúc" hoặc giảm đột
  ngột) → có khả năng liên quan tới `skip_lr_sched` trong `engine.py` (AMP
  scaler giảm scale khiến step đó bị bỏ qua `scheduler.step()`) xảy ra quá
  thường xuyên — đối chiếu tần suất này với log warning "gradient NaN/Inf
  trước clip" (mục 2.1) để xác nhận có đang do AMP overflow lặp lại không.
- `Weight_Decay/group_1` (nhóm `no_decay`: bias + tham số 1-D như BN
  weight/bias) phải luôn = 0.0 theo thiết kế trong `get_optimizer` — nếu thấy
  khác 0 trên biểu đồ, đó là dấu hiệu code cấu hình optimizer group đã bị đổi
  khác với thiết kế ban đầu.

---

## 6. EMA (`log_ema`, `log_ema_params`)

`EMA/current_decay`, `EMA/updates`, `EMA/warmup_progress`,
`EMA/param_norm`, `EMA/param_count`.

**Cách đọc:**
- `current_decay` tăng dần từ 0 tới gần `ema_decay` (0.9998) theo đúng công
  thức warmup (`decay * (1 - exp(-updates/warmup_updates))`) — nếu
  `warmup_progress` đã đạt 1.0 rất sớm (do `ema_warmup_updates=2000` nhỏ hơn
  nhiều so với tổng số step) thì EMA gần như hoạt động ở decay tối đa gần
  như suốt quá trình — cần biết điều này khi diễn giải **tại sao `val_loss`
  (tính trên EMA model) có thể "trễ" hơn nhiều so với `train_loss`**: EMA với
  decay cao nghĩa là model dùng để validate là trung bình trượt rất dài của
  hàng chục nghìn step gần nhất, nên **luôn chậm phản ứng hơn train_loss** —
  đây là hành vi kỳ vọng, không phải bug, nhưng cần nhớ khi so sánh 2 đường
  loss.
- `EMA/param_norm` giảm dần liên tục qua nhiều epoch trong khi
  `Weights_Stats/*/rms` của model gốc lại tăng → do decay cao, EMA "chưa bắt
  kịp" xu hướng thật của model gốc; đây là dấu hiệu bình thường ở early
  training, nhưng nếu khoảng cách này không thu hẹp lại về cuối training (khi
  model gốc đã ổn định), nên xem xét `ema_decay` có đang quá cao so với tổng
  số step huấn luyện hay không.

---

## 7. Hệ thống — GPU Memory (`log_gpu_memory`)

`System/GPU_memory_allocated_GB`, `_reserved_GB`, `_max_memory_allocated_GB`,
`_utilization` (= allocated / reserved).

**Cách đọc:**
- Đây là chỉ số **vận hành**, không phản ánh hành vi học, nhưng vẫn ảnh hưởng
  gián tiếp: `max_memory_allocated` tăng dần đều qua nhiều step rồi ổn định ở
  epoch 2 trở đi là bình thường (cache warm-up của CUDA allocator). Nếu tăng
  liên tục không dừng qua nhiều epoch → rò rỉ bộ nhớ (thường do giữ tensor có
  `requires_grad=True` ngoài ý muốn, ví dụ append `loss` thay vì
  `loss.item()` vào một list nào đó ở phần code khác).
- `GPU_memory_utilization` thấp kéo dài (< 0.5) trong khi `reserved` cao —
  bộ nhớ đã cấp phát nhưng không dùng hết, có thể tăng `batch_size` mà không
  lo OOM ngay, nhưng không liên quan tới chất lượng học.

---

## 8. BatchNorm (`log_batchnorm`)

`BN/{layer}/running_mean`, `running_var`, `gamma_mean`, `gamma_std`,
`beta_mean`, `beta_std` — chỉ ghi mỗi `histogram_interval` step vì tốn chi
phí duyệt toàn bộ module.

**Cách đọc:**
- `running_var` tiến rất gần 0 ở một layer → activation đi qua layer đó gần
  như là hằng số (không còn phân biệt được giữa các input khác nhau) — dấu
  hiệu **"BN collapse"**, thường là hệ quả của learning rate quá cao ở giai
  đoạn đầu làm activation bão hoà. Kết hợp xem `gamma_std` cùng layer: nếu
  `gamma_std` cũng rất nhỏ, gần như chắc chắn layer đó đang không đóng góp gì
  cho forward pass.
- `running_mean`/`running_var` dao động mạnh giữa các lần ghi (không hội tụ
  ổn định qua các epoch) ở layer nào đó trong khi các layer khác đã ổn định
  — có thể layer đó đang nhận input có phân phối thay đổi nhiều (ví dụ do vị
  trí trong neck nơi feature map bị concat từ nhiều nguồn khác scale nhau —
  đáng chú ý với `c2f_p4`/`c2f_n5` trong `PAFPN` vì đây là nơi nồng độ nhất
  của việc concat 2 nguồn feature khác tầng).

---

## 9. Activation (`ActivationTracker`, dùng thủ công qua forward hook)

`Activations/{layer}/{mean,std,max,min}` — không tự động bật trong vòng lặp
chính (`train_one_epoch`), cần chủ động gọi `register_hooks(model)` nếu muốn
dùng. Đáng cân nhắc bật tạm thời khi debug sâu, không nên bật thường trực vì
tốn chi phí hook trên mọi Conv2d/BatchNorm2d/SiLU.

**Cách đọc:**
- SiLU activation `mean` tiến về 0 và `std` rất nhỏ ở một layer → **dead
  unit** (layer gần như luôn output ~0 bất kể input, khác với ReLU nhưng
  hiện tượng tương tự vẫn có thể xảy ra ở vùng input rất âm của SiLU).
- `max` activation tăng không kiểm soát qua các layer sâu dần (không có dấu
  hiệu bão hoà) → có thể là tiền đề của gradient/weight explosion xuất hiện
  ở bước sau; nên bật activation tracking **trước khi** thấy `total_norm`
  gradient bất thường, để có dữ liệu truy vết ngược nguyên nhân.

---

## 10. Bảng chẩn đoán chéo (dùng khi có hiện tượng bất thường)

| Hiện tượng quan sát | Nhóm chỉ số cần đối chiếu tiếp | Kết luận khả dĩ |
|---|---|---|
| `loss` không giảm dù train nhiều epoch | `n_pos` (mục 1), `total_norm` gradient (2.1), `update_ratio` (4) | `n_pos≈0` → lỗi dữ liệu/assigner; `total_norm≈0` + `update_ratio≈0` → learning rate quá thấp hoặc optimizer không cập nhật đúng param group |
| `loss` giảm rồi tăng đột ngột (spike/NaN) | `total_norm` trước clip (2.1), `LR schedule` (5), warning NaN/Inf trong text log | Thường là LR đỉnh warmup quá cao hoặc 1 batch chứa box/label bất thường |
| `val_loss` (EMA) tệ hơn hẳn `train_loss` | `EMA/current_decay`, `warmup_progress` (6) | Có thể chỉ là EMA chưa "bắt kịp" do decay cao — không vội kết luận overfit |
| Một vài layer dường như "không học" | RMS gradient theo layer (2.2), `update_ratio` theo layer (4) | Xác nhận cả 2 đều gần 0 → layer chết thật; nếu chỉ update_ratio thấp nhưng RMS gradient bình thường → có thể do weight_decay/lr riêng nhóm đó |
| Nghi ngờ overfit sớm | `Update_Ratio` các layer head so với backbone (4), so `train_loss` vs `val_loss` theo epoch | update_ratio head cao bất thường kéo dài + val_loss tách xa train_loss dần |
| Nghi ngờ 1 kênh/layer "chết" | BN `gamma_std`/`running_var` (8) + Activation `std` cùng layer (9) | Cả 2 cùng gần 0 → gần như chắc chắn kênh/layer đó không đóng góp |

---

## Ghi chú chung khi phân tích

- Không có chỉ số nào nên được đọc **một mình**; hầu hết kết luận đáng tin
  cậy trong bảng trên đều cần ít nhất 2 nhóm chỉ số xác nhận lẫn nhau.
- Vì nhiều chỉ số (histogram, BN) chỉ ghi mỗi `histogram_interval` step
  (mặc định 100), khi debug sự cố xảy ra nhanh (vài chục step) có thể cần
  tạm giảm `histogram_interval` xuống thấp hơn để có đủ điểm dữ liệu quan
  sát, thay vì chỉ dựa vào các chỉ số ghi mỗi step.
- So sánh **theo epoch** (giá trị trung bình) đáng tin cậy hơn cho các kết
  luận dài hạn; so sánh **theo step** phù hợp hơn cho việc bắt các sự cố tức
  thời (spike, NaN, OOM).