# Pipeline training – NMSFreeDetector (YOLOv10-style)

## 1. Cấu trúc project sau khi chỉnh sửa

```
src/
├── backbone_neck.py 
├── blocks.py         
├── head.py           
├── model.py      
├── dataset.py              
├── config.py 
├── train_main.py            
├── test_pipeline.py        
└── train/
    ├── loss.py                  
    ├── ema.py                   
    └── engine.py                 
```

**Lý do di chuyển `loss.py` → `train/loss.py`:** `test_pipeline.py` gốc đã
viết sẵn `from train.loss import DetectionLoss`, tức là code kỳ vọng có
package `train`. Giữ nguyên cấu trúc này thay vì sửa `test_pipeline.py`.

## 2. Chạy training

```bash
python train_main.py \
  --data_dir ./data \
  --nc 80 \
  --epochs 150 \
  --batch_size 32 \
  --img_size 480 \
  --device cuda \
  --ckpt_dir ./checkpoints
```

Toàn bộ hyperparameter nằm trong `config.py` (`TrainConfig`), có thể override
qua CLI hoặc sửa trực tiếp file.

## 3. Các quyết định thiết kế chính

### a) Tách checkpoint "trunk" (backbone+neck) khỏi "head"
Vì model đang **pretrain** và sau này sẽ **đổi HEAD** (đổi nc, đổi reg_max,
hoặc đổi hẳn kiểu output), `model.py` được thêm:

- `model.trunk_state_dict()` / `model.save_trunk(path)` – lưu **riêng**
  backbone+neck, không dính head.
- `model.load_trunk(path)` – nạp lại backbone+neck đã pretrain vào model mới.
- `model.replace_head(nc=..., reg_max=..., strides=...)` – thay hẳn `DetectHead`
  bằng head mới, giữ nguyên trunk.
- `model.freeze_trunk(freeze=True)` – đóng băng backbone+neck khi finetune
  head mới ở vài epoch đầu (tránh phá vỡ đặc trưng đã học), rồi
  `freeze_trunk(False)` để finetune toàn bộ.

`train/engine.py` tự động lưu `best_trunk.pt` mỗi khi có val_loss tốt nhất,
song song với `best.pt` (checkpoint đầy đủ backbone+neck+head+optimizer).

**Quy trình đổi HEAD sau khi pretrain xong:**
```python
from model import NMSFreeDetector

model = NMSFreeDetector(nc=NEW_NC)          # tạo model cho task mới
model.load_trunk("checkpoints/best_trunk.pt")  # nạp backbone+neck đã pretrain
model.replace_head(nc=NEW_NC, reg_max=NEW_REG_MAX)  # (gọi lại cho chắc nếu đổi nc)
model.freeze_trunk(True)                     # tuỳ chọn: freeze vài epoch đầu
```

### b) Dataset: letterbox thay vì resize méo hình
Bản gốc `img.resize((size, size))` làm méo tỉ lệ khung hình, khiến object
detection học sai hình dạng thật của vật thể. Đã thay bằng `letterbox()`:
resize giữ tỉ lệ + pad màu xám (114,114,114) — đúng chuẩn YOLO. Box được map
qua cùng phép biến đổi (scale + pad offset).

### c) Augmentation (chỉ áp dụng tập train)
- Random horizontal flip (mặc định p=0.5)
- Color jitter: brightness/contrast/saturation (p=0.5)
- Tách `train_ds` (augment=True) và `val_ds` (augment=False) trên cùng
  `data_dir`, chia theo cùng bộ index (`split_dataset`) để không rò rỉ dữ liệu.

> Gợi ý mở rộng sau này nếu cần học biểu diễn mạnh hơn ở giai đoạn pretrain:
> Mosaic augmentation (ghép 4 ảnh) và MixUp — nên viết dưới dạng
> `IterableDataset`/wrapper riêng trong `train/augment.py` vì chúng cần truy
> cập nhiều sample cùng lúc, không hợp với `__getitem__` đơn ảnh hiện tại.

### d) Optimizer & LR schedule
- AdamW mặc định (SGD+momentum vẫn hỗ trợ qua `--optimizer sgd`).
- Tách param-group: **không** áp weight-decay lên BatchNorm và bias (chuẩn
  thực hành YOLO), chỉ áp lên trọng số conv/linear.
- LR: warmup tuyến tính (`warmup_epochs`) rồi cosine decay về
  `lr0 * lr_min_factor`.
- Gradient clipping (`grad_clip_norm`, mặc định 10.0) để tránh loss nổ khi
  cls loss lớn lúc đầu train (thấy rõ trong log giá trị cls loss cao vì bias
  init theo prior prob 0.01 — bình thường, sẽ giảm nhanh).

### e) EMA (Exponential Moving Average)
`train/ema.py` implement EMA kiểu Ultralytics: decay warm-up dần theo số
update. Khi có EMA, **validate() và lưu best checkpoint dùng trọng số EMA**
(ổn định hơn model đang được optimizer cập nhật trực tiếp) — rất quan trọng
cho giai đoạn pretrain chạy dài ngày.

### f) Mixed precision (AMP) + gradient scaler
Bật mặc định khi có GPU (`cfg.amp=True`), tự tắt an toàn trên CPU.

### g) Checkpoint & resume
- `last.pt`: checkpoint mỗi epoch (model, optimizer, scheduler, ema, epoch,
  best_val) — dùng để resume training bị gián đoạn (`--resume path/last.pt`).
- `best.pt` / `best_trunk.pt`: theo val_loss tốt nhất.

## 4. Những điểm cần lưu ý / TODO tiếp theo

1. **Validation hiện dùng loss, chưa có mAP.** Với dữ liệu thật, nên viết
   thêm `val.py` tính mAP@0.5 / mAP@0.5:0.95 (dùng `inference_nms_free` có
   sẵn trong `test_pipeline.py` làm điểm khởi đầu decode box, sau đó dùng
   `bbox_iou_plain` trong `loss.py` để match với GT).
2. **`dataset.py`** hiện raise lỗi nếu thiếu ảnh — với dataset lớn crawl từ
   nhiều nguồn nên cân nhắc `try/except` bỏ qua sample lỗi thay vì crash cả
   epoch.
3. **Multi-GPU:** engine hiện là single-device. Nếu cần scale, bọc model
   bằng `nn.parallel.DistributedDataParallel` và dùng `DistributedSampler`
   thay cho `shuffle=True` trong `build_dataloaders`.
4. **`n_pos=0` ở bước đầu** trong log là bình thường nếu ảnh/box quá nhỏ so
   với stride nhỏ nhất (P3 stride=8) — assigner chỉ gán positive khi anchor
   nằm trong GT box; kiểm tra lại nếu thấy `n_pos=0` kéo dài suốt training
   trên dữ liệu thật (dấu hiệu box GT bị lỗi toạ độ hoặc quá nhỏ).