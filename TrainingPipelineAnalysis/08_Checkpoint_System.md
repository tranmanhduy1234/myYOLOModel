# CHƯƠNG 8: CƠ CHẾ QUẢN LÝ CHECKPOINT VÀ KHÔI PHỤC TRẠNG THÁI (CHECKPOINT SYSTEM)

## 1. GIỚI THIỆU CHƯƠNG

Trong quá trình huấn luyện các mô hình Học sâu kéo dài nhiều giờ hoặc nhiều ngày trên các cụm máy tính cao cấp, **Hệ thống Quản lý Checkpoint (Checkpoint System)** đóng vai trò là "lưới an toàn" bảo vệ toàn bộ tiến trình tính toán. Checkpoint không chỉ đơn thuần là việc ghi đĩa các số thực trọng số của mạng thần kinh, mà là việc **đóng gói trạng thái tĩnh và động (Full State Persistence)** của toàn bộ hệ sinh thái huấn luyện.

Một hệ thống checkpoint thiếu sót (ví dụ: chỉ lưu trọng số mô hình mà quên lưu trạng thái của Optimizer hay GradScaler) sẽ khiến quá trình huấn luyện nối tiếp (**Resume Training**) bị mất ổn định nghiêm trọng: Optimizer bị reset động lượng về 0, tốc độ học bị nhảy cóc, và mô hình gặp hiện tượng đứt gãy đường cong hội tụ (Loss Spike).

Chương này phân tích kiến trúc quản lý checkpoint được cài đặt trong tệp [`src/utils/checkpoint.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/checkpoint.py), cơ chế lưu trữ **7 Thành phần Trạng thái**, kỹ thuật trích xuất phần thân **Trunk State-Dict (`best_trunk.pt`)** phục vụ cho Học chuyển giao (Transfer Learning), và quy trình tiếp tục huấn luyện (Resume Training) hoàn hảo.

---

## 2. NỘI DUNG PHÂN TÍCH HỆ THỐNG CHECKPOINT

### 2.1. Cấu Trúc Đóng Gói Checkpoint (Full State Dictionary Structure)

Tệp [`src/utils/checkpoint.py`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/utils/checkpoint.py#L5-L26) định nghĩa hàm `save_checkpoint`:

```python
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema,
    epoch: int,
    best_val: float,
    cfg,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "cfg": cfg.__dict__,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)
```

```text
+------------------------------------------------------------------------------------+
| Full State Checkpoint Dictionary Layout (.pt)                                     |
+------------------------------------------------------------------------------------+
| 1. "model"     : Dict[str, Tensor] -> State-dict trọng số hiện tại của NMSFreeDetector|
| 2. "optimizer" : Dict              -> State-dict động lượng (m_t, v_t) của AdamW    |
| 3. "scheduler" : Dict              -> State-dict số bước lặp của LambdaLR Scheduler |
| 4. "ema"       : Dict[str, Tensor] -> State-dict trọng số trung bình mượt ModelEMA  |
| 5. "epoch"     : int               -> Epoch chỉ số hiện tại (0..99)                |
| 6. "best_val"  : float             -> Giá trị Loss Validation tốt nhất từ trước     |
| 7. "cfg"       : Dict              -> Cấu hình siêu tham số toàn bộ TrainConfig     |
+------------------------------------------------------------------------------------+
```

#### Phân tích Chi tiết 7 Thành phần:

1. **`model` (`model.state_dict()`)**: Lưu trữ toàn bộ trọng số (Weights) và các tham số thống kê BatchNorm (`running_mean`, `running_var`) của mô hình huấn luyện chính.
2. **`optimizer` (`optimizer.state_dict()`)**: **Cực kỳ quan trọng**. Đối với AdamW, thành phần này chứa hai vector động lượng $m_t$ (First Moment) và $v_t$ (Second Moment) của từng tham số trong mô hình. Nếu không nạp lại thành phần này khi Resume, AdamW sẽ coi như bắt đầu từ $t=0$, tính toán lại $m_t, v_t$ từ đầu, làm biến động mạnh các bước cập nhật trọng số tiếp theo.
3. **`scheduler` (`scheduler.state_dict()`)**: Lưu trữ vị trí step hiện tại của bộ điều phối tốc độ học. Đảm bảo khi resume, LR tiếp tục suy giảm theo đúng đường cong Cosine Decay mà không bị nhảy về $\text{lr}_0$.
4. **`ema` (`ema.state_dict()`)**: Lưu trữ trọng số mượt của mô hình bóng `ModelEMA`. Giúp giữ nguyên chất lượng đánh giá validation mô hình EMA sau khi resume.
5. **`epoch`**: Ghi nhận chỉ số epoch vừa hoàn thành.
6. **`best_val`**: Kỷ lục loss validation tốt nhất đạt được, dùng làm mốc so sánh cho các epoch tiếp theo.
7. **`cfg`**: Ghi lại toàn bộ từ điển cấu hình `TrainConfig`, đảm bảo tính minh bạch tuyệt đối về siêu tham số đã dùng để tạo ra checkpoint đó.

---

### 2.2. Chiến Lược Lưu Checkpoint Định Kỳ Và Kỷ Lục (Best vs Last Checkpoint)

Trong hàm [`run_training`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/train/engine.py#L320-L331):

```python
is_best = (val_loss is not None) and (val_loss < best_val)
if is_best:
    best_val = val_loss
    save_checkpoint(os.path.join(cfg.ckpt_dir, "best.pt"),
                    model, optimizer, scheduler, ema, epoch, best_val, cfg)
    (ema.ema if ema is not None else model).save_trunk(
        os.path.join(cfg.ckpt_dir, "best_trunk.pt")
    )
    logger.info(f"[epoch {epoch}] -> best checkpoint mới (val_loss={best_val:.4f})")

if not cfg.save_best_only:
    save_checkpoint(os.path.join(cfg.ckpt_dir, "last.pt"),
                    model, optimizer, scheduler, ema, epoch, best_val, cfg)
```

#### Hai tệp checkpoint chính được duy trì:
1. **`best.pt`**: Chỉ được ghi đè khi `val_loss` đạt kỷ lục nhỏ hơn `best_val`. Tệp này đại diện cho mô hình có khả năng tổng quát hóa cao nhất.
2. **`last.pt`**: Được ghi đè ở cuối mỗi epoch (khi `save_best_only = False`). Tệp này lưu trữ trạng thái mới nhất của quá trình huấn luyện, đóng vai trò là điểm khôi phục tức thì nếu hệ thống bị ngắt điện hoặc gặp sự cố bất ngờ.

---

### 2.3. Trích Xuất Trunk State-Dict (`best_trunk.pt`) Cho Transfer Learning

Một thiết kế kiến trúc rất độc đáo được cài đặt trong [`NMSFreeDetector.trunk_state_dict`](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/model.py#L39-L56):

```python
def trunk_state_dict(self):
    return {
        "backbone": self.backbone.state_dict(),
        "neck": self.neck.state_dict(),
        "head": self.head.state_dict(),
        "meta": {
            "nc": self.nc,
            "reg_max": self.reg_max,
            "backbone_w": self.backbone_w,
            "backbone_n": self.backbone_n,
            "neck_n": self.neck_n,
            "neck_chs": self.neck_chs,
            "strides": self.strides,
        },
    }

def save_trunk(self, path):
    torch.save(self.trunk_state_dict(), path)
```

#### Tác dụng của `best_trunk.pt`:
Khi huấn luyện xong mô hình NMS-Free Detector trên tập dữ liệu lớn (như Object365), người dùng thường có nhu cầu chuyển giao tri thức (**Transfer Learning / Fine-tuning**) sang một bài toán mới với số lượng class khác ($nc_{\text{new}} \ne 80$, ví dụ: bài toán nhận diện khuôn mặt hoặc bài toán phát hiện xe cộ với 5 classes).

Tệp `best_trunk.pt`:
- Loại bỏ hoàn toàn các thông tin cồng kềnh của Optimizer, Scheduler hay trạng thái Epoch.
- Trích xuất riêng phần bộ trích xuất đặc trưng **Backbone + Neck** kèm metadata cấu hình mạng.
- Cho phép hàm `replace_head` trong `model.py` dễ dàng thay thế `DetectHead` mới, đóng băng Backbone (`freeze_trunk()`) và tiến hành fine-tune trên dữ liệu mới một cách cực kỳ gọn nhẹ.

---

### 2.4. Phân Tích Quy Trình Resume Training (Khôi Phục Hoàn Hảo)

Khi cấu hình `cfg.resume = "./checkpoints/last.pt"`, hàm `load_checkpoint` thực hiện khôi phục trạng thái:

```python
start_epoch, best_val = load_checkpoint(
    cfg.resume, model, optimizer, scheduler, ema, map_location=device
)
start_epoch += 1  # Tiếp tục từ epoch tiếp theo
```

#### Thứ tự khôi phục an toàn:
1. `model.load_state_dict(ckpt["model"])`: Nạp lại toàn bộ trọng số mô hình.
2. `optimizer.load_state_dict(ckpt["optimizer"])`: Khôi phục lại trạng thái động lượng AdamW.
3. `scheduler.load_state_dict(ckpt["scheduler"])`: Đưa chỉ số bước lặp trong LambdaLR về đúng vị trí.
4. `ema.load_state_dict(ckpt["ema"])`: Nạp lại trọng số mượt cho mô hình bóng EMA.
5. Trả về `start_epoch` và `best_val`, tiếp tục vòng lặp huấn luyện từ `start_epoch` đến `cfg.epochs` mà **không bị mất nét bất kỳ thông số nào**.

---

## 3. ĐÁNH GIÁ VÀ GIẢI THÍCH CHUYÊN SÂU

### 3.1. Bảng So Sánh Các Định Dạng Checkpoint Trong Dự Án

| Tệp Checkpoint | Dung Lượng Tương Đối | Thành Phần Lưu Trữ | Mục Đích Sử Dụng |
| :--- | :--- | :--- | :--- |
| **`best.pt`** | Lớn (~140 - 280 MB) | Full 7 thành phần (Model+Opt+Sched+EMA+Meta) | Khôi phục huấn luyện tốt nhất / Evaluate |
| **`last.pt`** | Lớn (~140 - 280 MB) | Full 7 thành phần (Trạng thái Epoch cuối) | Resume huấn luyện khi sự cố ngắt đứt |
| **`best_trunk.pt`** | Nhỏ (~40 - 90 MB) | Backbone + Neck + Meta config | Transfer Learning / Thay Head sang dataset mới |
| **Model Only Pt** | Nhỏ (~45 - 90 MB) | Chỉ `model.state_dict()` | Deploy suy luận (Inference / Export ONNX) |

---

## 4. KẾT LUẬN CHƯƠNG

Hệ thống quản lý Checkpoint trong dự án thể hiện sự hoàn thiện kỹ thuật cao và tư duy thiết kế phần mềm học sâu chuyên nghiệp. Bằng cách thực thi cơ chế **Đóng gói Trạng thái 7 Thành phần (Full State Persistence)**, phân tách minh bạch giữa tệp khôi phục huấn luyện (`best.pt`, `last.pt`) và tệp học chuyển giao (`best_trunk.pt`), hệ thống vừa đảm bảo tính an toàn tuyệt đối cho quá trình huấn luyện dài ngày, vừa mang lại sự linh hoạt tối đa cho việc tái sử dụng trọng số mô hình trong các bài toán thực tế về sau.
