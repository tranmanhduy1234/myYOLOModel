# Ghi chú tổ chức lại thư mục utils

## Cấu trúc mới

```
src/utils/
    __init__.py        # export gọn: set_seed, save/load_checkpoint, setup_logging, TrainingLogger, ...
    seed.py             # set_seed
    checkpoint.py       # save_checkpoint / load_checkpoint / load_model_only
    logging_setup.py    # setup_logging (Python text logging, khác TensorBoard)
    tb_logger.py         # TrainingLogger (hợp nhất, xem bên dưới) + TimeTracker/ActivationTracker/LossSmoother
docs/
    TENSORBOARD_GUIDE.md # hướng dẫn dùng (thay HYPERPARAMETERS_TRACKING.md + training_log.md)
    REORG_NOTES.md        # file này
```

## Việc đã làm với từng file cũ

- **`tb_logger.py` (cũ) + `training_logger.py`** → gộp làm một trong `src/utils/tb_logger.py`.
  Hai file này định nghĩa gần như cùng một bộ chức năng (log gradient, log
  weight, log loss...) với 2 API khác nhau — giữ cả hai sẽ luôn có nguy cơ
  sửa một bên mà quên bên kia. Bản gộp giữ **API dạng class `TrainingLogger`**
  của `training_logger.py` (vì có `log_interval`/`histogram_interval` để
  kiểm soát chi phí ghi log) và giữ **4 biểu đồ gộp nhiều-đường** của
  `log_loss_items` (dễ so sánh trực quan hơn ghi từng scalar riêng lẻ).

- **Đã sửa 1 lỗi tiềm ẩn**: `log_weight_updates`/`logWeight_update_ratio`
  bản cũ khớp `prev_params` với tham số hiện tại **theo vị trí (index)**
  trong `named_parameters()`, trong khi `prev_params` chỉ chứa các tham số
  có `grad is not None` tại thời điểm chụp. Nếu một layer chưa có gradient ở
  bước đó, toàn bộ các cặp tên/giá trị phía sau bị lệch. Bản mới dùng
  `Dict[name, tensor]` nên khớp đúng theo tên, không lệch.

- **`state_dict_handle.py`** → tách thành 2 file theo đúng 2 mối quan tâm
  khác nhau: `seed.py` (chỉ `set_seed`) và `checkpoint.py` (save/load
  checkpoint). Nhân tiện đổi tên tham số `schedular` → `scheduler` (lỗi
  chính tả trong bản gốc).

- **`log_setup.py`** → giữ nguyên logic, đổi tên thành `logging_setup.py`
  cho khỏi nhầm với `tb_logger.py` khi đọc lướt tên file.

- **`HYPERPARAMETERS_TRACKING.md`** → nội dung độc nhất (bảng tham chiếu,
  vòng lặp training mẫu, lệnh TensorBoard, checklist) được gộp vào
  `docs/TENSORBOARD_GUIDE.md`, cập nhật theo API mới. Phần code mẫu lặp lại
  y hệt trong `tb_logger.py` đã được bỏ — giờ guide chỉ trỏ tới code, không
  chép lại code.

- **`training_log.md`** → đây thực chất là bản nháp/thiết kế mà
  `training_logger.py` đã hiện thực hoá gần như đầy đủ. Các phần đã có
  trong code (gradient/weight/EMA/GPU/BN/activation tracking) bị bỏ khỏi
  tài liệu vì trùng lặp. 3 mục nhỏ **chưa** có trong `training_logger.py`
  được giữ lại và đưa vào `tb_logger.py` mới:
  - `LossSmoother` (moving-average của loss)
  - `log_lr_schedule` (đọc LR trực tiếp từ `scheduler.get_last_lr()`)
  - `log_activation_histograms` (histogram cho activation đã thu thập sẵn)

  File `training_log.md` gốc không cần giữ lại nữa — toàn bộ nội dung hữu
  ích đã nằm trong code + `TENSORBOARD_GUIDE.md`.

## Không đổi

- Toàn bộ tag TensorBoard (`Gradients/...`, `Weights_Stats/...`,
  `train/loss_total`, v.v.) giữ nguyên như cũ — các run cũ vẫn xem chung
  được với run mới trên cùng TensorBoard.