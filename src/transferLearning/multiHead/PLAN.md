# Kế hoạch multi-head face landmark

## Kiến trúc đã chốt

- `HEAD 4`: giữ nguyên toàn bộ `FaceLmkDetector` đã fine-tune, gồm backbone,
  neck và detection head 478 điểm. Toàn bộ nhánh này luôn frozen và ở `eval`.
- `HEAD 1`: nhận crop mắt trái, dự đoán 16 điểm mí mắt + 5 điểm iris.
- `HEAD 2`: nhận crop mắt phải, dự đoán 16 điểm mí mắt + 5 điểm iris.
- `HEAD 3`: nhận crop miệng, dự đoán 40 điểm môi.
- Ba specialist là ba mini-detector độc lập, mỗi mạng có backbone/neck nhỏ và
  giữ contract `o2m/o2o`, bbox, DFL, landmark anchor-offset giống pipeline cũ.
- Hai crop mắt dùng `128×128`; crop miệng dùng `160×160`. Crop phải lấy trực
  tiếp từ ảnh gốc độ phân giải cao, không crop từ ảnh đã resize xuống 480.

## Luồng dữ liệu đã triển khai

Một record ảnh tạo bốn tensor:

1. `full_face`: ảnh đầy đủ letterbox 480×480 cho HEAD 4.
2. `left_eye`: crop vuông có margin, resize/letterbox 128×128.
3. `right_eye`: crop vuông có margin, resize/letterbox 128×128.
4. `mouth`: crop vuông có margin, resize/letterbox 160×160.

Crop mắt chỉ dùng 16 điểm mí/góc mắt làm neo; 5 điểm iris vẫn
được dự đoán nhưng không được phép kéo lệch ROI ở inference.

Mỗi crop phải lưu metadata `crop_x1`, `crop_y1`, `scale`, `padding` để ánh xạ
landmark local về ảnh gốc. Crop được tạo từ landmark ground truth sau khi áp
dụng cùng phép biến đổi hình học cho ảnh và nhãn. Horizontal flip đang
được tắt để không tráo semantic mắt trái/phải.

## Các thành phần đã hoàn thành

1. `model_multihead.py`: HEAD4 frozen và ba mini-detector.
2. `data_multihead.py`: split theo record, crop dùng chung, target local và
   ma trận ánh xạ crop ↔ ảnh gốc.
3. `loss_multihead.py`: detection loss cũ + EAR + phạt nhầm trạng thái
   nhắm/mở mắt + tỷ lệ mở miệng.
4. `train_multihead.py`: train tuần tự `left_eye → right_eye → mouth`,
   optimizer/scheduler mới theo stage, validation, early stopping và resume.
5. Checkpoint chỉ lưu ba specialist (khoảng vài MB), kèm hash tham
   chiếu HEAD4, architecture signature, crop config và mapping landmark; không
   nhúng lại model global hàng trăm MB.
6. `inference_multihead.py`: HEAD4 chạy một full pass, ba specialist chạy
   theo batch vùng, fallback an toàn và demo ảnh/camera bằng Matplotlib.

## Điều kiện kiểm tra bắt buộc

- Mắt trái/phải có K=21; miệng có K=40.
- Output train/eval giữ đúng shape và key của `DetectHeadFaceLmk`.
- HEAD 4 không đổi parameter lẫn BatchNorm buffer sau mọi optimizer step.
- Mỗi thời điểm chỉ một specialist có `requires_grad=True`.
- Chuyển stage không làm thay đổi hai specialist đã frozen.
- Phép đổi tọa độ crop-local → ảnh gốc phải round-trip chính xác.
- Landmark ngoài vùng specialist không được thay đổi khi ghép inference.
