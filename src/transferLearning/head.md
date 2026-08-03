# Đặc tả đầu ra mô hình Face Detection + 478 Face Landmarks

Tài liệu này mô tả **hợp đồng dữ liệu đầu ra** của module
`src/transferLearning`, đặc biệt là API `FaceLandmarkInferencer.predict()` và ý
nghĩa của 478 landmark để các module phía sau có thể tính EAR, MAR và các đặc
trưng hình học khác.

> Tóm tắt quan trọng: với checkpoint được huấn luyện từ bộ nhãn hiện tại, phần
> tử `landmarks[i, j]` là tọa độ pixel `(x, y)` của landmark MediaPipe ID `j`
> trên khuôn mặt thứ `i`. Mô hình học lại tọa độ 2D theo thứ tự MediaPipe; mô
> hình **không chạy MediaPipe khi inference** và không trả về `z`, visibility,
> presence, blendshape hay transformation matrix.

## 1. API nên dùng khi tích hợp

```python
from src.transferLearning.inference import FaceLandmarkInferencer

inferencer = FaceLandmarkInferencer(
    weights_path="path/to/checkpoint.pt",
    device="cuda",             # hoặc "cpu"
    conf_threshold=0.25,
)

image_rgb, detections, letterbox_info = inferencer.predict(
    image_input,
    show=False,
)

boxes = detections["boxes"]          # (N, 4)
scores = detections["scores"]        # (N,)
landmarks = detections["landmarks"]  # (N, K, 2), thông thường K = 478
```

`predict()` trả về một tuple ba phần tử:

| Phần tử | Kiểu/shape | Ý nghĩa |
|---|---:|---|
| `image_rgb` | `np.ndarray[H,W,3]` | Ảnh gốc ở thứ tự kênh RGB. |
| `detections` | `dict[str, np.ndarray]` | Kết quả đã lọc và đã đưa về hệ tọa độ ảnh gốc. |
| `letterbox_info` | `dict` | Thông tin resize/padding trung gian; thường không cần dùng lại vì kết quả trong `detections` đã được khôi phục. |

Nếu không phát hiện được mặt, API vẫn trả đúng cấu trúc, với:

```text
boxes.shape     == (0, 4)
scores.shape    == (0,)
landmarks.shape == (0, K, 2)
```

Không nên coi `N` là hằng số. Một ảnh có thể không có mặt, có một mặt hoặc có
nhiều mặt.

## 2. Chi tiết `detections`

### 2.1 `boxes`: bounding box của khuôn mặt

```text
boxes[i] = [x1, y1, x2, y2]
shape    = (N, 4)
```

- Đơn vị là **pixel trên ảnh gốc**, không phải tọa độ chuẩn hóa `[0,1]`.
- Gốc tọa độ `(0,0)` ở góc trên-trái ảnh.
- `x` tăng từ trái sang phải; `y` tăng từ trên xuống dưới.
- `(x1,y1)` là góc trên-trái, `(x2,y2)` là góc dưới-phải.
- Giá trị là số thực, không mặc định làm tròn về số nguyên.
- Sau bước khôi phục, `x` được clip vào `[0,W]`, `y` được clip vào `[0,H]`.
- `boxes[i]`, `scores[i]` và `landmarks[i]` luôn mô tả cùng một khuôn mặt.

### 2.2 `scores`: độ tin cậy phát hiện mặt

```text
scores[i] = sigmoid(face_logit_i)
shape     = (N,)
range     = (0, 1)
```

Đây là confidence của **detection khuôn mặt**, không phải độ chính xác của từng
landmark. Mô hình chỉ có một class (`face`, class ID 0), vì vậy API không trả
`labels` hay `class_ids`.

Các candidate được xử lý như sau:

1. Giữ candidate có `score > conf_threshold` (dấu `>` nghiêm ngặt).
2. Nếu vượt `max_det` (mặc định 100), chỉ giữ `max_det` score lớn nhất.
3. Chỉ chạy NMS khi `iou_threshold > 0`. Mặc định `iou_threshold = 0`, tức nhánh
   one-to-one được dùng theo chế độ NMS-free.

Không dùng vị trí `i` như ID theo dõi qua các frame: thứ tự detection có thể
thay đổi. Nếu xử lý video, cần tracker hoặc ghép mặt bằng IoU/vị trí.

### 2.3 `landmarks`: 478 tọa độ mặt

```text
landmarks[i, j] = [x, y]
shape           = (N, K, 2), thông thường K = 478
```

- `i`: chỉ số khuôn mặt trong kết quả hiện tại.
- `j`: ID landmark; với bộ nhãn/checkpoint MediaPipe 478, `j` nằm trong
  `0..477` và trùng ID MediaPipe.
- chiều cuối: `[x, y]`, theo pixel ảnh gốc.
- cùng gốc tọa độ và chiều trục với `boxes`.
- tọa độ được clip vào biên ảnh gốc.
- không có chiều `z`.
- không có visibility/presence/confidence cho từng điểm.

Ví dụ, tọa độ đỉnh mũi của mặt đầu tiên:

```python
nose_tip_xy = detections["landmarks"][0, 1]  # [x, y]
```

## 3. Quan hệ chính xác với MediaPipe

### 3.1 Điều gì tương thích

Pipeline này dùng schema **MediaPipe Attention Face Mesh 478 điểm**:

- ID `0..467`: 468 đỉnh của Face Mesh cơ sở.
- ID `468..472`: 5 điểm iris của mắt phải của đối tượng.
- ID `473..477`: 5 điểm iris của mắt trái của đối tượng.
- Các ID mắt, môi, mũi được đặt trọng số loss theo đúng các tập ID MediaPipe.
- Khi augmentation lật ngang, code dùng một hoán vị song ánh 478 phần tử để
  đổi semantic trái/phải, thay vì chỉ đổi tọa độ `x`.

Vì vậy một đoạn code đang lấy `face_landmarks[j]` từ MediaPipe có thể lấy
`detections["landmarks"][i, j]` từ mô hình này cho các phép toán **chỉ cần
`x,y`**, sau khi chọn đúng khuôn mặt `i`.

### 3.2 Điều kiện bắt buộc của kết luận trên

Dataset loader không đọc trường `index` cho từng điểm; nó tin rằng phần tử thứ
`j` trong `landmarks_normalized` chính là landmark ID `j`. Nó kiểm tra số lượng
điểm đồng nhất, nhưng không thể phát hiện một list đủ 478 phần tử bị đảo thứ tự.

Do đó hợp đồng dữ liệu khi tạo nhãn phải luôn là:

```text
landmarks_normalized[0]   = MediaPipe landmark 0
landmarks_normalized[1]   = MediaPipe landmark 1
...
landmarks_normalized[477] = MediaPipe landmark 477
```

Checkpoint có `K != 478` sẽ được inferencer tự nhận từ shape của head. Khi đó
không được mặc định coi ID là MediaPipe 478. Luôn kiểm tra:

```python
assert detections["landmarks"].shape[1] == 478
```

### 3.3 Điều gì không tương thích trực tiếp

| MediaPipe Face Landmarker | Đầu ra mô hình này |
|---|---|
| `x,y` thường chuẩn hóa theo kích thước ảnh | `x,y` pixel ảnh gốc |
| Có `z` tương đối | Không có `z` |
| Có thể có visibility/presence | Không có |
| Có thể trả blendshapes | Không có |
| Có thể trả facial transformation matrix | Không có |
| Chạy MediaPipe graph ở inference | Chạy CNN detector/landmark head độc lập |

Hệ quả:

- EAR/MAR 2D có thể tính trực tiếp.
- Khoảng cách 2D, góc 2D, vùng mắt/miệng có thể tính trực tiếp.
- Không thể thay thế trực tiếp `z`, blendshape như `eyeBlinkLeft`, `jawOpen`,
  hoặc ma trận pose của MediaPipe bằng đầu ra này.
- Head pose vẫn có thể ước lượng bằng một bài toán `solvePnP` riêng với các điểm
  2D và một mô hình mặt 3D, nhưng kết quả đó không phải output có sẵn của model.

## 4. Quy ước trái/phải

Trong tài liệu này, **trái/phải là phía giải phẫu của người được chụp**, không
phải phía trái/phải của người đang nhìn ảnh:

- mắt trái của đối tượng: nhóm ID bắt đầu bằng `362`, `263`;
- mắt phải của đối tượng: nhóm ID bắt đầu bằng `33`, `133`.

Trong ảnh camera không mirror, mắt trái của đối tượng thường nằm bên phải ảnh.
Nếu UI lật gương ảnh để hiển thị, không được tự ý đổi hai nhóm ID; hãy tính trên
tọa độ/model output trước khi biến đổi hiển thị, hoặc áp dụng cùng một phép biến
đổi tọa độ có kiểm soát.

## 5. Bản đồ landmark quan trọng

MediaPipe định nghĩa 478 điểm như các **đỉnh của một mesh**, không phải 478 tên
giải phẫu độc lập. Các số ID cũng không được chia thành những khoảng liên tiếp
theo vùng: ví dụ ID của mắt và môi xen kẽ với các đỉnh bề mặt khác. Vì vậy cách
mô tả chính xác là bằng các đường contour/connection dưới đây.

### 5.1 Mắt phải của đối tượng

Hai cung khép thành viền mắt:

```text
33 - 7 - 163 - 144 - 145 - 153 - 154 - 155 - 133
33 - 246 - 161 - 160 - 159 - 158 - 157 - 173 - 133
```

Tập đầy đủ được pipeline đặt trọng số cao:

```text
[33, 7, 163, 144, 145, 153, 154, 155,
 133, 173, 157, 158, 159, 160, 161, 246]
```

Các điểm EAR thường dùng:

| Vai trò EAR | ID |
|---|---:|
| khóe ngoài/ngang `p1` | 33 |
| mí trên `p2` | 160 |
| mí trên `p3` | 158 |
| khóe trong/ngang `p4` | 133 |
| mí dưới `p5` | 153 |
| mí dưới `p6` | 144 |

### 5.2 Mắt trái của đối tượng

Hai cung khép thành viền mắt:

```text
263 - 249 - 390 - 373 - 374 - 380 - 381 - 382 - 362
263 - 466 - 388 - 387 - 386 - 385 - 384 - 398 - 362
```

Tập đầy đủ được pipeline đặt trọng số cao:

```text
[362, 382, 381, 380, 374, 373, 390, 249,
 263, 466, 388, 387, 386, 385, 384, 398]
```

Các điểm EAR thường dùng:

| Vai trò EAR | ID |
|---|---:|
| khóe trong/ngang `p1` | 362 |
| mí trên `p2` | 385 |
| mí trên `p3` | 387 |
| khóe ngoài/ngang `p4` | 263 |
| mí dưới `p5` | 373 |
| mí dưới `p6` | 380 |

### 5.3 Iris

| Mắt của đối tượng | Tâm iris | Bốn điểm vành iris |
|---|---:|---|
| Phải | 468 | 469, 470, 471, 472 |
| Trái | 473 | 474, 475, 476, 477 |

Các connection vòng iris là:

```text
iris phải: 469 - 470 - 471 - 472 - 469
iris trái : 474 - 475 - 476 - 477 - 474
```

Tâm iris không nằm trong các connection vòng, nhưng vẫn là một landmark riêng.
Mười điểm `468..477` chỉ tồn tại trong phiên bản 478 điểm/refined landmarks;
Face Mesh 468 điểm không có chúng.

### 5.4 Lông mày

```text
lông mày phải của đối tượng:
46 - 53 - 52 - 65 - 55
70 - 63 - 105 - 66 - 107

lông mày trái của đối tượng:
276 - 283 - 282 - 295 - 285
300 - 293 - 334 - 296 - 336
```

Mỗi bên có hai đoạn contour trong định nghĩa MediaPipe, vì vậy hai dòng của
cùng một bên không nên nối tùy ý thành một polygon duy nhất.

### 5.5 Môi và miệng

Bốn cung contour MediaPipe:

```text
viền ngoài 1:
61 - 146 - 91 - 181 - 84 - 17 - 314 - 405 - 321 - 375 - 291

viền ngoài 2:
61 - 185 - 40 - 39 - 37 - 0 - 267 - 269 - 270 - 409 - 291

viền trong 1:
78 - 95 - 88 - 178 - 87 - 14 - 317 - 402 - 318 - 324 - 308

viền trong 2:
78 - 191 - 80 - 81 - 82 - 13 - 312 - 311 - 310 - 415 - 308
```

Các mốc dễ nhớ:

| Vị trí | ID |
|---|---:|
| hai khóe ngoài của miệng | 61, 291 |
| hai khóe trong dùng cho MAR | 78, 308 |
| giữa môi trên ngoài | 0 |
| giữa môi dưới ngoài | 17 |
| giữa môi trên trong | 13 |
| giữa môi dưới trong | 14 |

Tập 40 ID môi được tăng trọng số trong loss của mô hình:

```text
[0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87,
 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405,
 409, 415]
```

### 5.6 Mũi

Các connection chính trong topology MediaPipe:

```text
sống/đường giữa:
168 - 6 - 197 - 195 - 5 - 4 - 1 - 19 - 94 - 2

phần dưới/cánh mũi:
98 - 97 - 2 - 326 - 327 - 294 - 278 - 344 - 440 - 275 - 4
4 - 45 - 220 - 115 - 48 - 64 - 98
```

ID `1` là đỉnh mũi quan trọng và được đặt trọng số loss cao nhất trong cấu hình
hiện tại. Các ID `168`, `6`, `197`, `195`, `5`, `4` đi dọc vùng sống mũi về
phía đỉnh mũi; các ID quanh `98` và `326` thuộc hai phía cánh/đáy mũi.

### 5.7 Viền khuôn mặt

Chuỗi contour khép kín, bắt đầu ở vùng đỉnh trán, đi vòng qua hai bên mặt và
cằm:

```text
10 - 338 - 297 - 332 - 284 - 251 - 389 - 356 - 454 - 323
- 361 - 288 - 397 - 365 - 379 - 378 - 400 - 377 - 152
- 148 - 176 - 149 - 150 - 136 - 172 - 58 - 132 - 93 - 234
- 127 - 162 - 21 - 54 - 103 - 67 - 109 - 10
```

Mốc dễ nhớ:

| Vị trí | ID |
|---|---:|
| đỉnh trên của face oval | 10 |
| đáy cằm | 152 |
| hai điểm ngoài vùng má/thái dương | 234, 454 |

Các điểm còn lại trong `0..467` là các đỉnh nội suy trên trán, má, quanh mắt,
quanh mũi và quanh miệng để tạo lưới tam giác. Không nên suy diễn vùng mặt từ
giá trị số ID; hãy dùng một tập ID/connection đã định nghĩa rõ.

## 6. Tính EAR

Với sáu điểm theo thứ tự `p1..p6`, Eye Aspect Ratio là:

```text
EAR = (distance(p2,p6) + distance(p3,p5))
      / (2 * distance(p1,p4))
```

Thứ tự được dùng thống nhất trong các module tính đặc trưng của workspace:

```python
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
```

Code tham khảo:

```python
import numpy as np

LEFT_EYE_EAR = np.array([362, 385, 387, 263, 373, 380])
RIGHT_EYE_EAR = np.array([33, 160, 158, 133, 153, 144])

def aspect_ratio_6pts(points):
    p1, p2, p3, p4, p5, p6 = points
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return np.nan
    return (
        np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    ) / (2.0 * horizontal)

def compute_ear(face_landmarks):
    # face_landmarks: (478, 2), pixel x/y của đúng một khuôn mặt
    ear_left = aspect_ratio_6pts(face_landmarks[LEFT_EYE_EAR])
    ear_right = aspect_ratio_6pts(face_landmarks[RIGHT_EYE_EAR])
    ear_mean = 0.5 * (ear_left + ear_right)
    return ear_left, ear_right, ear_mean
```

EAR là tỷ số nên bất biến với tịnh tiến và scale đồng đều. Tuy nhiên perspective,
quay đầu, nhắm lệch một mắt và lỗi landmark vẫn ảnh hưởng mạnh. Không nên lấy
một ngưỡng EAR cố định cho mọi người mà chưa hiệu chỉnh/đánh giá trên dữ liệu
thực tế; với video nên làm mượt theo thời gian.

## 7. Tính MAR

Biến thể MAR dùng ba khoảng cách dọc của môi trong và khoảng cách hai khóe trong:

```text
MAR = (d(81,178) + d(13,14) + d(311,402)) / (3 * d(78,308))
```

Thứ tự tám điểm:

```python
MOUTH_MAR = [78, 81, 13, 311, 308, 402, 14, 178]
```

Code tham khảo:

```python
MOUTH_MAR = np.array([78, 81, 13, 311, 308, 402, 14, 178])

def compute_mar(face_landmarks):
    p = face_landmarks[MOUTH_MAR]
    horizontal = np.linalg.norm(p[0] - p[4])       # 78--308
    if horizontal < 1e-6:
        return np.nan
    vertical = (
        np.linalg.norm(p[1] - p[7])                # 81--178
        + np.linalg.norm(p[2] - p[6])              # 13--14
        + np.linalg.norm(p[3] - p[5])              # 311--402
    )
    return vertical / (3.0 * horizontal)
```

Đây là **một định nghĩa MAR cụ thể**. Một số tài liệu dùng điểm viền ngoài hoặc
hệ số chuẩn hóa khác, nên khi so sánh threshold phải ghi rõ công thức và bộ ID.

## 8. Ví dụ xử lý an toàn một kết quả

```python
import numpy as np

_, detections, _ = inferencer.predict(frame, show=False)

if detections["scores"].size == 0:
    # Không có mặt: đánh dấu frame invalid; không tái dùng landmark cũ như dữ liệu mới.
    result = None
else:
    # Chính sách đơn giản: lấy mặt confidence cao nhất.
    # Với video nhiều người nên thay bằng tracker/ROI của tài xế.
    face_idx = int(np.argmax(detections["scores"]))
    face_landmarks = detections["landmarks"][face_idx]

    if face_landmarks.shape != (478, 2) or not np.isfinite(face_landmarks).all():
        result = None
    else:
        ear_left, ear_right, ear_mean = compute_ear(face_landmarks)
        mar = compute_mar(face_landmarks)
        result = {
            "face_index": face_idx,
            "face_score": float(detections["scores"][face_idx]),
            "ear_left": float(ear_left),
            "ear_right": float(ear_right),
            "ear_mean": float(ear_mean),
            "mar": float(mar),
        }
```

Trong hệ thống giám sát tài xế, chọn `argmax(score)` chưa chắc chọn đúng tài xế
nếu có hành khách. Nên khóa ROI ghế lái hoặc track identity/mặt qua thời gian.

## 9. Đầu ra mức thấp của `FaceLmkDetector.forward()`

Phần này dành cho người tích hợp trực tiếp PyTorch model, không qua `predict()`.
Với input mặc định `480x480`, ba feature scale có stride `8,16,32`, nên số vị
trí ứng viên là:

```text
A = 60*60 + 30*30 + 15*15 = 4725
```

Ở `model.eval()` và gọi mặc định `model(images)`, kết quả là:

```text
preds = {
  "o2o": {
    "cls":     Tensor[B, A, 1],
    "box":     Tensor[B, A, 4],
    "reg_raw": Tensor[B, 64, A],       # 4 * reg_max, reg_max=16
    "lmk":     Tensor[B, A, K, 2],     # landmark đã decode, pixel letterbox
    "lmk_raw": Tensor[B, 2*K, A],
  },
  "anchors": Tensor[A, 2],
  "strides": Tensor[A, 1],
}
```

Ý nghĩa:

- `o2o`: nhánh one-to-one dùng cho inference NMS-free.
- `cls`: logit raw, phải qua sigmoid để thành score.
- `box`: `[x1,y1,x2,y2]` đã decode, đơn vị pixel trong ảnh letterbox `480x480`.
- `lmk`: `(x,y)` đã decode, cũng trong ảnh letterbox; chưa phải tọa độ ảnh gốc.
- `reg_raw`: logits DFL của bốn khoảng cách trái/trên/phải/dưới.
- `lmk_raw`: logits raw; thứ tự channel sau reshape là
  `[x_0,y_0,x_1,y_1,...,x_(K-1),y_(K-1)]` cho mỗi anchor.
- `anchors`: tâm grid chưa nhân stride.
- `strides`: stride tương ứng mỗi anchor.

`predict()` thực hiện thêm sigmoid score, threshold, giới hạn `max_det`, NMS tùy
chọn và phép nghịch đảo letterbox. Vì vậy code nghiệp vụ nên dùng output của
`predict()` thay vì tự dùng `preds["o2o"]["lmk"]`.

Trong `train()` hoặc khi gọi `return_o2m=True`, model trả cả `o2m` và `o2o` để
tính loss. Ở đường này chỉ có `lmk_raw`, không có khóa `lmk` đã decode. Nhánh
`o2m` là tín hiệu huấn luyện one-to-many, không phải output khuyến nghị cho ứng
dụng cuối.

## 10. Cách landmark được giải mã

Với bbox dự đoán `(x1,y1,x2,y2)`, chiều rộng `w=x2-x1`, chiều cao `h=y2-y1` và
margin mặc định `m=0.05`, model tạo vùng mở rộng:

```text
x1e = x1 - m*w
y1e = y1 - m*h
we  = w*(1 + 2*m)
he  = h*(1 + 2*m)
```

Mỗi cặp logit raw của landmark `j` được sigmoid về `(tx,ty)` trong `(0,1)`, rồi:

```text
x_j = x1e + tx*we
y_j = y1e + ty*he
```

Do đó landmark được dự đoán tương đối theo bbox mở rộng 5% mỗi phía, không tương
đối theo toàn ảnh. Sau đó `predict()` mới bỏ padding letterbox, chia cho scale
và clip về ảnh gốc.

## 11. Các lưu ý chất lượng cho EAR/MAR

1. `scores[i]` cao chỉ nói rằng candidate giống khuôn mặt; nó không bảo đảm mọi
   landmark mắt/miệng đều chính xác.
2. Không có confidence từng điểm, nên nên kiểm tra hình học: mẫu số khác 0, giá
   trị hữu hạn, độ rộng hai mắt/miệng hợp lý so với bbox.
3. Tọa độ bị clip ở biên ảnh. Nếu mặt bị cắt, nhiều điểm có thể trùng `x=0`,
   `x=W`, `y=0` hoặc `y=H`; EAR/MAR khi đó không đáng tin.
4. Mặt nghiêng/che khuất có thể làm tỷ số 2D thay đổi dù trạng thái mắt/miệng
   không đổi.
5. Với video, cần đánh dấu frame thiếu mặt là invalid, làm mượt tín hiệu và dùng
   logic theo khoảng thời gian thay vì quyết định độc lập từng frame.
6. Model trả `float`; không ép landmark sang `int` trước khi tính khoảng cách vì
   sẽ gây sai số lượng tử hóa.
7. Threshold EAR/MAR phải được hiệu chỉnh trên đúng model/checkpoint và camera;
   không sao chép trực tiếp threshold được xây dựng từ một phiên bản MediaPipe
   hoặc công thức MAR khác.

## 12. Nguồn xác nhận trong mã dự án

Các kết luận trong tài liệu được đối chiếu trực tiếp từ:

- `src/transferLearning/inference.py`: cấu trúc API `predict()`, sigmoid score,
  threshold, NMS, inverse letterbox và dictionary `detections`.
- `src/transferLearning/model_lmk.py`: shape head, nhánh `o2o/o2m`, DFL bbox và
  công thức decode landmark theo bbox mở rộng.
- `src/transferLearning/dataset_lmk.py`: thứ tự landmark được giữ từ
  `landmarks_normalized`, phép letterbox và semantic horizontal flip.
- `src/transferLearning/config_lmk.py`: cấu hình MediaPipe 478 và các nhóm ID
  mắt, iris, môi, đỉnh mũi.
- `src/transferLearning/mediapipe_478.py`: hoán vị trái/phải cho đúng topology
  canonical MediaPipe 478.
- `src/transferLearning/loss_lmk.py`: cách encode target và trọng số landmark.
