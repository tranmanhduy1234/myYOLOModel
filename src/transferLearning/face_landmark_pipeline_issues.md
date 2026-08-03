# Báo cáo lỗi và rủi ro tích hợp Face Landmark Pipeline

Tài liệu này tổng hợp các lỗi, điểm không nhất quán và rủi ro kỹ thuật được phát hiện trong các file:

- `config_lmk.py`
- `dataset_lmk.py`
- `model_lmk.py`
- `loss_lmk.py`
- `inference.py`

Phạm vi kiểm tra:

```text
JSONL → Dataset → DataLoader → Model → Loss → Inference
```

---

# I. Lỗi bắt buộc sửa trước khi huấn luyện

## 1. Không khớp tên thư mục ảnh: `Images` và `images`

### Vị trí trong code

File: `dataset_lmk.py`

Trong hàm:

```python
class FaceLandmarkDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, transform: Optional[T.Transform] = None):
```

Dòng tạo đường dẫn ảnh:

```python
self.images_dir = os.path.join(cfg.root_dir, 'Images')
```

Trong khi pipeline tạo dataset hợp nhất trước đó lưu ảnh trong:

```text
MergedDataset/images/
```

### Lý do

Linux phân biệt chữ hoa và chữ thường:

```text
Images != images
```

Dataset sẽ đọc được JSONL nhưng thất bại khi mở ảnh vì nó tìm trong:

```text
MergedDataset/Images/
```

thay vì:

```text
MergedDataset/images/
```

Lỗi thường xuất hiện dưới dạng:

```text
FileNotFoundError
```

### Mức độ

**Critical**

---

## 2. Không khớp tên file JSONL mặc định

### Vị trí trong code

File: `config_lmk.py`

Trong:

```python
@dataclass
class DatasetConfig:
```

Có:

```python
jsonl_name: str = 'annotations_all.jsonl'
```

Trong:

```python
@dataclass
class TrainConfig:
```

Cũng có:

```python
jsonl_name: str = 'annotations_all.jsonl'
```

Trong khi pipeline hợp nhất tạo:

```text
merged_faces.jsonl
```

### Lý do

Nếu không ghi đè `jsonl_name`, dataset sẽ tìm:

```text
root_dir/annotations_all.jsonl
```

nhưng file thực tế là:

```text
root_dir/merged_faces.jsonl
```

Điều này khiến khởi tạo dataset thất bại ngay lập tức.

### Mức độ

**Critical**

---

## 3. Chính sách resize khi train và inference không đồng nhất

### Vị trí trong code

File: `dataset_lmk.py`

Transform mặc định:

```python
self.transform = transform or T.Compose(
    [
        T.ToImage(),
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToDtype(torch.float32, scale=True)
    ]
)
```

File: `inference.py`

Trong hàm:

```python
def letterbox(...)
```

Inference giữ nguyên aspect ratio và thêm padding.

### Lý do

Train hiện tại kéo trực tiếp mọi ảnh về hình vuông:

```text
W × H → 224 × 224
```

Inference lại:

```text
giữ aspect ratio → resize → padding
```

Hai pipeline tạo ra hai phân bố hình học khác nhau.

Nếu ảnh đầu vào không vuông:

- Train làm méo khuôn mặt.
- Inference không làm méo.
- Bbox và landmark học trên hình học khác với lúc suy luận.

Với dữ liệu toàn bộ là `480 × 480`, lỗi chưa biểu hiện. Nhưng pipeline không an toàn nếu sau này xuất hiện ảnh không vuông.

### Mức độ

**Critical nếu dataset có ảnh không vuông**

---

## 4. Classification bias được khởi tạo theo ảnh 640 thay vì ảnh 224

### Vị trí trong code

File: `model_lmk.py`

Trong:

```python
class ScaleHeadFaceLmk(nn.Module):
```

Hàm:

```python
def init_stride_bias(self, stride, img_size=640):
    value = math.log(5 / self.nc / (img_size / stride) ** 2)
```

Trong:

```python
class DetectHeadFaceLmk(nn.Module):
```

Có:

```python
for head, s in zip(self.heads, self.strides):
    head.init_stride_bias(s)
```

Không truyền `img_size`, nên luôn dùng mặc định `640`.

Trong khi cấu hình train:

```python
image_size: int = 224
```

### Lý do

Classification prior phụ thuộc số lượng cell trên từng scale.

Với stride 8:

```text
640 / 8 = 80
224 / 8 = 28
```

Bias tính theo 640 làm xác suất face ban đầu thấp hơn đáng kể so với thiết kế thực tế của ảnh 224.

Hệ quả:

- classification confidence đầu quá thấp;
- warmup khó hơn;
- gradient classification ban đầu có thể không tối ưu;
- model mất thêm thời gian để thoát khỏi prior sai.

### Mức độ

**High**

---

## 5. `trunk_feat_channels` có thể không khớp output thật của neck

### Vị trí trong code

File: `config_lmk.py`

```python
trunk_feat_channels: Tuple[int, int, int] = (224, 448, 640)
```

File: `model_lmk.py`

Trong:

```python
class FaceLmkDetector(nn.Module):
```

Có:

```python
self.head = DetectHeadFaceLmk(
    chs=cfg.trunk_feat_channels,
    cfg=cfg.face
)
```

### Lý do

Output channel thật của neck phụ thuộc:

```python
trunk_backbone_w
trunk_backbone_n
trunk_neck_n
```

Nhưng head lại nhận channel từ một tuple cấu hình thủ công khác.

Nếu thay đổi trunk mà quên cập nhật `trunk_feat_channels`, forward sẽ lỗi do channel mismatch:

```text
Expected input with C1 channels but received C2 channels
```

Đây là một contract bị lặp ở hai nơi và dễ lệch nhau.

### Mức độ

**High**

---

## 6. `load_trunk()` phụ thuộc một format checkpoint quá cụ thể

### Vị trí trong code

File: `model_lmk.py`

Trong:

```python
def load_trunk(self, path: str, map_location='cpu', strict: bool = True):
```

Có:

```python
sd = torch.load(path, map_location=map_location)

for name, module in (('backbone', self.backbone), ('neck', self.neck)):
    if name not in sd:
        raise KeyError(...)
```

### Lý do

Hàm chỉ chấp nhận checkpoint dạng:

```python
{
    "backbone": ...,
    "neck": ...
}
```

Nhưng nhiều pipeline lưu checkpoint dạng:

```python
{
    "model": model.state_dict()
}
```

với các key:

```text
backbone.xxx
neck.xxx
```

Hoặc:

```text
module.backbone.xxx
module.neck.xxx
```

Nếu checkpoint pretrain của bạn dùng format phổ biến này, `load_trunk()` sẽ báo thiếu khóa `backbone` hoặc `neck` dù trọng số thực tế có tồn tại.

### Mức độ

**High**

---

## 7. Dataset không xác thực toàn bộ số lượng landmark

### Vị trí trong code

File: `dataset_lmk.py`

Hàm:

```python
def _detect_num_landmarks(
    jsonl_path: str,
    offsets: np.ndarray,
    scan_limit: int = 2000
) -> int:
```

Có:

```python
for i in range(min(scan_limit, len(offsets))):
    ...
    if faces:
        return len(faces[0]['landmarks_normalized'])
```

### Lý do

Hàm chỉ lấy số landmark từ khuôn mặt hợp lệ đầu tiên trong tối đa 2.000 dòng.

Nếu dataset có lẫn:

```text
478-point Face Mesh
468-point Face Mesh
record lỗi thiếu điểm
```

thì hệ thống không phát hiện toàn cục.

Các record khác số điểm sẽ bị bỏ qua âm thầm trong `__getitem__`.

Hậu quả:

- mất dữ liệu không được thống kê đầy đủ;
- một số ảnh có thể trở thành target rỗng;
- khó phát hiện dataset bị trộn schema;
- train vẫn chạy nhưng dữ liệu thực tế ít hơn mong đợi.

### Mức độ

**High**

---

# II. Lỗi và rủi ro trong Dataset

## 8. API `transform` cho phép geometric augmentation nhưng không cập nhật target

### Vị trí trong code

File: `dataset_lmk.py`

Constructor:

```python
def __init__(
    self,
    cfg: DatasetConfig,
    transform: Optional[T.Transform] = None
):
```

Trong `__getitem__`:

```python
'image': self.transform(image),
```

Trong khi bbox và landmarks được tạo độc lập trước đó và không truyền qua transform.

### Lý do

Nếu người dùng truyền:

```text
RandomHorizontalFlip
RandomRotation
RandomAffine
RandomResizedCrop
Perspective
```

thì chỉ ảnh bị biến đổi.

Các target sau vẫn giữ nguyên:

```text
boxes
landmarks
```

Kết quả là annotation sai hoàn toàn so với ảnh.

Transform mặc định hiện tại chỉ resize nên chưa lỗi, nhưng API hiện tại rất dễ bị sử dụng sai.

### Mức độ

**High**

---

## 9. Horizontal flip cần đổi cả tọa độ lẫn semantic landmark index

### Vị trí trong code

File liên quan: `dataset_lmk.py`

Pipeline hiện chưa có horizontal flip, nhưng cấu trúc `transform` có thể khiến người dùng thêm trực tiếp.

### Lý do

Với landmark khuôn mặt, lật ngang không chỉ là:

```python
x_new = image_width - x_old
```

Mà còn phải remap landmark semantic:

```text
mắt trái ↔ mắt phải
iris trái ↔ iris phải
mép môi trái ↔ mép môi phải
các điểm contour trái ↔ contour phải
```

Nếu chỉ lật tọa độ mà không đổi index, model sẽ học semantic trái/phải sai.

### Mức độ

**High khi bổ sung augmentation**

---

## 10. `LandmarkMarginCoverageChecker` đọc toàn bộ JSONL vào RAM

### Vị trí trong code

File: `dataset_lmk.py`

Trong:

```python
class LandmarkMarginCoverageChecker:
```

Hàm:

```python
def run(self) -> None:
```

Có:

```python
with open(jsonl_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
```

### Lý do

JSONL chứa 478 landmark cho mỗi mặt có thể rất lớn.

`readlines()` sẽ đưa toàn bộ file vào RAM trước khi chỉ lấy `sample_size`.

Hệ quả:

- tiêu tốn nhiều GB RAM;
- có thể OOM;
- không tận dụng byte-offset index đã xây;
- thời gian khởi động không cần thiết.

### Mức độ

**High với dataset lớn**

---

## 11. `LandmarkMarginCoverageChecker` gần như mất ý nghĩa với bbox hiện tại

### Vị trí trong code

File: `dataset_lmk.py`

Trong `LandmarkMarginCoverageChecker.run()`:

```python
if not (x1e <= px <= x2e and y1e <= py <= y2e):
    n_outside[m] += 1
```

### Lý do

Bbox JSONL hiện tại được tạo bằng min/max của chính các landmark MediaPipe, sau đó còn thêm padding.

Do đó theo định nghĩa:

```text
landmark gần như luôn nằm trong bbox tại margin = 0
```

Checker sẽ báo coverage rất cao dù điều đó không chứng minh `lmk_margin` là tối ưu.

Checker chỉ có ý nghĩa khi bbox được lấy độc lập từ:

- face detector;
- annotation thủ công;
- RetinaFace/SCRFD;
- ground-truth bbox khác nguồn.

### Mức độ

**Medium về correctness của phân tích**

---

## 12. Dataset có thể âm thầm trả target rỗng

### Vị trí trong code

File: `dataset_lmk.py`

Trong `__getitem__`:

```python
if x2 - x1 < self.cfg.min_box_size_px or y2 - y1 < self.cfg.min_box_size_px:
    continue
```

Và:

```python
if len(pts) != self.num_landmarks:
    ...
    continue
```

Sau đó:

```python
if boxes:
    ...
else:
    boxes_t = torch.zeros((0, 4), ...)
```

### Lý do

Một record JSONL có `faces`, nhưng tất cả face có thể bị bỏ vì:

- bbox quá nhỏ;
- số landmark không khớp;
- bbox sai.

Khi đó ảnh vẫn được đưa vào train nhưng target rỗng.

Loss hỗ trợ ảnh không target, nên train không crash. Tuy nhiên:

- ảnh positive bị biến thành background;
- classification target bị sai;
- khó phát hiện lỗi dữ liệu;
- model có thể học nhầm ảnh mặt là negative.

### Mức độ

**High nếu xảy ra thường xuyên**

---

## 13. Cảnh báo landmark mismatch chỉ xuất hiện một lần

### Vị trí trong code

File: `dataset_lmk.py`

Có:

```python
self._warned_lmk_mismatch = False
```

Và:

```python
if not self._warned_lmk_mismatch:
    print(...)
    self._warned_lmk_mismatch = True
```

### Lý do

Nếu có hàng nghìn record sai số landmark, hệ thống chỉ in một cảnh báo.

Không có thống kê:

```text
bao nhiêu face bị bỏ
bao nhiêu ảnh trở thành target rỗng
tỷ lệ lỗi theo dataset nguồn
```

Điều này che giấu mức độ hỏng dữ liệu.

### Mức độ

**Medium**

---

## 14. File handle không có cơ chế đóng rõ ràng

### Vị trí trong code

File: `dataset_lmk.py`

Có:

```python
self._file_handle = None
```

Và:

```python
def _get_file(self):
    if self._file_handle is None:
        self._file_handle = open(...)
```

### Lý do

File handle được mở lazy nhưng không có:

```python
__del__
__getstate__
close
```

Trong DataLoader nhiều worker, đặc biệt khi dùng `persistent_workers=True`, handle có thể tồn tại lâu.

Thông thường hệ điều hành sẽ đóng khi process kết thúc, nhưng thiết kế này:

- khó kiểm soát resource;
- có thể gây vấn đề khi pickle/spawn;
- không reset handle rõ ràng khi dataset được serialize.

### Mức độ

**Low–Medium**

---

## 15. `prefetch_factor=4` có thể dùng RAM lớn

### Vị trí trong code

File: `dataset_lmk.py`

Trong:

```python
class FaceLandmarkDataModule:
```

Có:

```python
prefetch_factor=4 if cfg.num_workers > 0 else None
```

### Lý do

Với:

```text
batch_size = 32
num_workers = 4
prefetch_factor = 4
```

DataLoader có thể chuẩn bị trước lượng dữ liệu tương đương khoảng:

```text
4 × 4 × 32 = 512 ảnh
```

Mỗi ảnh còn đi kèm target 478 landmark.

Điều này có thể gây:

- RAM tăng mạnh;
- pinned memory lớn;
- áp lực CPU;
- không nhất thiết tăng throughput tương ứng.

### Mức độ

**Medium về hiệu năng**

---

# III. Lỗi và rủi ro trong Model

## 16. Landmark head được zero-initialize hoàn toàn

### Vị trí trong code

File: `model_lmk.py`

Trong:

```python
def _init_bias(self):
```

Có:

```python
for m in (self.lmk_o2m, self.lmk_o2o):
    nn.init.constant_(m.bias, 0.0)
    nn.init.constant_(m.weight, 0.0)
```

### Lý do

Output ban đầu bằng 0:

```text
sigmoid(0) = 0.5
```

nên mọi landmark ban đầu nằm ở tâm bbox mở rộng.

Điều này hợp lý về output ban đầu, nhưng vì weight final conv bằng 0:

```text
gradient về lmk_stem ở update đầu tiên bằng 0
```

Final landmark convolution học trước, còn stem chỉ bắt đầu nhận gradient sau khi final weight khác 0.

Không làm model hỏng, nhưng làm chậm một bước đầu và tạo gradient flow không tối ưu.

### Mức độ

**Medium**

---

## 17. `lmk_margin=0.15` có thể quá lớn so với bbox đã padding

### Vị trí trong code

File: `config_lmk.py`

```python
lmk_margin: float = 0.15
```

File: `model_lmk.py`

Trong:

```python
def decode_landmarks(...)
```

File: `loss_lmk.py`

Trong:

```python
def _encode_landmark_targets(...)
```

### Lý do

Bbox JSONL đã được tạo từ landmark và có padding.

Sau đó model lại mở rộng bbox thêm 15% mỗi phía:

```text
tổng chiều rộng vùng decode = 1.3 × bbox width
```

Landmark target sẽ chỉ sử dụng phần giữa của miền sigmoid, xấp xỉ:

```text
0.115 → 0.885
```

Hệ quả:

- giảm độ phân giải hiệu dụng của landmark output;
- tạo vùng thừa lớn;
- model ít sử dụng hai biên của sigmoid.

Margin vẫn có ích khi bbox dự đoán lệch, nhưng 0.15 có thể dư trong schema bbox hiện tại.

### Mức độ

**Medium**

---

## 18. Landmark head tạo tensor đầu ra rất lớn

### Vị trí trong code

File: `model_lmk.py`

Trong:

```python
self.lmk_o2m = nn.Conv2d(
    c_lmk_hidden,
    self.num_landmarks * 2,
    1
)
```

Và tương tự cho `lmk_o2o`.

### Lý do

Với:

```text
K = 478
K × 2 = 956 channels
```

Ở ảnh 224, tổng anchor:

```text
28² + 14² + 7² = 1029
```

Mỗi nhánh tạo gần một triệu giá trị landmark cho mỗi ảnh.

Hai nhánh o2m và o2o làm chi phí tăng gấp đôi trong train.

Hệ quả:

- VRAM cao;
- activation memory lớn;
- backward chậm;
- batch size 32 có thể không phù hợp;
- head landmark có thể chiếm phần lớn latency.

Đây không phải lỗi correctness, nhưng là rủi ro thiết kế lớn.

### Mức độ

**Medium–High về tài nguyên**

---

## 19. Model phụ thuộc `num_landmarks` phải được sync trước khi khởi tạo

### Vị trí trong code

File: `config_lmk.py`

```python
def require_num_landmarks(self) -> int:
    if self.num_landmarks is None:
        raise ValueError(...)
```

File: `model_lmk.py`

Trong constructor:

```python
self.num_landmarks = cfg.require_num_landmarks()
```

### Lý do

Nếu pipeline tạo model trước khi gọi:

```python
cfg.face.sync_num_landmarks(dataset.num_landmarks)
```

thì model crash ngay.

Đây là contract đúng nhưng dễ bị gọi sai thứ tự, vì `TrainConfig()` mặc định để:

```python
num_landmarks = None
```

Cần đảm bảo train entrypoint luôn:

```text
khởi tạo dataset
→ đọc K
→ sync config
→ khởi tạo model
→ khởi tạo loss
```

### Mức độ

**High nếu train entrypoint không đảm bảo thứ tự**

---

# IV. Lỗi và rủi ro trong Loss

## 20. `target_scores_sum.item()` gây GPU synchronization

### Vị trí trong code

File: `loss_lmk.py`

Trong:

```python
def _branch_loss(...):
```

Có:

```python
target_scores_sum = max(target_scores.sum().item(), 1)
```

### Lý do

`.item()` buộc GPU phải chờ tính toán xong rồi chuyển scalar về CPU.

Hàm `_branch_loss()` chạy hai lần:

```text
o2m
o2o
```

Do đó mỗi training step có ít nhất hai synchronization không cần thiết.

Điều này làm giảm throughput, đặc biệt khi model nhỏ hoặc batch time ngắn.

### Mức độ

**Medium về hiệu năng**

---

## 21. Logging `n_pos` và `n_lmk_pos` tiếp tục gây synchronization

### Vị trí trong code

File: `loss_lmk.py`

Trong `_branch_loss()`:

```python
n_lmk_pos = target_lmk_mask.sum().item()
```

Và:

```python
n_pos = fg_mask.sum().item()
```

### Lý do

Mỗi `.item()` tiếp tục đồng bộ GPU–CPU.

Vì `_branch_loss()` chạy hai nhánh, tổng số sync tăng thêm.

Không ảnh hưởng correctness, nhưng có thể làm pipeline chậm.

### Mức độ

**Low–Medium**

---

## 22. `lmk_gain=1.0` có nguy cơ làm landmark gradient quá yếu

### Vị trí trong code

File: `config_lmk.py`

```python
box_gain: float = 7.5
cls_gain: float = 0.5
dfl_gain: float = 1.5
lmk_gain: float = 1.0
```

File: `loss_lmk.py`

```python
loss_o2m = (
    self.box_gain * iou_m
    + self.cls_gain * cls_m
    + self.dfl_gain * dfl_m
    + self.lmk_gain * lmk_m
)
```

### Lý do

Landmark loss:

- tính trên tọa độ normalized `[0,1]`;
- được trung bình trên `478 × 2 = 956` thành phần;
- dùng Smooth L1;
- tiếp tục weight theo assignment score.

Trong khi box loss được nhân `7.5`.

Có khả năng gradient landmark nhỏ hơn đáng kể so với detection losses, khiến:

- model học bbox tốt nhưng landmark chậm;
- trunk chủ yếu tối ưu cho detection;
- dense landmarks không đạt độ chính xác cần thiết.

Chưa thể kết luận gain đúng chỉ từ code, nhưng cấu hình hiện tại có rủi ro mất cân bằng.

### Mức độ

**Medium–High, cần kiểm tra bằng gradient log**

---

## 23. Tất cả 478 landmark có trọng số bằng nhau

### Vị trí trong code

File: `loss_lmk.py`

Trong:

```python
per_point = F.smooth_l1_loss(
    pred_sel,
    target_sel,
    beta=0.05,
    reduction='none'
)
```

Sau đó trung bình toàn bộ điểm.

### Lý do

Các vùng landmark có mật độ và ý nghĩa khác nhau:

```text
mắt
iris
miệng
mũi
contour
má
trán
```

Tất cả điểm được xem quan trọng như nhau.

Nếu downstream tập trung vào:

- EAR;
- MAR;
- nháy mắt;
- ngáp;
- gaze;
- drowsiness;

thì mắt và miệng đáng lẽ cần trọng số lớn hơn contour.

Không phải lỗi baseline, nhưng có thể làm mục tiêu downstream chưa được tối ưu.

### Mức độ

**Low–Medium**

---

## 24. Loss landmark bị clamp nếu target nằm ngoài bbox mở rộng

### Vị trí trong code

File: `loss_lmk.py`

Trong:

```python
def _encode_landmark_targets(...):
```

Có:

```python
return torch.stack([tx, ty], dim=-1).clamp(0.0, 1.0)
```

### Lý do

Nếu landmark nằm ngoài bbox mở rộng, target bị ép về biên 0 hoặc 1.

Khi đó nhiều vị trí khác nhau bên ngoài bbox bị biến thành cùng một target biên.

Hậu quả:

- mất thông tin khoảng cách;
- gradient landmark bị méo;
- model không thể học chính xác landmark vượt bbox.

Với bbox hiện tại tạo từ landmark, trường hợp này hiếm. Nhưng nếu đổi bbox source sang detector độc lập, đây sẽ là vấn đề đáng kể.

### Mức độ

**Low hiện tại, có thể thành High khi đổi bbox**

---

# V. Lỗi và rủi ro trong Inference

## 25. Nhánh one-to-one vẫn dùng NMS

### Vị trí trong code

File: `inference.py`

Trong:

```python
def predict(...):
```

Có:

```python
if len(filtered_scores) > 0 and self.iou_threshold > 0:
    nms_indices = nms(
        filtered_boxes,
        filtered_scores,
        self.iou_threshold
    )
```

### Lý do

Model thiết kế nhánh one-to-one để suy luận NMS-free.

Nhưng inference vẫn chạy NMS.

Điều này:

- làm pipeline không còn NMS-free thực sự;
- tăng latency;
- có thể che giấu việc o2o chưa học tốt;
- benchmark không phản ánh đúng thiết kế.

NMS vẫn có thể được giữ tạm để an toàn, nhưng không nên mô tả pipeline là hoàn toàn NMS-free.

### Mức độ

**Medium**

---

## 26. `strict=False` có thể nạp checkpoint gần như sai hoàn toàn mà không dừng

### Vị trí trong code

File: `inference.py`

Trong:

```python
def load_weights(self, weights_path: str) -> None:
```

Có:

```python
missing_keys, unexpected_keys = self.model.load_state_dict(
    state_dict,
    strict=False
)
```

Sau đó chỉ in số lượng:

```python
print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")
```

### Lý do

Nếu checkpoint có prefix khác:

```text
module.
model.
ema.
```

thì hầu hết key có thể không khớp.

Do `strict=False`, chương trình vẫn chạy với nhiều trọng số random.

Inference vẫn trả output nhưng không có giá trị.

Đây là lỗi nguy hiểm vì không crash rõ ràng.

### Mức độ

**Critical**

---

## 27. Tự phát hiện `num_landmarks` từ checkpoint không xử lý prefix

### Vị trí trong code

File: `inference.py`

Có:

```python
for key in [
    'head.heads.0.lmk_o2o.weight',
    'head.heads.0.lmk_o2m.weight'
]:
    if key in state_dict:
        ...
```

### Lý do

Hàm chỉ phát hiện khi key đúng tuyệt đối.

Nếu checkpoint có:

```text
module.head.heads.0.lmk_o2o.weight
model.head.heads.0.lmk_o2o.weight
ema.head.heads.0.lmk_o2o.weight
```

thì không phát hiện được K.

Model có thể được khởi tạo với `478` trong khi checkpoint dùng số điểm khác.

### Mức độ

**High**

---

## 28. Giả định `checkpoint['ema']` là state dict trực tiếp

### Vị trí trong code

File: `inference.py`

Có:

```python
if 'ema' in checkpoint and checkpoint['ema'] is not None:
    state_dict = checkpoint['ema']
```

### Lý do

Một số checkpoint lưu EMA dạng:

```python
{
    "ema": {
        "updates": ...,
        "ema": state_dict
    }
}
```

Hoặc lưu một object EMA thay vì state dict thuần.

Nếu format không đúng giả định, `load_state_dict()` sẽ nhận dữ liệu sai.

Các file hiện tại không cung cấp hàm save checkpoint nên chưa thể xác nhận format.

### Mức độ

**High nếu format checkpoint khác**

---

## 29. Tạo matplotlib figure ngay cả khi `show=False`

### Vị trí trong code

File: `inference.py`

Trong `predict()`:

```python
plt.figure(figsize=(9, 9))
plt.imshow(annotated_img)
...
if show:
    plt.show()
```

### Lý do

Dù `show=False`, code vẫn tạo figure.

Figure không được đóng bằng:

```python
plt.close()
```

Khi inference nhiều ảnh:

- số figure tích lũy;
- RAM tăng dần;
- có thể xuất hiện cảnh báo quá nhiều figure;
- batch inference bị memory leak.

### Mức độ

**High khi inference hàng loạt**

---

## 30. `draw_detections()` nhận ảnh RGB nhưng dùng màu theo quy ước OpenCV

### Vị trí trong code

File: `inference.py`

Trong:

```python
def draw_detections(
    self,
    image_rgb: np.ndarray,
    ...
    box_color=(0, 255, 0),
    lmk_color=(255, 0, 0),
):
```

### Lý do

OpenCV ghi màu theo thứ tự BGR, nhưng ảnh đang ở RGB.

Màu xanh lá `(0,255,0)` không bị ảnh hưởng, nhưng:

```python
lmk_color=(255,0,0)
```

sẽ được ghi trực tiếp lên mảng RGB.

Kết quả hiển thị bằng matplotlib vẫn cho màu đỏ, nhưng nếu sau đó lưu bằng `cv2.imwrite()` mà không convert, màu sẽ bị đảo.

Hiện tại chưa có save trong code, nhưng contract màu chưa rõ ràng.

### Mức độ

**Low**

---

## 31. `_prepare_image()` giả định mọi NumPy input là RGB

### Vị trí trong code

File: `inference.py`

Trong:

```python
elif isinstance(image_input, np.ndarray):
    ...
    img_rgb = image_input.copy()
```

### Lý do

NumPy image trong hệ sinh thái OpenCV thường là BGR.

Hàm lại giả định NumPy input là RGB.

Nếu truyền trực tiếp:

```python
img = cv2.imread(...)
inferencer.predict(img)
```

thì model nhận ảnh BGR như RGB.

Hậu quả:

- màu sai;
- phân bố input lệch;
- confidence và landmark có thể giảm.

### Mức độ

**High về usability**

---

## 32. Demo inference dùng model random nhưng output dễ bị hiểu nhầm

### Vị trí trong code

File: `inference.py`

Trong:

```python
if __name__ == '__main__':
```

Có:

```python
inferencer = FaceLandmarkInferencer(
    weights_path=None,
    ...
)
```

Sau đó predict trên hình chữ nhật synthetic.

### Lý do

Model chưa nạp weights, nên output hoàn toàn ngẫu nhiên.

Demo chỉ kiểm tra:

```text
code có chạy
shape có hợp lệ
```

Nó không kiểm tra:

- khả năng detect;
- chất lượng landmark;
- restore coordinate;
- checkpoint;
- NMS-free behavior.

Tên và output demo có thể khiến người đọc tưởng đây là test inference chức năng.

### Mức độ

**Low**

---

## 33. Không giới hạn số detection sau threshold/NMS

### Vị trí trong code

File: `inference.py`

Trong `predict()`:

```python
keep_mask = scores > conf_thresh
```

Sau đó NMS, nhưng không có:

```text
max_det
top-k
```

### Lý do

Nếu model chưa hội tụ hoặc threshold thấp, có thể giữ rất nhiều anchor.

Với 1029 anchor, inference vẫn xử lý toàn bộ candidate.

Hệ quả:

- NMS chậm hơn;
- vẽ nhiều bbox và landmark;
- output lớn;
- có thể làm matplotlib rất chậm.

### Mức độ

**Medium**

---

# VI. Lỗi thiết kế liên module

## 34. Bbox target được tạo từ chính landmark target

### Vị trí trong pipeline

Nguồn JSONL tạo bbox bằng min/max landmark MediaPipe.

Dataset đọc bbox đó làm target detection.

Model đồng thời học:

```text
bbox
dense landmarks
```

### Lý do

Bbox không phải annotation độc lập.

Điều này tạo sự phụ thuộc:

```text
MediaPipe landmark lỗi
→ bbox cũng lỗi
→ cả hai target cùng sai theo một hướng
```

Model không học bbox mặt theo định nghĩa của detector độc lập mà học bounding rectangle của landmark mesh.

Không nhất thiết sai, nhưng cần hiểu đúng ý nghĩa bbox.

### Mức độ

**Medium về chất lượng nhãn**

---

## 35. MediaPipe là nguồn duy nhất cho cả detect và landmark

### Vị trí trong pipeline

Script curation sử dụng MediaPipe Face Mesh để:

```text
phát hiện số mặt
tạo bbox
tạo landmark
ước lượng yaw
lọc mặt nhỏ
```

### Lý do

Nếu MediaPipe fail:

```text
num_faces = 0
không có bbox
không có landmarks
ảnh bị loại
```

Một model duy nhất quyết định nhiều feature đồng thời.

Điều này tạo bias mạnh theo miền của MediaPipe và loại bỏ các ảnh khó mà MediaPipe không xử lý được.

### Mức độ

**Medium về dataset bias**

---

## 36. Không có kiểm tra leakage hoặc near-duplicate trong train/validation

### Vị trí trong pipeline

Các file hiện tại chỉ cung cấp:

```text
train_root_dir
val_root_dir
```

Không có logic group split hoặc duplicate-aware split.

### Lý do

Dataset ảnh khuôn mặt có thể chứa:

- frame liên tiếp;
- ảnh crop cùng nguồn;
- ảnh resize;
- ảnh cùng người;
- near-duplicate.

Nếu random split theo ảnh, train và validation có thể chứa ảnh gần giống nhau.

Metric validation sẽ cao giả tạo.

### Mức độ

**High về độ tin cậy đánh giá**

---

## 37. Không có augmentation photometric chuyên biệt

### Vị trí trong code

File: `dataset_lmk.py`

Transform mặc định:

```python
T.ToImage()
T.Resize(...)
T.ToDtype(...)
```

### Lý do

Pipeline không bổ sung robustness với:

```text
brightness
contrast
gamma
compression
noise
blur nhẹ
color jitter
```

Trong khi Data1 và Data2 có phân bố blur và chất lượng khác nhau.

Model có thể học bias theo nguồn dataset thay vì học representation tổng quát.

### Mức độ

**Medium về generalization**

---

# VII. Thứ tự ưu tiên sửa

## Nhóm 1 — Phải sửa trước khi chạy train

1. Sửa `Images` thành `images` hoặc đưa tên thư mục vào config.
2. Sửa `annotations_all.jsonl` thành `merged_faces.jsonl` hoặc truyền rõ trong config.
3. Đồng nhất resize giữa train và inference.
4. Truyền `image_size=224` vào `init_stride_bias()`.
5. Xác nhận `trunk_feat_channels` với output thật của neck.
6. Hỗ trợ đúng format checkpoint trong `load_trunk()`.
7. Xác thực toàn dataset dùng đúng một số lượng landmark.
8. Đảm bảo gọi `sync_num_landmarks()` trước khi khởi tạo model và loss.

## Nhóm 2 — Phải sửa trước khi inference thật

1. Không cho phép checkpoint load ratio thấp.
2. Normalize prefix checkpoint.
3. Xác thực format EMA.
4. Không tạo matplotlib figure khi `show=False`.
5. Làm rõ NumPy input là RGB hay BGR.
6. Thêm `max_det` hoặc top-k.

## Nhóm 3 — Nên sửa để train ổn định và đúng hơn

1. Xây augmentation cập nhật đồng thời ảnh, bbox và landmark.
2. Có permutation map cho horizontal flip.
3. Xem xét giảm `lmk_margin`.
4. Theo dõi weighted gradient của landmark loss.
5. Xem xét khởi tạo landmark final conv bằng normal rất nhỏ thay vì toàn 0.
6. Giảm `prefetch_factor` nếu RAM cao.
7. Loại `.item()` không cần thiết trong loss.

---

# VIII. Kết luận

Kiến trúc tổng thể có cơ sở tốt:

```text
YOLO/NMS-free trunk
+ o2m/o2o detection
+ DFL bbox
+ dense MediaPipe landmarks
+ bbox-relative landmark encoding
```

Các vấn đề nguy hiểm nhất không nằm ở công thức kiến trúc mà nằm ở contract giữa các module:

```text
tên thư mục
tên JSONL
resize policy
image-size-dependent bias
checkpoint format
feature channels
num_landmarks synchronization
```

Nếu chưa sửa các lỗi Critical và High, pipeline có thể:

- không đọc được dữ liệu;
- train trên target sai;
- load checkpoint không đầy đủ nhưng vẫn chạy;
- đánh giá validation không đáng tin;
- inference bằng model gần như random mà không phát hiện rõ.
