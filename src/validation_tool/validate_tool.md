# Tài liệu mô tả bộ công cụ Validation — NMSFreeDetector

> **Thư mục**: `src/validation_tool/`  
> **Cập nhật**: 2026-07-23  
> **Tổng số test-case**: **115** (27 Loss · 49 Model · 24 Pipeline · 15 Dataloader)

---

## Mục lục

1. [Tổng quan kiến trúc bộ công cụ](#1-tổng-quan-kiến-trúc-bộ-công-cụ)
2. [Cách chạy](#2-cách-chạy)
3. [validate_loss.py — 27 test-case](#3-validate_losspy--27-test-case)
4. [validate_model.py — 49 test-case](#4-validate_modelpy--49-test-case)
5. [validate_pipeline.py — 24 test-case](#5-validate_pipelinepy--24-test-case)
6. [validate_dataloader.py — 15 test-case](#6-validate_dataloaderpy--15-test-case)
7. [Phạm vi KHÔNG test](#7-phạm-vi-không-test)
8. [Quy ước trạng thái (PASS / FAIL / ERROR / SKIP)](#8-quy-ước-trạng-thái)

---

## 1. Tổng quan kiến trúc bộ công cụ

```
src/validation_tool/
├── validate_common.py        ← lớp Reporter, Skip, get_device() dùng chung
├── validate_loss.py          ← 27 test: bbox_iou, TAL, BboxLoss, DetectionLoss
├── validate_model.py         ← 49 test: blocks, backbone_neck, head, model
├── validate_pipeline.py      ← 24 test: EMA, engine, config
├── validate_dataloader.py    ← 15 test: letterbox, collate_fn, augmenter
└── run_all_validation.py     ← điểm vào duy nhất, gộp tất cả 4 suite
```

**Nguyên tắc thiết kế:**
- Mọi test đều là **zero-data**: tổng hợp tensor bằng `torch.randn` / `torch.rand`, không cần dataset thật.
- Toàn bộ dùng `Reporter` duy nhất từ `validate_common.py` — không tự định nghĩa Reporter ở mỗi file.
- File lưu checkpoint dùng `tempfile.TemporaryDirectory()` để không để lại file rác trên đĩa.
- Mỗi suite có hàm `run(device, verbose_traceback) -> Reporter` để `run_all_validation.py` gộp mà không cần monkey-patching.

---

## 2. Cách chạy

```bash
# Chạy toàn bộ (tự phát hiện thiết bị)
python -m src.validation_tool.run_all_validation

# Chỉ định thiết bị
python -m src.validation_tool.run_all_validation --device cuda

# Bỏ qua một số suite
python -m src.validation_tool.run_all_validation --skip loss,dataloader

# In full traceback khi ERROR
python -m src.validation_tool.run_all_validation --verbose-traceback

# Chạy từng suite riêng lẻ
python -m src.validation_tool.validate_loss
python -m src.validation_tool.validate_model
python -m src.validation_tool.validate_pipeline
python -m src.validation_tool.validate_dataloader
```

---

## 3. validate_loss.py — 27 test-case

Module được test: `src/train/loss.py`  
Các thành phần: `bbox_iou`, `dist2bbox`, `bbox2dist`, `TaskAlignedAssigner`, `BboxLoss`, `DetectionLoss`

---

### Nhóm 3.1: Tiện ích hình học (`bbox_utils`) — 9 test

#### TC-L-01: IoU hai box giống hệt nhau → ~1.0
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Box `a = b = [0,0,10,10]` (xyxy) |
| **Hàm** | `bbox_iou(a, a, CIoU=True)` |
| **Đầu ra mong đợi** | Giá trị trả về ∈ `[1.0 − 1e-5, 1.0]` |
| **Ý nghĩa** | Xác nhận điều kiện biên cơ bản nhất: hai box hoàn toàn trùng nhau phải có IoU bằng 1. Nếu fail → lỗi nghiêm trọng trong công thức giao/hợp. |

#### TC-L-02: IoU/CIoU hai box không giao nhau
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `a = [0,0,10,10]`, `b = [20,20,30,30]` (tách biệt hoàn toàn) |
| **Hàm** | `bbox_iou(a, b, CIoU=False)` và `bbox_iou(a, b, CIoU=True)` |
| **Đầu ra mong đợi** | IoU thường = 0.0 ; CIoU < 0.0 |
| **Ý nghĩa** | Khi hai box không giao nhau: IoU thường = 0 (không có phần giao). CIoU âm vì có penalty khoảng cách tâm — điều này quan trọng để loss CIoU phạt đúng hướng các dự đoán xa. |

#### TC-L-03: CIoU ≤ IoU thường (penalty đúng dấu)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `a = [0,0,10,10]`, `b = [5,5,15,15]` (giao nhau một phần) |
| **Hàm** | So sánh `bbox_iou(CIoU=False)` vs `bbox_iou(CIoU=True)` |
| **Đầu ra mong đợi** | `ciou ≤ iou + 1e-6` |
| **Ý nghĩa** | CIoU thêm penalty khoảng cách tâm và tỷ lệ aspect ratio — nên CIoU ≤ IoU thường. Nếu CIoU > IoU thường thì dấu penalty bị sai, loss sẽ không hội tụ đúng. |

#### TC-L-04: IoU đối xứng: iou(a,b) == iou(b,a)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `a = [10,20,50,80]`, `b = [30,10,70,60]` (hai box ngẫu nhiên có giao nhau) |
| **Hàm** | `bbox_iou(a, b)` và `bbox_iou(b, a)` |
| **Đầu ra mong đợi** | `|iou(a,b) - iou(b,a)| < 1e-5` |
| **Ý nghĩa** | IoU là hàm đối xứng theo định nghĩa toán học: `IoU(A,B) = IoU(B,A)`. Nếu không đối xứng → lỗi trong cách tính giao/hợp (ví dụ nhầm thứ tự clamp). |

#### TC-L-05: IoU box con nằm hoàn toàn trong box cha
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `outer = [0,0,10,10]` (diện tích 100), `inner = [2,2,8,8]` (diện tích 36) |
| **Hàm** | `bbox_iou(outer, inner, CIoU=False)` |
| **Đầu ra mong đợi** | `IoU = inter / union = 36 / 100 = 0.36` (vì union = outer_area vì inner ⊂ outer) |
| **Ý nghĩa** | Kiểm tra trường hợp box nằm gọn bên trong: phần giao = diện tích inner, phần hợp = diện tích outer. Thường bị lỗi nếu công thức dùng `max` sai thứ tự. |

#### TC-L-06: bbox_iou batch N=8: shape đúng, tất cả hữu hạn
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `a, b` mỗi cái shape `(8, 4)` — 8 cặp box ngẫu nhiên hợp lệ (x2>x1, y2>y1) |
| **Hàm** | `bbox_iou(a, b, CIoU=True)` |
| **Đầu ra mong đợi** | `out.shape == (8,)` ; `torch.isfinite(out).all() == True` |
| **Ý nghĩa** | Xác nhận `bbox_iou` xử lý batch nhiều cặp box cùng lúc mà không dùng vòng lặp Python. Cũng phát hiện NaN/Inf do chia cho union = 0 khi box degenerate. |

#### TC-L-07: dist2bbox / bbox2dist round-trip chính xác (xyxy mode)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `anchors = [[5,5],[12,8]]` ; `dist = [[2,2,3,3],[1,4,2,1]]` (ltrb) |
| **Hàm** | `dist2bbox(dist, anchors, xywh=False)` → `box` ; `bbox2dist(anchors, box, reg_max=16)` → `back` |
| **Đầu ra mong đợi** | `box == [[3,3,8,8],[11,4,14,9]]` (xyxy) ; `back == dist` |
| **Ý nghĩa** | `dist2bbox` và `bbox2dist` phải là nghịch đảo của nhau. Lỗi ở đây sẽ khiến việc decode box ra tọa độ pixel bị sai ngay cả khi model predict đúng distribution. |

#### TC-L-08: dist2bbox xywh mode: center và kích thước hợp lệ
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `anchors = [[8,8]]` ; `dist = [[2,3,4,5]]` (l,t,r,b) ; `xywh=True` |
| **Hàm** | `dist2bbox(dist, anchors, xywh=True)` |
| **Đầu ra mong đợi** | `cx = ax + (r-l)/2`, `cy = ay + (b-t)/2`, `w = l+r > 0`, `h = t+b > 0` |
| **Ý nghĩa** | Khi dùng ở chế độ xuất ONNX/inference, đầu ra là (cx,cy,w,h). Test này xác nhận công thức chuyển đổi xywh đúng và w,h dương (không âm do nhầm dấu). |

#### TC-L-09: bbox2dist: clamp khoảng cách, không NaN khi box vượt reg_max
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `anchors = [[5,5]]` ; `box = [[0,0,500,500]]` (khoảng cách tới anchor rất lớn, >> reg_max=16) |
| **Hàm** | `bbox2dist(anchors, box, reg_max=16)` |
| **Đầu ra mong đợi** | `torch.isfinite(out).all() == True` (không NaN/Inf) |
| **Ý nghĩa** | `bbox2dist` phải clamp khoảng cách về `[0, reg_max-1-ε]` trước khi trả về. Nếu không clamp, giá trị vượt `reg_max` sẽ thành index ngoài softmax → NaN, làm crash toàn bộ DFL loss. |

---

### Nhóm 3.2: Task-Aligned Assigner (`tal`) — 6 test

#### TC-L-10: topk đúng số lượng, ưu tiên anchor align_metric cao
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 GT box `[3,0,8,10]` ; A=20 anchor ; `pd_scores[0,5,1]=0.9`, `pd_scores[0,6,1]=0.85` (2 anchor điểm cao nhất) ; `topk=3` |
| **Hàm** | `TaskAlignedAssigner(topk=3)(...)` |
| **Đầu ra mong đợi** | `fg.sum() == 3` ; anchor 5 và 6 đều nằm trong `fg=True` |
| **Ý nghĩa** | Xác nhận TAL chọn đúng `topk` anchor tốt nhất và ưu tiên anchor có align_metric cao (tích score×iou). Nếu fail → model bị assign target sai, training không hội tụ. |

#### TC-L-11: Anchor dương phải nằm trong GT box
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | GT box `[3,0,8,10]` ; anchor phân bố đều trên trục x ; `topk=5` |
| **Hàm** | `TaskAlignedAssigner(topk=5)(...)` |
| **Đầu ra mong đợi** | Với mọi anchor `i` có `fg[0,i]==True`: `3 < anchor_x < 8` |
| **Ý nghĩa** | TAL chỉ cho phép anchor nằm **bên trong** GT box trở thành positive. Anchor ngoài GT box không bao giờ được assign — đây là constraint "center-in-gt" cốt lõi của TAL. |

#### TC-L-12: Anchor bị tranh chấp giữa 2 GT → giữ GT có IoU cao hơn
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 2 GT box chồng lên nhau `[1.8,0,3.2,1]` và `[0,0,5,1]` ; anchor nằm trong cả 2 box |
| **Hàm** | `TaskAlignedAssigner(topk=5)(...)` — kiểm tra `tgi` (target GT index) của mỗi anchor |
| **Đầu ra mong đợi** | Với anchor nằm trong **cả 2** GT box: `iou(pred_box, chosen_gt) >= iou(pred_box, other_gt) - 1e-5` |
| **Ý nghĩa** | Khi anchor bị tranh chấp, TAL phải giải quyết bằng cách chọn GT có IoU cao nhất (không hardcode index). Phiên bản cũ hardcode `tgi==1`, đây là bug vì phụ thuộc thứ tự GT trong tensor. |

#### TC-L-13: GT padding (mask_gt=False) không ảnh hưởng kết quả
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 2 GT: GT0 hợp lệ `mask_gt=True`, GT1 là padding `mask_gt=False` và box = `[0,0,0,0]` |
| **Hàm** | `TaskAlignedAssigner(topk=2)(...)` |
| **Đầu ra mong đợi** | `fg.sum() <= 2` ; mọi anchor positive đều được gán `tgi==0` (GT hợp lệ), không có anchor nào gán GT1 |
| **Ý nghĩa** | Batch có số GT khác nhau giữa các ảnh → phải dùng padding. Test xác nhận padding không bị assign nhầm làm target, tránh model học các GT giả. |

#### TC-L-14: Batch hoàn toàn không có GT (M=0)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `gt_bboxes.shape = (1,0,4)`, `mask_gt.shape = (1,0,1)` — không có GT nào |
| **Hàm** | `TaskAlignedAssigner(topk=3)(...)` |
| **Đầu ra mong đợi** | `fg.sum() == 0` ; `target_scores.sum() == 0` ; không crash |
| **Ý nghĩa** | Ảnh không có object nào (all-background). TAL phải xử lý edge case M=0 mà không gây index-out-of-bounds hay NaN. Quan trọng với dataset có nhiều ảnh background. |

#### TC-L-15: Multi-image batch (B=4, số GT khác nhau) không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=4 ảnh: ảnh 0 có 2 GT, ảnh 1 có 4 GT, **ảnh 2 không có GT** (mask_gt=False hết), ảnh 3 có 5 GT |
| **Hàm** | `TaskAlignedAssigner(topk=3)(...)` với batch B=4 |
| **Đầu ra mong đợi** | `fg.shape == (4, A)` ; `fg[2].sum() == 0` (ảnh không GT phải không có positive) |
| **Ý nghĩa** | Kiểm tra TAL hoạt động đúng với batch thực tế — mỗi ảnh có số GT khác nhau, xử lý đồng thời. Phát hiện lỗi broadcasting hay index nhầm giữa các ảnh trong batch. |

---

### Nhóm 3.3: BboxLoss (`bbox_loss`) — 4 test

#### TC-L-16: Loss hữu hạn, không âm, backward không NaN
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=1, A=4 anchor ; `fg_mask = [True, True, False, False]` ; 2 target box hợp lệ ; `pred_dist` ngẫu nhiên (seed=0) |
| **Hàm** | `BboxLoss(reg_max=16)(pred_dist, pred_bboxes, anchors, target_bboxes, target_scores, sum, fg_mask)` |
| **Đầu ra mong đợi** | `loss_iou >= 0` ; `loss_dfl >= 0` ; `isfinite(loss)` ; `loss.backward()` không NaN trong grad |
| **Ý nghĩa** | Kiểm tra điều kiện cơ bản của loss: không âm và hữu hạn. Nếu BboxLoss trả về âm → dấu CIoU bị sai. Nếu grad NaN → lỗi trong DFL (log của số âm, chia cho 0). |

#### TC-L-17: Trường hợp không có anchor dương (fg_mask rỗng)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `fg_mask = [False, False, False, False]` — không có anchor nào là positive |
| **Hàm** | `BboxLoss()(...)` với `fg_mask` toàn False |
| **Đầu ra mong đợi** | `loss_iou == 0.0` ; `loss_dfl == 0.0` ; `backward()` không crash |
| **Ý nghĩa** | Bước training đầu tiên hoặc ảnh chứa toàn background → không có anchor positive nào. BboxLoss phải trả về 0 (không phạt) và backward vẫn hoạt động không lỗi. |

#### TC-L-18: Gradient pred_dist: không dead, không exploding
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `pred_dist` ngẫu nhiên (seed=2) ; 2 anchor positive |
| **Hàm** | `BboxLoss()(...).backward()` ; đo `pred_dist.grad.norm()` |
| **Đầu ra mong đợi** | `grad_norm > 1e-8` (không dead) ; `grad_norm < 1e6` (không exploding) |
| **Ý nghĩa** | Phát hiện 2 vấn đề phổ biến: (1) **Dead gradient** — softmax + DFL bị vanish do logit quá âm; (2) **Exploding gradient** — chia cho số quá nhỏ. Cả 2 đều khiến training mất tác dụng. |

#### TC-L-19: loss_iou = 0 khi pred_box chính xác bằng target_box
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `pred_bboxes = target_bboxes` (dự đoán hoàn hảo) ; 2 anchor positive |
| **Hàm** | `BboxLoss()(..., pred_bboxes=target_bboxes, ...)` |
| **Đầu ra mong đợi** | `loss_iou < 1e-4` (xấp xỉ 0) |
| **Ý nghĩa** | Kiểm tra cần thiết: khi model dự đoán đúng hoàn toàn, CIoU loss phải bằng 0. Nếu không → loss có hằng số bias, model không thể đạt optimum dù dự đoán chuẩn xác. |

---

### Nhóm 3.4: DetectionLoss tích hợp (`detection_loss`) — 8 test

#### TC-L-20: Forward+backward bình thường, grad lan tới backbone
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=2 ảnh `(2,3,320,320)` ; ảnh 1 có 2 GT, ảnh 2 có 1 GT ; nc=7 |
| **Hàm** | `model.forward()` → `DetectionLoss()` → `total.backward()` |
| **Đầu ra mong đợi** | `total > 0` và hữu hạn ; mọi tham số (kể cả backbone) đều có `grad != None` và `isfinite(grad)` |
| **Ý nghĩa** | Test tích hợp đầu-cuối: loss tính được và gradient lan ngược qua toàn bộ mạng (head → neck → backbone). Phát hiện lỗi detach() nhầm chỗ hoặc `requires_grad=False` sai. |

#### TC-L-21: Cả batch không có GT nào (chỉ học negative)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=2 ảnh ; cả 2 ảnh đều có `boxes.shape = (0,4)` (không có GT) |
| **Hàm** | `DetectionLoss()(out, targets)` |
| **Đầu ra mong đợi** | `total` hữu hạn ; `o2m/n_pos == 0` ; `o2o/n_pos == 0` ; `backward()` không crash |
| **Ý nghĩa** | Batch full-background: loss chỉ từ classification (phạt false positive). Quan trọng với Object365 có nhiều ảnh không chứa object trong top-80 class. |

#### TC-L-22: Số lượng GT khác nhau giữa các ảnh trong batch
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=3 ảnh: ảnh 0 không GT, ảnh 1 có 1 GT, ảnh 2 có 5 GT |
| **Hàm** | `DetectionLoss(topk_o2o=1)(out, targets)` |
| **Đầu ra mong đợi** | `o2o/n_pos == 6` (= 0+1+5 = tổng GT thực trong batch với topk_o2o=1) |
| **Ý nghĩa** | Xác nhận padding/masking GT đúng khi số GT không đều giữa các ảnh. `topk_o2o=1` → mỗi GT tạo ra đúng 1 positive ở nhánh o2o, nên tổng positive = tổng GT thực. |

#### TC-L-23: [SANITY] Overfit 1 ảnh, loss phải giảm mạnh
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 ảnh `(1,3,320,320)` cố định ; 1 GT box `[50,50,150,180]` ; AdamW lr=1e-3 ; 60 bước |
| **Hàm** | Training loop 60 epoch với model nhỏ (w=(16,32,64,128,160)) |
| **Đầu ra mong đợi** | `losses[-1] < losses[0] * 0.5` (loss cuối < 50% loss đầu) |
| **Ý nghĩa** | **Sanity check quan trọng nhất**: nếu model và loss đều đúng, nó phải có thể overfit 1 sample đơn. Nếu fail → lỗi nghiêm trọng trong pipeline (gradient bị block, loss bị normalize sai, assign target bị lỗi). |

#### TC-L-24: GT box ở hai thái cực kích thước (rất nhỏ / rất lớn)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 ảnh `320×320` ; 2 GT: box nhỏ `[10,10,18,18]` (~8×8 px, nhỏ hơn cả stride P3=8) và box lớn `[5,5,315,315]` (~gần hết ảnh) |
| **Hàm** | `DetectionLoss()(out, targets)` |
| **Đầu ra mong đợi** | `isfinite(total)` ; `backward()` không crash |
| **Ý nghĩa** | Kiểm tra edge case kích thước cực đoan: box rất nhỏ có thể không có anchor nào chứa (n_pos=0 cho box đó), box rất lớn phủ hầu hết anchors. Xác nhận không crash do NaN khi n_pos=0. |

#### TC-L-25: items dict trả về đủ key (loss, o2m/\*, o2o/\*)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | B=1, 1 GT box |
| **Hàm** | `_, items = DetectionLoss()(out, targets)` |
| **Đầu ra mong đợi** | `items` chứa đủ 9 key: `{"loss", "o2m/iou", "o2m/cls", "o2m/dfl", "o2m/n_pos", "o2o/iou", "o2o/cls", "o2o/dfl", "o2o/n_pos"}` |
| **Ý nghĩa** | `engine.py` dùng `items` để log TensorBoard và console. Nếu thiếu key → KeyError ở bước logging, crash training ở cuối epoch đầu tiên. |

#### TC-L-26: Loss hữu hạn khi model ở eval() mode
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `model.eval()` ; B=2, 2 GT ; `torch.no_grad()` context |
| **Hàm** | `model(x)` → `DetectionLoss()(out, targets)` |
| **Đầu ra mong đợi** | `isfinite(total)` ; `"o2m" in out` (shortcut inference chưa bật → vẫn có o2m) |
| **Ý nghĩa** | Trong quá trình training, `engine.validate()` gọi `model.eval()` rồi tính validation loss. Test xác nhận loss vẫn tính được ở eval mode và output vẫn có `o2m` (cần cho DetectionLoss). |

#### TC-L-27: Loss > 0 khi có GT (không bị zero-out)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 ảnh ; 1 GT box `[10,10,200,200]` ; `torch.manual_seed(42)` |
| **Hàm** | `DetectionLoss()(out, targets)` |
| **Đầu ra mong đợi** | `total.item() > 0` |
| **Ý nghĩa** | Phát hiện lỗi mask bị sai khiến toàn bộ loss bị zero-out dù có GT. Nếu loss = 0 với GT → model không học được gì cả mà trainer không báo lỗi (chỉ thấy loss thấp bất thường). |

---

## 4. validate_model.py — 49 test-case

Modules được test: `src/blocks.py`, `src/backbone_neck.py`, `src/head.py`, `src/model.py`

---

### Nhóm 4.1: blocks.py — 22 test

#### TC-M-01: autopad tính đúng padding SAME
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `k=3` ; `k=1` ; `k=5` ; `k=3, p=0` (tường minh) |
| **Hàm** | `autopad(k)`, `autopad(k, p)` |
| **Đầu ra mong đợi** | `autopad(3)=1` ; `autopad(1)=0` ; `autopad(5)=2` ; `autopad(3,p=0)=0` |
| **Ý nghĩa** | `autopad` tính padding để giữ nguyên kích thước không gian (SAME). Nếu sai → mọi Conv với `autopad` sẽ cho output size sai, gây lỗi dimension mismatch ở skip connection. |

#### TC-M-02: autopad với dilation: tính đúng k_eff trước
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `k=3, d=2` (dilation=2) |
| **Hàm** | `autopad(k=3, d=2)` |
| **Đầu ra mong đợi** | `= 2` (vì `k_eff = d*(k-1)+1 = 5` → `pad = 5//2 = 2`) |
| **Ý nghĩa** | Khi dùng dilated convolution, kernel hiệu dụng rộng hơn. Nếu không tính `k_eff` → padding không đủ → feature map bị thu nhỏ → crash ở skip connection. |

#### TC-M-03: Conv (k=3, s=2) giảm kích thước đúng 1/2, đúng số kênh
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Conv(3, 16, 3, 2)` ; input `(2,3,32,32)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (2,16,16,16)` |
| **Ý nghĩa** | Conv với stride=2 phải giảm HxW đúng một nửa (32→16). Sai → toàn bộ Backbone bị sai kích thước P3/P4/P5. |

#### TC-M-04: Conv(act=False) không áp dụng SiLU
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Conv(3, 3, 1, 1, act=False)` |
| **Kiểm tra** | `isinstance(m.act, nn.Identity)` |
| **Đầu ra mong đợi** | `m.act` là `nn.Identity` |
| **Ý nghĩa** | Một số lớp (DWConv + pw trong CIB) không cần activation sau Conv. Nếu vẫn dùng SiLU → biến đổi không mong muốn, ảnh hưởng biểu diễn đặc trưng. |

#### TC-M-05: Conv: BatchNorm2d có mặt và bias=False trong Conv2d
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Conv(3, 16, 3)` |
| **Kiểm tra** | `isinstance(m.bn, nn.BatchNorm2d)` ; `m.conv.bias is None` |
| **Đầu ra mong đợi** | BN có mặt ; bias của Conv2d là None |
| **Ý nghĩa** | Theo chuẩn thiết kế: khi có BN thì Conv không cần bias (BN đã có learnable bias). Nếu Conv có thêm bias → tham số dư thừa, BN không normalize đúng. |

#### TC-M-06: DWConv đúng số groups = gcd(c1,c2) (depthwise)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DWConv(16, 16, 3, 1)` ; input `(1,16,8,8)` |
| **Kiểm tra** | `m.conv.groups == gcd(16,16) == 16` ; `output.shape == (1,16,8,8)` |
| **Đầu ra mong đợi** | `groups=16` ; shape không đổi |
| **Ý nghĩa** | Depthwise convolution phải có `groups = c1` (khi `c1==c2`). Nếu `groups` sai → không phải depthwise nữa, tốn nhiều tham số và sai kiến trúc CIB. |

#### TC-M-07: DWConv với c1≠c2: groups=gcd, shape đúng
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DWConv(12, 16, 3, 1)` (c1=12 ≠ c2=16) ; input `(1,12,8,8)` |
| **Kiểm tra** | `m.conv.groups == gcd(12,16) == 4` ; `output.shape == (1,16,8,8)` |
| **Đầu ra mong đợi** | `groups=4` ; shape `(1,16,8,8)` |
| **Ý nghĩa** | Trường hợp kênh vào và ra khác nhau: DWConv vẫn phải dùng `groups=gcd(c1,c2)` (grouped conv), không crash và cho output đúng số kênh. |

#### TC-M-08: Bottleneck: residual add chỉ khi c1==c2 và shortcut=True
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Bottleneck(16,16,shortcut=True)` ; `Bottleneck(16,32,shortcut=True)` ; `Bottleneck(16,16,shortcut=False)` |
| **Kiểm tra** | `m.add` flag + zero-weight residual test |
| **Đầu ra mong đợi** | `add=True` chỉ khi `c1==c2 AND shortcut=True` ; khi weights=0 → `output==input` |
| **Ý nghĩa** | Residual connection chỉ có ý nghĩa khi c1==c2 (cùng số kênh). Nếu add ngay cả khi c1≠c2 → dimension mismatch crash. Zero-weight test xác nhận skip connection thực sự hoạt động. |

#### TC-M-09: Bottleneck(c1≠c2): không add, output channels=c2
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Bottleneck(16, 32, shortcut=True)` ; input `(1,16,8,8)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (1,32,8,8)` (không crash, output theo c2) |
| **Ý nghĩa** | Khi c1≠c2, không có residual add → output shape = (B, c2, H, W). Xác nhận không có attempt add sai kiểu tensor. |

#### TC-M-10: C2f: đúng shape đầu ra, grad lan về input
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `C2f(32, 64, n=3, shortcut=True)` ; input `(2,32,16,16)` với `requires_grad=True` |
| **Hàm** | `y = m(x)` ; `y.sum().backward()` |
| **Đầu ra mong đợi** | `y.shape == (2,64,16,16)` ; `x.grad != None` và `isfinite(x.grad)` |
| **Ý nghĩa** | C2f là khối chính trong Backbone. Phải đổi kênh đúng (32→64) và giữ HxW. Backward xác nhận gradient lan qua cả split/concat operations. |

#### TC-M-11: C2f (n=0 bottlenecks) edge case không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `C2f(16, 32, n=0, shortcut=True)` ; input `(1,16,8,8)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (1,32,8,8)` ; không crash |
| **Ý nghĩa** | `n=0` nghĩa là không có Bottleneck nào (chỉ có cv1+cv2). Edge case này có thể crash nếu code dùng `torch.cat([...])` với list rỗng. |

#### TC-M-12: CIB / C2fCIB đúng shape đầu ra
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `CIB(32, 32, shortcut=True)` và `C2fCIB(32, 64, n=2)` ; input `(1,32,8,8)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `CIB output == input.shape` (residual) ; `C2fCIB output.shape == (1,64,8,8)` |
| **Ý nghĩa** | CIB là phiên bản inverted bottleneck trong Backbone stage3/4. Xác nhận residual connection đúng và kênh ra đúng với C2fCIB. |

#### TC-M-13: SPPF giữ nguyên HxW (maxpool stride=1)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `SPPF(64, 64, k=5)` ; input `(1,64,20,20)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (1,64,20,20)` |
| **Ý nghĩa** | SPPF dùng MaxPool với `stride=1, pad=k//2` để mở rộng receptive field mà không thu nhỏ feature map. Nếu stride sai → feature map bị thu nhỏ, không concat được. |

#### TC-M-14: SPPF: fuse multi-scale, output hữu hạn
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `SPPF(32, 32, k=5)` ; input `(1,32,10,10)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape` đúng ; `isfinite(output).all()` |
| **Ý nghĩa** | SPPF nối 4 bản (x, pool1, pool2, pool3) → fuse multi-scale. Output không được là NaN (BN + activation phải ổn định với input ngẫu nhiên). |

#### TC-M-15: DFL tính đúng kỳ vọng (soft-argmax) trên reg_max bin
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DFL(reg_max=8)` ; logit one-hot tại bin=5 (giá trị 20.0, còn lại -20.0) ; shape `(2, 4*8, 3)` |
| **Hàm** | `dfl(logits)` |
| **Đầu ra mong đợi** | Output xấp xỉ `5.0` với `atol=1e-3` ; `conv.weight` frozen (no grad) |
| **Ý nghĩa** | DFL tính kỳ vọng `E[bin] = sum(softmax(logit) * [0,1,...,reg_max-1])`. Khi one-hot tại bin=5 → kỳ vọng = 5. Nếu sai → box decoded bị offset. |

#### TC-M-16: DFL: conv.weight chính xác bằng arange(reg_max)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DFL(reg_max=12)` |
| **Kiểm tra** | `dfl.conv.weight.data == arange(12).reshape(1,12,1,1)` |
| **Đầu ra mong đợi** | Weights khớp tuyệt đối |
| **Ý nghĩa** | DFL dùng Conv1x1 với trọng số cố định `[0,1,...,reg_max-1]` để tính kỳ vọng. Nếu trọng số sai từ đầu → mọi box decode ra sai hoàn toàn. |

#### TC-M-17: Attention: assert dim chia hết cho num_heads
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Attention(32, num_heads=4)` (hợp lệ) ; `Attention(30, num_heads=4)` (30 % 4 ≠ 0) |
| **Hàm** | Constructor |
| **Đầu ra mong đợi** | Case hợp lệ: không lỗi ; Case không hợp lệ: `AssertionError` |
| **Ý nghĩa** | Multi-head attention yêu cầu `dim % num_heads == 0` để chia đều các head. Nếu không assert → lỗi shape ở runtime hoặc kết quả sai. |

#### TC-M-18: Attention giữ nguyên shape đầu vào/ra
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Attention(32, num_heads=4)` ; input `(1,32,10,10)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (1,32,10,10)` |
| **Ý nghĩa** | Attention là residual block (self-attention + FFN + skip). Phải giữ nguyên shape để nối vào phần còn lại của C2fPSA. |

#### TC-M-19: Attention: gradient đầy đủ, không dead neuron
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Attention(32, num_heads=4)` ; input `(1,32,8,8)` với `requires_grad=True` |
| **Hàm** | `m(x).sum().backward()` |
| **Đầu ra mong đợi** | `x.grad != None` và `isfinite` ; mọi tham số đều có `grad != None` |
| **Ý nghĩa** | Phát hiện dead neuron ngay từ khởi tạo (do initialization quá lớn → softmax saturate → gradient ≈ 0). Cũng xác nhận không có tham số bị detach nhầm. |

#### TC-M-20: C2fPSA: assert c1==c2, giữ nguyên shape khi hợp lệ
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `C2fPSA(32, 64)` (c1≠c2, không hợp lệ) ; `C2fPSA(32, 32, n=2)` (hợp lệ) |
| **Hàm** | Constructor + forward |
| **Đầu ra mong đợi** | `c1≠c2`: `AssertionError` ; `c1==c2`: `output.shape == input.shape` |
| **Ý nghĩa** | C2fPSA (C2f + Position-Sensitive Attention) chỉ hoạt động khi c1==c2 vì có residual connection. Giữ nguyên shape để nối với SPPF trong Stage4. |

#### TC-M-21: SCDown (pointwise + depthwise stride) đúng shape
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `SCDown(32, 64, 3, 2)` ; input `(1,32,16,16)` |
| **Hàm** | `m(x)` |
| **Đầu ra mong đợi** | `output.shape == (1,64,8,8)` (stride=2 giảm HxW một nửa) |
| **Ý nghĩa** | SCDown thay thế Conv stride-2 bằng pointwise+depthwise stride-2 để giảm compute. Phải cho output đúng kích thước để dùng trong PAFPN bottom-up path. |

#### TC-M-22: SCDown: cv1 pointwise k=1, cv2 depthwise groups=c2
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `SCDown(32, 64, 3, 2)` |
| **Kiểm tra** | `m.cv1.conv.kernel_size == (1,1)` ; `m.cv2.conv.groups == 64` |
| **Đầu ra mong đợi** | Cả 2 assertion đều đúng |
| **Ý nghĩa** | Xác nhận cấu trúc đúng thiết kế: cv1 là pointwise (k=1 mở rộng kênh), cv2 là depthwise stride-2 (k=3, groups=c2). Sai cấu trúc → sai số tham số và sai FLOPs. |

---

### Nhóm 4.2: backbone_neck.py — 7 test

#### TC-M-23: Backbone: đúng stride (8/16/32) và số kênh trên cả 3 output
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Backbone(w=(16,32,64,128,160), n=(1,1,1,1))` ; input `(2,3,256,256)` |
| **Hàm** | `p3, p4, p5 = backbone(x)` |
| **Đầu ra mong đợi** | `p3.shape==(2,64,32,32)` (256/8) ; `p4.shape==(2,128,16,16)` (256/16) ; `p5.shape==(2,160,8,8)` (256/32) |
| **Ý nghĩa** | Xác nhận tỉ lệ downsampling của từng stage (stem×2, stage1×2, stage2×2, stage3×2, stage4×2 = ×32). Sai stride → PAFPN upsample không khớp kích thước → crash concat. |

#### TC-M-24: Backbone: grad lan từ mọi output về input/stem
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `(p3.sum()+p4.sum()+p5.sum()).backward()` |
| **Đầu ra mong đợi** | `x.grad != None` ; `stem.parameters().grad != None` và `isfinite` |
| **Ý nghĩa** | Xác nhận gradient lan từ tất cả 3 output về input (không bị ngắt ở stage nào). Phát hiện lỗi `.detach()` hay `stop_gradient` nhầm giữa các stage. |

#### TC-M-25: Backbone: có đúng stem + stage1..stage4
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Backbone(...)` |
| **Kiểm tra** | `hasattr(bb, "stem/stage1/stage2/stage3/stage4")` |
| **Đầu ra mong đợi** | Tất cả 5 attribute tồn tại |
| **Ý nghĩa** | Xác nhận cấu trúc module đúng tên để các script khác (như `load_feature_extractor`) có thể truy cập đúng tên attribute. |

#### TC-M-26: Backbone: output khác nhau với input khác nhau
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 2 input ngẫu nhiên `x1`, `x2` ; `backbone.eval()` ; `no_grad` |
| **Kiểm tra** | `not allclose(p3(x1), p3(x2))` |
| **Đầu ra mong đợi** | P3 output từ 2 input khác nhau phải khác nhau |
| **Ý nghĩa** | Phát hiện lỗi "constant output" — thường do BN bị sai mode hoặc activation bị zero-out toàn bộ. Nếu output là hằng số → model không thực sự xử lý input. |

#### TC-M-27: PAFPN: shape đầu ra (P3,P4,P5) khớp đúng shape đầu vào
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `PAFPN(chs=(64,128,160), n=1)` ; `p3(2,64,32,32)`, `p4(2,128,16,16)`, `p5(2,160,8,8)` |
| **Hàm** | `o3,o4,o5 = neck(p3,p4,p5)` |
| **Đầu ra mong đợi** | `o3.shape == p3.shape` ; `o4.shape == p4.shape` ; `o5.shape == p5.shape` |
| **Ý nghĩa** | PAFPN phải giữ nguyên shape của mỗi level (top-down + bottom-up). Nếu sai → head không nhận được feature map đúng kích thước. |

#### TC-M-28: PAFPN: grad lan tới cả 3 input P3/P4/P5
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | p3, p4, p5 đều `requires_grad=True` ; `(o3+o4+o5).sum().backward()` |
| **Đầu ra mong đợi** | `p3.grad`, `p4.grad`, `p5.grad` đều `!= None` và `isfinite` |
| **Ý nghĩa** | Xác nhận gradient lan đúng qua cả đường top-down (P5→P4→P3) lẫn bottom-up (P3→P4→P5). Nếu một cấp không nhận grad → stage đó không được update. |

#### TC-M-29: PAFPN: P5 output khác P5 input (cross-scale fusion xảy ra)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `p5_in` ngẫu nhiên ; chạy `neck(p3, p4, p5_in)` → `p5_out` |
| **Kiểm tra** | `not allclose(p5_out, p5_in)` |
| **Đầu ra mong đợi** | `p5_out` khác `p5_in` |
| **Ý nghĩa** | Xác nhận cross-scale feature fusion đang xảy ra. Nếu P5 output == P5 input → PAFPN bị bypass (có thể do lỗi upsample hoặc concat bị bỏ qua), model không hưởng lợi từ multi-scale. |

---

### Nhóm 4.3: head.py — 9 test

#### TC-M-30: ScaleHead: tính cả 2 nhánh ở cả train lẫn eval
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `ScaleHead(c_in=32, nc=5, reg_max=8)` ; input `(2,32,10,10)` ; chạy cả `.train()` và `.eval()` |
| **Hàm** | `out_o2m, out_o2o = head(x)` |
| **Đầu ra mong đợi** | Cả `out_o2m` và `out_o2o` đều `!= None` ở cả 2 mode ; shape cls `(2,nc,10,10)`, reg `(2,4*reg_max,10,10)` |
| **Ý nghĩa** | Shortcut inference (bỏ o2m ở eval) chưa được bật. `engine.validate()` cần cả o2m để tính DetectionLoss. Nếu eval bỏ o2m → validate() crash với KeyError. |

#### TC-M-31: ScaleHead: khởi tạo bias cls theo công thức YOLOv8/v10
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `ScaleHead(c_in, nc=5, reg_max=8)` mới khởi tạo ; sau đó gọi `init_stride_bias(stride=8, img_size=640)` |
| **Kiểm tra** | Bias ban đầu `= -log((1-0.01)/0.01)` ; sau `init_stride_bias`: bias `= log(5/nc/(img/stride)^2)` |
| **Đầu ra mong đợi** | Cả 2 assertion đều đúng với `atol=1e-4` |
| **Ý nghĩa** | Bias initialization ảnh hưởng đến convergence tốc độ. Prior 0.01 cho mỗi class; `init_stride_bias` điều chỉnh theo mật độ object mong đợi ở từng scale. |

#### TC-M-32: ScaleHead: nhánh o2m và o2o có trọng số riêng
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `ScaleHead(...)` |
| **Kiểm tra** | `head.cls_stem_o2m is not head.cls_stem_o2o` ; tương tự cho reg_stem |
| **Đầu ra mong đợi** | Các module là object khác nhau (không share tham chiếu) |
| **Ý nghĩa** | o2m (one-to-many) và o2o (one-to-one) là 2 nhánh độc lập với trọng số riêng. Nếu share weight → 2 nhánh luôn cho output giống nhau, mất đi lợi thế của cơ chế NMS-free. |

#### TC-M-33: DetectHead.make_anchors: đúng số lượng, offset 0.5, thứ tự, stride
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `feats = [zeros(1,1,4,4), zeros(1,1,2,2)]` ; `strides=(8,16)` |
| **Hàm** | `DetectHead.make_anchors(feats, strides)` |
| **Đầu ra mong đợi** | `anchors.shape == (20,2)` ; `stride_t.shape == (20,1)` ; `anchors[0] == [0.5,0.5]` ; `anchors[3] == [3.5,0.5]` (row-major) ; stride đúng từng level |
| **Ý nghĩa** | Anchor points là tâm lưới với offset 0.5 pixel. Thứ tự row-major (x tăng trước). Stride đúng cho mỗi level để decode box ra đúng tọa độ pixel. |

#### TC-M-34: DetectHead.make_anchors: offset=0 → anchor đầu tại (0,0)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `feats = [zeros(1,1,3,3)]` ; `strides=(8,)` ; `offset=0.0` |
| **Hàm** | `DetectHead.make_anchors(feats, (8,), offset=0.0)` |
| **Đầu ra mong đợi** | `anchors[0] == [0.0, 0.0]` |
| **Ý nghĩa** | Xác nhận `offset` parameter hoạt động đúng. Cần thiết nếu tương lai chuyển sang convention anchor tại góc thay vì tâm ô lưới. |

#### TC-M-35: DetectHead.decode_box: shape đúng, box luôn hợp lệ (x2>=x1)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | A=6 anchor ; `stride = full(8.0)` ; `reg` ngẫu nhiên shape `(1,4*8,6)` |
| **Hàm** | `head.decode_box(reg, anchors, stride)` |
| **Đầu ra mong đợi** | `box.shape == (1,6,4)` ; `isfinite(box).all()` ; `x2>=x1` và `y2>=y1` cho mọi box |
| **Ý nghĩa** | DFL tính kỳ vọng dương (ltrb ≥ 0 sau softmax) → x1≤x2, y1≤y2. Nếu có box âm → anchor design sai hoặc DFL bị lỗi. |

#### TC-M-36: DetectHead.decode_box: box tỷ lệ tuyến tính với stride
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Cùng `reg`, `anchors` ; lần 1 `stride=8`, lần 2 `stride=16` |
| **Kiểm tra** | `width(stride=16) == 2 * width(stride=8)` |
| **Đầu ra mong đợi** | Tỷ lệ đúng `2×` với `atol=1e-3` |
| **Ý nghĩa** | Box ở grid level 16 phải lớn gấp đôi box cùng `reg` ở grid level 8 (vì 1 ô lưới P4=16px vs P3=8px). Đảm bảo tọa độ pixel sau decode đúng scale. |

#### TC-M-37: DetectHead.forward (train): shape mọi nhánh + strides đúng thứ tự
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DetectHead(chs=(16,32,64), nc=5, reg_max=8, strides=(8,16,32))` ; feats `[320/8, 320/16, 320/32]` |
| **Hàm** | `head.train(); out = head(feats)` |
| **Đầu ra mong đợi** | `o2m/cls.shape == (2,A,5)` ; `o2o/box.shape == (2,A,4)` ; `anchors.shape == (A,2)` ; `strides[:n_p3]==8`, `strides[n_p3:n_p3+n_p4]==16`, `strides[n_p3+n_p4:]==32` |
| **Ý nghĩa** | Gộp 3 scale đúng thứ tự P3→P4→P5 và stride tensor đúng block-wise. Nếu stride sai thứ tự → box decode ra đúng anchor nhưng nhân sai stride → tọa độ pixel sai. |

#### TC-M-38: DetectHead.forward (eval): vẫn trả về o2m
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `DetectHead(...)` ; `.eval()` ; `no_grad` |
| **Hàm** | `out = head(feats)` |
| **Đầu ra mong đợi** | `"o2m" in out` ; `"o2o" in out` ; `o2o["box"].shape == (1,100,4)` |
| **Ý nghĩa** | Shortcut inference (bỏ o2m) chưa kích hoạt. Trong giai đoạn training, `validate()` gọi `eval()` nhưng vẫn cần cả 2 nhánh để tính loss. |

---

### Nhóm 4.4: model.py (NMSFreeDetector) — 11 test

#### TC-M-39: NMSFreeDetector.forward (train): end-to-end shape + strides
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model nhỏ (w=(16,32,64,128,160)) ; input `(2,3,256,256)` ; `.train()` |
| **Hàm** | `out = m(x)` |
| **Đầu ra mong đợi** | `o2o/cls.shape == (2,A_total,nc)` ; `o2m/box.shape == (2,A_total,4)` ; strides đúng thứ tự P3(8)/P4(16)/P5(32) |
| **Ý nghĩa** | Test end-to-end đầu tiên: ảnh → backbone → neck → head → output. Nếu fail → lỗi ngay trong kiến trúc cơ bản, không tiến hành training được. |

#### TC-M-40: NMSFreeDetector.forward (eval): vẫn có o2m
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model `.eval()` ; input `(1,3,256,256)` ; `no_grad` |
| **Hàm** | `out = m(x)` |
| **Đầu ra mong đợi** | `"o2m" in out` ; o2o cls shape đúng |
| **Ý nghĩa** | Đảm bảo `validate()` trong training loop không bị KeyError `"o2m"`. |

#### TC-M-41: Grad end-to-end không None/NaN từ mọi nhánh
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Loss = sum của o2o cls + box + o2m cls + box ; `.backward()` |
| **Đầu ra mong đợi** | Mọi tham số (backbone/neck/head) có `grad != None` và `isfinite(grad)` |
| **Ý nghĩa** | Kiểm tra toàn bộ computation graph thông suốt từ tất cả output về mọi tham số. Phát hiện `.detach()` nhầm ở bất kỳ đâu. |

#### TC-M-42: save_trunk() / load_trunk(): round-trip, không leak tempfile
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model m1 ; dùng `tempfile.TemporaryDirectory()` ; sau load so sánh m2 vs m1 |
| **Đầu ra mong đợi** | Tất cả backbone+neck param của m2 khớp m1 ; head không bị đụng ; file tạm bị xóa sau with-block |
| **Ý nghĩa** | Đảm bảo checkpoint lưu/load đúng và không để lại file rác. Phiên bản cũ hardcode `/tmp/` → không cleanup → đầy đĩa sau nhiều lần chạy test. |

#### TC-M-43: load_feature_extractor(): chỉ load backbone+neck, head giữ nguyên
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Lưu trunk m1 ; thay đổi backbone của m2 ; `m2.load_feature_extractor(path)` |
| **Đầu ra mong đợi** | Backbone m2 khớp m1 ; head m2 giữ nguyên giá trị trước khi load |
| **Ý nghĩa** | Transfer learning: load backbone từ pretrained nhưng giữ nguyên head (sẽ replace_head sau). Nếu load_feature_extractor cũng load head → đè mất head đã được khởi tạo cho task mới. |

#### TC-M-44: replace_head(): thay head mới, giữ nguyên backbone/neck
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model nc=6 ; `m.replace_head(nc=3)` |
| **Đầu ra mong đợi** | `m.nc == 3` ; head mới có `cls_o2o.out_channels == 3` ; backbone param không đổi ; forward đúng nc=3 |
| **Ý nghĩa** | Fine-tuning sang dataset khác: thay head mới với số class khác mà không ảnh hưởng feature extractor. |

#### TC-M-45: freeze_trunk(): đóng/mở băng đúng phạm vi (backbone+neck, không đụng head)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `m.freeze_trunk(True)` ; sau đó `m.freeze_trunk(False)` |
| **Đầu ra mong đợi** | Freeze True: backbone+neck tất cả `requires_grad=False` ; head `requires_grad=True` ; Freeze False: bật lại tất cả |
| **Ý nghĩa** | Fine-tuning phase 1: chỉ train head, freeze backbone. Phải đúng phạm vi để optimizer không cập nhật sai tham số. |

#### TC-M-46: Toàn bộ tham số hữu hạn ngay sau __init__
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `NMSFreeDetector(**small_kwargs)` mới khởi tạo |
| **Kiểm tra** | `all(isfinite(p) for p in model.parameters())` |
| **Đầu ra mong đợi** | Không có tham số NaN/Inf nào |
| **Ý nghĩa** | Initialization sai (ví dụ conv weight quá lớn hoặc chia cho 0 trong custom init) sẽ tạo ra NaN ngay từ đầu → training diverge ngay bước đầu tiên. |

#### TC-M-47: Số tham số trong khoảng hợp lý (10K–10M với small config)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model nhỏ `w=(16,32,64,128,160)` |
| **Kiểm tra** | `10_000 < n_params < 10_000_000` |
| **Đầu ra mong đợi** | Số tham số trong khoảng |
| **Ý nghĩa** | Phát hiện lỗi kiến trúc gây ra model quá lớn (nhầm kênh, thêm layer thừa) hoặc quá nhỏ (thiếu layer). |

#### TC-M-48: [SANITY] Toàn model học được: overfit 1 mục tiêu đơn giản
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 ảnh cố định ; AdamW lr=5e-3 ; mục tiêu: `o2o_cls.sigmoid()` tiến về `[1,0,...,0]` ; 60 bước |
| **Đầu ra mong đợi** | `min(losses[-10:]) < losses[0] * 0.85` |
| **Ý nghĩa** | Sanity check toàn bộ kiến trúc: nếu model không thể overfit 1 bài toán đơn giản (MSE một mục tiêu) → lỗi trong backward, optimizer, hoặc architecture cơ bản. |

#### TC-M-49: Forward đúng với batch size khác nhau (B=1 và B=4)
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model `.eval()` ; input B=1 và B=4, cùng H=W=256 |
| **Hàm** | Forward 2 lần với B khác nhau |
| **Đầu ra mong đợi** | `out1["o2o"]["cls"].shape == (1,A,nc)` ; `out4["o2o"]["cls"].shape == (4,A,nc)` ; A_total giống nhau |
| **Ý nghĩa** | Xác nhận không có hardcode batch size. BN ở eval mode dùng running stats (không phụ thuộc B). |

---

## 5. validate_pipeline.py — 24 test-case

Modules được test: `src/train/ema.py`, `src/train/engine.py`, `src/config.py`

> **Không test**: `src/utils/` (state_dict_handle.py, engine_with_tb.py, log_setup.py, tb_logger.py) và `src/TransferLearning/`

---

### Nhóm 5.1: ModelEMA (`ema`) — 7 test

#### TC-P-01: ModelEMA.__init__: deepcopy đúng, đóng băng grad, eval()
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `NMSFreeDetector` + `ModelEMA(model, decay=0.999, warmup=10)` |
| **Kiểm tra** | Param EMA == Param model ; `requires_grad=False` ; `ema.ema.training=False` |
| **Đầu ra mong đợi** | Tất cả 3 điều kiện đúng |
| **Ý nghĩa** | EMA phải là bản sao độc lập của model ở chế độ eval và không tính grad. Nếu không deepcopy → EMA và model share param → cập nhật model ảnh hưởng trực tiếp EMA. |

#### TC-P-02: ModelEMA: vẫn ở eval() sau update
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model `.train()` ; thay đổi trọng số ; `ema.update(model)` |
| **Kiểm tra** | `ema.ema.training == False` |
| **Đầu ra mong đợi** | EMA ở eval() mode sau update |
| **Ý nghĩa** | `update()` không được bật train mode của EMA. Nếu EMA chuyển sang train mode → BN của EMA sẽ dùng batch stats thay vì running stats khi validate, gây kết quả không nhất quán. |

#### TC-P-03: _current_decay(): tăng đơn điệu, không vượt decay max
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `ModelEMA(decay=0.9998, warmup=100)` ; chạy 500 bước, đo decay sau mỗi bước |
| **Đầu ra mong đợi** | Dãy `d[i]` tăng đơn điệu ; `max(d) <= 0.9998` ; `d[499] > 0.9998 * 0.99` |
| **Ý nghĩa** | Warmup decay tăng từ ~0 → ~max để tránh EMA bị kéo mạnh ở đầu training. Nếu decay tăng sai → EMA không ổn định hoặc bắt đầu quá cao → EMA không học được. |

#### TC-P-04: update(): EMA = decay*ema + (1-decay)*model, không copy thẳng
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | EMA khởi tạo ; thay đổi model 0.1σ ; `ema.update(model)` |
| **Đầu ra mong đợi** | EMA param thay đổi (≠ giá trị ban đầu) ; EMA param ≠ model param (không copy thẳng) |
| **Ý nghĩa** | Công thức EMA là weighted average: nếu EMA copy thẳng model → mất tính "moving average", EMA chỉ là model bước trước. |

#### TC-P-05: update(): EMA không thay đổi khi model không thay đổi
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | EMA được khởi tạo từ model ; KHÔNG thay đổi model ; `ema.update(model)` |
| **Đầu ra mong đợi** | `ema param ≈ model param` (vì ema_before = model, sau update: `d*model + (1-d)*model = model`) |
| **Ý nghĩa** | Xác nhận công thức toán học: khi EMA và model cùng giá trị, sau update EMA vẫn là giá trị đó. |

#### TC-P-06: update(): buffer non-float được copy trực tiếp
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `bn.num_batches_tracked += 5` ; `ema.update(model)` |
| **Kiểm tra** | Tất cả buffer có `dtype.is_floating_point == False` trong EMA == buffer trong model |
| **Đầu ra mong đợi** | `ema.state_dict()["..num_batches_tracked"] == model.state_dict()["..num_batches_tracked"]` |
| **Ý nghĩa** | Buffers như `num_batches_tracked`, `running_mean/var` của BN (các cờ int) phải copy thẳng, không áp dụng công thức EMA (EMA của int không có ý nghĩa). |

#### TC-P-07: state_dict()/load_state_dict(): round-trip chính xác
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | EMA1 chạy 5 update ; `ema2.load_state_dict(ema1.state_dict())` |
| **Đầu ra mong đợi** | Tất cả param EMA1 == EMA2 |
| **Ý nghĩa** | Khi resume training, EMA được khôi phục từ checkpoint. Nếu state_dict không bao gồm đủ hoặc không load đúng → EMA sau resume sẽ bắt đầu lại từ model hiện tại thay vì giá trị accumulated. |

---

### Nhóm 5.2: engine.py — 9 test

#### TC-P-08: get_optimizer(): tách đúng nhóm weight_decay
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `NMSFreeDetector` + `TrainConfig(optimizer="adamw", weight_decay=0.05)` |
| **Hàm** | `get_optimizer(model, cfg)` |
| **Đầu ra mong đợi** | 2 param group ; ndim≤1 hoặc `.bias` → group no_decay ; còn lại → group decay |
| **Ý nghĩa** | Regularization chuẩn: bias và BN parameters không cần weight decay (không có ý nghĩa geometric). Nếu group sai → bias bị shrink về 0 → model dự đoán lệch. |

#### TC-P-09: get_optimizer(): optimizer='sgd' hoạt động đúng
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(optimizer="sgd", lr0=1e-2, momentum=0.9)` |
| **Hàm** | `get_optimizer(model, cfg)` ; `opt.step()` |
| **Đầu ra mong đợi** | `isinstance(opt, torch.optim.SGD)` ; `step()` không crash |
| **Ý nghĩa** | Người dùng có thể chọn SGD thay vì AdamW. Xác nhận code path này hoạt động đúng. |

#### TC-P-10: get_optimizer(): optimizer không hỗ trợ → ValueError
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(optimizer="rmsprop")` |
| **Hàm** | `get_optimizer(model, cfg)` |
| **Đầu ra mong đợi** | `ValueError` được raise |
| **Ý nghĩa** | Fail-fast với thông báo rõ ràng hơn là crash ở bước sau hoặc silently dùng default. |

#### TC-P-11: lr_lambda_factory(): warmup tuyến tính 0→1 rồi cosine decay
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(epochs=10, warmup_epochs=2.0, lr_min_factor=0.01)` ; `steps_per_epoch=100` |
| **Hàm** | `lam = lr_lambda_factory(cfg, steps_per_epoch)` ; đo `lam(0)`, `lam(warmup_steps)`, giá trị cuối |
| **Đầu ra mong đợi** | `lam(0)==0.0` ; `lam(200)~=1.0` ; warmup tăng đơn điệu ; `lam(999)~=0.01` |
| **Ý nghĩa** | Schedule LR là yếu tố quan trọng nhất cho convergence. Warmup từ 0 tránh gradient explosion ở đầu. Cosine decay giảm dần để fine-tune. |

#### TC-P-12: lr_lambda_factory(): epochs=1 edge case không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(epochs=1, warmup_epochs=0.5)` ; `steps_per_epoch=10` |
| **Hàm** | `lam(0)`, `lam(5)`, `lam(100)` |
| **Đầu ra mong đợi** | Không crash ; `lam(0)==0.0` ; `lam(100)>=lr_min_factor-ε` |
| **Ý nghĩa** | Edge case: chỉ chạy 1 epoch (debug / quick test). Lambda không được chia cho 0 khi `epochs-warmup_epochs` rất nhỏ. |

#### TC-P-13: save_checkpoint()/load_checkpoint(): round-trip đầy đủ
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Model + optimizer + scheduler + EMA đã chạy 3 bước ; lưu vào `tempfile` ; load vào model/opt/sch/ema mới |
| **Đầu ra mong đợi** | `epoch==3`, `best_val==1.234` ; model param khớp 100% ; `sch.get_last_lr()` khớp ; EMA param khớp |
| **Ý nghĩa** | Resume training phải khôi phục đúng mọi thứ: model weights, optimizer state (momentum buffers), scheduler step, EMA. Thiếu bất kỳ thành phần nào → resume không đúng điểm dừng. |

#### TC-P-14: save/load checkpoint không có EMA (ema=None) không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `save_checkpoint(..., ema=None, ...)` ; `load_checkpoint(..., ema=None, ...)` |
| **Đầu ra mong đợi** | Không crash ; epoch và best_val khớp |
| **Ý nghĩa** | Khi `use_ema=False` trong config, checkpoint không có EMA. Code phải handle None gracefully. |

#### TC-P-15: validate(): không sinh grad trên tham số model
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `validate(model, criterion, loader, device)` với DataLoader mock 2 sample |
| **Kiểm tra** | Sau khi gọi: tất cả `p.grad is None` |
| **Đầu ra mong đợi** | `val_loss` là float ≥ 0 ; không có grad nào |
| **Ý nghĩa** | validate() phải dùng `@torch.no_grad()`. Nếu không → tính grad không cần thiết → tốn 2x memory, có thể OOM khi validate với batch lớn. |

#### TC-P-16: run_training(): logic fallback device đúng (else 'cpu')
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Source code của `engine.run_training` qua `inspect.getsource()` |
| **Kiểm tra** | Kiểm tra string `'torch.cuda.is_available() else "cuda"'` (bug) không tồn tại |
| **Đầu ra mong đợi** | Bug cũ không có mặt ; `else "cpu"` có mặt |
| **Ý nghĩa** | Bug cũ: `device = cfg.device if cuda_available else "cuda"` — khi không có CUDA lại fallback về "cuda" → crash ngay. Đây là test phát hiện regression của bug này. |

---

### Nhóm 5.3: config.py (`config`) — 8 test

#### TC-P-17: TrainConfig() mặc định không lỗi
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig()` không tham số |
| **Hàm** | Constructor + `__post_init__` |
| **Đầu ra mong đợi** | Không raise exception ; `nc > 0` ; `reg_max > 0` |
| **Ý nghĩa** | Config mặc định phải hợp lệ để có thể chạy ngay. |

#### TC-P-18: __post_init__: bắt đúng số phần tử các tuple augment
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 4 trường hợp: `shiftScaleRotate=(0.1,0.1,5)` (3 thay vì 4), `hueSaturationValue=(1,2,3)` (3/4), `gaussNoise=(1,2)` (2/3), `blur=(3,)` (1/2) |
| **Đầu ra mong đợi** | `AssertionError` ở tất cả 4 trường hợp |
| **Ý nghĩa** | Albumentations nhận tuple với số phần tử cố định. Nếu sai → runtime error ở bước augment, khó debug. `__post_init__` phải bắt lỗi sớm. |

#### TC-P-19: __post_init__: num_workers âm phải bị chặn
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(num_workers=-1)` |
| **Đầu ra mong đợi** | `AssertionError` |
| **Ý nghĩa** | `num_workers=-1` sẽ crash ở `DataLoader.__init__` với thông báo khó hiểu. Better fail fast với message rõ ràng. |

#### TC-P-20: __post_init__: tự động sửa persistent_workers/prefetch_factor khi num_workers=0
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig(num_workers=0, persistent_workers=True, prefetch_factor=4)` |
| **Đầu ra mong đợi** | Sau `__post_init__`: `persistent_workers=False` ; `prefetch_factor=None` |
| **Ý nghĩa** | PyTorch yêu cầu `num_workers>0` để dùng `persistent_workers` và `prefetch_factor`. Tự động sửa tránh crash DataLoader với thông báo confusing. |

#### TC-P-21: TrainConfig: có trường 'device' kiểu str
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig()` |
| **Kiểm tra** | `hasattr(cfg, "device")` ; `isinstance(cfg.device, str)` |
| **Đầu ra mong đợi** | Cả 2 True |
| **Ý nghĩa** | `engine.run_training()` truy cập `cfg.device`. Nếu field bị đổi tên → AttributeError khó trace ở runtime. |

#### TC-P-22: TrainConfig: có trường 'amp' kiểu bool
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig()` |
| **Kiểm tra** | `hasattr(cfg, "amp")` ; `isinstance(cfg.amp, bool)` |
| **Đầu ra mong đợi** | Cả 2 True |
| **Ý nghĩa** | Mixed precision (AMP) được bật/tắt qua `cfg.amp`. Nếu field không tồn tại → crash engine khi tạo GradScaler. |

#### TC-P-23: TrainConfig: backbone_w(5), backbone_n(4), strides(3) đúng số phần tử
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig()` |
| **Kiểm tra** | `len(cfg.backbone_w)==5` ; `len(cfg.backbone_n)==4` ; `len(cfg.strides)==3` |
| **Đầu ra mong đợi** | Tất cả đúng |
| **Ý nghĩa** | `Backbone(w, n)` và `DetectHead(strides)` đều unpack trực tiếp. Sai số phần tử → `ValueError: not enough values to unpack`. |

#### TC-P-24: TrainConfig: các trường LR có giá trị dương hợp lý
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `TrainConfig()` |
| **Kiểm tra** | `lr0>0` ; `lr_min_factor>0` ; `weight_decay>=0` ; `warmup_epochs>=0` |
| **Đầu ra mong đợi** | Tất cả đúng |
| **Ý nghĩa** | LR âm hay bằng 0 → model không học. weight_decay âm → khuếch đại trọng số thay vì regularize. Phát hiện typo trong config. |

---

## 6. validate_dataloader.py — 15 test-case

Modules được test: `src/train/dataloader1_obj365.py`  
Thành phần: `letterbox()`, `collate_fn()`, `DetectionAugmenter`

> **Không test**: `ObjectDetectionDataset`, `build_dataloaders()` (cần dữ liệu Object365 thật)

---

### Nhóm 6.1: letterbox() — 6 test

#### TC-D-01: letterbox: luôn ra hình vuông, scale theo cạnh dài hơn
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img.shape=(90,160,3)` (ảnh ngang 16:9) ; `new_size=128` |
| **Hàm** | `letterbox(img, 128)` → `canvas, scale, pad_left, pad_top` |
| **Đầu ra mong đợi** | `canvas.shape==(128,128,3)` ; `scale==128/160==0.8` ; `pad_left==0` |
| **Ý nghĩa** | Letterbox resize ảnh về `new_size×new_size` mà không cắt hay méo. Scale theo cạnh dài hơn đảm bảo không crop nội dung. |

#### TC-D-02: letterbox: pad căn giữa ảnh trong canvas vuông
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img.shape=(50,100,3)` ; `new_size=100` |
| **Hàm** | `letterbox(img, 100)` |
| **Đầu ra mong đợi** | `pad_top == (100 - new_h) // 2` ; `pad_left == 0` |
| **Ý nghĩa** | Padding phải căn giữa ảnh (không padding một phía). Căn giữa giúp anchor distribution đều hơn và preprocessing nhất quán với inference. |

#### TC-D-03: letterbox: ảnh đã vuông khớp kích thước → không biến dạng
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img.shape=(64,64,3)` ngẫu nhiên ; `new_size=64` |
| **Hàm** | `letterbox(img, 64)` |
| **Đầu ra mong đợi** | `scale==1.0` ; `pad_left==0` ; `pad_top==0` |
| **Ý nghĩa** | Không resize hay pad không cần thiết → không mất thông tin và không chậm. |

#### TC-D-04: letterbox: nội dung pixel ảnh gốc được giữ nguyên
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img` xám đồng đều (giá trị=128) `(40,80,3)` ; `new_size=80` |
| **Hàm** | `letterbox(img, 80)` ; đo giá trị trung bình vùng nội dung |
| **Đầu ra mong đợi** | `mean(canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w]) ≈ 128 ± 10` |
| **Ý nghĩa** | Xác nhận letterbox dùng ảnh thực (không phải toàn màu padding). Nếu nội dung bị tràn sang vùng padding hoặc màu sắc sai → augmentation không đúng. |

#### TC-D-05: letterbox: ảnh portrait (cao > rộng) scale theo chiều cao
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img.shape=(160,90,3)` (portrait 9:16) ; `new_size=128` |
| **Hàm** | `letterbox(img, 128)` |
| **Đầu ra mong đợi** | `canvas.shape==(128,128,3)` ; `scale==128/160==0.8` ; `pad_left > 0` (padding ngang) |
| **Ý nghĩa** | Xác nhận xử lý đúng chiều cao > chiều rộng — cần pad ở 2 bên ngang. |

#### TC-D-06: letterbox: scale up ảnh nhỏ hơn kích thước đích
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `img.shape=(30,40,3)` (nhỏ hơn 100) ; `new_size=100` |
| **Hàm** | `letterbox(img, 100)` |
| **Đầu ra mong đợi** | `canvas.shape==(100,100,3)` ; `scale > 1.0` |
| **Ý nghĩa** | letterbox phải scale UP ảnh nhỏ. Nếu chỉ scale down → ảnh nhỏ bị đặt góc trong canvas trống, mất nhiều không gian. |

---

### Nhóm 6.2: collate_fn() — 5 test

#### TC-D-07: collate_fn: stack ảnh, giữ targets dạng list
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 3 sample: `(img_32x32, target_i)` với số GT: 2, 0, 1 |
| **Hàm** | `collate_fn(batch)` |
| **Đầu ra mong đợi** | `images.shape==(3,3,32,32)` ; `isinstance(targets, list)` ; `len(targets)==3` ; `targets[1]["boxes"].shape==(0,4)` |
| **Ý nghĩa** | Ảnh có thể stack (cùng kích thước), nhưng targets không thể stack vì mỗi ảnh có số GT khác nhau. Giữ dưới dạng list để DetectionLoss xử lý linh hoạt. |

#### TC-D-08: collate_fn: boxes tensor có dtype float
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 2 sample với boxes `[[0.,0.,10.,10.]]` |
| **Hàm** | `collate_fn(batch)` |
| **Kiểm tra** | `targets[i]["boxes"].dtype in (float32, float64, float16)` |
| **Đầu ra mong đợi** | Tất cả là float |
| **Ý nghĩa** | `bbox_iou()` và loss functions yêu cầu float. Nếu boxes là int64 (từ JSON) → TypeError ở loss computation. |

#### TC-D-09: collate_fn: hoạt động đúng khi gắn vào DataLoader thật
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `Dataset` 4 sample, `DataLoader(batch_size=2, collate_fn=collate_fn)` |
| **Hàm** | `list(loader)` |
| **Đầu ra mong đợi** | 2 batch ; `images.shape[0]==2` ; `len(targets)==2` |
| **Ý nghĩa** | Xác nhận `collate_fn` tương thích với PyTorch DataLoader interface (không chỉ hoạt động khi gọi trực tiếp). |

#### TC-D-10: collate_fn: batch_size=1 edge case không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 sample duy nhất |
| **Hàm** | `collate_fn([sample])` |
| **Đầu ra mong đợi** | `images.shape==(1,3,64,64)` ; không crash |
| **Ý nghĩa** | Debug/inference thường chạy B=1. Collate_fn với list 1 phần tử không được gây lỗi (vd: `.unsqueeze` hay `.stack` hành xử khác với 1 phần tử). |

#### TC-D-11: collate_fn: batch lớn (B=16) với số GT từ 0 đến 10 không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 16 sample: sample `i` có `i%11` GT boxes (0, 1, ..., 10, 0, 1, ..., 5) |
| **Hàm** | `collate_fn(batch)` |
| **Đầu ra mong đợi** | `images.shape[0]==16` ; `len(targets)==16` ; tổng GT đúng |
| **Ý nghĩa** | Stress test với nhiều sample và số GT đa dạng trong cùng batch. |

---

### Nhóm 6.3: DetectionAugmenter — 4 test

> **Phụ thuộc**: `albumentations` + `opencv-python`. Nếu thiếu → tất cả test báo SKIP.

#### TC-D-12: DetectionAugmenter: bỏ qua augment khi không có box
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | Ảnh ngẫu nhiên 64×64 ; `boxes=[]` ; `labels=[]` |
| **Hàm** | `aug(img, [], [])` |
| **Đầu ra mong đợi** | `out_boxes == []` ; `out_labels == []` ; không crash |
| **Ý nghĩa** | Albumentations crash nếu truyền list box rỗng vào `BboxParams`. Augmenter phải short-circuit khi không có box, tránh lỗi ngoài ý muốn. |

#### TC-D-13: DetectionAugmenter: box/label luôn khớp số lượng, không sinh thêm box
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | `cfg.horizontalFlip=1.0` (ép augment chắc chắn) ; 2 boxes |
| **Hàm** | `aug(img, boxes, labels)` |
| **Đầu ra mong đợi** | `len(out_boxes)==len(out_labels)` ; `len(out_boxes) <= 2` ; `out_img.shape == img.shape` |
| **Ý nghĩa** | Augment có thể lọc bỏ box bị crop mất (min_visibility), nhưng không thể sinh thêm box. Mất đồng bộ box/label → IndexError trong loss. |

#### TC-D-14: DetectionAugmenter: tọa độ box sau augment nằm trong ảnh
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 2 boxes trong ảnh 128×128 ; `cfg.horizontalFlip=1.0` |
| **Hàm** | `aug(img, boxes, labels)` |
| **Kiểm tra** | Với mọi `(x1,y1,x2,y2)` trong `out_boxes`: `x1>=0`, `y1>=0`, `x2<=W`, `y2<=H`, `x2>x1`, `y2>y1` |
| **Đầu ra mong đợi** | Tất cả box hợp lệ |
| **Ý nghĩa** | Phát hiện lỗi clipping: sau flip/rotate, box có thể vượt biên ảnh. Nếu không clip → `bbox_iou` nhận box âm → NaN. |

#### TC-D-15: DetectionAugmenter: augment với 1 box không crash
| Mục | Nội dung |
|-----|---------|
| **Đầu vào** | 1 box `[5,5,55,55]` ; label `[3]` |
| **Hàm** | `aug(img, [[5,5,55,55]], [3])` |
| **Đầu ra mong đợi** | `out_img.shape == img.shape` ; `len(out_boxes)==len(out_labels)` ; không crash |
| **Ý nghĩa** | Albumentations có thể hành xử khác với list 1 phần tử (vd: khi box bị squeeze thành 1D). Test đảm bảo không lỗi trong trường hợp thông thường nhất. |

---

## 7. Phạm vi KHÔNG test

| Folder / Script | Lý do |
|----------------|-------|
| `src/utils/state_dict_handle.py` | Chỉ chứa `set_seed()` — tiện ích ngắn, logic trivial (gọi `torch.manual_seed`, `np.random.seed`) |
| `src/utils/engine_with_tb.py` | File template/tham khảo, không phải module production |
| `src/utils/log_setup.py` | Wrapper logging chuẩn Python, không có logic domain |
| `src/utils/tb_logger.py` | Wrapper TensorBoard, phụ thuộc `tensorboard` ngoài scope |
| `src/TransferLearning/dataloader_tfl.py` | Chưa integrate vào training pipeline chính |
| `src/TransferLearning/head_tfl.py` | Chưa integrate vào training pipeline chính |
| `src/TransferLearning/loss_tfl.py` | Chưa integrate vào training pipeline chính |
| `ObjectDetectionDataset` / `build_dataloaders()` | Cần dữ liệu Object365 thật trên đĩa — không phù hợp zero-data test |

---

## 8. Quy ước trạng thái

| Trạng thái | Ý nghĩa | Xử lý |
|-----------|---------|-------|
| **PASS** | Test chạy xong không exception, assertion đúng | Tốt ✓ |
| **FAIL** | `AssertionError` — logic sai hoặc kết quả ngoài kỳ vọng | Cần sửa code |
| **ERROR** | Exception bất ngờ (TypeError, RuntimeError, ImportError...) | Cần debug |
| **SKIP** | `Skip` exception — thiếu dependency tùy chọn (cv2, albumentations, cuda) | Không tính vào FAIL |

Bộ công cụ coi là **PASS toàn bộ** khi không có test nào ở trạng thái FAIL hoặc ERROR (SKIP được chấp nhận).
