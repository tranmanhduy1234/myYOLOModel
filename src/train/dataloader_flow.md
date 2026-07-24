=====================================================================================================================
graph TD
    classDef storage fill:#1f2937,stroke:#4b5563,stroke-width:2px,color:#fff
    classDef process fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef data fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    classDef alert fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#fff
    classDef note fill:#374151,stroke:#9ca3af,stroke-width:1px,color:#f3f4f6,stroke-dasharray: 3 3

    subgraph Phase1 ["1. ĐỌC CẤU TRÚC & TẠO INDEX (Chạy 1 lần - Lưu Cache)"]
        Raw_Info[("images_info.jsonl")]:::storage
        Raw_Ann[("annotations.jsonl")]:::storage
        Raw_Map[("image_path_map.jsonl")]:::storage
        Raw_Cat[("categories.jsonl")]:::storage

        Idx_Info["build_id_offset_index()<br/>Offset của từng ID trong file"]:::process
        Idx_Ann["build_annotation_group_index()<br/>Group danh sách Offset theo image_id"]:::process
        Idx_Map["load_image_path_map()<br/>Dict mapping: image_name -> path"]:::process
        Idx_Cat["load_categories()<br/>Dict mapping: cat_id -> idx (0..N-1)"]:::process

        Cache_Idx[("Index Cache .pkl<br/>Tốc độ load RAM cực nhanh")]:::storage

        Raw_Info --> Idx_Info --> Cache_Idx
        Raw_Ann --> Idx_Ann --> Cache_Idx
        Raw_Map --> Idx_Map --> Cache_Idx
        Raw_Cat --> Idx_Cat

        N_CatMap["📌 cat_id (JSON gốc, VD: 372) khác với\nlabel idx dùng để train (0..nc-1).\nMapping này chỉ tạo Ở ĐÂY, sẽ được\ntra cứu lại ở Phase 2."]:::note
        Idx_Cat -.- N_CatMap
    end

    subgraph Phase2 ["2. TRUY XUẤT DỮ LIỆU BỞI DATASET (__getitem__)"]
        Idx_Fetch["Truy xuất Index (image_id)"]:::process

        Disk_Image[("Image File (.jpg/.png)<br/>Đọc ngẫu nhiên bằng cv2.imread")]:::storage

        f_Seek1["Read Offset -> Parse Json"]:::process
        f_Seek2["Read List Offsets -> Parse Bboxes"]:::process
        MapCat["Map category_id gốc -> label idx<br/>(tra dict từ Idx_Cat)"]:::process

        Raw_BGR["Image Array (BGR)<br/>Mặc định OpenCV"]:::data
        Bbox_Raw["Bboxes gốc + label idx<br/>(COCO Format)"]:::data

        Idx_Fetch -->|"f.seek(offset)"| f_Seek1
        Idx_Fetch -->|"f.seek(offsets)"| f_Seek2
        Idx_Fetch -->|"Lookup Path"| Disk_Image

        f_Seek1 --> Raw_BGR
        Disk_Image --> Raw_BGR
        f_Seek2 --> MapCat --> Bbox_Raw

        N_BGR["📌 KÊNH ẢNH tại bước này: BGR\n(OpenCV mặc định), dtype uint8 [0..255],\nlayout HWC, KÍCH THƯỚC GỐC (chưa resize)"]:::note
        N_XYWH["📌 FORMAT BBOX tại bước này: [x, y, w, h]\n(COCO) - x,y là góc TRÊN-TRÁI, đơn vị pixel\nTUYỆT ĐỐI theo ẢNH GỐC (chưa scale/pad).\nLabel: đã map sang idx 0..N-1 (không còn category_id gốc)"]:::note
        Raw_BGR -.- N_BGR
        Bbox_Raw -.- N_XYWH
    end

    subgraph Phase3 ["3. TIỀN XỬ LÝ & BIẾN ĐỔI (Transform & Augment)"]
        CVT["cv2.cvtColor()<br/>Đổi BGR -> RGB"]:::process
        Raw_BGR --> CVT

        N_CVT["📌 Sau bước này: kênh ảnh đổi từ\nBGR -> RGB, layout vẫn HWC, dtype vẫn\nuint8, KÍCH THƯỚC VẪN CHƯA ĐỔI"]:::note
        CVT -.- N_CVT

        Letterbox["letterbox()<br/>- Scale giữ tỷ lệ aspect ratio<br/>- Pad viền xám (114, 114, 114)<br/>- Resize về hình vuông (imgsz x imgsz)"]:::process
        Bbox_Rescale["Biến đổi & Chuyển tọa độ Bbox<br/>- Chuyển x, y, w, h thành x1, y1, x2, y2<br/>- Tính lại theo Scale & Pad Offset<br/>- Clip biên [0, imgsz]<br/>- Lọc bỏ box rác (w <= 0 hoặc h <= 0)"]:::process

        CVT --> Letterbox
        Img_Padded["Image Resized (RGB)"]:::data
        Bbox_Padded["Bboxes (Pascal VOC)"]:::data

        Letterbox --> Img_Padded
        Bbox_Raw --> Bbox_Rescale --> Bbox_Padded

        N_Letterbox["📌 Ảnh sau letterbox: vẫn RGB, uint8, HWC,\nnhưng kích thước CỐ ĐỊNH [imgsz, imgsz, 3]\n(có viền pad màu xám 114 nếu ảnh gốc không vuông)"]:::note
        Img_Padded -.- N_Letterbox

        N_XYXY["📌 Sau bước này: bbox đổi format\nxywh -> xyxy (Pascal VOC: x1,y1 góc trên-trái,\nx2,y2 góc dưới-phải), ĐÃ nhân theo hệ số scale\ncủa letterbox + cộng offset pad, đơn vị pixel\nTUYỆT ĐỐI theo ẢNH ĐÃ RESIZE [imgsz, imgsz]"]:::note
        Bbox_Padded -.- N_XYXY

        Augmenter{"DetectionAugmenter<br/>(Albumentations)<br/>Flip, Rotate, Blur, HSV..."}:::process

        Img_Padded & Bbox_Padded --> Augmenter

        Aug_Failed["[Warning] Bỏ qua Augment<br/>Giữ nguyên data gốc nếu dính Exception"]:::alert
        Augmenter -.->|Xảy ra ngoại lệ| Aug_Failed

        Img_Aug["Augmented Image (RGB)"]:::data
        Bbox_Aug["Augmented Bboxes & Labels"]:::data

        Augmenter -->|Thành công| Img_Aug & Bbox_Aug

        N_Transform["📌 NOTE TỔNG HỢP (sau Augment):\n• Kênh ảnh: RGB, uint8, HWC, [imgsz, imgsz, 3]\n• Bbox format: [x1, y1, x2, y2] (Pascal VOC),\n  Albumentations dùng bbox_params(format='pascal_voc')\n  nên augment (rotate/shift/scale) tự re-tính lại xyxy,\n  KHÔNG đổi ngược về xywh\n• Label: vẫn giữ idx 0..N-1 đã map từ Phase 2"]:::note
        Img_Aug -.- N_Transform
        Bbox_Aug -.- N_Transform
    end

    subgraph Phase4 ["4. ĐÓNG GÓI PYTORCH TENSORS & COLLATE"]
        ToTensor["Chuyển đổi Tensor & Normalization<br/>- Permute kênh: (H, W, C) -> (C, H, W)<br/>- Chuẩn hóa pixel: uint8 [0..255] -> float32 [0.0..1.0]"]:::process

        Tensor_Img["Image Tensor [3, H, W]"]:::data
        Tensor_Target["Target Dict<br/>- boxes: Tensor [N, 4]<br/>- labels: Tensor [N,]"]:::data

        Img_Aug --> ToTensor --> Tensor_Img
        Bbox_Aug --> ToTensor --> Tensor_Target

        N_ToTensor_Img["📌 Ảnh: kênh vẫn RGB (không đổi ở bước này),\nchỉ đổi layout HWC -> CHW và dtype\nuint8[0,255] -> float32[0.0,1.0]"]:::note
        Tensor_Img -.- N_ToTensor_Img

        N_ToTensor_Box["📌 Bbox: format VẪN GIỮ xyxy (không đổi ở\nbước này), chỉ convert sang torch.Tensor\nfloat32, đơn vị PIXEL tuyệt đối theo [imgsz,imgsz]\n(không normalize về [0,1]). Label: torch.Tensor int64"]:::note
        Tensor_Target -.- N_ToTensor_Box

        Collate["collate_fn()<br/>Đóng gói theo Batch"]:::process

        Tensor_Img & Tensor_Target --> Collate

        Batch_Output[("Batch Output DataLoader<br/>- Images: Tensor [B, 3, H, W]<br/>- Targets: List[Dict] độ dài B")]:::storage

        N_Tensor["📌 NOTE ĐẦU RA CUỐI CÙNG:\n• Image Tensor: kênh RGB, Shape [3, imgsz, imgsz],\n  value range [0.0, 1.0], images.stack() -> [B,3,H,W]\n• Box Tensor: Shape [N, 4], format (x1,y1,x2,y2)\n  (xyxy) theo Pixel tọa độ TUYỆT ĐỐI (KHÔNG phải xywh,\n  KHÔNG normalize)\n• Label Tensor: Shape [N,], dtype int64 (0..num_classes-1)\n• Target: List[Dict] (không stack được vì N mỗi ảnh\n  khác nhau) - collate_fn chỉ stack phần Image"]:::note
        Batch_Output -.- N_Tensor
    end
=====================================================================================================================
__get_item__()
flowchart TD
    A["1. Đọc thông tin file & lấy đường dẫn<br/><b>Input:</b> index (chỉ số mẫu)"] --> B["2. Đọc ảnh bằng OpenCV (cv2.imread)<br/>• <b>Image Shape:</b> (H_orig, W_orig, 3)<br/>• <b>Thứ tự chiều:</b> HWC<br/>• <b>Hệ màu:</b> BGR<br/>• <b>Kiểu dữ liệu:</b> uint8 [0 - 255]"]
    
    B --> C["3. Chuyển đổi hệ màu (cv2.cvtColor)<br/>• <b>Image Shape:</b> (H_orig, W_orig, 3)<br/>• <b>Thứ tự chiều:</b> HWC<br/>• <b>Hệ màu:</b> RGB<br/>• <b>Kiểu dữ liệu:</b> uint8 [0 - 255]"]
    
    C --> D["4. Căn chỉnh kích thước (Letterbox)<br/>• <b>Image Shape:</b> (imgsz, imgsz, 3)<br/>• <b>Thứ tự chiều:</b> HWC | <b>Hệ màu:</b> RGB<br/>• <b>Tính toán:</b> scale, pad_left, pad_top"]
    
    D --> E["5. Đọc & Chuyển đổi Bounding Boxes<br/>• <b>Input Box:</b> [x, y, w, h] (xywh - top-left, pixel gốc)<br/>• <b>Biến đổi:</b> Scale & Offset theo Letterbox<br/>• <b>Output Box:</b> [x1, y1, x2, y2] (xyxy - pixel imgsz)<br/>• <b>Lọc:</b> iscrowd, isfake, box dị dạng, out-of-bounds"]
    
    E --> F{"Có Augmenter không?"}
    
    F -- "Có (Train)" --> G["6. Tăng cường dữ liệu (DetectionAugmenter)<br/>• <b>Image:</b> (imgsz, imgsz, 3) | HWC | RGB | uint8<br/>• <b>Boxes:</b> List [x1, y1, x2, y2] (xyxy, pixel)<br/>• <b>Labels:</b> List ID lớp (0 -> N-1)"]
    F -- "Không (Val/Test)" --> H["Bỏ qua Augmentation"]
    
    G --> I["7. Chuyển đổi Image sang PyTorch Tensor<br/>• <b>Thao tác:</b> permute(2, 0, 1) & float() / 255.0<br/>• <b>Image Shape:</b> (3, imgsz, imgsz)<br/>• <b>Thứ tự chiều:</b> CHW<br/>• <b>Hệ màu:</b> RGB<br/>• <b>Kiểu dữ liệu:</b> float32 [0.0 - 1.0]"]
    
    H --> I
    
    I --> J["8. Chuyển Bounding Boxes & Labels sang Tensor<br/>• <b>boxes_tensor:</b> Shape (N, 4) | float32 | xyxy (pixel)<br/>• <b>labels_tensor:</b> Shape (N,) | int64 | [0 -> N-1]"]
    
    J --> K["9. Trả về kết quả (Output)<br/><b>return</b> img_tensor, {'boxes': ..., 'labels': ...}"]

=======================================================================================================================
flowchart TD

    subgraph PREP["🗂️ GIAI ĐOẠN CHUẨN BỊ (chạy 1 lần, cache bằng pickle)"]
        A1["annotations.jsonl<br/>mỗi dòng 1 object<br/>bbox = xywh, (x,y) là góc TRÊN-TRÁI<br/>vd: [356.96, 216.07, 108.51, 295.83]"]
        A2["images_info.jsonl<br/>mỗi dòng 1 ảnh<br/>chỉ có file_name (bare), không có path"]
        A3["images_train.jsonl (path map)<br/>image_name → path tương đối"]
        A1 --> B1["build_annotation_group_index()<br/>group theo image_id<br/>→ dict{image_id: [byte_offset,...]}"]
        A2 --> B2["build_id_offset_index()<br/>→ dict{image_id: byte_offset}"]
        A3 --> B3["load_image_path_map()<br/>→ dict{file_name: relative_path}"]
        B1 & B2 & B3 -.pickle cache.-> CACHE[("index_cache_dir/*.pkl")]
    end

    subgraph GETITEM["🔄 __getitem__(index) — xử lý 1 SAMPLE"]
        C1["image_id = image_ids[index]"]
        C1 --> C2["_read_image_info(image_id)<br/>seek(offset) + readline + json.loads<br/>KHÔNG load cả file vào RAM"]
        C2 --> C3["cv2.imread(img_path)<br/>⚠️ định dạng: BGR, HWC (H,W,3), dtype=uint8"]
        C3 --> C4["cv2.cvtColor(BGR2RGB)<br/>✅ RGB, HWC (H,W,3), dtype=uint8"]

        C4 --> C5{"ảnh đã đúng<br/>imgsz x imgsz?"}
        C5 -- "không" --> C6["letterbox(image, new_size)<br/>resize giữ tỉ lệ + pad màu (114,114,114)<br/>→ RGB, HWC (imgsz,imgsz,3)<br/>trả về scale, pad_left, pad_top"]
        C5 -- "có" --> C7["giữ nguyên, scale=1, pad=0"]

        C1 --> D1["_read_annotations(image_id)<br/>seek từng offset trong list + readline"]
        D1 --> D2["lọc iscrowd=1 / isfake=1 (nếu bật)<br/>lọc w<=0 hoặc h<=0"]
        D2 --> D3["bbox: xywh → xyxy<br/>x1,y1,x2,y2 = x, y, x+w, y+h<br/>(vẫn ở toạ độ ẢNH GỐC, chưa letterbox)"]
        D3 --> D4["áp dụng scale + pad_left/pad_top<br/>lên cả x1,y1,x2,y2<br/>rồi clip về [0, imgsz]<br/>✅ bbox giờ là XYXY, PIXEL space<br/>của ẢNH ĐÃ LETTERBOX (imgsz x imgsz)"]
        D4 --> D5["map category_id gốc → index liên tục<br/>qua cat_id_to_idx"]

        C6 & C7 --> E1
        D5 --> E1{"self.augmenter<br/>(chỉ bật khi train)"}
        E1 -- "có" --> E2["DetectionAugmenter<br/>albumentations Compose<br/>bbox_params format='pascal_voc' (=xyxy)<br/>HFlip/ShiftScaleRotate/BrightnessContrast/<br/>HueSatVal/GaussNoise/Blur<br/>vẫn RGB, HWC, bbox vẫn XYXY pixel"]
        E1 -- "không (val)" --> E3["giữ nguyên"]

        E2 & E3 --> F1["np.ascontiguousarray(uint8).copy()<br/>vẫn RGB, HWC"]
        F1 --> F2["torch.from_numpy(...)<br/>.permute(2,0,1)<br/>⚠️ HWC → CHW (3,imgsz,imgsz)<br/>.float() / 255.0<br/>✅ float32, giá trị trong [0,1], vẫn kênh RGB"]

        D5 --> G1["boxes: list→ torch.as_tensor float32<br/>shape (N,4), XYXY, PIXEL space<br/>labels: torch.as_tensor int64, shape (N,)<br/>nếu N=0 → tensor rỗng shape (0,4)/(0,)"]

        F2 --> H["return img_tensor (3,H,W) float32 RGB<br/>+ target = {'boxes':(N,4) xyxy pixel,<br/>'labels':(N,)}"]
        G1 --> H
    end

    subgraph COLLATE["📦 collate_fn — gộp thành BATCH"]
        H --> I1["torch.stack(imgs, dim=0)<br/>✅ images: (B, 3, H, W), CHW, RGB, float32 [0,1]"]
        H --> I2["targets = list các dict, độ dài = B<br/>MỖI ảnh N khác nhau → KHÔNG pad<br/>(padding để training thực hiện ở nơi khác,<br/>vd trong assigner của loss.py)"]
    end

    CACHE -.dùng lại giữa các epoch.-> C1

    style C3 fill:#ffe0e0
    style C4 fill:#e0ffe0
    style D3 fill:#fff3cd
    style D4 fill:#fff3cd
    style F2 fill:#e0f0ff
    style I1 fill:#e0f0ff

=========================================================================================================================