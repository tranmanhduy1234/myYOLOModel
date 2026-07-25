"""
face_landmark_dataset_v3.py
=============================
PyTorch Dataset / DataLoader cho annotations_all.jsonl (sinh bởi
process_dataset_parallel.py + merge_jsonl.py), khớp trực tiếp với định
dạng targets mà DetectHeadFaceLmk / FaceLandmarkDetectionLoss (v3) mong
đợi.

THAY ĐỔI SO VỚI v2 (fix theo review):
------------------------------------------------------------------------
1. VALIDATE số landmark mỗi mặt so với `self.num_landmarks` (đã dò lúc
   khởi tạo dataset). v2 giả định NGẦM rằng mọi record đều có cùng số
   điểm landmark (chỉ dò từ 2000 dòng đầu) - nếu có vài record lệch (vd
   chạy MediaPipe với cấu hình refine_landmarks khác nhau -> 468 vs 478
   điểm), v2 sẽ crash muộn, khó hiểu, ngay tại torch.tensor(landmarks)
   hoặc thậm chí muộn hơn ở phía loss. v3 kiểm tra NGAY trong
   __getitem__, bỏ qua (skip) riêng mặt bị lệch và in CẢNH BÁO 1 LẦN duy
   nhất (tránh spam log khi dataset lớn), thay vì để cả batch/cả lần
   train crash.
2. CLAMP box + landmark về trong biên ảnh [0, image_size] sau khi quy
   đổi pixel. Dữ liệu normalized đôi khi hơi vượt [0,1] (nhiễu số hoặc
   do model gốc sinh annotation không tuyệt đối chính xác) - clamp nhẹ
   này chỉ để tránh toạ độ pixel âm/vượt biên ảnh, KHÔNG liên quan đến
   margin của bbox->landmark (đó là việc của head/loss).
3. Bổ sung `check_lmk_margin_coverage()` (import từ file riêng
   check_lmk_margin_coverage.py) được nhắc trong docstring `main()` -
   nên chạy 1 lần trên data thật trước khi train dài hạn, để chọn đúng
   `lmk_margin` cho FaceLmkConfig (xem file đó để biết lý do).
4. Demo ở `main()` cập nhật để minh hoạ đúng luồng đồng bộ:
       cfg = FaceLmkConfig(...)
       dataset = FaceLandmarkDataset(...)
       cfg.sync_num_landmarks(dataset.num_landmarks)

CÁC ĐIỂM GIỮ NGUYÊN TỪ v2 (không đổi):
------------------------------------------------------------------------
- Hỗ trợ nhiều mặt / ảnh (record["faces"] là 1 danh sách).
- Trả về PIXEL-SPACE, không phải [0,1] normalized.
- Bỏ toạ độ z (depth).
- `face_landmark_collate` gộp batch thành list[dict] khớp thẳng với
  FaceLandmarkDetectionLoss.forward(preds, targets), không cần default_collate.
- Bỏ qua (skip) các face có bbox suy biến (width/height <= min_box_size_px).
- Toàn bộ kỹ thuật index byte-offset, lazy file handle theo worker, dò số
  lượng landmark từ dữ liệu (không hard-code 478), DataLoader hiện đại
  (pin_memory, persistent_workers, prefetch_factor).

CÀI ĐẶT:
    pip install torch torchvision matplotlib numpy pillow

CHẠY DEMO:
    python3 face_landmark_dataset_v3.py --root-dir /duong/dan/DataPretrain
"""

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import v2 as T
except ImportError as e:
    raise ImportError(
        "Cần torchvision bản mới có transforms.v2. Cài: pip install -U torchvision"
    ) from e

# ---------------------------------------------------------------------------
# Xây / nạp index byte-offset cho file JSONL lớn (giống bản gốc)
# ---------------------------------------------------------------------------

def _build_or_load_offsets(jsonl_path: str) -> np.ndarray:
    idx_path = jsonl_path + ".idx.npy"

    needs_rebuild = True
    if os.path.exists(idx_path):
        jsonl_mtime = os.path.getmtime(jsonl_path)
        idx_mtime = os.path.getmtime(idx_path)
        if idx_mtime >= jsonl_mtime:
            needs_rebuild = False

    if not needs_rebuild:
        return np.load(idx_path)

    print(f"[Dataset] Đang xây index cho {jsonl_path} (chỉ chạy 1 lần, lần sau sẽ cache)...")
    t0 = time.time()
    offsets: List[int] = []
    with open(jsonl_path, "rb") as f:
        offset = f.tell()
        for line in f:
            if line.strip():
                offsets.append(offset)
            offset = f.tell()
    offsets_arr = np.array(offsets, dtype=np.int64)
    np.save(idx_path, offsets_arr)
    print(f"[Dataset] Xong: {len(offsets_arr)} ảnh, mất {time.time() - t0:.1f}s. "
          f"Index lưu tại {idx_path}")
    return offsets_arr

def _detect_num_landmarks(jsonl_path: str, offsets: np.ndarray, scan_limit: int = 2000) -> int:
    """Dò số điểm landmark thực tế trong data (thường 478 với MediaPipe
    FaceLandmarker có iris), thay vì hard-code."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i in range(min(scan_limit, len(offsets))):
            f.seek(int(offsets[i]))
            record = json.loads(f.readline())
            faces = record.get("faces", [])
            if faces:
                return len(faces[0]["landmarks_normalized"])
    print("[Dataset] CẢNH BÁO: không tìm thấy ảnh nào có mặt trong "
          f"{scan_limit} dòng đầu để dò số landmark, dùng mặc định 478.")
    return 478


# ---------------------------------------------------------------------------
# Dataset (multi-face)
# ---------------------------------------------------------------------------

class FaceLandmarkDataset(Dataset):
    """
    Mỗi sample trả về:
        image            : FloatTensor (3, H, W), giá trị [0, 1]
        boxes            : FloatTensor (N, 4)      -- xyxy, PIXEL trong không gian (H, W)
        labels           : LongTensor  (N,)        -- toàn 0 ("face"), dự sau này thêm class
        landmarks        : FloatTensor (N, K, 2)   -- (x, y) PIXEL, K = số landmark/mặt
        landmarks_valid  : BoolTensor  (N,)         -- True = mặt này có nhãn landmark đầy đủ
        file_name        : str
        orig_size        : LongTensor (2,)         -- (width, height) ảnh gốc
        (N có thể = 0 nếu ảnh không có mặt nào)
    """

    def __init__(
        self,
        root_dir: str,
        jsonl_name: str = "annotations_all.jsonl",
        image_size: int = 224,
        transform: Optional[T.Transform] = None,
        min_box_size_px: float = 2.0,
    ):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "Images")
        self.jsonl_path = os.path.join(root_dir, jsonl_name)
        self.min_box_size_px = min_box_size_px  # bỏ qua bbox quá nhỏ/suy biến sau khi quy pixel

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(
                f"Không tìm thấy {self.jsonl_path}. "
                "Chạy merge_jsonl.py merge trước để gộp các shard."
            )

        self.offsets = _build_or_load_offsets(self.jsonl_path)
        self.num_landmarks = _detect_num_landmarks(self.jsonl_path, self.offsets)
        self.image_size = image_size

        # cờ chỉ để in cảnh báo "số landmark lệch" đúng 1 lần (fix #1),
        # tránh spam hàng nghìn dòng log nếu dataset lớn có nhiều record lệch.
        self._warned_lmk_mismatch = False

        # ---- Transform mặc định: KHÔNG augmentation, chỉ decode + resize + chuẩn hoá ----
        # Muốn thêm augmentation (vd RandomHorizontalFlip), xem ghi chú
        # `hflip_sample()` ở cuối file - KHÔNG chèn trực tiếp vào đây vì
        # box/landmark cần được lật ĐỒNG BỘ theo ảnh, transforms.v2 ảnh
        # đơn thuần không tự làm việc đó cho toạ độ ngoài luồng.
        self.transform = transform or T.Compose([
            T.ToImage(),
            T.Resize((image_size, image_size)),
            T.ToDtype(torch.float32, scale=True),
        ])

        self._file_handle = None  # mở lazy, riêng cho mỗi worker process

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_file(self):
        if self._file_handle is None:
            self._file_handle = open(self.jsonl_path, "r", encoding="utf-8")
        return self._file_handle

    def __getitem__(self, i: int):
        f = self._get_file()
        f.seek(int(self.offsets[i]))
        record = json.loads(f.readline())

        img_path = os.path.join(self.images_dir, record["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        S = float(self.image_size)
        faces = record.get("faces", [])

        boxes, labels, landmarks, landmarks_valid = [], [], [], []
        for face in faces:
            bb = face["bounding_box_normalized"]
            # (fix #2) clamp nhẹ về biên ảnh [0, S] - chỉ để chặn nhiễu số
            # (normalized hơi ngoài [0,1]), KHÔNG liên quan tới lmk_margin.
            x1 = min(max(bb["xmin"] * S, 0.0), S)
            y1 = min(max(bb["ymin"] * S, 0.0), S)
            x2 = min(max(bb["xmax"] * S, 0.0), S)
            y2 = min(max(bb["ymax"] * S, 0.0), S)
            if (x2 - x1) < self.min_box_size_px or (y2 - y1) < self.min_box_size_px:
                continue  # bbox suy biến -> bỏ qua

            pts = face["landmarks_normalized"]
            if len(pts) != self.num_landmarks:
                # (fix #1) record lệch số landmark so với phần còn lại của
                # dataset -> bỏ QUA RIÊNG mặt này (không làm crash cả
                # batch/cả lần train), chỉ cảnh báo 1 lần duy nhất.
                if not self._warned_lmk_mismatch:
                    print(
                        f"[Dataset] CẢNH BÁO: {record['file_name']} có "
                        f"{len(pts)} điểm landmark, khác với {self.num_landmarks} "
                        "đã dò lúc khởi tạo dataset - bỏ qua mặt này. "
                        "(cảnh báo này chỉ in 1 lần, có thể còn record khác bị lệch)"
                    )
                    self._warned_lmk_mismatch = True
                continue

            lm = [
                [min(max(p["x"] * S, 0.0), S), min(max(p["y"] * S, 0.0), S)]
                for p in pts
            ]

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # 1 class duy nhất: "face"
            landmarks.append(lm)
            landmarks_valid.append(True)  # MediaPipe luôn xuất đủ landmark khi phát hiện mặt

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            landmarks_t = torch.tensor(landmarks, dtype=torch.float32)  # (N, K, 2)
            valid_t = torch.tensor(landmarks_valid, dtype=torch.bool)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
            landmarks_t = torch.zeros((0, self.num_landmarks, 2), dtype=torch.float32)
            valid_t = torch.zeros((0,), dtype=torch.bool)

        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "boxes": boxes_t,
            "labels": labels_t,
            "landmarks": landmarks_t,
            "landmarks_valid": valid_t,
            "file_name": record["file_name"],
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate: gộp N mặt khác nhau mỗi ảnh thành đúng định dạng FaceLandmarkDetectionLoss cần
# ---------------------------------------------------------------------------

def face_landmark_collate(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)

    targets = [
        {
            "boxes": b["boxes"],
            "labels": b["labels"],
            "landmarks": b["landmarks"],
            "landmarks_valid": b["landmarks_valid"],
        }
        for b in batch
    ]

    file_names = [b["file_name"] for b in batch]
    orig_sizes = torch.stack([b["orig_size"] for b in batch], dim=0)

    return {
        "image": images,
        "targets": targets,     # <-- đưa thẳng vào loss_fn(preds, batch["targets"])
        "file_name": file_names,
        "orig_size": orig_sizes,
    }


def make_dataloader(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = FaceLandmarkDataset(root_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=face_landmark_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# (Tuỳ chọn, MẶC ĐỊNH KHÔNG bật) Random horizontal flip
# ---------------------------------------------------------------------------
"""
CẢNH BÁO QUAN TRỌNG khi tự thêm RandomHorizontalFlip cho landmark dạng
MESH (478 điểm MediaPipe FaceMesh, khác với 5/68-điểm thường gặp):

Lật ngang ảnh KHÔNG CHỈ cần đổi x -> (image_size - x). Với mesh dense,
MỖI INDEX có Ý NGHĨA CỐ ĐỊNH (vd index 33 luôn là "khoé mắt trái" trong
hệ toạ độ CHUẨN của MediaPipe). Sau khi lật ảnh, điểm từng là "mắt trái"
giờ nằm ở VỊ TRÍ của "mắt phải" -> cần HOÁN VỊ cả CHỈ SỐ, không chỉ đổi
dấu toạ độ, nếu không model sẽ học nhầm tưởng với 2 phần bức mặt.

MediaPipe có công bố bảng tương ứng trái-phải chính thức (canonical face
mesh symmetry) nhưng file này KHÔNG hard-code lại bảng đó (có thể sai
lệch version, rủi ro cao nếu sai). Nếu bạn có bảng flip_index_map đúng
(list độ dài K, flip_index_map[i] = index điểm đối xứng của điểm i),
dùng hàm dưới đây; nếu KHÔNG có, ĐỪNG bật flip cho dữ liệu mesh này -
tốt hơn là thiếu augmentation còn hơn là augmentation sai làm mô hình
học sai cấu trúc khuôn mặt.
"""

def hflip_sample(sample: dict, image_size: int, flip_index_map: Optional[List[int]] = None) -> dict:
    """
    sample: 1 phần tử trả về bởi FaceLandmarkDataset.__getitem__ (TRƯỚC
            khi collate). Lật ngang image + box + landmark.
    flip_index_map: BẮT BUỘC nếu landmarks là mesh dense (xem cảnh báo
                     trên). Với sơ đồ landmark ĐỐI XỨNG TỰ NHIÊN qua chỉ
                     số (vd 5-điểm RetinaFace nếu bạn tự quy ước
                     [0,1,2,3,4] = [mắt_trái,mắt_phải,mũi,miệng_trái,
                     miệng_phải]) thì map = [1,0,2,4,3].
    """
    img = sample["image"]
    flipped_img = torch.flip(img, dims=[-1])  # lật trục W

    boxes = sample["boxes"].clone()
    if boxes.numel():
        x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
        boxes[:, 0] = image_size - x2
        boxes[:, 2] = image_size - x1

    landmarks = sample["landmarks"].clone()
    if landmarks.numel():
        landmarks[..., 0] = image_size - landmarks[..., 0]
        if flip_index_map is not None:
            landmarks = landmarks[:, flip_index_map, :]
        # else: CHỈ đổi trục x, KHÔNG hoán vị chỉ số - dùng được cho các
        # điểm đối xứng qua đường giữa mặt (vd chóp mũi, cạnh môi giữa)
        # nhưng SAI cho các điểm có index gắn với "trái/phải" cố định.

    out = dict(sample)
    out["image"] = flipped_img
    out["boxes"] = boxes
    out["landmarks"] = landmarks
    return out


# ---------------------------------------------------------------------------
# Visualize: kiểm tra DataLoader hoạt động đúng (nhiều mặt / ảnh)
# ---------------------------------------------------------------------------

def visualize_batch(loader: DataLoader, save_path: str = "dataloader_demo.png", max_images: int = 8):
    """Lấy 1 batch từ loader, vẽ ảnh + TẤT CẢ mặt (box + landmark) trong ảnh, lưu PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    batch = next(iter(loader))
    images = batch["image"]        # (B,3,H,W)
    targets = batch["targets"]     # list[dict], độ dài B
    file_names = batch["file_name"]

    n = min(max_images, images.shape[0])
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))

        boxes = targets[i]["boxes"].numpy()
        landmarks = targets[i]["landmarks"].numpy()
        n_faces = boxes.shape[0]

        for fidx in range(n_faces):
            bx = boxes[fidx]
            rect = patches.Rectangle(
                (bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                linewidth=1.5, edgecolor="yellow", facecolor="none",
            )
            ax.add_patch(rect)
            lm = landmarks[fidx]
            ax.scatter(lm[:, 0], lm[:, 1], s=2, c="lime", alpha=0.8)

        status = f"{n_faces} mặt" if n_faces else "KHÔNG có mặt"
        ax.set_title(f"{file_names[i]}\n({status})", fontsize=8)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualize] Đã lưu demo tại: {save_path}")


# ---------------------------------------------------------------------------
# Demo chạy trực tiếp (fix #4: minh hoạ luồng đồng bộ cfg <-> dataset)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True,
                         help="Thư mục chứa Images/ và annotations_all.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="dataloader_demo.png")
    args = parser.parse_args()

    dataset = FaceLandmarkDataset(args.root_dir, image_size=args.image_size)
    print(f"Tổng số ảnh trong dataset : {len(dataset)}")
    print(f"Số landmark / mặt         : {dataset.num_landmarks}")

    # ---- Đồng bộ config dùng chung cho head/loss (xem face_lmk_config.py) ----
    from face_lmk_config import FaceLmkConfig
    cfg = FaceLmkConfig(nc=1, reg_max=16)
    cfg.sync_num_landmarks(dataset.num_landmarks)
    print(f"FaceLmkConfig đã đồng bộ  : num_landmarks={cfg.num_landmarks}, "
          f"lmk_margin={cfg.lmk_margin}")
    print("-> dùng CHÍNH cfg này để tạo DetectHeadFaceLmk(chs=..., cfg=cfg) "
          "và FaceLandmarkDetectionLoss(cfg=cfg).")

    sample = dataset[0]
    print("Shape 1 sample:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} ({v.dtype})")
        else:
            print(f"  {k}: {v}")

    loader = make_dataloader(
        args.root_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    t0 = time.time()
    batch = next(iter(loader))
    print(f"\nLấy 1 batch mất {time.time() - t0:.3f}s")
    print(f"  image shape : {tuple(batch['image'].shape)}")
    n_faces_per_img = [t["boxes"].shape[0] for t in batch["targets"]]
    print(f"  số mặt / ảnh trong batch: {n_faces_per_img}")

    visualize_batch(loader, save_path=args.save_path, max_images=min(8, args.batch_size))

if __name__ == "__main__":
    main()
