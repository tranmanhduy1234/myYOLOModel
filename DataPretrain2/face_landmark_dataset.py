"""
face_landmark_dataset.py
=========================

PyTorch Dataset / DataLoader cho bộ dữ liệu do process_dataset_parallel.py +
merge_jsonl.py tạo ra:

    <root_dir>/Images/<file_name>.jpg
    <root_dir>/annotations_all.jsonl      (mỗi dòng: 1 object JSON cho 1 ảnh)

CÔNG NGHỆ / KỸ THUẬT ĐƯỢC ÁP DỤNG
---------------------------------
1. RANDOM-ACCESS TRÊN JSONL LỚN BẰNG BYTE-OFFSET INDEX
   - Với annotations_all.jsonl có thể chứa 1 triệu dòng, ta KHÔNG load toàn
     bộ file vào RAM. Ta chỉ quét 1 lần để lưu vị trí byte (offset) đầu mỗi
     dòng vào 1 file index nhỏ (.idx.npy). Lần sau __getitem__(i) chỉ cần
     f.seek(offset[i]) rồi đọc đúng 1 dòng -> O(1), không phụ thuộc kích
     thước dataset.
   - Index được cache lại trên đĩa, chỉ build lại nếu file .jsonl mới hơn.

2. LAZY FILE HANDLE THEO WORKER (an toàn với num_workers > 0)
   - File handle KHÔNG mở trong __init__ (vì Dataset bị pickle sang các
     worker process). Mỗi worker tự mở file riêng ở lần __getitem__ đầu
     tiên -> không chia sẻ file descriptor giữa các process.

3. torchvision.transforms.v2 (API transform mới nhất của torchvision,
   thay cho `transforms` cũ) để decode ảnh -> tensor + resize.

4. DataLoader hiện đại: pin_memory, persistent_workers, prefetch_factor.

5. Landmark được lưu ở dạng NORMALIZED (x, y là tỉ lệ theo width/height gốc)
   nên khi resize ảnh về kích thước cố định, tọa độ landmark vẫn đúng
   KHÔNG cần biến đổi lại (vì x chia theo width, y chia theo height riêng).

6. CHƯA áp dụng augmentation (theo yêu cầu) - chỗ để thêm augmentation sau
   này (RandomHorizontalFlip, ColorJitter, v.v.) được đánh dấu rõ trong code.

7. Hàm visualize_batch(): lấy 1 batch từ DataLoader, vẽ ảnh + landmark +
   bounding box bằng matplotlib, lưu ra 1 file PNG để kiểm tra DataLoader
   hoạt động đúng.

CÀI ĐẶT:
    pip install torch torchvision matplotlib numpy pillow

CHẠY DEMO:
    python3 face_landmark_dataset.py --root-dir /run/media/tranmanhduy/Data/DataPretrain
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
# Xây / nạp index byte-offset cho file JSONL lớn
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
    """Dò số điểm landmark thực tế trong data (thường là 478 với MediaPipe
    FaceLandmarker có iris), thay vì hard-code, để không lệ thuộc version model."""
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
# Dataset
# ---------------------------------------------------------------------------

class FaceLandmarkDataset(Dataset):
    """
    Mỗi sample trả về:
        image      : FloatTensor (3, H, W), giá trị [0, 1]
        landmarks  : FloatTensor (num_landmarks, 3)  -- (x, y, z) normalized
        bbox       : FloatTensor (4,)                -- (xmin, ymin, xmax, ymax) normalized
        has_face   : BoolTensor  ()                  -- False nếu ảnh không có mặt
        file_name  : str
        orig_size  : LongTensor (2,)                 -- (width, height) ảnh gốc
    """

    def __init__(
        self,
        root_dir: str,
        jsonl_name: str = "annotations_all.jsonl",
        image_size: int = 224,
        transform: Optional[T.Transform] = None,
    ):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "Images")
        self.jsonl_path = os.path.join(root_dir, jsonl_name)

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(
                f"Không tìm thấy {self.jsonl_path}. "
                "Chạy merge_jsonl.py merge trước để gộp các shard."
            )

        self.offsets = _build_or_load_offsets(self.jsonl_path)
        self.num_landmarks = _detect_num_landmarks(self.jsonl_path, self.offsets)
        self.image_size = image_size

        # ---- Transform mặc định: KHÔNG augmentation, chỉ decode + resize + chuẩn hoá ----
        # Muốn thêm augmentation sau này, chèn vào list dưới đây, ví dụ:
        #   T.RandomHorizontalFlip(p=0.5),   (nhớ: phải lật lại landmark tương ứng!)
        #   T.ColorJitter(brightness=0.2, contrast=0.2),
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

        faces = record.get("faces", [])
        has_face = len(faces) > 0

        if has_face:
            face = faces[0]  # dataset này giả định 1 mặt / ảnh (num_faces=1 lúc trích xuất)
            landmarks = torch.tensor(
                [[p["x"], p["y"], p["z"]] for p in face["landmarks_normalized"]],
                dtype=torch.float32,
            )
            bb = face["bounding_box_normalized"]
            bbox = torch.tensor([bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]], dtype=torch.float32)
        else:
            landmarks = torch.zeros((self.num_landmarks, 3), dtype=torch.float32)
            bbox = torch.zeros(4, dtype=torch.float32)

        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "landmarks": landmarks,
            "bbox": bbox,
            "has_face": torch.tensor(has_face, dtype=torch.bool),
            "file_name": record["file_name"],
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.long),
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Visualize: kiểm tra DataLoader hoạt động đúng
# ---------------------------------------------------------------------------

def visualize_batch(loader: DataLoader, save_path: str = "dataloader_demo.png", max_images: int = 8):
    """Lấy 1 batch từ loader, vẽ ảnh + landmark + bbox, lưu ra file PNG."""
    import matplotlib
    matplotlib.use("Agg")  # không cần môi trường có màn hình
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    batch = next(iter(loader))
    images = batch["image"]          # (B, 3, H, W) float [0,1]
    landmarks = batch["landmarks"]   # (B, N, 3)
    has_face = batch["has_face"]     # (B,)
    bbox = batch["bbox"]             # (B, 4)
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
        h, w = img.shape[:2]

        if has_face[i]:
            lm = landmarks[i].numpy()
            xs, ys = lm[:, 0] * w, lm[:, 1] * h
            ax.scatter(xs, ys, s=2, c="lime", alpha=0.8)

            bx = bbox[i].numpy()
            rect = patches.Rectangle(
                (bx[0] * w, bx[1] * h),
                (bx[2] - bx[0]) * w,
                (bx[3] - bx[1]) * h,
                linewidth=1.5, edgecolor="yellow", facecolor="none",
            )
            ax.add_patch(rect)
            status = "có mặt"
        else:
            status = "KHÔNG có mặt"

        ax.set_title(f"{file_names[i]}\n({status})", fontsize=8)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualize] Đã lưu demo tại: {save_path}")


# ---------------------------------------------------------------------------
# Demo chạy trực tiếp
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
    print(f"  image shape    : {tuple(batch['image'].shape)}")
    print(f"  landmarks shape: {tuple(batch['landmarks'].shape)}")
    print(f"  has_face       : {batch['has_face'].tolist()}")

    visualize_batch(loader, save_path=args.save_path, max_images=min(8, args.batch_size))

if __name__ == "__main__":
    main()