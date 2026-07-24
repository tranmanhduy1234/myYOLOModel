"""
face_landmark_dataset_v2.py
=============================
PyTorch Dataset / DataLoader cho annotations_all.jsonl (sinh boi
process_dataset_parallel.py + merge_jsonl.py), CHINH SUA de KHOP TRUC
TIEP voi dinh dang targets ma DetectHeadFaceLmk / FaceLandmarkDetectionLoss
(head_face_landmark.py / loss_face_landmark.py) mong doi.

THAY DOI SO VOI BAN GOC (face_landmark_dataset.py)
---------------------------------------------------------------------
1. HO TRO NHIEU MAT / ANH (ban goc chi lay faces[0], gia dinh 1 mat/anh).
   `record["faces"]` von di la MOT DANH SACH (extract_face_data() trong
   process_dataset_parallel.py duyet toan bo result.face_landmarks), nen
   du du lieu duoc trich xuat voi --num-faces 1 hay > 1, dataset nay deu
   doc dung TOAN BO danh sach do, khong bi cat con 1 mat.

2. TRA VE PIXEL-SPACE, KHONG PHAI [0,1] NORMALIZED.
   DetectHeadFaceLmk / FaceLandmarkDetectionLoss lam viec voi toa do
   PIXEL cua anh dau vao mang. Vi transform resize ve dung
   (image_size, image_size) (khong giu ti le, xem diem 5 trong docstring
   ban goc: x chia rieng theo width, y chia rieng theo height nen KHONG
   BI LECH khi nhan lai voi image_size), buoc quy doi rat don gian:
       pixel = normalized * image_size     (ca x va y, vi anh vuong sau resize)

3. BO TOA DO Z (DEPTH). Head/loss hien tai chi xu ly landmark 2D (x, y).
   z van con trong file JSONL goc (landmarks_normalized co z) neu sau
   nay can dung (vd uoc luong pose 3D), nhung o day khong dua vao tensor
   landmarks tra ve.

4. __getitem__ tra ve SO LUONG MAT KHAC NHAU MOI ANH (N co the = 0, 1, 2...)
   -> KHONG THE dung default_collate (no doi moi sample cung shape). Vi
   vay file nay dinh nghia `face_landmark_collate` rieng, gop batch thanh:
       images : (B, 3, H, W)  - stack binh thuong (cung kich thuoc sau resize)
       targets: list[dict] do dai B, MOI PHAN TU dung dinh dang ma
                FaceLandmarkDetectionLoss.forward(preds, targets) can:
                    {"boxes": (N,4) xyxy pixel,
                     "labels": (N,) long,
                     "landmarks": (N, K, 2) xyxy pixel,
                     "landmarks_valid": (N,) bool}
   -> Dua thang batch["targets"] vao loss_fn(preds, batch["targets"]),
      KHONG can xu ly gi them.

5. Bo qua (skip, khong dua vao targets) cac face co bbox suy bien
   (width hoac height <= 0 sau khi tinh tu min/max landmark) de tranh
   NaN/Inf khi tinh CIoU hoac khi chuan hoa landmark theo box trong loss.

6. Giu nguyen toan bo ky thuat cua ban goc: byte-offset index cho JSONL
   lon, lazy file handle theo worker, dò so luong landmark tu du lieu
   (khong hard-code 478), DataLoader hien dai (pin_memory, persistent_workers,
   prefetch_factor).

CAI DAT:
    pip install torch torchvision matplotlib numpy pillow

CHAY DEMO:
    python3 face_landmark_dataset_v2.py --root-dir /duong/dan/DataPretrain
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
    Mỗi sample trả về (LƯU Ý: khác bản gốc, N mặt thay vì 1 mặt cố định):
        image            : FloatTensor (3, H, W), giá trị [0, 1]
        boxes            : FloatTensor (N, 4)      -- xyxy, PIXEL trong khong gian (H, W)
        labels           : LongTensor  (N,)        -- toan 0 ("face"), du sau nay them class
        landmarks        : FloatTensor (N, K, 2)   -- (x, y) PIXEL, K = so landmark/mat
        landmarks_valid  : BoolTensor  (N,)         -- True = mat nay co nhan landmark day du
        file_name        : str
        orig_size        : LongTensor (2,)         -- (width, height) ảnh gốc
        (N co the = 0 neu anh khong co mat nao)
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
        self.min_box_size_px = min_box_size_px  # bo qua bbox qua nho/suy bien sau khi quy pixel

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(
                f"Không tìm thấy {self.jsonl_path}. "
                "Chạy merge_jsonl.py merge trước để gộp các shard."
            )

        self.offsets = _build_or_load_offsets(self.jsonl_path)
        self.num_landmarks = _detect_num_landmarks(self.jsonl_path, self.offsets)
        self.image_size = image_size

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

        S = self.image_size
        faces = record.get("faces", [])

        boxes, labels, landmarks, landmarks_valid = [], [], [], []
        for face in faces:
            bb = face["bounding_box_normalized"]
            x1, y1 = bb["xmin"] * S, bb["ymin"] * S
            x2, y2 = bb["xmax"] * S, bb["ymax"] * S
            if (x2 - x1) < self.min_box_size_px or (y2 - y1) < self.min_box_size_px:
                continue  # bbox suy bien (vd tat ca landmark trung 1 diem) -> bo qua

            pts = face["landmarks_normalized"]
            lm = [[p["x"] * S, p["y"] * S] for p in pts]

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # 1 class duy nhat: "face"
            landmarks.append(lm)
            landmarks_valid.append(True)  # MediaPipe luon xuat du landmark khi phat hien mat

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
# Collate: gop N mat khac nhau moi anh thanh dung dinh dang FaceLandmarkDetectionLoss can
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
        "targets": targets,     # <-- dua thang vao loss_fn(preds, batch["targets"])
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
# (Tuy chon, MAC DINH KHONG bat) Random horizontal flip
# ---------------------------------------------------------------------------
"""
CANH BAO QUAN TRONG khi tu them RandomHorizontalFlip cho landmark dang
MESH (478 diem MediaPipe FaceMesh, khac voi 5/68-diem thuong gap):

Lat ngang anh KHONG CHI can doi x -> (image_size - x). Voi mesh dense,
MOI INDEX co Y NGHIA CO DINH (vd index 33 luon la "khoe mat trai" trong
he toa do CHUAN cua MediaPipe). Sau khi lat anh, diem tung la "mat trai"
gio nam o VI TRI cua "mat phai" -> can HOAN VI ca CHI SO, khong chi doi
dau toa do, neu khong model se hoc nhan tuong tu voi 2 phan buc mat.

MediaPipe co cong bo bang tuong ung trai-phai chinh thuc (canonical face
mesh symmetry) nhung file nay KHONG hard-code lai bang do (co the sai
lech version, rui ro cao neu sai). Neu ban co bang flip_index_map dung
(list do dai K, flip_index_map[i] = index diem doi xung cua diem i),
dung ham duoi day; neu KHONG co, DUNG bat flip cho du lieu mesh nay -
tot hon la thieu augmentation con hon la augmentation sai lam mo hinh
hoc sai cau truc khuon mat.
"""

def hflip_sample(sample: dict, image_size: int, flip_index_map: Optional[List[int]] = None) -> dict:
    """
    sample: 1 phan tu tra ve boi FaceLandmarkDataset.__getitem__ (TRUOC
            khi collate). Lat ngang image + box + landmark.
    flip_index_map: BAT BUOC neu landmarks la mesh dense (xem canh bao
                     tren). Voi so do landmark DOI XUNG TU NHIEN qua chi
                     so (vd 5-diem RetinaFace neu ban tu quy uoc
                     [0,1,2,3,4] = [mat_trai,mat_phai,mui,mieng_trai,
                     mieng_phai]) thi map = [1,0,2,4,3].
    """
    img = sample["image"]
    flipped_img = torch.flip(img, dims=[-1])  # lat truc W

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
        # else: CHI doi truc x, KHONG hoan vi chi so - dung duoc cho cac
        # diem doi xung qua duong giua mat (vd chop mui, canh moi giua)
        # nhung SAI cho cac diem co index gan voi "trai/phai" co dinh.

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
    targets = batch["targets"]     # list[dict], do dai B
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
    print(f"  image shape : {tuple(batch['image'].shape)}")
    n_faces_per_img = [t["boxes"].shape[0] for t in batch["targets"]]
    print(f"  số mặt / ảnh trong batch: {n_faces_per_img}")

    visualize_batch(loader, save_path=args.save_path, max_images=min(8, args.batch_size))

if __name__ == "__main__":
    main()