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
    raise ImportError('Cần torchvision bản mới có transforms.v2. Cài: pip install -U torchvision') from e

def _build_or_load_offsets(jsonl_path: str) -> np.ndarray:
    idx_path = jsonl_path + '.idx.npy'
    needs_rebuild = True
    if os.path.exists(idx_path):
        jsonl_mtime = os.path.getmtime(jsonl_path)
        idx_mtime = os.path.getmtime(idx_path)
        if idx_mtime >= jsonl_mtime:
            needs_rebuild = False
    if not needs_rebuild:
        return np.load(idx_path)
    print(f'[Dataset] Đang xây index cho {jsonl_path} (chỉ chạy 1 lần, lần sau sẽ cache)...')
    t0 = time.time()
    offsets: List[int] = []
    with open(jsonl_path, 'rb') as f:
        offset = f.tell()
        for line in f:
            if line.strip():
                offsets.append(offset)
            offset = f.tell()
    offsets_arr = np.array(offsets, dtype=np.int64)
    np.save(idx_path, offsets_arr)
    print(f'[Dataset] Xong: {len(offsets_arr)} ảnh, mất {time.time() - t0:.1f}s. Index lưu tại {idx_path}')
    return offsets_arr

def _detect_num_landmarks(jsonl_path: str, offsets: np.ndarray, scan_limit: int=2000) -> int:
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i in range(min(scan_limit, len(offsets))):
            f.seek(int(offsets[i]))
            record = json.loads(f.readline())
            faces = record.get('faces', [])
            if faces:
                return len(faces[0]['landmarks_normalized'])
    print(f'[Dataset] CẢNH BÁO: không tìm thấy ảnh nào có mặt trong {scan_limit} dòng đầu để dò số landmark, dùng mặc định 478.')
    return 478

class FaceLandmarkDataset(Dataset):

    def __init__(self, root_dir: str, jsonl_name: str='annotations_all.jsonl', image_size: int=224, transform: Optional[T.Transform]=None, min_box_size_px: float=2.0):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'Images')
        self.jsonl_path = os.path.join(root_dir, jsonl_name)
        self.min_box_size_px = min_box_size_px
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f'Không tìm thấy {self.jsonl_path}. Chạy merge_jsonl.py merge trước để gộp các shard.')
        self.offsets = _build_or_load_offsets(self.jsonl_path)
        self.num_landmarks = _detect_num_landmarks(self.jsonl_path, self.offsets)
        self.image_size = image_size
        self._warned_lmk_mismatch = False
        self.transform = transform or T.Compose([T.ToImage(), T.Resize((image_size, image_size)), T.ToDtype(torch.float32, scale=True)])
        self._file_handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_file(self):
        if self._file_handle is None:
            self._file_handle = open(self.jsonl_path, 'r', encoding='utf-8')
        return self._file_handle

    def __getitem__(self, i: int):
        f = self._get_file()
        f.seek(int(self.offsets[i]))
        record = json.loads(f.readline())
        img_path = os.path.join(self.images_dir, record['file_name'])
        image = Image.open(img_path).convert('RGB')
        (orig_w, orig_h) = image.size
        S = float(self.image_size)
        faces = record.get('faces', [])
        (boxes, labels, landmarks, landmarks_valid) = ([], [], [], [])
        for face in faces:
            bb = face['bounding_box_normalized']
            x1 = min(max(bb['xmin'] * S, 0.0), S)
            y1 = min(max(bb['ymin'] * S, 0.0), S)
            x2 = min(max(bb['xmax'] * S, 0.0), S)
            y2 = min(max(bb['ymax'] * S, 0.0), S)
            if x2 - x1 < self.min_box_size_px or y2 - y1 < self.min_box_size_px:
                continue
            pts = face['landmarks_normalized']
            if len(pts) != self.num_landmarks:
                if not self._warned_lmk_mismatch:
                    print(f"[Dataset] CẢNH BÁO: {record['file_name']} có {len(pts)} điểm landmark, khác với {self.num_landmarks} đã dò lúc khởi tạo dataset - bỏ qua mặt này. (cảnh báo này chỉ in 1 lần, có thể còn record khác bị lệch)")
                    self._warned_lmk_mismatch = True
                continue
            lm = [[min(max(p['x'] * S, 0.0), S), min(max(p['y'] * S, 0.0), S)] for p in pts]
            boxes.append([x1, y1, x2, y2])
            labels.append(0)
            landmarks.append(lm)
            landmarks_valid.append(True)
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            landmarks_t = torch.tensor(landmarks, dtype=torch.float32)
            valid_t = torch.tensor(landmarks_valid, dtype=torch.bool)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
            landmarks_t = torch.zeros((0, self.num_landmarks, 2), dtype=torch.float32)
            valid_t = torch.zeros((0,), dtype=torch.bool)
        image_tensor = self.transform(image)
        return {'image': image_tensor, 'boxes': boxes_t, 'labels': labels_t, 'landmarks': landmarks_t, 'landmarks_valid': valid_t, 'file_name': record['file_name'], 'orig_size': torch.tensor([orig_w, orig_h], dtype=torch.long)}

def face_landmark_collate(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    targets = [{'boxes': b['boxes'], 'labels': b['labels'], 'landmarks': b['landmarks'], 'landmarks_valid': b['landmarks_valid']} for b in batch]
    file_names = [b['file_name'] for b in batch]
    orig_sizes = torch.stack([b['orig_size'] for b in batch], dim=0)
    return {'image': images, 'targets': targets, 'file_name': file_names, 'orig_size': orig_sizes}

def make_dataloader(root_dir: str, batch_size: int=32, image_size: int=224, num_workers: int=4, shuffle: bool=True) -> DataLoader:
    dataset = FaceLandmarkDataset(root_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=face_landmark_collate, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=False)

def hflip_sample(sample: dict, image_size: int, flip_index_map: Optional[List[int]]=None) -> dict:
    img = sample['image']
    flipped_img = torch.flip(img, dims=[-1])
    boxes = sample['boxes'].clone()
    if boxes.numel():
        (x1, x2) = (boxes[:, 0].clone(), boxes[:, 2].clone())
        boxes[:, 0] = image_size - x2
        boxes[:, 2] = image_size - x1
    landmarks = sample['landmarks'].clone()
    if landmarks.numel():
        landmarks[..., 0] = image_size - landmarks[..., 0]
        if flip_index_map is not None:
            landmarks = landmarks[:, flip_index_map, :]
    out = dict(sample)
    out['image'] = flipped_img
    out['boxes'] = boxes
    out['landmarks'] = landmarks
    return out

def visualize_batch(loader: DataLoader, save_path: str='dataloader_demo.png', max_images: int=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    batch = next(iter(loader))
    images = batch['image']
    targets = batch['targets']
    file_names = batch['file_name']
    n = min(max_images, images.shape[0])
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    (fig, axes) = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    for i in range(n):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        boxes = targets[i]['boxes'].numpy()
        landmarks = targets[i]['landmarks'].numpy()
        n_faces = boxes.shape[0]
        for fidx in range(n_faces):
            bx = boxes[fidx]
            rect = patches.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], linewidth=1.5, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            lm = landmarks[fidx]
            ax.scatter(lm[:, 0], lm[:, 1], s=2, c='lime', alpha=0.8)
        status = f'{n_faces} mặt' if n_faces else 'KHÔNG có mặt'
        ax.set_title(f'{file_names[i]}\n({status})', fontsize=8)
        ax.axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[Visualize] Đã lưu demo tại: {save_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True, help='Thư mục chứa Images/ và annotations_all.jsonl')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='dataloader_demo.png')
    args = parser.parse_args()
    dataset = FaceLandmarkDataset(args.root_dir, image_size=args.image_size)
    print(f'Tổng số ảnh trong dataset : {len(dataset)}')
    print(f'Số landmark / mặt         : {dataset.num_landmarks}')
    from face_lmk_config import FaceLmkConfig
    cfg = FaceLmkConfig(nc=1, reg_max=16)
    cfg.sync_num_landmarks(dataset.num_landmarks)
    print(f'FaceLmkConfig đã đồng bộ  : num_landmarks={cfg.num_landmarks}, lmk_margin={cfg.lmk_margin}')
    print('-> dùng CHÍNH cfg này để tạo DetectHeadFaceLmk(chs=..., cfg=cfg) và FaceLandmarkDetectionLoss(cfg=cfg).')
    sample = dataset[0]
    print('Shape 1 sample:')
    for (k, v) in sample.items():
        if torch.is_tensor(v):
            print(f'  {k}: {tuple(v.shape)} ({v.dtype})')
        else:
            print(f'  {k}: {v}')
    loader = make_dataloader(args.root_dir, batch_size=args.batch_size, image_size=args.image_size, num_workers=args.num_workers, shuffle=True)
    t0 = time.time()
    batch = next(iter(loader))
    print(f'\nLấy 1 batch mất {time.time() - t0:.3f}s')
    print(f"  image shape : {tuple(batch['image'].shape)}")
    n_faces_per_img = [t['boxes'].shape[0] for t in batch['targets']]
    print(f'  số mặt / ảnh trong batch: {n_faces_per_img}')
    visualize_batch(loader, save_path=args.save_path, max_images=min(8, args.batch_size))
if __name__ == '__main__':
    main()
