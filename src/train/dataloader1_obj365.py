import os 
import json
import random
import pickle
import inspect
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from src.config import TrainConfig

def _make_gauss_noise(var_limit, p):
    params = inspect.signature(A.GaussNoise.__init__).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=var_limit, p=p)
    elif "std_range" in params:
        var_min, var_max = var_limit
        std_min = max(0.0, min(1.0, (var_min ** 0.5) / 255.0))
        std_max = max(0.0, min(1.0, (var_max ** 0.5) / 255.0))
        return A.GaussNoise(std_range=(std_min, std_max), p=p)
    
    raise RuntimeError("Phiên bản albumentations hiện tại không hỗ trợ tham số GaussNoise đã biết.")

def _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, p, fill_color=(114, 114, 114)):
    params = inspect.signature(A.ShiftScaleRotate.__init__).parameters
    kwargs = dict(
        shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
        border_mode=cv2.BORDER_CONSTANT, p=p
    )
    if "value" in params:
        kwargs["value"] = fill_color
    elif "fill" in params:
        kwargs["fill"] = fill_color
    return A.ShiftScaleRotate(**kwargs)

class DetectionAugmenter:
    def __init__(self, cfg: TrainConfig):
        shift_limit, scale_limit, rotate_limit, ssr_p = cfg.shiftScaleRotate
        hue_shift, sat_shift, val_shift, hsv_p = cfg.hueSaturationValue
        var_min, var_max, gn_p = cfg.gaussNoise
        blur_limit, blur_p = cfg.blur
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=cfg.horizontalFlip),
            _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, ssr_p),
            A.RandomBrightnessContrast(p=cfg.randomBrightnessContrast),
            A.HueSaturationValue(
                hue_shift_limit=hue_shift, sat_shift_limit=sat_shift,
                val_shift_limit=val_shift, p=hsv_p
            ),
            _make_gauss_noise((var_min, var_max), p=gn_p),
            A.Blur(blur_limit=blur_limit, p=blur_p)
        ], bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category_ids"], min_visibility=0.4
        ))
    
    def __call__(self, image, boxes, labels):
        boxes = np.array(boxes, dtype=np.float32).tolist()
        labels = np.array(labels, dtype=np.int64).tolist()
        
        if len(boxes) == 0:
            return image, boxes, labels
        try: 
            augmented = self.transform(image=image, bboxes=boxes, category_ids=labels)
            return augmented["image"], augmented["bboxes"], augmented["category_ids"]
        
        except Exception as e:
            print(f"[Augmenter][Warning] Bỏ qua augment do lỗi: {e}")
            return image, boxes, labels
        
def letterbox(image, new_size, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top

def build_id_offset_index(jsonl_path, id_field="id", cache_path=None, force_rebuild=False):
    if cache_path and os.path.isfile(cache_path) and not force_rebuild:
        print(f"[Data] Load id->offset index đã cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[Data] Đang build id->offset index từ: {jsonl_path} ...")
    index = {}
    with open(jsonl_path, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            if line.strip():
                record = json.loads(line)
                index[record[id_field]] = offset
            offset = f.tell()
            line = f.readline()
    print(f"[Data] Xong: {len(index):,} dòng đã được index.")
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Data] Đã lưu cache tại: {cache_path}")
    return index

def build_annotation_group_index(annotations_path, cache_path=None, force_rebuild=False):
    if cache_path and os.path.isfile(cache_path) and not force_rebuild:
        print(f"[Data] Load annotation group index đã cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
        
    print(f"[Data] Đang build annotation index (group theo image_id) từ: {annotations_path} ...")
    index = defaultdict(list)
    with open(annotations_path, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            if line.strip():
                record = json.loads(line)
                index[record["image_id"]].append(offset)
            offset = f.tell()
            line = f.readline()
    index = dict(index)
    n_ann = sum(len(v) for v in index.values())
    print(f"[Data] Xong: {len(index):,} ảnh có annotation, tổng {n_ann:,} annotation.")
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Data] Đã lưu cache tại: {cache_path}")
    return index

def load_image_path_map(path_map_file, cache_path=None, force_rebuild=False):
    if cache_path and os.path.isfile(cache_path) and not force_rebuild:
        print(f"[Data] Load image_path_map đã cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
        
    print(f"[Data] Đang load image_path_map từ: {path_map_file} ...")
    mapping = {}
    with open(path_map_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mapping[rec["image_name"]] = rec["path"]
    print(f"[Data] Xong: {len(mapping):,} ảnh trong image_path_map.")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Data] Đã lưu cache tại: {cache_path}")
    return mapping

def load_categories(categories_path):
    records = []
    with open(categories_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) == 0:
        raise RuntimeError(f"'{categories_path}' tồn tại nhưng rỗng - không có class nào được khai báo.")
    records.sort(key=lambda r: r["id"])
    cat_id_to_idx = {r["id"]: i for i, r in enumerate(records)}
    return cat_id_to_idx, records, len(records)

class ObjectDetectionDataset(Dataset):
    def __init__(self, images_info_path, annotations_path, image_path_map,
                 images_root_dir, images_split_dir, cat_id_to_idx, image_ids,
                 images_offset_index, ann_group_index,
                 imgsz=480, augmenter=None,
                 skip_iscrowd=True, skip_isfake=True):
        self.images_info_path = images_info_path
        self.annotations_path = annotations_path
        self.image_path_map = image_path_map
        self.images_split_root = os.path.join(images_root_dir, images_split_dir)
        self.cat_id_to_idx = cat_id_to_idx
        self.image_ids = image_ids                    
        self.images_offset_index = images_offset_index
        self.ann_group_index = ann_group_index           
        self.imgsz = imgsz
        self.augmenter = augmenter
        self.skip_iscrowd = skip_iscrowd
        self.skip_isfake = skip_isfake
        
        n_ann = sum(len(self.ann_group_index.get(i, [])) for i in self.image_ids)
        print(f"[Data] Dataset sẵn sàng với {len(self.image_ids):,} ảnh, {n_ann:,} annotation "
              f"(augment={'ON' if augmenter is not None else 'OFF'}). RAM cho pixel data: ~0MB.")

    def __len__(self):
        return len(self.image_ids)
    
    def _read_image_info(self, image_id):
        offset = self.images_offset_index[image_id]
        with open(self.images_info_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            return json.loads(f.readline())
        
    def _read_annotations(self, image_id):
        offsets = self.ann_group_index.get(image_id, [])
        if not offsets:
            return []
        records = []
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            for off in offsets:
                f.seek(off)
                records.append(json.loads(f.readline()))
        return records

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self._read_image_info(image_id)
        
        file_name = info["file_name"]
        rel_path = self.image_path_map.get(file_name)
        if rel_path is None:
            raise KeyError(
                f"Không tìm thấy file_name='{file_name}' (image_id={image_id}) trong image_path_map."
            )
        img_path = os.path.join(self.images_split_root, rel_path)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc tệp ảnh tại đường dẫn: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if img_rgb.shape[0] == self.imgsz and img_rgb.shape[1] == self.imgsz:
            img_resized = img_rgb
            scale, pad_left, pad_top = 1.0, 0, 0
        else:
            img_resized, scale, pad_left, pad_top = letterbox(img_rgb, self.imgsz)
            
        boxes, labels = [], []
        for ann in self._read_annotations(image_id):
            if self.skip_iscrowd and ann.get("iscrowd", 0) == 1:
                continue
            if self.skip_isfake and ann.get("isfake", 0) == 1:
                continue
            
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            x1_new = x1 * scale + pad_left
            y1_new = y1 * scale + pad_top
            x2_new = x2 * scale + pad_left
            y2_new = y2 * scale + pad_top
            
            x1_new = float(np.clip(x1_new, 0, self.imgsz))
            y1_new = float(np.clip(y1_new, 0, self.imgsz))
            x2_new = float(np.clip(x2_new, 0, self.imgsz))
            y2_new = float(np.clip(y2_new, 0, self.imgsz))
            
            if x2_new <= x1_new or y2_new <= y1_new:
                continue
            
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_idx:
                continue
            
            boxes.append([x1_new, y1_new, x2_new, y2_new])
            labels.append(self.cat_id_to_idx[cat_id])

        if self.augmenter is not None:
            img_resized, boxes, labels = self.augmenter(img_resized, boxes, labels)
            
        img_numpy = np.ascontiguousarray(img_resized, dtype=np.uint8).copy()
        img_tensor = torch.from_numpy(img_numpy).permute(2, 0, 1).float() / 255.0
        
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target
    
def collate_fn(batch):
    imgs, targets = zip(*batch)
    images = torch.stack(imgs, dim=0)
    targets = list(targets)
    return images, targets

def _build_split_dataset(cfg: TrainConfig, split_dir, is_train, image_path_map_filename, images_split_dir, cat_id_to_idx):
    images_info_path = os.path.join(split_dir, cfg.images_info_filename)
    annotations_path = os.path.join(split_dir, cfg.annotations_filename)
    image_path_map_path = os.path.join(split_dir, image_path_map_filename)
    
    os.makedirs(cfg.index_cache_dir, exist_ok=True)
    split_name = os.path.basename(os.path.normpath(split_dir))
    
    images_offset_index = build_id_offset_index(
        images_info_path, id_field="id",
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_images_info.idx.pkl"),
        force_rebuild=cfg.rebuild_index
    )
    ann_group_index = build_annotation_group_index(
        annotations_path,
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_annotations_group.idx.pkl"),
        force_rebuild=cfg.rebuild_index
    )
    image_path_map = load_image_path_map(
        image_path_map_path,
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_image_path_map.pkl"),
        force_rebuild=cfg.rebuild_index 
    )
    
    image_ids = list(images_offset_index.keys())
    if not cfg.include_images_without_annotations:
        image_ids = [i for i in image_ids if ann_group_index.get(i)]
        
    augmenter = DetectionAugmenter(cfg) if is_train else None
    return ObjectDetectionDataset(
        images_info_path=images_info_path,
        annotations_path=annotations_path,
        image_path_map=image_path_map,
        images_root_dir=cfg.images_root_dir,
        images_split_dir=images_split_dir,
        cat_id_to_idx=cat_id_to_idx,
        image_ids=image_ids,
        images_offset_index=images_offset_index,
        ann_group_index=ann_group_index,
        imgsz=cfg.img_size,
        augmenter=augmenter,
        skip_iscrowd=cfg.skip_iscrowd,
        skip_isfake=cfg.skip_isfake,
    )

def build_dataloaders(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    train_dir = os.path.join(cfg.labels_root, cfg.train_subdir)
    val_dir = os.path.join(cfg.labels_root, cfg.val_subdir)
    
    cat_id_to_idx, classes, num_classes = load_categories(
        os.path.join(train_dir, cfg.categories_filename)
    )
    
    train_dataset = _build_split_dataset(
        cfg, train_dir, is_train=True,
        image_path_map_filename=cfg.train_image_path_map_filename,
        images_split_dir=cfg.images_train_subdir, cat_id_to_idx=cat_id_to_idx
    )
    
    val_dataset = None
    val_images_info = os.path.join(val_dir, cfg.images_info_filename)
    if os.path.isfile(val_images_info):
        val_dataset = _build_split_dataset(
            cfg, val_dir, is_train=False,
            image_path_map_filename=cfg.val_image_path_map_filename,
            images_split_dir=cfg.images_val_subdir, cat_id_to_idx=cat_id_to_idx
        )
    else:
        print(f"[Data][Notice] Không tìm thấy '{val_images_info}' - bỏ qua val_loader.")

    def _dl_kwargs(is_train):
        kwargs = dict(
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle if is_train else False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last if is_train else False,
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        )
        if cfg.num_workers > 0:
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return kwargs

    train_loader = DataLoader(train_dataset, **_dl_kwargs(is_train=True))
    val_loader = DataLoader(val_dataset, **_dl_kwargs(is_train=False)) if val_dataset is not None else None

    print(f"[Data] Train: {len(train_dataset):,} ảnh | "
          f"Val: {len(val_dataset) if val_dataset else 0:,} ảnh | num_classes={num_classes}")

    return train_loader, val_loader, classes, num_classes

if __name__ == "__main__":
    cfg = TrainConfig()
    train_loader, val_loader, classes, num_classes = build_dataloaders(cfg)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for batch_idx, (images, targets) in enumerate(train_loader):
        num_show = min(4, len(images))
        """
            targets : list[dict] do dai = batch_size:
                {"boxes": (N,4), [PIXEL], xyxy, "labels": (N,)}
        """
        fig, axes = plt.subplots(1, num_show, figsize=(6 * num_show, 6))
        if num_show == 1:
            axes = [axes]

        for ax, image, target in zip(axes, images[:num_show], targets[:num_show]):
            # CHW -> HWC
            img = image.permute(1, 2, 0).cpu().numpy()
            img = img.clip(0, 1)  # bỏ dòng này nếu ảnh chưa về [0,1]
            ax.imshow(img)
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1),x2 - x1,y2 - y1,linewidth=2,edgecolor="red",facecolor="none")
                ax.add_patch(rect)
                ax.text(x1,y1,str(int(label)),color="white",fontsize=9,bbox=dict(facecolor="red", alpha=0.7, pad=1))
            ax.set_title(f"{len(boxes)} objects")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        if batch_idx >= 5:
            break