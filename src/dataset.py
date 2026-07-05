import os
import json
import random
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image, ImageEnhance


def letterbox(img, img_size, fill=(114, 114, 114)):
    """
    Resize giu nguyen ty le khung hinh (khong meo anh), pad phan con lai
    bang mau xam trung tinh (kieu YOLO). Tra ve anh moi + he so bien doi
    de map box tu toa do goc sang toa do sau letterbox.
    """
    org_w, org_h = img.size
    scale = min(img_size / org_w, img_size / org_h)
    new_w, new_h = int(round(org_w * scale)), int(round(org_h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (img_size, img_size), fill)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


class YOLOv10Dataset(Dataset):
    """
    Dataset class for loading YOLOv10 data from a custom directory structure.
    
    Data Structure:
    data_dir/
    ├── Images/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── annotations.jsonl   # Line format: {"image_name": "...", "object": [{"x1":..., "y1":..., "x2":..., "y2":..., "id_class":...}]}
    └── classes.jsonl       # Line format: {"id": 0, "name_class": "person"} or {"0": "person"}

    Tham so augment:
        augment=True   -> bat cac augmentation nhe (chi dung cho tap train):
                           random horizontal flip, color jitter (brightness/
                           contrast/saturation). Anh luon duoc resize kieu
                           "letterbox" (giu ty le, pad xam) thay vi resize
                           meo hinh nhu truoc, giup box khong bi bien dang.
    """
    def __init__(self, data_dir, annotations_file="annotations.jsonl", classes_file="classes.jsonl",
                 img_size=480, augment=False, hflip_p=0.5, color_jitter_p=0.5):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.img_size = img_size
        self.augment = augment
        self.hflip_p = hflip_p
        self.color_jitter_p = color_jitter_p

        # Load classes mapping
        classes_path = os.path.join(data_dir, classes_file)
        self.classes = self._load_classes(classes_path)
        
        # Load annotations
        annotations_path = os.path.join(data_dir, annotations_file)
        self.annotations = self._load_annotations(annotations_path)
        
    def _load_classes(self, file_path):
        """
        Parses the classes JSONL file. 
        Supports format: {"id": 0, "name_class": "name"} or {"0": "name"}
        """
        classes = {}
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Classes file not found at: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                # Check for {"id": ..., "name_class": ...}
                if "id" in data and ("name_class" in data or "name" in data):
                    class_id = int(data["id"])
                    class_name = data.get("name_class") or data.get("name")
                    classes[class_id] = class_name
                # Check for {"id_class": ..., "name_class": ...}
                elif "id_class" in data and "name_class" in data:
                    class_id = int(data["id_class"])
                    class_name = data["name_class"]
                    classes[class_id] = class_name
                # Check for direct mapping {"0": "name"}
                else:
                    for k, v in data.items():
                        try:
                            class_id = int(k)
                            classes[class_id] = v
                        except ValueError:
                            pass
        return classes

    def _load_annotations(self, file_path):
        """
        Parses the annotations JSONL file.
        Each line represents one image and its bounding box annotations.
        """
        annotations = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Annotations file not found at: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "image_name" in data:
                    annotations.append(data)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image_name = anno["image_name"]
        img_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        img = Image.open(img_path).convert("RGB")
        org_w, org_h = img.size

        img_lb, scale, pad_x, pad_y = letterbox(img, self.img_size)

        do_hflip = self.augment and random.random() < self.hflip_p
        if do_hflip:
            img_lb = img_lb.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment and random.random() < self.color_jitter_p:
            img_lb = self._color_jitter(img_lb)

        img_np = np.array(img_lb, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        boxes = []
        labels = []

        objects = anno.get("object", [])
        for obj in objects:
            x1 = float(obj["x1"]) * scale + pad_x
            y1 = float(obj["y1"]) * scale + pad_y
            x2 = float(obj["x2"]) * scale + pad_x
            y2 = float(obj["y2"]) * scale + pad_y

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            if do_hflip:
                x1, x2 = self.img_size - x2, self.img_size - x1

            x1 = max(0.0, min(x1, self.img_size))
            y1 = max(0.0, min(y1, self.img_size))
            x2 = max(0.0, min(x2, self.img_size))
            y2 = max(0.0, min(y2, self.img_size))

            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            class_id = int(obj["id_class"])
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return img_tensor, boxes_tensor, labels_tensor, (org_h, org_w), image_name

    @staticmethod
    def _color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3):
        if brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-brightness, brightness))
        if contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-contrast, contrast))
        if saturation > 0:
            img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-saturation, saturation))
        return img


def split_dataset(dataset, val_ratio=0.1, seed=42):
    """Tra ve (train_idx, val_idx) de dung voi torch.utils.data.Subset."""
    n = len(dataset)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Pads bounding boxes and labels to match the image in the batch with the maximum number of objects.
    
    Args:
        batch: List of tuples (image_tensor, boxes_tensor, labels_tensor, original_size, image_name)
        
    Returns:
        dict containing:
            - "images": Tensor of shape (B, 3, H, W)
            - "gt_boxes": Tensor of shape (B, M, 4) - Padded bounding boxes
            - "gt_labels": Tensor of shape (B, M) - Padded class labels
            - "gt_mask": Tensor of shape (B, M) - Boolean mask indicating active (True) vs padded (False) boxes
            - "image_names": List of image names in the batch
            - "org_sizes": List of original image dimensions (height, width)
    """
    images = []
    boxes_list = []
    labels_list = []
    img_names = []
    org_sizes = []
    
    for img, boxes, labels, org_size, img_name in batch:
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
        img_names.append(img_name)
        org_sizes.append(org_size)
        
    # Stack images into shape (B, 3, img_size, img_size)
    images = torch.stack(images, dim=0)
    
    # Find the maximum number of bounding boxes in this batch
    max_objs = max(boxes.shape[0] for boxes in boxes_list)
    # If all images in the batch are empty of objects, default max_objs to 1 to avoid empty dimensions
    if max_objs == 0:
        max_objs = 1
        
    B = len(batch)
    padded_boxes = torch.zeros(B, max_objs, 4, dtype=torch.float32)
    padded_labels = torch.zeros(B, max_objs, dtype=torch.long)
    padded_mask = torch.zeros(B, max_objs, dtype=torch.bool)
    
    for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        num_objs = boxes.shape[0]
        if num_objs > 0:
            padded_boxes[i, :num_objs] = boxes
            padded_labels[i, :num_objs] = labels
            padded_mask[i, :num_objs] = True
            
    return {
        "images": images,
        "gt_boxes": padded_boxes,
        "gt_labels": padded_labels,
        "gt_mask": padded_mask,
        "image_names": img_names,
        "org_sizes": org_sizes
    }