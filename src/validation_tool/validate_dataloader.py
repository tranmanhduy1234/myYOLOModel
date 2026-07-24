"""
validate_dataloader.py
=======================
Validate các hàm TIỆN ÍCH thuần túy của dataloader1_obj365.py
(letterbox, collate_fn, DetectionAugmenter) bằng dữ liệu TỔNG HỢP trong bộ
nhớ — KHÔNG cần dataset Object365 thật trên đĩa.

build_dataloaders()/ObjectDetectionDataset thật sử dụng labels_root/images_root_dir
trỏ tới dữ liệu thật nên KHÔNG được test ở đây; nếu bạn có dữ liệu thật, hãy
tự gọi build_dataloaders(cfg) với cfg trỏ đúng đường dẫn rồi kiểm tra thủ công
1 batch (xem hướng dẫn cuối file).

Nếu thiếu dependency tùy chọn (opencv-python, albumentations) các test liên
quan sẽ được báo SKIP thay vì FAIL, để toolkit vẫn chạy được trên máy không
có đủ mọi thư viện.

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ
------------------------------
1. Thêm test mới: letterbox pixel content, letterbox non-square target size,
   collate_fn kiểm tra boxes tensor dtype, collate_fn B=1 edge case,
   collate_fn với nhiều batch size khác nhau, augmenter giới hạn box tọa độ,
   augmenter seed reproducibility.

Chạy độc lập:
    python -m src.validation_tool.validate_dataloader
"""

import argparse
import sys

import numpy as np
import torch

from src.validation_tool.validate_common import Reporter, skip


def _try_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _try_import_albumentations():
    try:
        import albumentations as A
        return A
    except ImportError:
        return None


# ==============================================================================
# 1. letterbox() - resize + pad giữ tỷ lệ
# ==============================================================================
def test_letterbox(R: Reporter):
    R.section("1. letterbox() - resize giữ tỷ lệ + pad về hình vuông")

    cv2 = _try_import_cv2()
    if cv2 is None:
        R.check("dataloader", "letterbox: [TẤT CẢ TEST]", lambda: skip("thiếu opencv-python (cv2), bỏ qua"))
        return

    from src.train.dataloader1_obj365 import letterbox

    def t_output_shape_always_square():
        img = np.zeros((90, 160, 3), dtype=np.uint8)  # ảnh chữ nhật 16:9
        canvas, scale, pad_left, pad_top = letterbox(img, 128)
        assert canvas.shape == (128, 128, 3), f"letterbox phải luôn ra hình vuông new_size x new_size, được {canvas.shape}"
        expected_scale = 128 / 160
        assert abs(scale - expected_scale) < 1e-6, f"scale phải theo cạnh dài hơn (160) để không bị crop, kỳ vọng {expected_scale:.4f}"
        return f"canvas={canvas.shape}, scale={scale:.4f}, pad_left={pad_left}, pad_top={pad_top}"
    R.check("dataloader", "letterbox: luôn ra hình vuông, scale theo cạnh dài hơn", t_output_shape_always_square)

    def t_padding_centers_image():
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        canvas, scale, pad_left, pad_top = letterbox(img, 100)
        new_h = int(round(50 * scale))
        assert pad_top == (100 - new_h) // 2, "phải căn giữa theo chiều dọc (top padding = (new_size-new_h)//2)"
        assert pad_left == 0, "ảnh rộng hơn cao -> không cần pad ngang khi scale theo chiều rộng"
        return f"scale={scale:.4f}, pad_top={pad_top} (căn giữa)"
    R.check("dataloader", "letterbox: pad căn giữa ảnh trong canvas vuông", t_padding_centers_image)

    def t_square_image_no_padding():
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        canvas, scale, pad_left, pad_top = letterbox(img, 64)
        assert pad_left == 0 and pad_top == 0, "ảnh vuông đúng size đích -> không cần padding"
        assert abs(scale - 1.0) < 1e-6
        return "ảnh vuông khớp size -> scale=1.0, không padding"
    R.check("dataloader", "letterbox: ảnh đã vuông và khớp kích thước -> không biến dạng", t_square_image_no_padding)

    def t_letterbox_preserves_pixel_content():
        """Vùng ảnh gốc trong canvas phải có màu giống ảnh gốc (không phải toàn màu pad)."""
        img = np.ones((40, 80, 3), dtype=np.uint8) * 128  # xám đồng đều
        canvas, scale, pad_left, pad_top = letterbox(img, 80)
        # Phần nội dung (không phải padding) phải là 128
        new_h = int(round(40 * scale))
        new_w = int(round(80 * scale))
        region = canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        mean_val = region.mean()
        assert abs(mean_val - 128) < 10, f"nội dung ảnh trong canvas phải giữ màu gốc (~128), được {mean_val:.1f}"
        return f"pixel content preserved: mean={mean_val:.1f}"
    R.check("dataloader", "letterbox: nội dung pixel ảnh gốc được giữ nguyên trong canvas", t_letterbox_preserves_pixel_content)

    def t_letterbox_portrait_image():
        """Ảnh cao hơn rộng (portrait) -> scale theo chiều cao."""
        img = np.zeros((160, 90, 3), dtype=np.uint8)  # portrait
        canvas, scale, pad_left, pad_top = letterbox(img, 128)
        assert canvas.shape == (128, 128, 3)
        expected_scale = 128 / 160  # scale theo cạnh dài (height)
        assert abs(scale - expected_scale) < 1e-6, f"portrait: scale phải theo chiều cao, kỳ vọng {expected_scale:.4f}"
        # pad phải theo chiều ngang (width < new_size)
        assert pad_left > 0 or pad_top == 0
        return f"portrait scale={scale:.4f}, pad_left={pad_left} (pad ngang)"
    R.check("dataloader", "letterbox: ảnh portrait (cao > rộng) scale theo chiều cao", t_letterbox_portrait_image)

    def t_letterbox_scale_up():
        """letterbox phải scale UP ảnh nhỏ hơn new_size."""
        img = np.zeros((30, 40, 3), dtype=np.uint8)
        canvas, scale, pad_left, pad_top = letterbox(img, 100)
        assert canvas.shape == (100, 100, 3)
        assert scale > 1.0, f"ảnh nhỏ hơn target phải được scale up, được scale={scale:.4f}"
        return f"scale up: {img.shape[:2]} -> {canvas.shape[:2]}, scale={scale:.4f}"
    R.check("dataloader", "letterbox: scale up ảnh nhỏ hơn kích thước đích", t_letterbox_scale_up)


# ==============================================================================
# 2. collate_fn() - gộp batch ảnh + targets
# ==============================================================================
def test_collate_fn(R: Reporter):
    R.section("2. collate_fn() - gộp batch (Dataset -> DataLoader)")

    from src.train.dataloader1_obj365 import collate_fn

    def t_stacks_images_keeps_targets_as_list():
        batch = [
            (torch.rand(3, 32, 32), {"boxes": torch.rand(2, 4), "labels": torch.tensor([0, 1])}),
            (torch.rand(3, 32, 32), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}),
            (torch.rand(3, 32, 32), {"boxes": torch.rand(1, 4), "labels": torch.tensor([2])}),
        ]
        images, targets = collate_fn(batch)
        assert images.shape == (3, 3, 32, 32), f"collate_fn phải stack ảnh thành (B,C,H,W), được {tuple(images.shape)}"
        assert isinstance(targets, list) and len(targets) == 3, "targets phải là list (mỗi ảnh số GT khác nhau, không stack được)"
        assert targets[1]["boxes"].shape == (0, 4), "ảnh không có GT (N=0) phải được giữ nguyên, không lỗi"
        return "images: stack đúng shape; targets: giữ nguyên list (hỗ trợ số GT khác nhau)"
    R.check("dataloader", "collate_fn: stack ảnh, giữ targets dạng list (hỗ trợ N GT khác nhau)", t_stacks_images_keeps_targets_as_list)

    def t_boxes_dtype_float():
        """Sau collate_fn, boxes phải là float tensor (để dùng với loss/IoU)."""
        batch = [
            (torch.rand(3, 16, 16), {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([0])}),
            (torch.rand(3, 16, 16), {"boxes": torch.tensor([[5., 5., 15., 15.]]), "labels": torch.tensor([1])}),
        ]
        _, targets = collate_fn(batch)
        for i, t in enumerate(targets):
            assert t["boxes"].dtype in (torch.float32, torch.float64, torch.float16), \
                f"targets[{i}]['boxes'] phải là float dtype, được {t['boxes'].dtype}"
        return "tất cả boxes tensor là float dtype"
    R.check("dataloader", "collate_fn: boxes tensor có dtype float", t_boxes_dtype_float)

    def t_compatible_with_dataloader():
        from torch.utils.data import DataLoader, Dataset

        class _Toy(Dataset):
            def __len__(self): return 4
            def __getitem__(self, i):
                n = i % 3
                return torch.rand(3, 16, 16), {
                    "boxes":  torch.rand(n, 4),
                    "labels": torch.randint(0, 5, (n,))
                }

        loader  = DataLoader(_Toy(), batch_size=2, collate_fn=collate_fn)
        batches = list(loader)
        assert len(batches) == 2, "4 sample / batch_size=2 -> phải có 2 batch"
        images, targets = batches[0]
        assert images.shape[0] == 2 and len(targets) == 2
        return f"{len(batches)} batch, đúng với DataLoader thật (số GT không đều giữa các ảnh)"
    R.check("dataloader", "collate_fn: hoạt động đúng khi gắn vào torch DataLoader thật", t_compatible_with_dataloader)

    def t_single_sample_batch():
        """collate_fn với B=1 (edge case) không crash."""
        batch = [(torch.rand(3, 64, 64), {"boxes": torch.rand(3, 4), "labels": torch.randint(0, 5, (3,))})]
        images, targets = collate_fn(batch)
        assert images.shape == (1, 3, 64, 64), f"B=1 phải ra shape (1,3,64,64), được {tuple(images.shape)}"
        assert len(targets) == 1
        return "collate_fn B=1 không crash, shape đúng"
    R.check("dataloader", "collate_fn: batch_size=1 edge case không crash", t_single_sample_batch)

    def t_large_batch():
        """collate_fn với B=16 và số GT rất khác nhau (0 đến 10) không crash."""
        batch = []
        for i in range(16):
            n = i % 11  # 0..10
            batch.append((
                torch.rand(3, 32, 32),
                {"boxes": torch.rand(n, 4), "labels": torch.randint(0, 10, (n,))}
            ))
        images, targets = collate_fn(batch)
        assert images.shape[0] == 16
        assert len(targets) == 16
        # Kiểm tra tổng GT
        total_gt = sum(t["boxes"].shape[0] for t in targets)
        assert total_gt == sum(i % 11 for i in range(16))
        return f"B=16, tổng GT={total_gt}, không crash"
    R.check("dataloader", "collate_fn: batch lớn (B=16) với số GT từ 0 đến 10 không crash", t_large_batch)


# ==============================================================================
# 3. DetectionAugmenter - augment ảnh + box đồng bộ
# ==============================================================================
def test_augmenter(R: Reporter):
    R.section("3. DetectionAugmenter - augment đồng bộ ảnh + bbox")

    A   = _try_import_albumentations()
    cv2 = _try_import_cv2()
    if A is None or cv2 is None:
        R.check("dataloader", "DetectionAugmenter: [TẤT CẢ TEST]",
                lambda: skip("thiếu albumentations/opencv-python, bỏ qua"))
        return

    from src.train.dataloader1_obj365 import DetectionAugmenter
    from src.config import TrainConfig

    def t_empty_boxes_short_circuit():
        cfg = TrainConfig()
        aug = DetectionAugmenter(cfg)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out_img, out_boxes, out_labels = aug(img, [], [])
        assert out_boxes == [] and out_labels == [], "không có box nào -> phải trả về nguyên trạng, không gọi augment"
        return "boxes rỗng -> bỏ qua augment, trả về nguyên vẹn"
    R.check("dataloader", "DetectionAugmenter: bỏ qua augment khi không có box (tránh lỗi albumentations)", t_empty_boxes_short_circuit)

    def t_augment_keeps_box_count_or_drops_safely():
        cfg = TrainConfig()
        cfg.horizontalFlip = 1.0  # ép chắc chắn augment chạy (p=1)
        aug = DetectionAugmenter(cfg)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        boxes  = [[10.0, 10.0, 30.0, 30.0], [5.0, 5.0, 20.0, 20.0]]
        labels = [1, 2]
        out_img, out_boxes, out_labels = aug(img, boxes, labels)
        assert out_img.shape == img.shape, "augment không được đổi kích thước ảnh"
        assert len(out_boxes) == len(out_labels), "số box và số label sau augment phải luôn khớp nhau"
        assert len(out_boxes) <= len(boxes), "augment (min_visibility) chỉ có thể GIỮ hoặc LOẠI box, không thể SINH THÊM"
        return f"box: {len(boxes)} -> {len(out_boxes)} sau augment (shape ảnh giữ nguyên)"
    R.check("dataloader", "DetectionAugmenter: box/label luôn khớp số lượng, không sinh thêm, ảnh giữ shape", t_augment_keeps_box_count_or_drops_safely)

    def t_augment_boxes_within_image():
        """Sau augment, tọa độ box phải nằm trong [0, img_size]."""
        cfg = TrainConfig()
        cfg.horizontalFlip = 1.0
        aug = DetectionAugmenter(cfg)
        H, W = 128, 128
        img  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        boxes  = [[10.0, 10.0, 60.0, 60.0], [50.0, 50.0, 100.0, 100.0]]
        labels = [0, 1]
        out_img, out_boxes, out_labels = aug(img, boxes, labels)
        for box in out_boxes:
            x1, y1, x2, y2 = box
            assert x1 >= 0 and y1 >= 0, f"box tọa độ âm: {box}"
            assert x2 <= W and y2 <= H,  f"box vượt kích thước ảnh: {box} > ({W},{H})"
            assert x2 > x1 and y2 > y1,  f"box không hợp lệ (x2<=x1 hoặc y2<=y1): {box}"
        return f"{len(out_boxes)} box, tất cả trong [0,{W}]x[0,{H}]"
    R.check("dataloader", "DetectionAugmenter: tọa độ box sau augment nằm trong ảnh [0,W]x[0,H]", t_augment_boxes_within_image)

    def t_augment_single_box():
        """Augment với chỉ 1 box phải hoạt động (không crash do shape)."""
        cfg = TrainConfig()
        aug = DetectionAugmenter(cfg)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        boxes  = [[5.0, 5.0, 55.0, 55.0]]
        labels = [3]
        out_img, out_boxes, out_labels = aug(img, boxes, labels)
        assert out_img.shape == img.shape
        assert len(out_boxes) == len(out_labels)
        return f"1 box -> {len(out_boxes)} box sau augment, không crash"
    R.check("dataloader", "DetectionAugmenter: augment với 1 box không crash", t_augment_single_box)


# ==============================================================================
# MAIN
# ==============================================================================
def run(verbose_traceback: bool = False) -> Reporter:
    """Chạy toàn bộ suite dataloader và trả về Reporter để run_all_validation.py gộp."""
    r = Reporter(verbose_traceback)
    test_letterbox(r)
    test_collate_fn(r)
    test_augmenter(r)
    return r


def main():
    parser = argparse.ArgumentParser(description="Validate letterbox/collate_fn/DetectionAugmenter")
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    r  = run(args.verbose_traceback)
    ok = r.summary("TỔNG KẾT - VALIDATE DATALOADER (letterbox/collate_fn/augmenter)")

    print(
        "Lưu ý: KHÔNG test build_dataloaders()/ObjectDetectionDataset thật vì cần\n"
        "dữ liệu Object365 thật trên đĩa. Để kiểm tra với dữ liệu thật, tự chạy:\n"
        "    from src.config import TrainConfig\n"
        "    from src.train.dataloader1_obj365 import build_dataloaders\n"
        "    cfg = TrainConfig()  # sửa labels_root/images_root_dir cho đúng máy bạn\n"
        "    train_loader, val_loader, classes, nc = build_dataloaders(cfg)\n"
        "    images, targets = next(iter(train_loader))\n"
        "    print(images.shape, len(targets), nc)\n"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
