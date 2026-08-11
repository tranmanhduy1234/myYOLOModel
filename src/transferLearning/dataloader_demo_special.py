#!/usr/bin/env python3
"""Kiểm tra horizontal flip có phải phép đối hợp: flip(flip(x)) == x."""

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from src.transferLearning.config_lmk import TrainConfig

try:
    from src.transferLearning.dataloader_lmk import (
        FaceLandmarkDataModule,
        horizontal_flip_targets,
    )
except ImportError:
    from src.transferLearning.dataloader_lmk import (
        FaceLandmarkDataModule,
        _flip_targets as horizontal_flip_targets,
    )


def draw_sample(ax, image, boxes, landmarks, valid, title: str) -> None:
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    for box, points, is_valid in zip(boxes, landmarks, valid):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5)
        )

        if bool(is_valid):
            points = points.cpu().numpy()
            ax.scatter(points[:, 0], points[:, 1], s=2)

    ax.set_title(title)
    ax.axis("off")


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max())


def main() -> None:
    defaults = TrainConfig(require_pretrained_trunk=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--jsonl", default=defaults.jsonl_name)
    parser.add_argument("--cache", default=defaults.index_cache_dir)
    parser.add_argument("--sample-index", type=int, default=defaults.demo_sample_index)
    parser.add_argument("--min-box-size", type=float, default=defaults.demo_min_box_size_px)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_root_dir=args.root,
        jsonl_name=args.jsonl,
        index_cache_dir=args.cache,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        min_box_size_px=args.min_box_size,
        require_pretrained_trunk=False,
    )

    # Tắt toàn bộ augmentation và horizontal flip tự động.
    data_cfg = cfg.dataset_config(args.root, train=False)
    data_module = FaceLandmarkDataModule(data_cfg)

    try:
        sample = data_module.dataset[args.sample_index]
        image_0 = sample["image"]
        boxes_0 = sample["boxes"]
        landmarks_0 = sample["landmarks"]
        labels_0 = sample["labels"]
        valid_0 = sample["landmarks_valid"]

        permutation = data_module.dataset.flip_permutation
        image_size = data_cfg.image_size

        image_1 = torch.flip(image_0, dims=(-1,))
        boxes_1, landmarks_1 = horizontal_flip_targets(
            boxes_0, landmarks_0, image_size, permutation
        )

        image_2 = torch.flip(image_1, dims=(-1,))
        boxes_2, landmarks_2 = horizontal_flip_targets(
            boxes_1, landmarks_1, image_size, permutation
        )

        permutation_is_involution = torch.equal(
            permutation[permutation],
            torch.arange(len(permutation), dtype=permutation.dtype),
        )

        image_equal = torch.equal(image_0, image_2)
        labels_equal = torch.equal(labels_0, labels_0.clone())
        valid_equal = torch.equal(valid_0, valid_0.clone())
        box_error = max_abs_error(boxes_0, boxes_2)
        lmk_error = max_abs_error(landmarks_0, landmarks_2)

        print("=" * 72)
        print(f"file                  : {sample['file_name']}")
        print(f"permutation involution: {permutation_is_involution}")
        print(f"image exact equal     : {image_equal}")
        print(f"labels exact equal    : {labels_equal}")
        print(f"valid exact equal     : {valid_equal}")
        print(f"bbox max abs error    : {box_error:.8f}")
        print(f"landmark max abs error: {lmk_error:.8f}")
        print("=" * 72)

        ok = (
            permutation_is_involution
            and image_equal
            and torch.allclose(
                boxes_0,
                boxes_2,
                atol=cfg.demo_flip_atol,
                rtol=cfg.demo_flip_rtol,
            )
            and torch.allclose(
                landmarks_0,
                landmarks_2,
                atol=cfg.demo_flip_atol,
                rtol=cfg.demo_flip_rtol,
            )
        )
        print("KẾT QUẢ:", "PASS" if ok else "FAIL")

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(3 * cfg.demo_plot_cell_size, cfg.demo_plot_cell_size),
        )
        draw_sample(axes[0], image_0, boxes_0, landmarks_0, valid_0, "Ảnh gốc")
        draw_sample(axes[1], image_1, boxes_1, landmarks_1, valid_0, "Lật lần 1")
        draw_sample(
            axes[2],
            image_2,
            boxes_2,
            landmarks_2,
            valid_0,
            f"Lật lần 2 | {'PASS' if ok else 'FAIL'}",
        )
        plt.tight_layout()
        plt.show()

    finally:
        data_module.dataset.close()


if __name__ == "__main__":
    main()

"""
python -m src.transferLearning.dataloader_demo_special \
  --root /run/media/tranmanhduy/Data/DataTransferSplit/train \
  --jsonl annotations.jsonl \
  --sample-index 0
"""
