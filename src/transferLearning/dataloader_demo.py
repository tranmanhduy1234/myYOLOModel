#!/usr/bin/env python3
"""Demo nhanh DataLoader và geometric augmentation bằng matplotlib."""

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.transferLearning.config_lmk import TrainConfig
from src.transferLearning.dataloader_lmk import FaceLandmarkDataModule

def draw_batch(batch: dict, cfg: TrainConfig, max_images: int) -> None:
    images = batch["image"]
    targets = batch["targets"]
    count = min(max_images, len(images))
    cols = min(cfg.demo_plot_columns, count)
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cfg.demo_plot_cell_size * cols, cfg.demo_plot_cell_size * rows),
        squeeze=False,
    )

    for i in range(count):
        ax = axes[i // cols][i % cols]
        image = images[i].permute(1, 2, 0).cpu().numpy()
        target = targets[i]

        ax.imshow(image)

        for box, landmarks, valid in zip(
            target["boxes"], target["landmarks"], target["landmarks_valid"]
        ):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False))

            if bool(valid):
                points = landmarks.cpu().numpy()
                ax.scatter(points[:, 0], points[:, 1], s=2)

        ax.set_title(
            f"{batch['file_name'][i]}\n"
            f"flip={bool(batch['was_flipped'][i])} | "
            f"aug={batch['geometric_aug'][i]}"
        )
        ax.axis("off")

    for i in range(count, rows * cols):
        axes[i // cols][i % cols].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    defaults = TrainConfig(require_pretrained_trunk=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root chứa images/ và JSONL")
    parser.add_argument("--jsonl", default=defaults.jsonl_name, help="Tên file JSONL trong dataset root")
    parser.add_argument("--cache", default=defaults.index_cache_dir)
    parser.add_argument("--batches", type=int, default=defaults.demo_batches)
    parser.add_argument("--max-images", type=int, default=defaults.demo_max_images)
    parser.add_argument("--batch-size", type=int, default=defaults.demo_batch_size)
    parser.add_argument("--min-box-size", type=float, default=defaults.demo_min_box_size_px)
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Ép mỗi sample chạy affine + perspective + radial",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        train_root_dir=args.root,
        jsonl_name=args.jsonl,
        index_cache_dir=args.cache,
        batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        min_box_size_px=args.min_box_size,
        require_pretrained_trunk=False,
    )

    if args.force_all:
        cfg.train_geometric_probability = 1.0
        cfg.train_affine_probability = 1.0
        cfg.train_perspective_probability = 1.0
        cfg.train_radial_distortion_probability = 1.0

    data_cfg = cfg.dataset_config(args.root, train=True)
    data_module = FaceLandmarkDataModule(data_cfg)
    loader = data_module.loader()

    try:
        for batch_idx, batch in enumerate(loader):
            print(
                f"batch={batch_idx} | image={tuple(batch['image'].shape)} | "
                f"augmentation={batch['geometric_aug']}"
            )
            draw_batch(batch, cfg, max_images=args.max_images)

            if batch_idx + 1 >= args.batches:
                break
    finally:
        data_module.dataset.close()

if __name__ == "__main__":
    main()

"""
python -m src.transferLearning.dataloader_demo \
--root /run/media/tranmanhduy/Data/DataTransferSplit/val \
--jsonl annotations.jsonl
"""
