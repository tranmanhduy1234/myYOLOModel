# NMS-Free Object Detection and Face Landmark Learning

This repository implements a custom PyTorch computer-vision pipeline for object detection and face landmark estimation. The base model follows a YOLOv10-style, end-to-end design with one-to-many and one-to-one detection branches. The one-to-one branch is intended to produce final detections without Non-Maximum Suppression (NMS).

The project supports three related workflows:

- pretraining an 80-class object detector on Object365-style JSONL data;
- fine-tuning the detector on another object-detection dataset, such as MS COCO;
- transferring the pretrained backbone and neck to face detection and 478-point MediaPipe-style facial landmark estimation.

> This is a research project. Dataset locations, checkpoint locations, and most experiment settings are configured directly in Python dataclasses and must be updated for the target machine before training.

## Main Features

- Custom backbone built from `C2f`, `CIB`, `C2fCIB`, `SCDown`, `SPPF`, and `C2fPSA` blocks.
- PAFPN neck with P3, P4, and P5 feature maps at strides 8, 16, and 32.
- Dual one-to-many (O2M) and one-to-one (O2O) detection heads.
- Task-Aligned Assignment, CIoU box loss, classification loss, and Distribution Focal Loss (DFL).
- Mixed-precision training, gradient clipping, Exponential Moving Average (EMA), warmup, and cosine learning-rate decay.
- Step-based validation, TensorBoard logging, checkpoint rotation, and training resume support.
- Byte-offset JSONL indices for large datasets without loading all annotations into memory.
- Two-stage face-landmark transfer learning: head-only training followed by full-model fine-tuning.
- Optional specialist heads for the left eye, right eye, and mouth regions.
- CPU and CUDA validation suites for model, loss, data-loader, and training-pipeline behavior.

## Architecture

```text
Input image (3 x 480 x 480)
          |
          v
Custom backbone
  C2f -> C2f -> C2fCIB -> SPPF -> C2fPSA
          |
          v
PAFPN neck
  P3 (stride 8), P4 (stride 16), P5 (stride 32)
          |
          v
Dual detection head at each scale
  +----------------------+----------------------+
  | O2M branch           | O2O branch           |
  | dense supervision    | end-to-end prediction|
  | training             | training + inference |
  +----------------------+----------------------+
          |
          v
Class scores + DFL-decoded bounding boxes
```

For face-landmark transfer learning, the object-detection head is replaced by a face head that predicts one face class, a bounding box, and landmark offsets for every detected face. The default configuration uses 478 landmarks.

## Repository Layout

```text
CNNModel/
├── src/
│   ├── blocks.py                     # Reusable CNN building blocks
│   ├── backbone_neck.py              # Backbone and PAFPN
│   ├── head.py                       # O2M/O2O detection head and DFL decoding
│   ├── model.py                      # NMSFreeDetector
│   ├── config.py                     # Object-detection training configuration
│   ├── train/
│   │   ├── training.py               # Object-detection training entry point
│   │   ├── engine.py                 # Training and validation loop
│   │   ├── loss.py                   # Assigner and detection losses
│   │   └── dataloader1_obj365.py     # Object365-style JSONL loader
│   ├── finetune/                     # Object-detection fine-tuning pipeline
│   ├── transferLearning/
│   │   ├── config_lmk.py             # Face-landmark configuration
│   │   ├── model_lmk.py              # Face and landmark model
│   │   ├── dataloader_lmk.py         # Face-landmark JSONL loader
│   │   ├── train_lmk.py              # Two-stage landmark trainer
│   │   ├── finetune.py               # Landmark checkpoint fine-tuning
│   │   ├── inference.py              # Image and camera inference
│   │   └── multiHead/                 # Eye and mouth specialist heads
│   ├── evaluation/                   # Detection metric accumulation
│   ├── validation_tool/              # Automated validation suites
│   └── utils/                         # Checkpoints, EMA, logging, and seeds
├── TrainingPipelineAnalysis/         # Detailed technical documentation
├── DataPretrain1/                    # Local datasets and preprocessing assets
└── checkpoints*/                     # Local experiment outputs
```

## Requirements

Python 3.10 or newer is recommended. Training is designed for CUDA, but the core model and validation tools can fall back to CPU.

The project does not currently include a pinned dependency file. Create an environment and install a PyTorch build appropriate for the local CPU or CUDA version, followed by the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install torch and torchvision using the command recommended at:
# https://pytorch.org/get-started/locally/

python -m pip install numpy opencv-python pillow albumentations \
  tensorboard tqdm matplotlib
```

All commands below assume they are run from the repository root so that imports beginning with `src.` resolve correctly.

## Object-Detection Dataset Format

The base loader expects separate label and image roots with pre-split `train` and `val` data:

```text
labels_root/
├── train/
│   ├── images_info.jsonl
│   ├── annotations.jsonl
│   ├── categories.jsonl
│   └── images_train.jsonl
└── val/
    ├── images_info.jsonl
    ├── annotations.jsonl
    ├── categories.jsonl
    └── images_val.jsonl

images_root_dir/
├── train/
│   ├── patch0/
│   └── ...
└── valid/
    ├── patch0/
    └── ...
```

Each JSONL file contains one JSON object per line. The principal records have these forms:

```json
{"id": 42, "width": 1280, "height": 720, "file_name": "image_0042.jpg"}
```

```json
{"id": 1001, "image_id": 42, "category_id": 0, "bbox": [120.0, 80.0, 240.0, 180.0], "iscrowd": 0, "isfake": 0}
```

```json
{"id": 0, "name": "Person"}
```

```json
{"image_name": "image_0042.jpg", "path": "patch0/image_0042.jpg"}
```

Bounding boxes use absolute `[x, y, width, height]` coordinates in the source image. Category IDs are remapped to contiguous model labels according to the sorted category records.

## Train the Base Detector

First, edit `src/config.py` and set at least:

- `labels_root` and `images_root_dir`;
- the train/validation subdirectory and JSONL filenames if they differ;
- `index_cache_dir`;
- model, batch-size, device, and optimization settings;
- `resume` when continuing from an existing checkpoint.

Then start training:

```bash
python -m src.train.training
```

The default configuration uses 480 × 480 inputs, 80 classes, AdamW, AMP, EMA, linear warmup, and cosine learning-rate decay. CUDA is selected when available; otherwise the training engine falls back to CPU.

To monitor a run:

```bash
tensorboard --logdir runs
```

Checkpoints are written to the configured `ckpt_dir`. Depending on the run settings, the directory may contain:

- `last.pt` for the most recent complete state;
- `best.pt` for the best validation loss;
- periodic `ckpt_step*.pt` files;
- `best_trunk.pt`, containing transferable backbone, neck, and detection-head weights.

## Fine-Tune the Object Detector

Edit `src/finetune/finetune_config.py` to point to the pretrained checkpoint, architecture manifest, and target detection dataset. Start fine-tuning with:

```bash
python -m src.finetune.run_finetune
```

The fine-tuning engine first freezes the feature extractor, then unfreezes it and trains with separate learning rates for the backbone and detection head.

## Face-Landmark Dataset Format

Each train or validation root must contain an image directory and one annotation JSONL file:

```text
face_dataset/
├── images/
│   ├── face_0001.jpg
│   └── ...
└── merged_faces.jsonl
```

A record uses normalized coordinates in `[0, 1]`:

```json
{
  "file_name": "face_0001.jpg",
  "image_width": 1280,
  "image_height": 720,
  "faces": [
    {
      "bounding_box_normalized": {
        "xmin": 0.20,
        "ymin": 0.10,
        "xmax": 0.65,
        "ymax": 0.90
      },
      "landmarks_normalized": [
        {"x": 0.31, "y": 0.35},
        {"x": 0.32, "y": 0.36}
      ]
    }
  ]
}
```

Every face in the dataset must contain the same number of landmarks. With the default MediaPipe configuration and paired horizontal flipping, each face must contain exactly 478 points; the abbreviated example above only illustrates the schema.

The loader validates the complete dataset on startup, applies letterboxing, and can apply synchronized color, affine, perspective, radial-distortion, and landmark-aware horizontal-flip augmentations.

## Train the Face-Landmark Model

Configure a run in Python with the pretrained trunk and dataset roots:

```python
from src.transferLearning.config_lmk import TrainConfig
from src.transferLearning.train_lmk import Trainer

config = TrainConfig(
    trunk_ckpt="checkpoints/best_trunk.pt",
    train_root_dir="/path/to/face_train",
    val_root_dir="/path/to/face_val",
    batch_size=4,
    device="cuda",
)

Trainer(config).fit()
```

The two stages are configured in `src/transferLearning/config_lmk.py`:

1. `stage1_head_only` freezes the backbone and neck and trains only the new face-landmark head.
2. `stage2_finetune` unfreezes the full model and uses a lower learning rate for the trunk.

To run the module-level entry point instead, fill in the empty dataset and checkpoint fields in `TrainConfig`, then run:

```bash
python -m src.transferLearning.train_lmk
```

For continued landmark fine-tuning, edit the constants at the bottom of `src/transferLearning/finetune.py` and run:

```bash
python -m src.transferLearning.finetune
```

## Face-Landmark Inference

Image and camera inference can be called directly from Python:

```python
from src.transferLearning.inference import main

main(
    mode="image",
    image_input="/path/to/input.jpg",
    weights_path="checkpoints_face_lmk/best.pt",
    device="cuda",
    conf_threshold=0.25,
)
```

Set `mode="camera"` and provide `camera_index` for live-camera inference. The O2O branch is NMS-free by default when `inference_iou_threshold` is `0.0`; a positive IoU threshold enables optional NMS in the face inference wrapper.

## Multi-Head Regional Refinement

`src/transferLearning/multiHead/` adds specialist models and losses for the left eye, right eye, and mouth while retaining the global face-landmark model. To train these heads, edit the experiment constants at the bottom of `train_multihead.py`, then run:

```bash
python -m src.transferLearning.multiHead.train_multihead
```

The associated `inference_multihead.py` combines the global detector with the regional specialists.

## Validation

Run all built-in validation suites:

```bash
python -m src.validation_tool.run_all_validation
```

Useful variants include:

```bash
python -m src.validation_tool.run_all_validation --device cpu
python -m src.validation_tool.run_all_validation --device cuda
python -m src.validation_tool.run_all_validation --skip loss,dataloader
python -m src.validation_tool.run_all_validation --verbose-traceback
```

The validation runner checks mathematical loss behavior, model tensor shapes, optimizer and checkpoint logic, EMA, letterboxing, augmentation, and collation without requiring the full training dataset.

## Configuration and Reproducibility Notes

- Configuration is code-based; there is no CLI override layer for the main training scripts.
- Several checked-in defaults use absolute paths from the original development environment. Replace them before running an experiment.
- The model input size is 480 by default and is fixed to 480 in the face-landmark pipeline.
- Seeds are set for Python, NumPy, and PyTorch, but complete GPU determinism can still depend on the CUDA and cuDNN environment.
- Checkpoint compatibility is validated for face-landmark runs through a saved model signature and training plan.
- Large datasets, model weights, cache files, logs, and TensorBoard runs are intended to remain local and are excluded by `.gitignore`.

## Technical Documentation

The `TrainingPipelineAnalysis/` directory contains an in-depth discussion of the training lifecycle, data pipeline, initialization, optimizer, learning-rate schedule, dual-head losses, logging, checkpoints, technology stack, and design trade-offs. Additional implementation notes and diagrams are available under `src/image_sketch/`, `src/train/`, and `src/utils/`.
