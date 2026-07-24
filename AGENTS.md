# AGENTS.md

OpenCode guidance for NMS-Free Object Detector — PyTorch implementation with dual-branch head (o2o/o2m).

## Commands

```bash
# Training
python -m src.train.training

# Run validation suite
python -m src.validation_tool.run_all_validation
python -m src.validation_tool.run_all_validation --device cuda --skip dataloader

# Model benchmark
python -m src.model

# TensorBoard
tensorboard --logdir runs
```

## Architecture Quick Reference

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| Config | `src/config.py` | `TrainConfig` (dataclass with validation) |
| Model | `src/model.py` | `NMSFreeDetector` |
| Backbone+Neck | `src/backbone_neck.py` | `Backbone`, `PAFPN` |
| Head | `src/head.py` | `DetectHead` (dual branch: o2m + o2o) |
| Loss | `src/train/loss.py` | `DetectionLoss` |
| Engine | `src/train/engine.py` | `run_training()`, `train_one_epoch()` |
| Data | `src/train/dataloader1_obj365.py` | `ObjectDetectionDataset` |
| EMA | `src/train/ema.py` | `ModelEMA` |

## Critical Design Patterns

### Dual-Branch Head (o2o/o2m)
- **o2o (one-to-one)**: `topk=1` → one prediction per GT → enables NMS-free inference
- **o2m (one-to-many)**: `topk=10` → multiple positives per GT → stabilizes training
- In `eval()` mode, o2m stems are skipped entirely (FLOPs savings)
- Loss weights: `total = w_o2m * loss_o2m + w_o2o * loss_o2o`

### Coordinate Spaces (documented in `loss.py`)
- **`[PIXEL]`**: Input image space (0–640). Used for GT boxes and TAL assignment.
- **`[GRID]`**: Pixel divided by stride. Used for CIoU and DFL loss computation.

### Data Loading
- JSONL format with byte-offset indexing (`.pkl` cache in `index_cache_dir`)
- Train/val split via subdirectories, not random split
- Labels and images have separate root paths (`labels_root`, `images_root_dir`)

## Entry Points & Execution Flow

```
src/train/training.py:main()
    ↓
TrainConfig()  ← src/config.py
    ↓
run_training(cfg)  ← src/train/engine.py
    ↓
    ├─ get_dataloader() → build_dataloaders()  ← dataloader1_obj365.py
    ├─ get_model() → NMSFreeDetector()  ← model.py
    ├─ get_criterion() → DetectionLoss()  ← loss.py
    ├─ get_optimizer() → AdamW/SGD
    │   ↓
    train_one_epoch() ──→ validate()
        ↓
        ModelEMA.update()  ← ema.py
        Checkpoint save  ← state_dict_handle.py
```

## Common Tasks

### Modify config
Edit `src/config.py` → `TrainConfig` dataclass. All hyperparameters with defaults.

### Run single validation module
```python
# In Python shell or script
from src.validation_tool.validate_model import run
reporter = run()  # Returns Reporter with stats
```

### Save/load checkpoints
```python
from src.utils.state_dict_handle import save_checkpoint, load_checkpoint
# Automatic handling of EMA, optimizer, scheduler states
```

## Repository Structure

```
DataPretrain1/Object365/     # Object365 dataset (labels/ + images/)
DataPretrain2/               # Data processing scripts
src/
    blocks.py              # Conv, C2f, C2fCIB, SPPF, Attention, etc.
    backbone_neck.py       # Backbone, PAFPN
    head.py                # DetectHead (dual-branch)
    model.py               # NMSFreeDetector
    config.py              # TrainConfig
    train/
        training.py        # Entry point
        engine.py          # Main training loop
        loss.py            # DetectionLoss
        ema.py             # ModelEMA
        dataloader1_obj365.py
    validation_tool/       # Comprehensive test suites
    utils/
    TransferLearning/      # Standalone variants (not imported by main)
checkpoints/               # Saved checkpoints
runs/                      # TensorBoard logs
```

## Notes

- No `requirements.txt` — dependencies must be inferred from imports (PyTorch, albumentations, etc.)
- `TransferLearning/` contains standalone variants; not imported by main pipeline
- Validation suite can run without real data (`dataloader` suite uses synthetic inputs)
