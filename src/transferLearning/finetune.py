"""Fine-tune face landmark model từ ``best.pt`` trên dataset mới.

Dataset mới dùng cùng cấu trúc với pipeline hiện tại::

    root_dir/
    ├── images/
    └── annotations.jsonl

File này khởi tạo một lượt training mới từ trọng số tốt nhất (ưu tiên EMA),
không resume optimizer/global_step/best_val của lượt training cũ.
"""

import copy
import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.transferLearning.config_lmk import TrainConfig, TrainingStageConfig
from src.transferLearning.dataloader_lmk import face_landmark_collate
from src.transferLearning.train_lmk import Trainer


logger = logging.getLogger('train_face_lmk')


def _subset_loader(
    dataset,
    indices: list[int],
    dataset_cfg,
    *,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    """Tạo DataLoader từ một phần dataset nhưng giữ nguyên collate hiện tại."""
    if not indices:
        raise ValueError('Subset không được rỗng.')

    workers_enabled = dataset_cfg.num_workers > 0
    kwargs = {
        'dataset': Subset(dataset, indices),
        'batch_size': dataset_cfg.batch_size,
        'shuffle': shuffle,
        'num_workers': dataset_cfg.num_workers,
        'collate_fn': face_landmark_collate,
        'pin_memory': dataset_cfg.pin_memory and torch.cuda.is_available(),
        'persistent_workers': dataset_cfg.persistent_workers and workers_enabled,
        'drop_last': drop_last,
    }
    if shuffle:
        kwargs['generator'] = torch.Generator().manual_seed(seed)
    if workers_enabled and dataset_cfg.prefetch_factor is not None:
        kwargs['prefetch_factor'] = dataset_cfg.prefetch_factor
    return DataLoader(**kwargs)


def _install_internal_train_val_split(
    trainer: Trainer,
    val_ratio: float,
) -> dict:
    """Chia theo record gốc để bản paired/augmented không lọt sang validation."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('VAL_RATIO phải nằm trong khoảng (0, 1).')
    if trainer.val_dm is None:
        raise RuntimeError('Cần tạo validation dataset trước khi chia nội bộ.')

    train_dataset = trainer.train_dm.dataset
    val_dataset = trainer.val_dm.dataset
    num_records = len(train_dataset.offsets)
    if num_records < 2:
        raise ValueError('Dataset cần ít nhất 2 record để chia train/validation.')
    if len(val_dataset.offsets) != num_records:
        raise RuntimeError('Train dataset và validation dataset không cùng số record.')

    generator = torch.Generator().manual_seed(trainer.cfg.seed)
    permutation = torch.randperm(num_records, generator=generator).tolist()
    num_val_records = max(1, min(num_records - 1, round(num_records * val_ratio)))
    val_record_indices = sorted(permutation[:num_val_records])
    train_record_indices = sorted(permutation[num_val_records:])

    # Dataset train ở mode paired có hai sample (gốc + lật) cho mỗi record.
    if trainer.train_dm.cfg.horizontal_flip_mode == 'paired':
        train_sample_indices = [
            sample_index
            for record_index in train_record_indices
            for sample_index in (2 * record_index, 2 * record_index + 1)
        ]
    else:
        train_sample_indices = train_record_indices

    # Validation luôn được Trainer tạo với augment=False và flip='off', nên
    # sample index của nó trùng với record index trong JSONL.
    val_sample_indices = val_record_indices
    trainer.train_loader = _subset_loader(
        train_dataset,
        train_sample_indices,
        trainer.train_dm.cfg,
        shuffle=True,
        drop_last=trainer.train_dm.cfg.drop_last,
        seed=trainer.cfg.seed,
    )
    trainer.val_loader = _subset_loader(
        val_dataset,
        val_sample_indices,
        trainer.val_dm.cfg,
        shuffle=False,
        drop_last=False,
        seed=trainer.cfg.seed,
    )

    stats = {
        'total_records': num_records,
        'train_records': len(train_record_indices),
        'val_records': len(val_record_indices),
        'train_samples': len(train_sample_indices),
        'val_samples': len(val_sample_indices),
    }
    logger.info(
        '[FINETUNE SPLIT] seed=%s | val_ratio=%.3f | total=%s record | '
        'train=%s record/%s sample | val=%s record/%s sample | overlap=0',
        trainer.cfg.seed,
        val_ratio,
        stats['total_records'],
        stats['train_records'],
        stats['train_samples'],
        stats['val_records'],
        stats['val_samples'],
    )
    return stats


def _normalize_state_dict(state_dict: dict) -> dict:
    """Bỏ prefix sinh bởi wrapper nhưng giữ nguyên toàn bộ model state."""
    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError('State dict chứa key không phải chuỗi.')
        clean = key
        while clean.startswith(('module.', 'model.', 'ema.', '_orig_mod.')):
            clean = clean.split('.', 1)[1]
        if clean in normalized:
            raise KeyError(f'Trùng key sau khi chuẩn hóa checkpoint: {clean!r}.')
        normalized[clean] = value
    return normalized


def load_best_weights(trainer: Trainer, best_path: str) -> dict:
    """Nạp full model từ best checkpoint, ưu tiên EMA và bỏ optimizer cũ."""
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Không tìm thấy best checkpoint: '{best_path}'.")

    logger.info("[FINETUNE LOAD] START | file='%s'", best_path)
    checkpoint = torch.load(
        best_path,
        map_location='cpu',
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError('best.pt phải là checkpoint dict.')

    saved_signature = checkpoint.get('model_signature')
    current_signature = trainer.cfg.checkpoint_model_signature()
    if saved_signature is None:
        raise ValueError(
            'best.pt không có model_signature; không thể xác minh kiến trúc '
            'landmark trước khi fine-tune.'
        )
    if saved_signature != current_signature:
        raise ValueError(
            'Model signature không khớp: '
            f'checkpoint={saved_signature}, current={current_signature}.'
        )

    if checkpoint.get('ema') is not None:
        source_name = "checkpoint['ema']"
        state_dict = checkpoint['ema']
    elif checkpoint.get('model') is not None:
        source_name = "checkpoint['model']"
        state_dict = checkpoint['model']
    elif checkpoint.get('state_dict') is not None:
        source_name = "checkpoint['state_dict']"
        state_dict = checkpoint['state_dict']
    else:
        raise KeyError('best.pt không chứa ema, model hoặc state_dict.')

    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f'{source_name} không phải state_dict hợp lệ.')
    state_dict = _normalize_state_dict(state_dict)

    model_state = trainer.model.state_dict()
    source_keys = set(state_dict)
    model_keys = set(model_state)
    missing = sorted(model_keys - source_keys)
    unexpected = sorted(source_keys - model_keys)
    bad_shapes = sorted(
        (
            key,
            tuple(state_dict[key].shape) if isinstance(state_dict[key], torch.Tensor) else None,
            tuple(model_state[key].shape),
        )
        for key in source_keys & model_keys
        if not isinstance(state_dict[key], torch.Tensor)
        or state_dict[key].shape != model_state[key].shape
    )
    if missing or unexpected or bad_shapes:
        raise RuntimeError(
            'best.pt không tương thích hoàn toàn: '
            f'missing={len(missing)}, unexpected={len(unexpected)}, '
            f'bad_shape={len(bad_shapes)}; ví dụ missing={missing[:5]}, '
            f'unexpected={unexpected[:5]}, bad_shape={bad_shapes[:5]}.'
        )

    tensor_count = len(state_dict)
    element_count = sum(value.numel() for value in state_dict.values())
    trainer.model.load_state_dict(state_dict, strict=True)

    # EMA của run mới bắt đầu từ cùng bộ trọng số tốt nhất và warmup lại từ 0.
    if trainer.ema is not None:
        trainer.ema.ema.load_state_dict(state_dict, strict=True)
        trainer.ema.ema.eval()
        trainer.ema.updates = 0

    # Đây là fine-tune mới: optimizer/scaler do Trainer vừa tạo được giữ nguyên.
    trainer.start_epoch = 0
    trainer.global_step = 0
    trainer.best_val = float('inf')
    trainer._active_stage_name = None
    trainer._configure_stage(0, force=True)

    metadata = {
        'source': source_name,
        'source_epoch': checkpoint.get('epoch'),
        'source_global_step': checkpoint.get('global_step'),
        'source_best_val': checkpoint.get('best_val'),
        'tensor_count': tensor_count,
        'element_count': element_count,
    }
    logger.info(
        '[FINETUNE LOAD] PASS | source=%s | source_epoch=%s | '
        'source_global_step=%s | source_best_val=%s | tensors=%s | '
        'elements=%s | optimizer mới | global_step=0',
        source_name,
        metadata['source_epoch'],
        metadata['source_global_step'],
        metadata['source_best_val'],
        f'{tensor_count:,}',
        f'{element_count:,}',
    )
    return metadata


def finetune_landmarks(
    best_path: str,
    train_root_dir: str,
    val_root_dir: Optional[str] = None,
    *,
    val_ratio: float = 0.15,
    base_cfg: Optional[TrainConfig] = None,
    ckpt_dir: str = './checkpoints_face_lmk_finetune',
    log_dir: str = './logs_face_lmk_finetune',
    run_name: str = 'face_lmk_finetune',
):
    """Tạo Trainer, nạp ``best.pt`` rồi fine-tune trên dataset mới."""
    if not train_root_dir:
        raise ValueError('Phải điền train_root_dir của dataset fine-tune.')
    if not best_path:
        raise ValueError('Phải điền đường dẫn best.pt.')

    cfg = copy.deepcopy(base_cfg or TrainConfig())
    cfg.train_root_dir = train_root_dir
    use_internal_split = val_root_dir is None
    # Trainer tạo riêng hai FaceLandmarkDataset: bản train có augmentation,
    # bản validation không augmentation. Với dataset chưa chia, cả hai cùng
    # đọc một root rồi DataLoader sẽ được giới hạn bằng hai bộ index độc lập.
    cfg.val_root_dir = train_root_dir if use_internal_split else val_root_dir
    cfg.resume = '/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/checkpoints_face_lmk_finetune/best.pt'
    cfg.ckpt_dir = ckpt_dir
    cfg.log_dir = log_dir
    cfg.run_name = run_name

    dataset_roots = [('train', cfg.train_root_dir)]
    if cfg.val_root_dir:
        dataset_roots.append(('val', cfg.val_root_dir))
    for split_name, root_dir in dataset_roots:
        images_dir = os.path.join(root_dir, cfg.images_dir_name)
        jsonl_path = os.path.join(root_dir, cfg.jsonl_name)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Dataset {split_name}: không tìm thấy thư mục ảnh '{images_dir}'."
            )
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(
                f"Dataset {split_name}: không tìm thấy annotation '{jsonl_path}'. "
                "Hãy kiểm tra ANNOTATION_FILE trong hàm main."
            )

    # Trainer cần trunk trước khi tạo optimizer. Dùng chính best.pt làm nguồn
    # trunk, sau đó load_best_weights sẽ nạp full model EMA gồm cả HEAD.
    cfg.trunk_ckpt = best_path
    cfg.require_pretrained_trunk = True

    trainer = Trainer(cfg)
    if use_internal_split:
        _install_internal_train_val_split(trainer, val_ratio)
    load_best_weights(trainer, best_path)
    return trainer.fit()


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Cấu hình tối thiểu cho lượt fine-tune; chỉnh trực tiếp, không dùng CLI.
    # ------------------------------------------------------------------
    BEST_PT = '/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/checkpoint_transfer/best.pt'
    FINETUNE_TRAIN_ROOT = '/run/media/tranmanhduy/Data/Datafinetune2'
    FINETUNE_VAL_ROOT = None
    IMAGES_DIR_NAME = 'images'
    ANNOTATION_FILE = 'annotations.jsonl'
    VAL_RATIO = 0.15

    NUM_EPOCHS = 50
    HEAD_ONLY_EPOCHS = 5
    BATCH_SIZE = 4
    DEVICE = 'cuda'

    # Fine-tune từ best.pt nên dùng LR thấp hơn lượt train ban đầu.
    HEAD_ONLY_LR = 1e-4
    FINETUNE_HEAD_LR = 5e-5
    FINETUNE_TRUNK_LR = 5e-6

    if NUM_EPOCHS < 2:
        raise ValueError('NUM_EPOCHS phải >= 2 để có đủ hai stage fine-tune.')
    if not 1 <= HEAD_ONLY_EPOCHS < NUM_EPOCHS:
        raise ValueError('HEAD_ONLY_EPOCHS phải nằm trong [1, NUM_EPOCHS).')

    full_model_epochs = NUM_EPOCHS - HEAD_ONLY_EPOCHS
    config = TrainConfig(
        batch_size=BATCH_SIZE,
        device=DEVICE,
        images_dir_name=IMAGES_DIR_NAME,
        jsonl_name=ANNOTATION_FILE,
        stage1=TrainingStageConfig(
            epochs=HEAD_ONLY_EPOCHS,
            head_lr=HEAD_ONLY_LR,
            trunk_lr=0.0,
            warmup_epochs=min(0.5, HEAD_ONLY_EPOCHS / 2),
            min_lr_factor=0.20,
        ),
        stage2=TrainingStageConfig(
            epochs=full_model_epochs,
            head_lr=FINETUNE_HEAD_LR,
            trunk_lr=FINETUNE_TRUNK_LR,
            warmup_epochs=min(1.0, full_model_epochs / 2),
            min_lr_factor=0.05,
        ),
        
    )

    finetune_landmarks(
        best_path=BEST_PT,
        train_root_dir=FINETUNE_TRAIN_ROOT,
        val_root_dir=FINETUNE_VAL_ROOT,
        val_ratio=VAL_RATIO,
        base_cfg=config,
    )