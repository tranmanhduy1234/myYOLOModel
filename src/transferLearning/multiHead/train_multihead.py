"""Train tuần tự ba specialist mắt trái, mắt phải và miệng.

HEAD 4 (model 478 điểm đã fine-tune) chỉ được dùng làm mốc kiến trúc và luôn
``eval``/frozen. Mỗi stage tạo optimizer, scheduler và AMP scaler mới, chỉ
chứa parameter của một mini-detector đang active.

Checkpoint của file này cố ý **không** chứa state 465 MB của HEAD 4. Nó chỉ
lưu ba specialist nhẹ cùng chữ ký/hash và đường dẫn tham chiếu tới checkpoint
global để inference có thể kiểm tra nghiêm ngặt.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.transferLearning.config_lmk import TrainConfig
from src.transferLearning.model_lmk import FaceLmkDetector
from src.transferLearning.multiHead.data_multihead import (
    MultiHeadDatasetConfig,
    build_multihead_loaders,
)
from src.transferLearning.multiHead.loss_multihead import (
    MultiHeadRegionLoss,
    RegionGeometryLossConfig,
)
from src.transferLearning.multiHead.model_multihead import (
    LEFT_EYE,
    MOUTH,
    RIGHT_EYE,
    SPECIALIST_NAMES,
    SpecializedMultiHeadFaceLandmark,
)


logger = logging.getLogger('train_multihead_face_lmk')
_CHECKPOINT_KIND = 'multihead_specialists'
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class SpecialistStageConfig:
    """Cấu hình một stage; mỗi stage luôn có optimizer/scheduler mới."""

    epochs: int = 25
    learning_rate: float = 3e-4
    warmup_epochs: float = 1.0
    min_lr_factor: float = 0.05
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError('epochs của specialist phải > 0.')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate phải > 0.')
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError('warmup_epochs phải nằm trong [0, epochs).')
        if not 0 < self.min_lr_factor <= 1:
            raise ValueError('min_lr_factor phải nằm trong (0, 1].')
        if self.early_stopping_patience <= 0:
            raise ValueError('early_stopping_patience phải > 0.')
        if self.early_stopping_min_delta < 0:
            raise ValueError('early_stopping_min_delta không được âm.')


@dataclass
class MultiHeadTrainerConfig:
    """Cấu hình trainer multi-head, độc lập với TrainConfig hai stage cũ."""

    global_checkpoint_path: str = ''
    dataset: Optional[MultiHeadDatasetConfig] = None
    global_model_cfg: TrainConfig = field(
        default_factory=lambda: TrainConfig(require_pretrained_trunk=False)
    )

    left_eye_stage: SpecialistStageConfig = field(
        default_factory=SpecialistStageConfig
    )
    right_eye_stage: SpecialistStageConfig = field(
        default_factory=SpecialistStageConfig
    )
    mouth_stage: SpecialistStageConfig = field(
        default_factory=SpecialistStageConfig
    )
    geometry_loss: RegionGeometryLossConfig = field(
        default_factory=RegionGeometryLossConfig
    )

    optimizer: str = 'adamw'
    weight_decay: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.937
    eps: float = 1e-8
    amp: bool = True
    grad_clip_norm: float = 10.0
    device: str = 'cuda'
    seed: int = 42

    checkpoint_dir: str = './checkpoints_multihead'
    log_dir: str = './logs_multihead'
    run_name: str = 'specialized_face_landmarks'
    log_interval: int = 25
    resume_path: str = ''
    # Chọn checkpoint theo sai số landmark thực, không theo tổng loss pha trộn.
    selection_metric: str = 'landmark_nme'

    def __post_init__(self) -> None:
        self.optimizer = self.optimizer.lower()
        if not self.global_checkpoint_path:
            raise ValueError('Phải cấu hình global_checkpoint_path tới best.pt.')
        if self.dataset is None:
            raise ValueError('Phải cấu hình MultiHeadDatasetConfig trong dataset.')
        if self.optimizer not in {'adamw', 'sgd'}:
            raise ValueError("optimizer phải là 'adamw' hoặc 'sgd'.")
        if self.weight_decay < 0 or self.eps <= 0:
            raise ValueError('weight_decay phải >= 0 và eps phải > 0.')
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError('betas phải có hai giá trị trong [0, 1).')
        if not 0 <= self.momentum < 1:
            raise ValueError('momentum phải nằm trong [0, 1).')
        if self.grad_clip_norm <= 0 or self.log_interval <= 0:
            raise ValueError('grad_clip_norm và log_interval phải > 0.')
        if self.selection_metric not in {'loss', 'landmark_nme'}:
            raise ValueError("selection_metric phải là 'loss' hoặc 'landmark_nme'.")
        if not (
            self.device == 'cpu'
            or self.device == 'cuda'
            or self.device.startswith('cuda:')
        ):
            raise ValueError("device phải là 'cpu', 'cuda' hoặc 'cuda:<index>'.")

    def stage(self, name: str) -> SpecialistStageConfig:
        stages = {
            LEFT_EYE: self.left_eye_stage,
            RIGHT_EYE: self.right_eye_stage,
            MOUTH: self.mouth_stage,
        }
        if name not in stages:
            raise KeyError(f'Specialist không hợp lệ: {name!r}.')
        return stages[name]


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch mà không ép deterministic kernels chậm."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover - dataset ảnh luôn cần NumPy.
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError('State dict chứa key không phải chuỗi.')
        clean_key = key
        while clean_key.startswith(('module.', 'model.', 'ema.', '_orig_mod.')):
            clean_key = clean_key.split('.', 1)[1]
        if clean_key in normalized:
            raise KeyError(f'Trùng key sau chuẩn hóa checkpoint: {clean_key!r}.')
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'State dict key {key!r} không chứa tensor.')
        normalized[clean_key] = value
    if not normalized:
        raise ValueError('State dict checkpoint rỗng.')
    return normalized


def _state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash canonical toàn bộ tensor để kiểm chứng HEAD 4 bitwise."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{key!r} không phải tensor khi tính state hash.')
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(cpu_tensor.dtype).encode('ascii'))
        digest.update(str(tuple(cpu_tensor.shape)).encode('ascii'))
        if cpu_tensor.numel():
            raw = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file_handle:
        while True:
            chunk = file_handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _to_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def load_frozen_global_detector(
    cfg: TrainConfig,
    checkpoint_path: str,
) -> tuple[FaceLmkDetector, dict]:
    """Strict-load full HEAD 4, ưu tiên EMA và xác minh lại từng tensor."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy global best checkpoint: '{checkpoint_path}'."
        )
    logger.info("[GLOBAL LOAD] START | file='%s'", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError('Global checkpoint phải là dict.')

    expected_signature = cfg.checkpoint_model_signature()
    saved_signature = checkpoint.get('model_signature')
    if saved_signature is None:
        raise ValueError(
            'Global checkpoint không có model_signature; không thể strict-load.'
        )
    if saved_signature != expected_signature:
        raise ValueError(
            'Global model signature không khớp: '
            f'checkpoint={saved_signature}, current={expected_signature}.'
        )

    if checkpoint.get('ema') is not None:
        source_name = "checkpoint['ema']"
        raw_state = checkpoint['ema']
    elif checkpoint.get('model') is not None:
        source_name = "checkpoint['model']"
        raw_state = checkpoint['model']
    elif checkpoint.get('state_dict') is not None:
        source_name = "checkpoint['state_dict']"
        raw_state = checkpoint['state_dict']
    else:
        raise KeyError('Global checkpoint không chứa ema/model/state_dict.')
    if not isinstance(raw_state, Mapping):
        raise TypeError(f'{source_name} không phải state_dict.')
    source_state = _normalize_state_dict(raw_state)

    detector = FaceLmkDetector(cfg)
    target_state = detector.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    bad_shapes = sorted(
        (key, tuple(source_state[key].shape), tuple(target_state[key].shape))
        for key in set(source_state) & set(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if missing or unexpected or bad_shapes:
        raise RuntimeError(
            'Global checkpoint không tương thích hoàn toàn: '
            f'missing={len(missing)}, unexpected={len(unexpected)}, '
            f'bad_shape={len(bad_shapes)}; missing[:5]={missing[:5]}, '
            f'unexpected[:5]={unexpected[:5]}, bad_shape[:5]={bad_shapes[:5]}.'
        )

    source_hash = _state_sha256(source_state)
    detector.load_state_dict(source_state, strict=True)
    loaded_hash = _state_sha256(detector.state_dict())
    if loaded_hash != source_hash:
        raise RuntimeError('Global HEAD 4 không khớp bitwise sau strict load.')
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    detector.eval()

    component_stats = {}
    for component_name in ('backbone', 'neck', 'head'):
        prefix = f'{component_name}.'
        component_tensors = {
            key: tensor
            for key, tensor in source_state.items()
            if key.startswith(prefix)
        }
        if not component_tensors:
            raise RuntimeError(
                f'Global checkpoint strict-load nhung thieu component '
                f'{component_name!r}.'
            )
        component_stats[component_name] = {
            'tensor_count': len(component_tensors),
            'element_count': sum(
                tensor.numel() for tensor in component_tensors.values()
            ),
            'state_sha256': _state_sha256(component_tensors),
        }
        logger.info(
            '[GLOBAL LOAD] component=%s | tensors=%s | elements=%s | '
            'sha256=%s | strict=PASS',
            component_name,
            f"{component_stats[component_name]['tensor_count']:,}",
            f"{component_stats[component_name]['element_count']:,}",
            component_stats[component_name]['state_sha256'][:16],
        )

    metadata = {
        'path': os.path.abspath(checkpoint_path),
        'file_sha256': _file_sha256(checkpoint_path),
        'state_sha256': loaded_hash,
        'model_signature': saved_signature,
        'source': source_name,
        'source_epoch': checkpoint.get('epoch'),
        'source_global_step': checkpoint.get('global_step'),
        'source_best_val': checkpoint.get('best_val'),
        'tensor_count': len(source_state),
        'element_count': sum(tensor.numel() for tensor in source_state.values()),
        'components': component_stats,
    }
    logger.info(
        '[GLOBAL LOAD] PASS | source=%s | tensors=%s | elements=%s | '
        'state_sha256=%s | HEAD4 frozen/eval',
        source_name,
        f"{metadata['tensor_count']:,}",
        f"{metadata['element_count']:,}",
        loaded_hash[:16],
    )
    return detector, metadata


class SequentialMultiHeadTrainer:
    """Train đúng thứ tự left_eye -> right_eye -> mouth."""

    def __init__(self, cfg: MultiHeadTrainerConfig) -> None:
        self.cfg = cfg
        self._setup_logging()
        set_seed(cfg.seed)
        requested_device = torch.device(cfg.device)
        if requested_device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning('CUDA không sẵn có; chuyển multi-head trainer sang CPU.')
            requested_device = torch.device('cpu')
        self.device = requested_device
        if cfg.dataset.num_workers > 0 and cfg.dataset.persistent_workers:
            logger.warning(
                'persistent_workers=True: resume van dung split/checkpoint '
                'chinh xac, nhung crop-jitter stream khong dam bao bitwise. '
                'Dung mac dinh False neu can tai lap trajectory theo epoch.'
            )

        loader_bundle = build_multihead_loaders(cfg.dataset)
        self.train_loader, self.val_loader = self._resolve_loaders(loader_bundle)
        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise ValueError('Train loader và validation loader phải có ít nhất 1 batch.')
        self.dataset_metadata = self._build_dataset_metadata(loader_bundle)

        global_detector, self.global_checkpoint_metadata = (
            load_frozen_global_detector(
                cfg.global_model_cfg,
                cfg.global_checkpoint_path,
            )
        )
        self.model = SpecializedMultiHeadFaceLandmark(global_detector).to(self.device)
        self.loss_fn = MultiHeadRegionLoss(
            self.model.specialists,
            self.model.specs,
            cfg.geometry_loss,
        ).to(self.device)
        # Hash lại sau khi chuyển device để làm mốc bất biến suốt ba stage.
        self._global_state_hash = _state_sha256(
            self.model.global_detector.state_dict()
        )
        if self._global_state_hash != self.global_checkpoint_metadata['state_sha256']:
            raise RuntimeError('HEAD 4 thay đổi bitwise khi chuyển sang training device.')

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=os.path.join(cfg.log_dir, cfg.run_name)
        )
        self.global_step = 0
        self.completed_stages: list[str] = []
        self.stage_metrics: dict[str, list[dict]] = {
            name: [] for name in SPECIALIST_NAMES
        }
        self._resume_training_state: Optional[dict] = None

        summary = self.model.parameter_summary()
        logger.info(
            '[DATASET] root=%s | records=%s (train=%s val=%s) | '
            'batches=%s/%s | annotation_sha256=%s | split_seed=%s',
            self.dataset_metadata['root_dir'],
            self.dataset_metadata['record_count'],
            self.dataset_metadata['train_record_count'],
            self.dataset_metadata['val_record_count'],
            self.dataset_metadata['train_batches'],
            self.dataset_metadata['val_batches'],
            self.dataset_metadata['annotation_sha256'][:16],
            self.dataset_metadata['split_seed'],
        )
        logger.info(
            '[MODEL] HEAD4 frozen=%s params | left_eye=%s | right_eye=%s | '
            'mouth=%s | specialists_total=%s',
            f"{summary['head4_global_frozen']:,}",
            f"{summary[LEFT_EYE]:,}",
            f"{summary[RIGHT_EYE]:,}",
            f"{summary[MOUTH]:,}",
            f"{summary['specialists_total']:,}",
        )
        self._assert_global_frozen(check_bitwise=True)
        if cfg.resume_path:
            self._load_resume(cfg.resume_path)

    @staticmethod
    def _resolve_loaders(loader_bundle):
        """Adapter nhỏ để trainer không phụ thuộc chi tiết dataclass bundle."""
        if hasattr(loader_bundle, 'train_loader') and hasattr(
            loader_bundle, 'val_loader'
        ):
            return loader_bundle.train_loader, loader_bundle.val_loader
        if isinstance(loader_bundle, Mapping):
            train_loader = loader_bundle.get('train_loader', loader_bundle.get('train'))
            val_loader = loader_bundle.get('val_loader', loader_bundle.get('val'))
            if train_loader is not None and val_loader is not None:
                return train_loader, val_loader
        if isinstance(loader_bundle, (tuple, list)) and len(loader_bundle) >= 2:
            return loader_bundle[0], loader_bundle[1]
        raise TypeError(
            'build_multihead_loaders phải trả object .train_loader/.val_loader, '
            'mapping hoặc tuple(train_loader, val_loader).'
        )

    def _setup_logging(self) -> None:
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        if logger.handlers:
            return
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            os.path.join(self.cfg.log_dir, f'{self.cfg.run_name}.log'),
            encoding='utf-8',
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _checkpoint_specs(self) -> dict:
        return {
            name: {
                'name': spec.name,
                'input_size': spec.input_size,
                'num_landmarks': spec.num_landmarks,
                'global_landmark_indices': tuple(spec.global_landmark_indices),
                'crop_anchor_indices': tuple(spec.crop_anchor_indices),
            }
            for name, spec in self.model.specs.items()
        }

    @staticmethod
    def _index_sequence_sha256(indices) -> str:
        digest = hashlib.sha256()
        for index in indices:
            digest.update(int(index).to_bytes(8, 'little', signed=False))
        return digest.hexdigest()

    def _build_dataset_metadata(self, loader_bundle) -> dict:
        """Khoa split/dataset de resume khong vo tinh doi validation set."""
        split = getattr(loader_bundle, 'split', None)
        if split is None:
            raise TypeError('Loader bundle phai cung cap RecordSplit de resume an toan.')
        annotation_path = os.path.realpath(
            os.path.join(
                self.cfg.dataset.root_dir,
                self.cfg.dataset.jsonl_name,
            )
        )
        train_indices = tuple(split.train_indices)
        val_indices = tuple(split.val_indices)
        return {
            'root_dir': os.path.realpath(self.cfg.dataset.root_dir),
            'images_dir_name': self.cfg.dataset.images_dir_name,
            'jsonl_name': self.cfg.dataset.jsonl_name,
            'annotation_sha256': _file_sha256(annotation_path),
            'record_count': len(train_indices) + len(val_indices),
            'train_record_count': len(train_indices),
            'val_record_count': len(val_indices),
            'train_indices_sha256': self._index_sequence_sha256(train_indices),
            'val_indices_sha256': self._index_sequence_sha256(val_indices),
            'val_ratio': self.cfg.dataset.val_ratio,
            'split_seed': self.cfg.dataset.seed,
            'batch_size': self.cfg.dataset.batch_size,
            'num_workers': self.cfg.dataset.num_workers,
            'persistent_workers': self.cfg.dataset.persistent_workers,
            'train_drop_last': self.cfg.dataset.train_drop_last,
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
        }

    def _training_plan(self) -> dict:
        return {
            'stage_order': tuple(SPECIALIST_NAMES),
            'stages': {
                name: asdict(self.cfg.stage(name)) for name in SPECIALIST_NAMES
            },
            'optimizer': self.cfg.optimizer,
            'weight_decay': self.cfg.weight_decay,
            'betas': tuple(self.cfg.betas),
            'momentum': self.cfg.momentum,
            'eps': self.cfg.eps,
            'amp': self.cfg.amp,
            'grad_clip_norm': self.cfg.grad_clip_norm,
            'selection_metric': self.cfg.selection_metric,
            'geometry_loss': asdict(self.cfg.geometry_loss),
            'trainer_seed': self.cfg.seed,
            'dataset': dict(self.dataset_metadata),
        }

    def _assert_global_frozen(self, *, check_bitwise: bool) -> None:
        detector = self.model.global_detector
        trainable = [name for name, p in detector.named_parameters() if p.requires_grad]
        gradients = [name for name, p in detector.named_parameters() if p.grad is not None]
        if trainable or gradients or detector.training:
            raise RuntimeError(
                'Vi phạm freeze HEAD4: '
                f'trainable={trainable[:5]}, gradients={gradients[:5]}, '
                f'training={detector.training}.'
            )
        if check_bitwise:
            current_hash = _state_sha256(detector.state_dict())
            if current_hash != self._global_state_hash:
                raise RuntimeError(
                    'HEAD4/backbone/neck đã thay đổi bitwise trong lúc train.'
                )

    def _assert_only_active_is_trainable(self, active_name: str) -> None:
        for name, specialist in self.model.specialists.items():
            trainable = any(p.requires_grad for p in specialist.parameters())
            if trainable != (name == active_name):
                raise RuntimeError(
                    f'requires_grad sai ở {name}: trainable={trainable}, '
                    f'active={active_name}.'
                )
            if name != active_name and specialist.training:
                raise RuntimeError(f'Specialist frozen {name} không ở eval mode.')
        self._assert_global_frozen(check_bitwise=False)

    def _build_optimizer(self, name: str):
        parameters = list(self.model.trainable_parameters())
        if not parameters:
            raise RuntimeError(f'{name} không có parameter trainable.')
        stage_cfg = self.cfg.stage(name)
        if self.cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters,
                lr=stage_cfg.learning_rate,
                betas=self.cfg.betas,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                parameters,
                lr=stage_cfg.learning_rate,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay,
                nesterov=True,
            )
        return optimizer, parameters

    def _build_scheduler(self, optimizer, stage_cfg: SpecialistStageConfig):
        steps_per_epoch = len(self.train_loader)
        total_steps = max(stage_cfg.epochs * steps_per_epoch, 1)
        warmup_steps = int(stage_cfg.warmup_epochs * steps_per_epoch)

        def lr_factor(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
            decay_step = max(step - warmup_steps, 0)
            decay_steps = max(total_steps - warmup_steps - 1, 1)
            progress = min(decay_step / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                stage_cfg.min_lr_factor
                + (1.0 - stage_cfg.min_lr_factor) * cosine
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    @staticmethod
    def _average_items(sums: dict[str, float], count: int) -> dict[str, float]:
        return {key: value / max(count, 1) for key, value in sums.items()}

    @staticmethod
    def _accumulate_items(sums: dict[str, float], items: Mapping[str, Any]) -> None:
        for key, value in items.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)

    def _extract_region_batch(self, batch, name: str):
        if not isinstance(batch, Mapping) or 'regions' not in batch:
            raise KeyError("Batch phải chứa batch['regions'].")
        regions = batch['regions']
        if name not in regions:
            raise KeyError(f"Batch thiếu batch['regions']['{name}'].")
        region = regions[name]
        if 'images' not in region or 'targets' not in region:
            raise KeyError(f'{name} phải chứa images và targets.')
        images, targets = region['images'], region['targets']
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise TypeError(f'{name}.images phải là Tensor [B,3,H,W].')
        if not isinstance(targets, (list, tuple)) or len(targets) != images.shape[0]:
            raise ValueError(
                f'{name}.targets phải là list dài B={images.shape[0]}.'
            )
        return images.to(self.device, non_blocking=True), targets

    @torch.no_grad()
    def _landmark_nme(self, name: str, predictions, targets) -> tuple[float, int]:
        """NME của candidate o2o confidence cao nhất, chuẩn hóa theo ROI GT.

        Đây là metric sát trực tiếp độ chính xác landmark hơn tổng detection
        loss vốn trộn classification, bbox, DFL và geometry với nhiều gain.
        """
        branch = predictions['o2o']
        specialist = self.model.specialists[name]
        decoded = specialist.head.decode_landmarks(
            branch['lmk_raw'],
            predictions['anchors'],
            predictions['strides'],
        )
        scores = torch.sigmoid(branch['cls']).squeeze(-1)
        best_anchor = scores.argmax(dim=1)
        batch_indices = torch.arange(decoded.shape[0], device=decoded.device)
        selected = decoded[batch_indices, best_anchor].float()

        nme_sum = 0.0
        valid_count = 0
        for batch_index, target in enumerate(targets):
            valid = target.get('landmarks_valid')
            if valid is not None and not bool(valid.reshape(-1)[0].item()):
                continue
            ground_truth = target['landmarks'][0].to(
                device=selected.device,
                dtype=selected.dtype,
            )
            if ground_truth.shape != selected[batch_index].shape:
                raise RuntimeError(
                    f'{name}: pred/GT landmark shape không khớp: '
                    f'{tuple(selected[batch_index].shape)} và '
                    f'{tuple(ground_truth.shape)}.'
                )
            roi_diagonal = torch.linalg.vector_norm(
                ground_truth.amax(dim=0) - ground_truth.amin(dim=0)
            ).clamp_min(1.0)
            nme = torch.linalg.vector_norm(
                selected[batch_index] - ground_truth,
                dim=-1,
            ).mean() / roi_diagonal
            if torch.isfinite(nme):
                nme_sum += float(nme.item())
                valid_count += 1
        return nme_sum, valid_count

    def _seed_train_epoch(self, name: str, local_epoch: int) -> int:
        """Cho shuffle va crop jitter cung trajectory khi resume cung epoch."""
        stage_index = SPECIALIST_NAMES.index(name)
        epoch_seed = (
            int(self.cfg.seed)
            + (stage_index + 1) * 1_000_003
            + int(local_epoch) * 10_007
        )
        dataset = getattr(self.train_loader, 'dataset', None)
        if hasattr(dataset, 'set_augmentation_seed'):
            dataset.set_augmentation_seed(epoch_seed)
        loader_generator = getattr(self.train_loader, 'generator', None)
        if isinstance(loader_generator, torch.Generator):
            loader_generator.manual_seed(epoch_seed)
        sampler = getattr(self.train_loader, 'sampler', None)
        sampler_generator = getattr(sampler, 'generator', None)
        if (
            isinstance(sampler_generator, torch.Generator)
            and sampler_generator is not loader_generator
        ):
            sampler_generator.manual_seed(epoch_seed)
        return epoch_seed

    def _train_epoch(
        self,
        name: str,
        local_epoch: int,
        optimizer,
        scheduler,
        scaler,
        trainable_parameters,
    ) -> dict[str, float]:
        stage_cfg = self.cfg.stage(name)
        self.model.train(True)
        self._assert_only_active_is_trainable(name)
        use_amp = self.cfg.amp and self.device.type == 'cuda'
        sums: dict[str, float] = {}
        successful_batches = 0
        start_time = time.time()
        epoch_seed = self._seed_train_epoch(name, local_epoch)
        progress = tqdm(
            self.train_loader,
            desc=f'Train {name} {local_epoch + 1}/{stage_cfg.epochs}',
            unit='batch',
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            images, targets = self._extract_region_batch(batch, name)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                predictions = self.model.forward_specialist(name, images)
                total_loss, items = self.loss_fn(name, predictions, targets)
            if not torch.isfinite(total_loss):
                logger.warning(
                    '[%s epoch=%s batch=%s] loss không hữu hạn (%s), bỏ batch.',
                    name,
                    local_epoch,
                    batch_index,
                    total_loss.detach().item(),
                )
                continue

            optimizer_step_succeeded = True
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    self.cfg.grad_clip_norm,
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_succeeded = scaler.get_scale() >= previous_scale
            else:
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    self.cfg.grad_clip_norm,
                )
                optimizer.step()

            if optimizer_step_succeeded:
                scheduler.step()
                self.global_step += 1
            successful_batches += 1
            self._accumulate_items(sums, items)
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix(
                loss=f"{items['loss']:.4f}",
                lmk=f"{items['o2o/lmk']:.4f}",
                geometry=f"{items['geometry']:.4f}",
                state=f"{items['geometry_state']:.4f}",
                lr=f'{current_lr:.2e}',
            )

            if self.global_step and self.global_step % self.cfg.log_interval == 0:
                logger.info(
                    '[TRAIN] region=%s epoch=%s batch=%s/%s step=%s | '
                    'loss=%.5f detection=%.5f | '
                    'geometry(total=%.5f aperture=%.5f state=%.5f) | '
                    'o2m(iou=%.5f cls=%.5f dfl=%.5f lmk=%.5f) | '
                    'o2o(iou=%.5f cls=%.5f dfl=%.5f lmk=%.5f) | '
                    'lr=%.3e grad_norm=%.4f',
                    name,
                    local_epoch,
                    batch_index + 1,
                    len(self.train_loader),
                    self.global_step,
                    items['loss'],
                    items['loss_detection'],
                    items['geometry'],
                    items['geometry_aperture'],
                    items['geometry_state'],
                    items['o2m/iou'],
                    items['o2m/cls'],
                    items['o2m/dfl'],
                    items['o2m/lmk'],
                    items['o2o/iou'],
                    items['o2o/cls'],
                    items['o2o/dfl'],
                    items['o2o/lmk'],
                    current_lr,
                    float(grad_norm),
                )
                for key, value in items.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(
                            f'{name}/train/{key}', value, self.global_step
                        )
                self.writer.add_scalar(
                    f'{name}/train/learning_rate', current_lr, self.global_step
                )
                self.writer.add_scalar(
                    f'{name}/train/grad_norm', float(grad_norm), self.global_step
                )

        if successful_batches == 0:
            raise RuntimeError(f'{name}: không có train batch hữu hạn nào.')
        averages = self._average_items(sums, successful_batches)
        averages['learning_rate'] = optimizer.param_groups[0]['lr']
        averages['seconds'] = time.time() - start_time
        logger.info(
            '[TRAIN EPOCH] region=%s epoch=%s/%s | loss=%.5f | '
            'detection=%.5f | geometry=%.5f (aperture=%.5f state=%.5f) | '
            'batches=%s | seed=%s | %.1fs',
            name,
            local_epoch + 1,
            stage_cfg.epochs,
            averages['loss'],
            averages['loss_detection'],
            averages['geometry'],
            averages['geometry_aperture'],
            averages['geometry_state'],
            successful_batches,
            epoch_seed,
            averages['seconds'],
        )
        self._assert_global_frozen(check_bitwise=False)
        return averages

    @torch.no_grad()
    def _validate(self, name: str, local_epoch: int) -> dict[str, float]:
        self.model.eval()
        self._assert_global_frozen(check_bitwise=False)
        use_amp = self.cfg.amp and self.device.type == 'cuda'
        sums: dict[str, float] = {}
        count = 0
        nme_sum = 0.0
        nme_count = 0
        progress = tqdm(
            self.val_loader,
            desc=f'Val {name} {local_epoch + 1}',
            unit='batch',
            dynamic_ncols=True,
        )
        for batch in progress:
            images, targets = self._extract_region_batch(batch, name)
            with torch.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                predictions = self.model.forward_specialist(name, images)
                total_loss, items = self.loss_fn(name, predictions, targets)
            if not torch.isfinite(total_loss):
                logger.warning('[VAL] %s gặp loss không hữu hạn; bỏ batch.', name)
                continue
            batch_nme_sum, batch_nme_count = self._landmark_nme(
                name, predictions, targets
            )
            nme_sum += batch_nme_sum
            nme_count += batch_nme_count
            count += 1
            self._accumulate_items(sums, items)
            progress.set_postfix(
                loss=f"{sums['loss'] / count:.4f}",
                nme=(f'{nme_sum / nme_count:.4f}' if nme_count else 'n/a'),
                geometry=f"{sums['geometry'] / count:.4f}",
                state=f"{sums['geometry_state'] / count:.4f}",
            )
        if count == 0:
            raise RuntimeError(f'{name}: không có validation batch hữu hạn nào.')
        averages = self._average_items(sums, count)
        if nme_count == 0:
            raise RuntimeError(f'{name}: không có landmark hợp lệ để tính NME.')
        averages['landmark_nme'] = nme_sum / nme_count
        logger.info(
            '[VALIDATE] region=%s epoch=%s | loss=%.5f detection=%.5f '
            'landmark_nme=%.6f | '
            'geometry=%.5f (aperture=%.5f state=%.5f) | '
            'o2m(lmk=%.5f n_pos=%.1f) | '
            'o2o(lmk=%.5f n_pos=%.1f) | batches=%s',
            name,
            local_epoch + 1,
            averages['loss'],
            averages['loss_detection'],
            averages['landmark_nme'],
            averages['geometry'],
            averages['geometry_aperture'],
            averages['geometry_state'],
            averages['o2m/lmk'],
            averages['o2m/n_lmk_pos'],
            averages['o2o/lmk'],
            averages['o2o/n_lmk_pos'],
            count,
        )
        for key, value in averages.items():
            self.writer.add_scalar(f'{name}/val/{key}', value, self.global_step)
        return averages

    def _specialists_state(self) -> dict:
        return {
            name: {
                key: value.detach().cpu()
                for key, value in specialist.state_dict().items()
            }
            for name, specialist in self.model.specialists.items()
        }

    def _save_checkpoint(
        self,
        path: str,
        *,
        active_stage: Optional[str],
        next_epoch: int,
        best_val: float,
        bad_epochs: int,
        optimizer=None,
        scheduler=None,
        scaler=None,
        best_specialist_state: Optional[dict] = None,
    ) -> None:
        training_state = {
            'active_stage': active_stage,
            'next_epoch': int(next_epoch),
            'global_step': int(self.global_step),
            'best_val': float(best_val),
            'bad_epochs': int(bad_epochs),
            'optimizer': _to_cpu_tree(optimizer.state_dict()) if optimizer else None,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None,
            'best_specialist': _to_cpu_tree(best_specialist_state),
        }
        checkpoint = {
            'format_version': _CHECKPOINT_VERSION,
            'kind': _CHECKPOINT_KIND,
            'global_checkpoint': dict(self.global_checkpoint_metadata),
            'architecture_signature': self.model.architecture_signature(),
            # Inference phải tái tạo crop bằng đúng geometry của train.
            'crop_config': asdict(self.cfg.dataset.crop),
            'specs': self._checkpoint_specs(),
            'specialists': self._specialists_state(),
            'completed_stages': tuple(self.completed_stages),
            'stage_metrics': self.stage_metrics,
            'training_plan': self._training_plan(),
            'training_state': training_state,
        }
        # Không dùng model.state_dict(): nó sẽ kéo toàn bộ HEAD4 vào checkpoint.
        if 'model' in checkpoint or 'global_detector' in checkpoint:
            raise RuntimeError('Checkpoint nhẹ vô tình chứa full global model.')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary_path = f'{path}.tmp'
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
        logger.info(
            '[CHECKPOINT] saved=%s | %.2f MB | completed=%s | active=%s',
            path,
            os.path.getsize(path) / (1024 ** 2),
            self.completed_stages,
            active_stage,
        )

    def _validate_checkpoint_header(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get('kind') != _CHECKPOINT_KIND:
            raise ValueError(f"Checkpoint kind không phải '{_CHECKPOINT_KIND}'.")
        if checkpoint.get('format_version') != _CHECKPOINT_VERSION:
            raise ValueError(
                f'Checkpoint format_version phải là {_CHECKPOINT_VERSION}.'
            )
        if checkpoint.get('specs') != self._checkpoint_specs():
            raise ValueError('Checkpoint specs/index mapping khác model hiện tại.')
        if checkpoint.get('architecture_signature') != (
            self.model.architecture_signature()
        ):
            raise ValueError('Architecture signature của specialist không khớp.')
        if checkpoint.get('crop_config') != asdict(self.cfg.dataset.crop):
            raise ValueError('Crop config của checkpoint khác dataloader hiện tại.')
        saved_global = checkpoint.get('global_checkpoint', {})
        if saved_global.get('model_signature') != self.global_checkpoint_metadata[
            'model_signature'
        ]:
            raise ValueError('Checkpoint specialist tham chiếu model signature khác.')
        if saved_global.get('state_sha256') != self._global_state_hash:
            raise ValueError('Checkpoint specialist tham chiếu HEAD4 state hash khác.')
        if checkpoint.get('training_plan') != self._training_plan():
            raise ValueError('Training plan của resume checkpoint đã thay đổi.')

    def _load_resume(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Không tìm thấy resume checkpoint: '{path}'.")
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError('Resume checkpoint phải là dict.')
        self._validate_checkpoint_header(checkpoint)
        specialist_states = checkpoint.get('specialists')
        if not isinstance(specialist_states, Mapping) or set(
            specialist_states
        ) != set(SPECIALIST_NAMES):
            raise ValueError('Resume checkpoint không chứa đủ ba specialist.')
        for name in SPECIALIST_NAMES:
            self.model.specialists[name].load_state_dict(
                specialist_states[name], strict=True
            )

        completed = list(checkpoint.get('completed_stages', ()))
        if completed != list(SPECIALIST_NAMES[:len(completed)]):
            raise ValueError(
                f'completed_stages phải là prefix của {SPECIALIST_NAMES}, '
                f'nhận {completed}.'
            )
        self.completed_stages = completed
        saved_metrics = checkpoint.get('stage_metrics', {})
        self.stage_metrics = {
            name: list(saved_metrics.get(name, [])) for name in SPECIALIST_NAMES
        }
        training_state = checkpoint.get('training_state') or {}
        self.global_step = int(training_state.get('global_step', 0))
        active_stage = training_state.get('active_stage')
        if active_stage is not None and active_stage not in SPECIALIST_NAMES:
            raise ValueError(f'active_stage resume không hợp lệ: {active_stage!r}.')
        if active_stage in self.completed_stages:
            raise ValueError(
                f'active_stage={active_stage!r} da nam trong completed_stages.'
            )
        next_stage = (
            SPECIALIST_NAMES[len(self.completed_stages)]
            if len(self.completed_stages) < len(SPECIALIST_NAMES)
            else None
        )
        if active_stage is not None and active_stage != next_stage:
            raise ValueError(
                'active_stage phai dung stage ke tiep sau completed_stages: '
                f'expected={next_stage!r}, received={active_stage!r}.'
            )
        if next_stage is None and active_stage is not None:
            raise ValueError('Da completed ca ba stage nhung checkpoint van active.')
        if active_stage is not None:
            next_epoch = int(training_state.get('next_epoch', 0))
            stage_epochs = self.cfg.stage(active_stage).epochs
            if not 0 <= next_epoch <= stage_epochs:
                raise ValueError(
                    f'next_epoch={next_epoch} ngoai [0,{stage_epochs}] '
                    f'cho stage {active_stage}.'
                )
            if next_epoch > 0:
                if not isinstance(training_state.get('best_specialist'), Mapping):
                    raise ValueError(
                        'Resume giua stage phai co best_specialist state.'
                    )
                if training_state.get('optimizer') is None:
                    raise ValueError('Resume giua stage thieu optimizer state.')
                if training_state.get('scheduler') is None:
                    raise ValueError('Resume giua stage thieu scheduler state.')
                if not math.isfinite(
                    float(training_state.get('best_val', float('inf')))
                ):
                    raise ValueError('Resume giua stage co best_val khong huu han.')
        self._resume_training_state = (
            dict(training_state) if active_stage is not None else None
        )
        self.model.activate_specialist(None)
        self._assert_global_frozen(check_bitwise=True)
        logger.info(
            '[RESUME] file=%s | completed=%s | active=%s | next_epoch=%s | '
            'global_step=%s',
            path,
            self.completed_stages,
            active_stage,
            training_state.get('next_epoch'),
            self.global_step,
        )

    def _run_stage(self, name: str) -> None:
        stage_cfg = self.cfg.stage(name)
        self.model.activate_specialist(name)
        self.model.train(True)
        self._assert_only_active_is_trainable(name)
        optimizer, trainable_parameters = self._build_optimizer(name)
        scheduler = self._build_scheduler(optimizer, stage_cfg)
        scaler = torch.amp.GradScaler(
            'cuda', enabled=self.cfg.amp and self.device.type == 'cuda'
        )
        start_epoch = 0
        best_val = float('inf')
        bad_epochs = 0
        best_specialist_state = None

        resume_state = self._resume_training_state
        if resume_state and resume_state.get('active_stage') == name:
            start_epoch = int(resume_state.get('next_epoch', 0))
            best_val = float(resume_state.get('best_val', float('inf')))
            bad_epochs = int(resume_state.get('bad_epochs', 0))
            if resume_state.get('optimizer') is not None:
                optimizer.load_state_dict(resume_state['optimizer'])
            if resume_state.get('scheduler') is not None:
                scheduler.load_state_dict(resume_state['scheduler'])
            if resume_state.get('scaler') is not None:
                scaler.load_state_dict(resume_state['scaler'])
            best_specialist_state = resume_state.get('best_specialist')
            logger.info(
                '[STAGE RESUME] %s từ local_epoch=%s | best_val=%.6f | '
                'bad_epochs=%s',
                name,
                start_epoch,
                best_val,
                bad_epochs,
            )
        self._resume_training_state = None
        if start_epoch >= stage_cfg.epochs:
            logger.warning(
                '%s đã đủ %s epoch trong resume; chuyển thẳng stage.',
                name,
                stage_cfg.epochs,
            )

        trainable_count = sum(p.numel() for p in trainable_parameters)
        logger.info(
            '[STAGE START] region=%s | epochs=%s | start_epoch=%s | '
            'lr=%.3e | trainable=%s | aperture_gain=%.3f | state_gain=%.3f',
            name,
            stage_cfg.epochs,
            start_epoch,
            stage_cfg.learning_rate,
            f'{trainable_count:,}',
            (
                self.cfg.geometry_loss.eye_aperture_gain
                if name in (LEFT_EYE, RIGHT_EYE)
                else self.cfg.geometry_loss.mouth_aperture_gain
            ),
            (
                self.cfg.geometry_loss.eye_state_gain
                if name in (LEFT_EYE, RIGHT_EYE)
                else 0.0
            ),
        )

        best_path = os.path.join(self.cfg.checkpoint_dir, f'best_{name}.pt')
        for local_epoch in range(start_epoch, stage_cfg.epochs):
            train_metrics = self._train_epoch(
                name,
                local_epoch,
                optimizer,
                scheduler,
                scaler,
                trainable_parameters,
            )
            val_metrics = self._validate(name, local_epoch)
            selection_value = val_metrics[self.cfg.selection_metric]
            improved = selection_value < (
                best_val - stage_cfg.early_stopping_min_delta
            )
            if improved:
                best_val = selection_value
                bad_epochs = 0
                best_specialist_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.specialists[name]
                    .state_dict()
                    .items()
                }
            else:
                bad_epochs += 1

            epoch_metrics = {
                'epoch': local_epoch,
                'global_step': self.global_step,
                'train': train_metrics,
                'val': val_metrics,
                'best_val': best_val,
                'improved': improved,
            }
            self.stage_metrics[name].append(epoch_metrics)
            checkpoint_kwargs = dict(
                active_stage=name,
                next_epoch=local_epoch + 1,
                best_val=best_val,
                bad_epochs=bad_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_specialist_state=best_specialist_state,
            )
            if improved:
                self._save_checkpoint(best_path, **checkpoint_kwargs)
                logger.info(
                    '[BEST] region=%s epoch=%s metric=%s value=%.6f | '
                    'val_loss=%.6f nme=%.6f geometry=%.6f',
                    name,
                    local_epoch + 1,
                    self.cfg.selection_metric,
                    best_val,
                    val_metrics['loss'],
                    val_metrics['landmark_nme'],
                    val_metrics['geometry'],
                )
            self._save_checkpoint(
                os.path.join(self.cfg.checkpoint_dir, 'last.pt'),
                **checkpoint_kwargs,
            )
            self.writer.flush()
            if bad_epochs >= stage_cfg.early_stopping_patience:
                logger.info(
                    '[EARLY STOP] region=%s tại epoch=%s | metric=%s '
                    'best=%.6f | '
                    'patience=%s',
                    name,
                    local_epoch + 1,
                    self.cfg.selection_metric,
                    best_val,
                    stage_cfg.early_stopping_patience,
                )
                break

        if best_specialist_state is None:
            raise RuntimeError(
                f'{name} chưa có best state hợp lệ; không được chuyển stage.'
            )
        self.model.specialists[name].load_state_dict(
            best_specialist_state, strict=True
        )
        if name not in self.completed_stages:
            self.completed_stages.append(name)
        self.model.activate_specialist(None)
        self._assert_global_frozen(check_bitwise=True)
        self._save_checkpoint(
            os.path.join(self.cfg.checkpoint_dir, f'{name}_final.pt'),
            active_stage=None,
            next_epoch=0,
            best_val=best_val,
            bad_epochs=0,
        )
        self._save_checkpoint(
            os.path.join(self.cfg.checkpoint_dir, 'last.pt'),
            active_stage=None,
            next_epoch=0,
            best_val=best_val,
            bad_epochs=0,
        )
        logger.info(
            '[STAGE DONE] region=%s | metric=%s restored_best=%.6f | '
            'completed=%s | '
            'HEAD4 sha256=%s',
            name,
            self.cfg.selection_metric,
            best_val,
            self.completed_stages,
            self._global_state_hash[:16],
        )

    def fit(self):
        logger.info(
            '[TRAIN PLAN] order=%s | device=%s | train_batches=%s | '
            'val_batches=%s',
            SPECIALIST_NAMES,
            self.device,
            len(self.train_loader),
            len(self.val_loader),
        )
        try:
            for name in SPECIALIST_NAMES:
                if name in self.completed_stages:
                    logger.info('[SKIP] specialist %s đã hoàn tất trong resume.', name)
                    continue
                self._run_stage(name)
            self.model.activate_specialist(None)
            self._assert_global_frozen(check_bitwise=True)
            final_path = os.path.join(
                self.cfg.checkpoint_dir, 'multihead_final.pt'
            )
            final_best = {
                name: (
                    min(
                        (
                            entry['val'][self.cfg.selection_metric]
                            for entry in metrics
                        ),
                        default=float('inf'),
                    )
                )
                for name, metrics in self.stage_metrics.items()
            }
            self._save_checkpoint(
                final_path,
                active_stage=None,
                next_epoch=0,
                best_val=max(final_best.values()),
                bad_epochs=0,
            )
            logger.info(
                '[TRAIN COMPLETE] final=%s | selection_metric=%s | '
                'best_by_region=%s | '
                'HEAD4 unchanged=%s',
                final_path,
                self.cfg.selection_metric,
                final_best,
                self._global_state_hash[:16],
            )
            return self.model, final_path
        finally:
            self.writer.close()


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Cấu hình trực tiếp tại đây; không dùng CLI.
    # ------------------------------------------------------------------
    GLOBAL_BEST_PT = (
        '/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/'
        'checkpoints_face_lmk_finetune/best.pt'
    )
    DATASET_ROOT = '/run/media/tranmanhduy/Data/Datafinetune2'
    IMAGES_DIR_NAME = 'images'
    ANNOTATION_FILE = 'annotations.jsonl'

    BATCH_SIZE = 4
    NUM_WORKERS = 4
    VAL_RATIO = 0.15
    DEVICE = 'cuda'
    SEED = 42

    # Model moi lon hon can them runway; early stopping van ngan overfit.
    LEFT_EYE_EPOCHS = 25
    RIGHT_EYE_EPOCHS = 25
    MOUTH_EPOCHS = 25
    SPECIALIST_LR = 3e-4
    EYE_GEOMETRY_GAIN = 8.0
    EYE_STATE_GAIN = 20.0
    EYE_CLOSED_THRESHOLD = 0.15
    EYE_OPEN_THRESHOLD = 0.20
    MOUTH_GEOMETRY_GAIN = 6.0

    dataset_config = MultiHeadDatasetConfig(
        root_dir=DATASET_ROOT,
        images_dir_name=IMAGES_DIR_NAME,
        jsonl_name=ANNOTATION_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    global_config = TrainConfig(require_pretrained_trunk=False)
    trainer_config = MultiHeadTrainerConfig(
        global_checkpoint_path=GLOBAL_BEST_PT,
        dataset=dataset_config,
        global_model_cfg=global_config,
        left_eye_stage=SpecialistStageConfig(
            epochs=LEFT_EYE_EPOCHS,
            learning_rate=SPECIALIST_LR,
            warmup_epochs=1.0,
            early_stopping_patience=7,
        ),
        right_eye_stage=SpecialistStageConfig(
            epochs=RIGHT_EYE_EPOCHS,
            learning_rate=SPECIALIST_LR,
            warmup_epochs=1.0,
            early_stopping_patience=7,
        ),
        mouth_stage=SpecialistStageConfig(
            epochs=MOUTH_EPOCHS,
            learning_rate=SPECIALIST_LR,
            warmup_epochs=1.0,
            early_stopping_patience=7,
        ),
        geometry_loss=RegionGeometryLossConfig(
            eye_aperture_gain=EYE_GEOMETRY_GAIN,
            eye_state_gain=EYE_STATE_GAIN,
            eye_closed_threshold=EYE_CLOSED_THRESHOLD,
            eye_open_threshold=EYE_OPEN_THRESHOLD,
            mouth_aperture_gain=MOUTH_GEOMETRY_GAIN,
        ),
        device=DEVICE,
        seed=SEED,
    )
    SequentialMultiHeadTrainer(trainer_config).fit()


__all__ = (
    'SpecialistStageConfig',
    'MultiHeadTrainerConfig',
    'SequentialMultiHeadTrainer',
    'load_frozen_global_detector',
)
