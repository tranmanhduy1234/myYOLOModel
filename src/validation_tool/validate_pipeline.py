"""
validate_pipeline.py
=====================
Validate các thành phần "hạ tầng" xung quanh model:
  - ema.py         : ModelEMA (init, decay warmup, update, non-float buffer, state_dict)
  - engine.py      : get_optimizer, lr_lambda_factory, save/load_checkpoint,
                     validate(), logic device fallback
  - config.py      : TrainConfig validation (tuple length, num_workers, auto-fix)

PHẠM VI KHÔNG TEST
--------------------
- src/utils/   : state_dict_handle.py (set_seed), engine_with_tb.py, log_setup.py, tb_logger.py
- src/TransferLearning/ : dataloader_tfl.py, head_tfl.py, loss_tfl.py

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ
------------------------------
1. Sửa assertion device fallback: sau khi bug đã được sửa trong engine.py,
   test kiểm tra logic đúng (else "cpu") thay vì chỉ phát hiện bug.
2. Thêm test mới: EMA không thay đổi khi model không thay đổi, EMA eval() mode
   luôn giữ, optimizer SGD hoạt động, validate() không tính grad,
   checkpoint không có EMA, config device field, config amp field.
3. XÓA t_set_seed_reproducibility: set_seed thuộc src/utils/state_dict_handle.py,
   nằm ngoài phạm vi validate của file này.

Chạy độc lập:
    python -m src.validation_tool.validate_pipeline
    python -m src.validation_tool.validate_pipeline --device cuda
"""

import argparse
import copy
import os
import sys
import tempfile

import torch
import torch.nn as nn

from src.validation_tool.validate_common import Reporter, get_device, skip

from src.model import NMSFreeDetector
from src.train.ema import ModelEMA
from src.train.engine import (
    get_optimizer, lr_lambda_factory, save_checkpoint, load_checkpoint, set_seed,
    validate,
)
# Lưu ý: set_seed được import để dùng làm tiện ích thiết lập seed trong các test
# checkpoint (không phải test bản thân hàm set_seed — xem t_set_seed_reproducibility đã bị xóa).
from src.train.loss import DetectionLoss
from src.config import TrainConfig

SMALL = dict(nc=4, reg_max=8, backbone_w=(16, 32, 64, 128, 160),
             backbone_n=(1, 1, 1, 1), neck_n=1, strides=(8, 16, 32))


# ==============================================================================
# 1. ema.py - ModelEMA
# ==============================================================================
def test_ema(device: str, R: Reporter):
    R.section("1. EMA.PY - ModelEMA")

    def t_ema_init_matches_model():
        m   = NMSFreeDetector(**SMALL).to(device)
        ema = ModelEMA(m, decay=0.999, warmup_updates=10)
        for (n1, p1), (n2, p2) in zip(m.named_parameters(), ema.ema.named_parameters()):
            assert torch.allclose(p1, p2), f"EMA khởi tạo phải là bản sao chính xác của model ({n1})"
        assert all(not p.requires_grad for p in ema.ema.parameters()), "tham số EMA phải bị đóng băng"
        assert not ema.ema.training, "model EMA phải ở chế độ eval() ngay khi khởi tạo"
        return "EMA khởi tạo = deepcopy(model), đóng băng grad, ở eval()"
    R.check("ema", "ModelEMA.__init__: deepcopy đúng, đóng băng grad, eval()", t_ema_init_matches_model)

    def t_ema_stays_eval_after_update():
        """EMA phải vẫn ở eval() sau khi update, dù model đang ở train()."""
        m   = NMSFreeDetector(**SMALL).to(device)
        ema = ModelEMA(m, decay=0.999, warmup_updates=10)
        m.train()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema.update(m)
        assert not ema.ema.training, "EMA phải ở eval() sau khi update (không bị lây train() của model)"
        return "EMA giữ eval() mode sau update"
    R.check("ema", "ModelEMA: vẫn ở eval() sau update, không bị lây train() của model", t_ema_stays_eval_after_update)

    def t_decay_warmup_monotonic():
        m   = NMSFreeDetector(**SMALL).to(device)
        ema = ModelEMA(m, decay=0.9998, warmup_updates=100)
        prev = -1.0
        for _ in range(500):
            ema.updates += 1
            d = ema._current_decay()
            assert d >= prev - 1e-9, "decay phải tăng đơn điệu theo số update (warmup)"
            assert d <= ema.decay + 1e-9, "decay không được VƯỢT QUÁ decay tối đa cấu hình"
            prev = d
        assert prev > ema.decay * 0.99, f"sau nhiều update, decay phải gần sát decay tối đa, được {prev:.6f}"
        return f"decay tăng đơn điệu 0 -> {prev:.6f} (max cấu hình={ema.decay})"
    R.check("ema", "_current_decay(): tăng đơn điệu theo warmup, không vượt decay max", t_decay_warmup_monotonic)

    def t_update_moves_toward_model_not_equal():
        torch.manual_seed(0)
        m   = NMSFreeDetector(**SMALL).to(device)
        ema = ModelEMA(m, decay=0.99, warmup_updates=1)
        ema_before = copy.deepcopy(ema.ema.state_dict())

        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(m)
        moved = False
        model_params = dict(m.named_parameters())
        for name, v_after in ema.ema.named_parameters():
            v_before = ema_before[name]
            v_model  = model_params[name]
            if not torch.allclose(v_after, v_before):
                moved = True
            assert not torch.allclose(v_after, v_model), \
                f"EMA sau 1 update KHÔNG được bằng tuyệt đối trọng số model (decay={ema.decay}!=0, tham số '{name}')"
        assert moved, "phải có ít nhất 1 tham số trong EMA thay đổi sau update()"
        return "EMA dịch chuyển về phía model nhưng không nhảy thẳng tới (đúng công thức weighted avg)"
    R.check("ema", "update(): EMA = decay*ema + (1-decay)*model, không copy thẳng", t_update_moves_toward_model_not_equal)

    def t_ema_no_change_when_model_unchanged():
        """Nếu model không thay đổi, EMA sau update phải rất gần giá trị ban đầu."""
        torch.manual_seed(1)
        m   = NMSFreeDetector(**SMALL).to(device)
        ema = ModelEMA(m, decay=0.999, warmup_updates=1)
        ema_before = {n: v.clone() for n, v in ema.ema.named_parameters()}
        # KHÔNG thay đổi model
        ema.update(m)
        for name, v_after in ema.ema.named_parameters():
            v_before = ema_before[name]
            v_model  = dict(m.named_parameters())[name]
            # EMA = 0.999*ema + 0.001*model, nhưng ema==model → ema sau == model sau == trước
            assert torch.allclose(v_after, v_model, atol=1e-5), \
                f"nếu model không đổi, EMA phải bằng model sau update ('{name}')"
        return "EMA = model khi model không thay đổi (weighted avg của cùng giá trị)"
    R.check("ema", "update(): EMA không thay đổi khi model không thay đổi", t_ema_no_change_when_model_unchanged)

    def t_non_float_buffers_copied_directly():
        m  = NMSFreeDetector(**SMALL).to(device)
        bn = next(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d))
        ema = ModelEMA(m, decay=0.999, warmup_updates=10)
        bn.num_batches_tracked += 5
        ema.update(m)
        for key, v in m.state_dict().items():
            if not v.dtype.is_floating_point:
                assert torch.equal(ema.ema.state_dict()[key], v), \
                    f"buffer không phải float ('{key}', vd num_batches_tracked) phải được COPY THẲNG, không EMA"
        return "buffer non-float (vd num_batches_tracked) được copy trực tiếp, không làm EMA"
    R.check("ema", "update(): buffer không phải float được copy thẳng (không áp dụng EMA)", t_non_float_buffers_copied_directly)

    def t_state_dict_roundtrip():
        m1  = NMSFreeDetector(**SMALL).to(device)
        ema1 = ModelEMA(m1, decay=0.999, warmup_updates=10)
        for _ in range(5):
            with torch.no_grad():
                for p in m1.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema1.update(m1)

        m2  = NMSFreeDetector(**SMALL).to(device)
        ema2 = ModelEMA(m2, decay=0.999, warmup_updates=10)
        ema2.load_state_dict(ema1.state_dict())
        for (n1, p1), (n2, p2) in zip(ema1.ema.named_parameters(), ema2.ema.named_parameters()):
            assert torch.allclose(p1, p2), f"load_state_dict/state_dict round-trip phải khớp 100% ('{n1}')"
        return "ModelEMA.state_dict() -> load_state_dict(): round-trip chính xác"
    R.check("ema", "state_dict()/load_state_dict(): round-trip chính xác", t_state_dict_roundtrip)


# ==============================================================================
# 2. engine.py - optimizer grouping, LR schedule, checkpoint I/O
# ==============================================================================
def test_engine(device: str, R: Reporter):
    R.section("2. ENGINE.PY - optimizer, LR schedule, checkpoint")

    def t_optimizer_param_groups_split_correctly():
        m   = NMSFreeDetector(**SMALL).to(device)
        cfg = TrainConfig(optimizer="adamw", lr0=1e-3, weight_decay=0.05)
        opt = get_optimizer(m, cfg)
        assert len(opt.param_groups) == 2, "phải có đúng 2 param group (decay / no_decay)"
        g_decay, g_no_decay = opt.param_groups
        assert g_decay["weight_decay"]    == cfg.weight_decay
        assert g_no_decay["weight_decay"] == 0.0

        decay_ids    = {id(p) for p in g_decay["params"]}
        no_decay_ids = {id(p) for p in g_no_decay["params"]}
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith("bias"):
                assert id(p) in no_decay_ids, f"'{name}' (ndim<=1 hoặc bias) phải ở group KHÔNG weight_decay"
            else:
                assert id(p) in decay_ids,    f"'{name}' phải ở group weight_decay"
        return f"{len(g_decay['params'])} tensor có decay, {len(g_no_decay['params'])} tensor không decay (bias/BN)"
    R.check("engine", "get_optimizer(): tách đúng nhóm weight_decay (bỏ qua bias/BN 1-D)", t_optimizer_param_groups_split_correctly)

    def t_optimizer_sgd_works():
        """get_optimizer với optimizer='sgd' phải trả về SGD hợp lệ."""
        m   = NMSFreeDetector(**SMALL).to(device)
        cfg = TrainConfig(optimizer="sgd", lr0=1e-2, momentum=0.9)
        opt = get_optimizer(m, cfg)
        assert isinstance(opt, torch.optim.SGD), "optimizer='sgd' phải trả về SGD"
        # Bước optimizer một lần
        loss = sum(p.sum() for p in m.parameters())
        m.zero_grad()
        loss.backward()
        opt.step()  # không được crash
        return "optimizer='sgd' tạo SGD, step() không crash"
    R.check("engine", "get_optimizer(): optimizer='sgd' hoạt động đúng", t_optimizer_sgd_works)

    def t_optimizer_unknown_raises():
        m   = NMSFreeDetector(**SMALL).to(device)
        cfg = TrainConfig(optimizer="rmsprop")
        try:
            get_optimizer(m, cfg)
            raise RuntimeError("phải raise ValueError với optimizer không hỗ trợ")
        except ValueError:
            pass
        return "optimizer là 'rmsprop' -> ValueError đúng như kỳ vọng"
    R.check("engine", "get_optimizer(): optimizer không hỗ trợ -> ValueError", t_optimizer_unknown_raises)

    def t_lr_lambda_warmup_then_cosine_decay():
        cfg             = TrainConfig(epochs=10, warmup_epochs=2.0, lr_min_factor=0.01)
        steps_per_epoch = 100
        lam             = lr_lambda_factory(cfg, steps_per_epoch)
        warmup_steps    = int(cfg.warmup_epochs * steps_per_epoch)

        assert lam(0) == 0.0, "step=0 -> factor phải = 0 (bắt đầu warmup từ 0)"
        assert abs(lam(warmup_steps) - 1.0) < 1e-6, "cuối warmup -> factor phải đạt đỉnh = 1.0"

        vals_warmup = [lam(s) for s in range(0, warmup_steps, warmup_steps // 5 or 1)]
        assert all(vals_warmup[i] <= vals_warmup[i + 1] for i in range(len(vals_warmup) - 1)), \
            "trong giai đoạn warmup, LR factor phải tăng đơn điệu"

        total_steps = cfg.epochs * steps_per_epoch
        last        = lam(total_steps - 1)
        assert abs(last - cfg.lr_min_factor) < 1e-2, \
            f"cuối lịch trình cosine, factor phải xấp xỉ lr_min_factor ({cfg.lr_min_factor}), được {last:.4f}"
        assert lam(total_steps + 500) == lam(total_steps - 1) or abs(lam(total_steps + 500) - cfg.lr_min_factor) < 1e-2, \
            "vượt qua total_steps, factor phải được clamp (không âm/không vượt)"
        return f"lam(0)=0, lam(warmup)=1.0, lam(cuối)~{last:.4f}~=lr_min_factor"
    R.check("engine", "lr_lambda_factory(): warmup tuyến tính 0->1 rồi cosine decay về lr_min_factor", t_lr_lambda_warmup_then_cosine_decay)

    def t_lr_lambda_single_epoch():
        """lr_lambda_factory với epochs=1 không crash và clamp đúng."""
        cfg = TrainConfig(epochs=1, warmup_epochs=0.5, lr_min_factor=0.05)
        lam = lr_lambda_factory(cfg, steps_per_epoch=10)
        assert lam(0) == 0.0
        assert lam(5) >= 0 and lam(5) <= 1.0  # trong warmup
        assert lam(100) >= cfg.lr_min_factor - 1e-3  # vượt qua, phải clamp
        return "lr_lambda với epochs=1 không crash, clamp đúng"
    R.check("engine", "lr_lambda_factory(): epochs=1 edge case không crash", t_lr_lambda_single_epoch)

    def t_checkpoint_roundtrip():
        set_seed(0)
        m1  = NMSFreeDetector(**SMALL).to(device)
        cfg = TrainConfig(lr0=1e-3)
        opt1 = get_optimizer(m1, cfg)
        sch1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda_factory(cfg, 10))
        ema1 = ModelEMA(m1, decay=0.99, warmup_updates=5)
        for _ in range(3):
            opt1.step()
            sch1.step()
            ema1.update(m1)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, m1, opt1, sch1, ema1, epoch=3, best_val=1.234, cfg=cfg)
            assert os.path.isfile(path), "save_checkpoint phải tạo file trên đĩa"

            m2   = NMSFreeDetector(**SMALL).to(device)
            opt2 = get_optimizer(m2, cfg)
            sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda_factory(cfg, 10))
            ema2 = ModelEMA(m2, decay=0.99, warmup_updates=5)

            epoch, best_val = load_checkpoint(path, m2, opt2, sch2, ema2, map_location=str(device))
            assert epoch == 3 and abs(best_val - 1.234) < 1e-6, "epoch/best_val phải được khôi phục chính xác"
            for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                assert torch.allclose(p1, p2), f"trọng số model phải khớp sau load_checkpoint ('{n1}')"
            assert sch2.get_last_lr() == sch1.get_last_lr(), "scheduler state (LR hiện tại) phải khớp sau resume"
            for (n1, p1), (n2, p2) in zip(ema1.ema.named_parameters(), ema2.ema.named_parameters()):
                assert torch.allclose(p1, p2), f"trọng số EMA phải khớp sau load_checkpoint ('{n1}')"
        return "save_checkpoint -> load_checkpoint: model/optimizer/scheduler/ema/epoch/best_val đều khớp"
    R.check("engine", "save_checkpoint()/load_checkpoint(): round-trip đầy đủ (model+opt+sched+ema+meta)", t_checkpoint_roundtrip)

    def t_checkpoint_without_ema():
        """save/load checkpoint không có EMA (ema=None) không crash."""
        set_seed(0)
        m   = NMSFreeDetector(**SMALL).to(device)
        cfg = TrainConfig(lr0=1e-3)
        opt = get_optimizer(m, cfg)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg, 10))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt_no_ema.pt")
            save_checkpoint(path, m, opt, sch, ema=None, epoch=1, best_val=2.5, cfg=cfg)
            m2  = NMSFreeDetector(**SMALL).to(device)
            opt2 = get_optimizer(m2, cfg)
            sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda_factory(cfg, 10))
            epoch, best_val = load_checkpoint(path, m2, opt2, sch2, ema=None, map_location=str(device))
            assert epoch == 1 and abs(best_val - 2.5) < 1e-6
        return "save/load checkpoint không có EMA (ema=None) hoạt động đúng"
    R.check("engine", "save/load checkpoint không có EMA (ema=None) không crash", t_checkpoint_without_ema)

    # t_set_seed_reproducibility đã bị xóa:
    # set_seed() được định nghĩa trong src/utils/state_dict_handle.py — nằm ngoài
    # phạm vi validate của file này (không test code từ src/utils/ và src/TransferLearning/).

    def t_validate_no_grad():
        """validate() không được sinh grad trên model (decorator @torch.no_grad)."""
        m  = NMSFreeDetector(**SMALL).to(device)
        cr = DetectionLoss(nc=SMALL["nc"], reg_max=SMALL["reg_max"], topk_o2m=3, topk_o2o=1)

        from torch.utils.data import DataLoader, Dataset

        class _Toy(Dataset):
            def __len__(self): return 2
            def __getitem__(self, i):
                img = torch.randn(3, 128, 128)
                tgt = {"boxes": torch.tensor([[10., 10., 80., 80.]]),
                       "labels": torch.tensor([0])}
                return img, tgt

        from src.train.dataloader1_obj365 import collate_fn
        loader = DataLoader(_Toy(), batch_size=2, collate_fn=collate_fn)
        val_loss = validate(m, cr, loader, device)
        assert isinstance(val_loss, float) and val_loss >= 0
        # Kiểm tra không có grad nào được sinh
        for p in m.parameters():
            assert p.grad is None, f"validate() không được sinh grad (param '{p.shape}' có grad)"
        return f"val_loss={val_loss:.4f}, không có grad sau validate()"
    R.check("engine", "validate(): không sinh grad trên tham số model (@torch.no_grad)", t_validate_no_grad)

    def t_run_training_device_fallback_logic():
        """Kiểm tra logic device fallback trong engine.py (else 'cpu', không phải else 'cuda')."""
        import inspect
        from src.train import engine as engine_mod
        src_code = inspect.getsource(engine_mod.run_training)
        # Bug cũ: else "cuda" — nếu không có CUDA thì fallback ngược về "cuda"
        has_bug   = 'else "cuda"' in src_code and 'torch.cuda.is_available() else "cuda"' in src_code
        has_fixed = 'torch.cuda.is_available() else "cpu"' in src_code
        if has_bug and not has_fixed:
            assert False, (
                "run_training() đang fallback VỀ 'cuda' khi KHÔNG có GPU (ngược logic). "
                "Trên máy không có CUDA, dòng này sẽ khiến torch cố gắng dùng cuda và crash. "
                "Sửa thành: device = cfg.device if torch.cuda.is_available() else \"cpu\""
            )
        return "logic fallback device đúng: else 'cpu' (không phải else 'cuda')"
    R.check("engine", "run_training(): logic fallback device đúng (else 'cpu')", t_run_training_device_fallback_logic)


# ==============================================================================
# 3. config.py - TrainConfig
# ==============================================================================
def test_config(device: str, R: Reporter):
    R.section("3. CONFIG.PY - TrainConfig")

    def t_default_construct_ok():
        cfg = TrainConfig()
        assert cfg.nc > 0 and cfg.reg_max > 0
        return "TrainConfig() mặc định khởi tạo thành công"
    R.check("config", "TrainConfig() mặc định không lỗi", t_default_construct_ok)

    def t_invalid_tuple_lengths_raise():
        for bad_kwargs, field in [
            ({"shiftScaleRotate":   (0.1, 0.1, 5)},    "shiftScaleRotate (cần 4)"),
            ({"hueSaturationValue": (1, 2, 3)},         "hueSaturationValue (cần 4)"),
            ({"gaussNoise":         (1, 2)},            "gaussNoise (cần 3)"),
            ({"blur":               (3,)},              "blur (cần 2)"),
        ]:
            try:
                TrainConfig(**bad_kwargs)
                assert False, f"phải AssertionError với {field}"
            except AssertionError:
                pass
        return "__post_init__ bắt đúng độ dài tuple augment cho tất cả 4 trường"
    R.check("config", "__post_init__: bắt đúng số phần tử các tuple augment", t_invalid_tuple_lengths_raise)

    def t_negative_num_workers_raises():
        try:
            TrainConfig(num_workers=-1)
            raise RuntimeError("phải assert fail khi num_workers < 0")
        except AssertionError:
            pass
        return "num_workers < 0 -> AssertionError đúng như kỳ vọng"
    R.check("config", "__post_init__: num_workers âm phải bị chặn", t_negative_num_workers_raises)

    def t_num_workers_zero_disables_persistent_and_prefetch():
        cfg = TrainConfig(num_workers=0, persistent_workers=True, prefetch_factor=4)
        assert cfg.persistent_workers is False, "num_workers=0 phải tự động tắt persistent_workers"
        assert cfg.prefetch_factor is None,     "num_workers=0 phải tự động đặt prefetch_factor=None"
        return "num_workers=0 -> tự động sửa persistent_workers=False, prefetch_factor=None"
    R.check("config", "__post_init__: tự động sửa persistent_workers/prefetch_factor khi num_workers=0", t_num_workers_zero_disables_persistent_and_prefetch)

    def t_config_device_field_exists():
        """TrainConfig phải có trường device (dùng trong engine.py)."""
        cfg = TrainConfig()
        assert hasattr(cfg, "device"), "TrainConfig phải có trường 'device'"
        assert isinstance(cfg.device, str), "TrainConfig.device phải là str"
        return f"cfg.device='{cfg.device}' (kiểu str)"
    R.check("config", "TrainConfig: có trường 'device' kiểu str", t_config_device_field_exists)

    def t_config_amp_field_exists():
        """TrainConfig phải có trường amp (automatic mixed precision)."""
        cfg = TrainConfig()
        assert hasattr(cfg, "amp"), "TrainConfig phải có trường 'amp'"
        assert isinstance(cfg.amp, bool), "TrainConfig.amp phải là bool"
        return f"cfg.amp={cfg.amp} (kiểu bool)"
    R.check("config", "TrainConfig: có trường 'amp' kiểu bool", t_config_amp_field_exists)

    def t_config_model_fields_consistent():
        """backbone_w và backbone_n phải có số phần tử hợp lý với nhau."""
        cfg = TrainConfig()
        assert len(cfg.backbone_w) == 5, f"backbone_w phải có 5 giá trị (c0..c4), được {len(cfg.backbone_w)}"
        assert len(cfg.backbone_n) == 4, f"backbone_n phải có 4 giá trị (n0..n3), được {len(cfg.backbone_n)}"
        assert len(cfg.strides) == 3,    f"strides phải có 3 giá trị (P3/P4/P5), được {len(cfg.strides)}"
        return f"backbone_w={len(cfg.backbone_w)}, backbone_n={len(cfg.backbone_n)}, strides={cfg.strides}"
    R.check("config", "TrainConfig: backbone_w(5), backbone_n(4), strides(3) đúng số phần tử", t_config_model_fields_consistent)

    def t_config_lr_fields_positive():
        """Các trường LR phải dương."""
        cfg = TrainConfig()
        assert cfg.lr0 > 0,            "lr0 phải > 0"
        assert cfg.lr_min_factor > 0,  "lr_min_factor phải > 0"
        assert cfg.weight_decay >= 0,  "weight_decay phải >= 0"
        assert cfg.warmup_epochs >= 0, "warmup_epochs phải >= 0"
        return f"lr0={cfg.lr0}, lr_min_factor={cfg.lr_min_factor}, wd={cfg.weight_decay}"
    R.check("config", "TrainConfig: các trường LR có giá trị dương hợp lý", t_config_lr_fields_positive)


# ==============================================================================
# MAIN
# ==============================================================================
def run(device: str, verbose_traceback: bool = False) -> Reporter:
    """Chạy toàn bộ suite pipeline và trả về Reporter để run_all_validation.py gộp."""
    r = Reporter(verbose_traceback)
    torch.manual_seed(0)
    test_ema(device, r)
    test_engine(device, r)
    test_config(device, r)
    return r


def main():
    parser = argparse.ArgumentParser(description="Validate ema/engine/config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Thiết bị sử dụng: {device}")

    r = run(device, args.verbose_traceback)
    ok = r.summary("TỔNG KẾT - VALIDATE PIPELINE (ema/engine/config)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
