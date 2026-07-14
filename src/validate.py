"""
validate_model.py
==================
Bo cong cu validate toan dien cho NMSFreeDetector (backbone_neck.py, blocks.py,
head.py, model.py).

Muc tieu:
  1. TINH DUNG DAN (Correctness)
     - Shape / dtype cua tung block rieng le (Conv, C2f, CIB, SPPF, Attention, SCDown, DFL...)
     - Shape cua toan bo pipeline Backbone -> PAFPN -> DetectHead (train & eval mode)
     - Rang buoc kich thuoc dau vao (phai chia het cho stride tong = 32)
     - Rang buoc kenh cho Attention (dim % num_heads == 0) khi doi backbone_w
     - Backward pass: khong None-grad, khong NaN/Inf trong gradient
     - Numerical stability: khong NaN/Inf trong output voi nhieu batch size
     - DFL: gia tri decode nam trong khoang hop ly, la ham don dieu theo expectation
     - Anchor generation: doi chieu voi cong thuc thu cong
     - save_trunk / load_trunk: round-trip trong so giong het nhau
     - freeze_trunk: dong bang dung backbone+neck, khong dong bang head (tru DFL co dinh)
     - replace_head: doi nc/reg_max/strides dung, shape dau ra khop

  2. HIEU QUA (Efficiency)
     - Dem tham so theo tung submodule (backbone / neck / head / total)
     - FLOPs / MACs (dung thop, fallback bao loi neu chua cai)
     - Do tre suy luan (latency) & thong luong (throughput) tren nhieu batch size
     - Uoc luong bo nho GPU (neu co CUDA)
     - So sanh FP32 vs FP16 tren CUDA (neu co)

Cach chay:
    python validate_model.py                      # cau hinh mac dinh
    python validate_model.py --device cuda         # ep chay GPU
    python validate_model.py --imgsz 640 --nc 80    # tuy chinh
    python validate_model.py --skip-bench           # bo qua benchmark toc do (chi test dung)

File nay gia dinh duoc dat cung cap voi thu muc `src/` chua backbone_neck.py,
blocks.py, head.py, model.py (dung cau truc import "from src.xxx import ...").
"""

import argparse
import sys
import time
import traceback
from contextlib import contextmanager

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Import model & cac block can test rieng le
# ----------------------------------------------------------------------------
try:
    from src.model import NMSFreeDetector
    from src.backbone_neck import Backbone, PAFPN
    from src.head import DetectHead, ScaleHead
    from src.blocks import (
        Conv, DWConv, Bottleneck, C2f, CIB, C2fCIB, SPPF, DFL,
        Attention, C2fPSA, SCDown,
    )
except ImportError as e:
    print(f"[LOI IMPORT] Khong the import module tu 'src': {e}")
    print("  -> Dam bao validate_model.py duoc dat ngang hang voi thu muc 'src/'")
    print("     va thu muc 'src/' co file __init__.py (co the rong).")
    sys.exit(1)


# ==============================================================================
# TIEN ICH BAO CAO KET QUA
# ==============================================================================
class Reporter:
    def __init__(self):
        self.results = []  # (section, name, passed, detail)

    def check(self, section, name, fn):
        try:
            detail = fn()
            self.results.append((section, name, True, detail or ""))
            print(f"  [PASS] {name}" + (f" -> {detail}" if detail else ""))
        except AssertionError as e:
            self.results.append((section, name, False, str(e)))
            print(f"  [FAIL] {name} -> AssertionError: {e}")
        except Exception as e:
            self.results.append((section, name, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {name} -> {type(e).__name__}: {e}")
            if VERBOSE_TRACEBACK:
                traceback.print_exc()

    def section(self, title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r[2])
        failed = total - passed
        print(f"\n{'=' * 78}\nTONG KET\n{'=' * 78}")
        print(f"Tong so kiem tra : {total}")
        print(f"  Dat (PASS)     : {passed}")
        print(f"  Khong dat (FAIL): {failed}")
        if failed:
            print("\nDanh sach kiem tra KHONG DAT:")
            for section, name, ok, detail in self.results:
                if not ok:
                    print(f"  - [{section}] {name}: {detail}")
        print()
        return failed == 0


VERBOSE_TRACEBACK = False
R = Reporter()


@contextmanager
def no_grad_eval(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield
    model.train(was_training)


def count_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters())


def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


# ==============================================================================
# 1. KIEM TRA TUNG BLOCK CO BAN (blocks.py)
# ==============================================================================
def test_basic_blocks(device):
    R.section("1. KIEM TRA TUNG BLOCK CO BAN (blocks.py)")

    x = torch.randn(2, 16, 32, 32, device=device)

    def t_conv():
        m = Conv(16, 32, 3, 2).to(device)
        y = m(x)
        assert y.shape == (2, 32, 16, 16), f"shape sai: {y.shape}"
        assert not has_nan_or_inf(y), "output co NaN/Inf"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "Conv (stride 2, downsample)", t_conv)

    def t_dwconv():
        m = DWConv(16, 32, 3, 1).to(device)
        y = m(x)
        assert y.shape == (2, 32, 32, 32)
        return f"out={tuple(y.shape)}"
    R.check("blocks", "DWConv", t_dwconv)

    def t_bottleneck_add():
        m = Bottleneck(16, 16, shortcut=True).to(device)
        y = m(x)
        assert y.shape == x.shape
        assert m.add is True, "shortcut phai duoc bat khi c1==c2 va shortcut=True"
        return "residual add OK"
    R.check("blocks", "Bottleneck (co residual, c1==c2)", t_bottleneck_add)

    def t_bottleneck_noadd():
        m = Bottleneck(16, 32, shortcut=True).to(device)
        assert m.add is False, "khong duoc cong residual khi c1 != c2"
        y = m(x)
        assert y.shape == (2, 32, 32, 32)
        return "khong residual khi lech kenh, OK"
    R.check("blocks", "Bottleneck (khac kenh -> tu tat residual)", t_bottleneck_noadd)

    def t_c2f():
        m = C2f(16, 32, n=3, shortcut=True).to(device)
        y = m(x)
        assert y.shape == (2, 32, 32, 32)
        n_bottleneck = len(m.m)
        assert n_bottleneck == 3, f"so luong Bottleneck sai: {n_bottleneck}"
        return f"out={tuple(y.shape)}, n_blocks={n_bottleneck}"
    R.check("blocks", "C2f (n=3)", t_c2f)

    def t_cib():
        m = CIB(16, 16, shortcut=True).to(device)
        y = m(x)
        assert y.shape == x.shape
        return f"out={tuple(y.shape)}"
    R.check("blocks", "CIB", t_cib)

    def t_c2fcib():
        m = C2fCIB(16, 32, n=2, shortcut=False).to(device)
        y = m(x)
        assert y.shape == (2, 32, 32, 32)
        assert all(isinstance(b, CIB) for b in m.m), "C2fCIB.m phai la danh sach cac CIB"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "C2fCIB (n=2)", t_c2fcib)

    def t_sppf():
        m = SPPF(16, 32, k=5).to(device)
        y = m(x)
        assert y.shape == (2, 32, 32, 32)
        return f"out={tuple(y.shape)}"
    R.check("blocks", "SPPF", t_sppf)

    def t_scdown_s1():
        m = SCDown(16, 32, 3, 1).to(device)
        y = m(x)
        assert y.shape == (2, 32, 32, 32), "SCDown stride=1 phai giu nguyen H,W"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "SCDown (stride=1)", t_scdown_s1)

    def t_scdown_s2():
        m = SCDown(16, 32, 3, 2).to(device)
        y = m(x)
        assert y.shape == (2, 32, 16, 16), "SCDown stride=2 phai giam 1/2 H,W"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "SCDown (stride=2)", t_scdown_s2)

    def t_attention_divisible():
        m = Attention(dim=64, num_heads=4).to(device)
        xin = torch.randn(2, 64, 8, 8, device=device)
        y = m(xin)
        assert y.shape == xin.shape, "Attention phai giu nguyen shape (residual)"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "Attention (dim chia het num_heads)", t_attention_divisible)

    def t_attention_not_divisible():
        raised = False
        try:
            Attention(dim=66, num_heads=4)
        except AssertionError:
            raised = True
        assert raised, "Attention phai raise AssertionError khi dim % num_heads != 0"
        return "raise dung nhu ky vong khi dim khong chia het cho num_heads"
    R.check("blocks", "Attention (dim KHONG chia het -> phai bao loi)", t_attention_not_divisible)

    def t_c2fpsa():
        m = C2fPSA(64, 64, n=1, e=0.5).to(device)
        xin = torch.randn(2, 64, 8, 8, device=device)
        y = m(xin)
        assert y.shape == xin.shape
        return f"out={tuple(y.shape)}"
    R.check("blocks", "C2fPSA", t_c2fpsa)

    def t_dfl_shape():
        reg_max = 16
        m = DFL(reg_max).to(device)
        a = 100  # so anchor
        reg = torch.randn(2, 4 * reg_max, a, device=device)
        y = m(reg)
        assert y.shape == (2, 4, a), f"shape DFL sai: {y.shape}"
        for p in m.parameters():
            assert p.requires_grad is False, "trong so DFL phai duoc dong bang (fixed conv)"
        return f"out={tuple(y.shape)}, requires_grad=False (dung)"
    R.check("blocks", "DFL (shape + trong so co dinh)", t_dfl_shape)

    def t_dfl_expectation_range():
        # Voi logit dong nhat (uniform), expectation phai xap xi trung binh (0..reg_max-1)/2
        reg_max = 16
        m = DFL(reg_max).to(device)
        a = 10
        reg = torch.zeros(1, 4 * reg_max, a, device=device)  # logits bang nhau -> softmax uniform
        y = m(reg)
        expected = (reg_max - 1) / 2.0
        max_err = (y - expected).abs().max().item()
        assert max_err < 1e-3, f"DFL voi logits dong nhat phai cho ra trung binh cong ~{expected}, sai lech {max_err}"
        return f"expectation~{expected:.2f}, sai_lech_max={max_err:.2e}"
    R.check("blocks", "DFL (kiem tra dung cong thuc expectation)", t_dfl_expectation_range)


# ==============================================================================
# 2. KIEM TRA BACKBONE / NECK / HEAD RIENG LE
# ==============================================================================
def test_submodules(device, imgsz, backbone_w, backbone_n, neck_n, nc, reg_max, strides):
    R.section("2. KIEM TRA BACKBONE / PAFPN / DETECTHEAD RIENG LE")

    x = torch.randn(2, 3, imgsz, imgsz, device=device)
    c0, c1, c2, c3, c4 = backbone_w

    backbone = Backbone(w=backbone_w, n=backbone_n).to(device)

    def t_backbone_shapes():
        p3, p4, p5 = backbone(x)
        s = imgsz
        assert p3.shape == (2, c2, s // 8, s // 8), f"p3 sai shape: {p3.shape}"
        assert p4.shape == (2, c3, s // 16, s // 16), f"p4 sai shape: {p4.shape}"
        assert p5.shape == (2, c4, s // 32, s // 32), f"p5 sai shape: {p5.shape}"
        return f"p3={tuple(p3.shape)}, p4={tuple(p4.shape)}, p5={tuple(p5.shape)}"
    R.check("backbone", "Backbone stride 8/16/32", t_backbone_shapes)

    with torch.no_grad():
        p3, p4, p5 = backbone(x)

    neck = PAFPN(chs=(c2, c3, c4), n=neck_n).to(device)

    def t_neck_shapes():
        with torch.no_grad():
            n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape == p3.shape, f"neck p3_out shape khong khop backbone p3: {n3.shape} vs {p3.shape}"
        assert n4.shape == p4.shape, f"neck p4_out shape khong khop backbone p4: {n4.shape} vs {p4.shape}"
        assert n5.shape == p5.shape, f"neck p5_out shape khong khop backbone p5: {n5.shape} vs {p5.shape}"
        return f"n3={tuple(n3.shape)}, n4={tuple(n4.shape)}, n5={tuple(n5.shape)}"
    R.check("neck", "PAFPN giu nguyen so kenh dau ra so voi dau vao", t_neck_shapes)

    with torch.no_grad():
        n3, n4, n5 = neck(p3, p4, p5)

    head = DetectHead(chs=(c2, c3, c4), nc=nc, reg_max=reg_max, strides=strides).to(device)

    def t_head_train_shapes():
        head.train()
        out = head([n3, n4, n5])
        total_anchors = sum((imgsz // s) * (imgsz // s) for s in strides)
        assert "o2m" in out and "o2o" in out, "che do train phai co ca o2m va o2o"
        for key in ("o2m", "o2o"):
            assert out[key]["cls"].shape == (2, total_anchors, nc), f"{key} cls sai shape: {out[key]['cls'].shape}"
            assert out[key]["box"].shape == (2, total_anchors, 4), f"{key} box sai shape: {out[key]['box'].shape}"
        assert out["anchors"].shape == (total_anchors, 2)
        assert out["strides"].shape == (total_anchors, 1)
        return f"total_anchors={total_anchors}, cls={tuple(out['o2o']['cls'].shape)}"
    R.check("head", "DetectHead che do TRAIN (co o2m + o2o)", t_head_train_shapes)

    def t_head_eval_shapes():
        head.eval()
        with torch.no_grad():
            out = head([n3, n4, n5])
        assert "o2m" not in out, "che do eval KHONG duoc tra ve nhanh o2m (de toi uu toc do)"
        assert "o2o" in out
        return "eval mode chi tra ve o2o (dung thiet ke NMS-free 1-to-1)"
    R.check("head", "DetectHead che do EVAL (chi o2o, tiet kiem tai nguyen)", t_head_eval_shapes)

    def t_cls_bias_init():
        import math
        prior = 0.01
        expected_bias = -math.log((1 - prior) / prior)
        for sh in head.heads:
            for m in (sh.cls_o2m, sh.cls_o2o):
                err = (m.bias - expected_bias).abs().max().item()
                assert err < 1e-4, f"bias khoi tao cls sai: ky vong {expected_bias}, lech {err}"
        return f"bias init = {expected_bias:.4f} (prior_prob=0.01, dung chuan RetinaNet/YOLO)"
    R.check("head", "Khoi tao bias lop phan loai (focal-loss prior)", t_cls_bias_init)

    def t_anchor_formula():
        feats = [n3, n4, n5]
        anchors, stride_t = DetectHead.make_anchors(feats, strides)
        # tu tinh tay va so sanh
        exp_anchors = []
        exp_strides = []
        for f, s in zip(feats, strides):
            h, w = f.shape[-2:]
            for yy in range(h):
                for xx in range(w):
                    exp_anchors.append((xx + 0.5, yy + 0.5))
                    exp_strides.append(s)
        exp_anchors_t = torch.tensor(exp_anchors, dtype=torch.float, device=device)
        exp_strides_t = torch.tensor(exp_strides, dtype=torch.float, device=device).view(-1, 1)
        assert torch.allclose(anchors, exp_anchors_t), "toa do anchor khong khop cong thuc thu cong"
        assert torch.allclose(stride_t, exp_strides_t), "stride cua anchor khong khop"
        return f"{anchors.shape[0]} anchor(s) khop 100% voi tinh tay"
    R.check("head", "make_anchors dung cong thuc (offset 0.5)", t_anchor_formula)


# ==============================================================================
# 3. KIEM TRA TOAN BO PIPELINE (model.py)
# ==============================================================================
def test_full_model(device, imgsz, backbone_w, backbone_n, neck_n, nc, reg_max, strides):
    R.section("3. KIEM TRA TOAN BO PIPELINE NMSFreeDetector")

    model = NMSFreeDetector(
        nc=nc, reg_max=reg_max, backbone_w=backbone_w,
        backbone_n=backbone_n, neck_n=neck_n, strides=strides,
    ).to(device)

    total_anchors = sum((imgsz // s) * (imgsz // s) for s in strides)

    def t_forward_eval_multi_batch():
        model.eval()
        details = []
        for bs in (1, 2, 8):
            x = torch.randn(bs, 3, imgsz, imgsz, device=device)
            with torch.no_grad():
                out = model(x)
            assert out["o2o"]["cls"].shape == (bs, total_anchors, nc)
            assert out["o2o"]["box"].shape == (bs, total_anchors, 4)
            assert not has_nan_or_inf(out["o2o"]["cls"]), f"NaN/Inf trong cls (batch={bs})"
            assert not has_nan_or_inf(out["o2o"]["box"]), f"NaN/Inf trong box (batch={bs})"
            details.append(f"bs={bs} OK")
        return ", ".join(details)
    R.check("model", "Forward EVAL voi nhieu batch size (1,2,8), khong NaN/Inf", t_forward_eval_multi_batch)

    def t_forward_train_multi_batch():
        model.train()
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        out = model(x)
        for key in ("o2m", "o2o"):
            assert out[key]["cls"].shape == (2, total_anchors, nc)
            assert out[key]["box"].shape == (2, total_anchors, 4)
        return "train mode: o2m & o2o dung shape"
    R.check("model", "Forward TRAIN (co o2m/o2o)", t_forward_train_multi_batch)

    def t_input_size_must_divide_32():
        model.eval()
        raised = False
        x_bad = torch.randn(1, 3, imgsz + 1, imgsz - 1, device=device)  # khong chia het 32
        try:
            with torch.no_grad():
                model(x_bad)
        except RuntimeError:
            raised = True
        assert raised, (
            "KY VONG loi khi input khong chia het cho 32 (do upsample/concat trong PAFPN). "
            "Neu khong loi, kiem tra lai logic PAFPN."
        )
        return "xac nhan dung: input KHONG chia het 32 se gay loi shape mismatch trong PAFPN (nhu du kien)"
    R.check(
        "model",
        "[CANH BAO THIET KE] input phai la boi so cua 32 (stride tong = 2^5)",
        t_input_size_must_divide_32,
    )

    def t_backward_no_nan():
        model.train()
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, 3, imgsz, imgsz, device=device)
        out = model(x)
        loss = (
            out["o2m"]["cls"].sum() + out["o2m"]["box"].sum()
            + out["o2o"]["cls"].sum() + out["o2o"]["box"].sum()
        )
        loss.backward()
        none_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nan_grad = [n for n, p in model.named_parameters() if p.grad is not None and has_nan_or_inf(p.grad)]
        assert not none_grad, f"co {len(none_grad)} tham so khong nhan gradient: {none_grad[:5]}..."
        assert not nan_grad, f"co {len(nan_grad)} tham so co NaN/Inf trong gradient: {nan_grad[:5]}..."
        return f"backward OK, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} tham so co grad hop le"
    R.check("model", "Backward pass: khong None-grad, khong NaN/Inf-grad", t_backward_no_nan)

    def t_save_load_trunk_roundtrip():
        import tempfile, os
        m1 = NMSFreeDetector(nc=nc, reg_max=reg_max, backbone_w=backbone_w,
                              backbone_n=backbone_n, neck_n=neck_n, strides=strides).to(device)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "trunk.pt")
            m1.save_trunk(path)

            m2 = NMSFreeDetector(nc=nc, reg_max=reg_max, backbone_w=backbone_w,
                                  backbone_n=backbone_n, neck_n=neck_n, strides=strides).to(device)
            m2.load_trunk(path, map_location=str(device))

        for (n1, p1), (n2, p2) in zip(m1.backbone.named_parameters(), m2.backbone.named_parameters()):
            assert torch.equal(p1, p2), f"trong so backbone khong khop sau load: {n1}"
        for (n1, p1), (n2, p2) in zip(m1.neck.named_parameters(), m2.neck.named_parameters()):
            assert torch.equal(p1, p2), f"trong so neck khong khop sau load: {n1}"
        return "save_trunk -> load_trunk cho trong so backbone+neck giong het 100%"
    R.check("model", "save_trunk / load_trunk round-trip", t_save_load_trunk_roundtrip)

    def t_freeze_trunk():
        m = NMSFreeDetector(nc=nc, reg_max=reg_max, backbone_w=backbone_w,
                             backbone_n=backbone_n, neck_n=neck_n, strides=strides).to(device)
        m.freeze_trunk(True)
        backbone_frozen = all(not p.requires_grad for p in m.backbone.parameters())
        neck_frozen = all(not p.requires_grad for p in m.neck.parameters())
        # head phai con trainable, TRU tham so co-dinh cua DFL (thiet ke co y, khong phai bug)
        head_trainable = all(
            p.requires_grad for n, p in m.head.named_parameters() if "dfl" not in n
        )
        assert backbone_frozen, "freeze_trunk(True) phai dong bang toan bo backbone"
        assert neck_frozen, "freeze_trunk(True) phai dong bang toan bo neck"
        assert head_trainable, "freeze_trunk(True) KHONG duoc dong bang head (tru DFL co dinh theo thiet ke)"

        m.freeze_trunk(False)
        assert all(p.requires_grad for p in m.backbone.parameters()), "freeze_trunk(False) phai mo lai backbone"
        assert all(p.requires_grad for p in m.neck.parameters()), "freeze_trunk(False) phai mo lai neck"
        return "freeze/unfreeze backbone+neck hoat dong dung, head khong bi anh huong"
    R.check("model", "freeze_trunk(True/False)", t_freeze_trunk)

    def t_replace_head():
        m = NMSFreeDetector(nc=nc, reg_max=reg_max, backbone_w=backbone_w,
                             backbone_n=backbone_n, neck_n=neck_n, strides=strides).to(device)
        new_nc = nc + 5
        m.replace_head(nc=new_nc)
        assert m.nc == new_nc
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out["o2o"]["cls"].shape == (1, total_anchors, new_nc), \
            f"sau replace_head, so lop phai la {new_nc}, nhung ra {out['o2o']['cls'].shape}"
        return f"replace_head(nc={new_nc}) -> output khop dung so lop moi"
    R.check("model", "replace_head thay doi so class dung", t_replace_head)

    def t_config_channel_mismatch_guard():
        # Kiem tra rang neu backbone_w khong chia het cho Attention.num_heads (mac dinh 4)
        # thi model phai bao loi ro rang thay vi silent-fail
        raised = False
        try:
            NMSFreeDetector(backbone_w=(32, 64, 128, 256, 302))  # 302*0.5=151, 151%4 !=0
        except AssertionError:
            raised = True
        assert raised, (
            "khi backbone_w[-1]*0.5 khong chia het cho so head cua Attention (mac dinh 4), "
            "phai bao AssertionError ro rang de nguoi dung biet cach chon w hop le"
        )
        return "xac nhan: can chon backbone_w[-1] sao cho (w[-1]*0.5) % num_heads == 0"
    R.check(
        "model",
        "[RANG BUOC CAU HINH] backbone_w[-1] phai tuong thich Attention.num_heads",
        t_config_channel_mismatch_guard,
    )

    return model, total_anchors


# ==============================================================================
# 4. HIEU QUA: THAM SO, FLOPs
# ==============================================================================
def report_efficiency_params(model, imgsz, device):
    R.section("4. HIEU QUA - THAM SO & FLOPs")

    n_backbone = count_params(model.backbone)
    n_neck = count_params(model.neck)
    n_head = count_params(model.head)
    n_total = count_params(model)

    print(f"  Backbone : {n_backbone:>12,} tham so ({n_backbone/1e6:6.2f} M)")
    print(f"  Neck     : {n_neck:>12,} tham so ({n_neck/1e6:6.2f} M)")
    print(f"  Head     : {n_head:>12,} tham so ({n_head/1e6:6.2f} M)")
    print(f"  TONG     : {n_total:>12,} tham so ({n_total/1e6:6.2f} M)")

    def t_param_sum_consistency():
        assert n_backbone + n_neck + n_head == n_total, "tong tham so tung phan phai bang tong toan mo hinh"
        return "tong khop 100%"
    R.check("efficiency", "Tong tham so backbone+neck+head == tong model", t_param_sum_consistency)

    try:
        from thop import profile
        model.eval()
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        gflops = macs * 2 / 1e9
        print(f"\n  MACs (thop)   : {macs/1e9:.3f} G  (~ {gflops:.3f} GFLOPs @ input {imgsz}x{imgsz})")
        print(f"  Params (thop) : {params/1e6:.2f} M (doi chieu voi {n_total/1e6:.2f} M dem thu cong)")
    except ImportError:
        print("\n  [BO QUA] Chua cai 'thop' -> khong tinh duoc FLOPs.")
        print("  Cai dat bang: pip install thop --break-system-packages")
    except Exception as e:
        print(f"\n  [CANH BAO] Khong tinh duoc FLOPs bang thop: {e}")


# ==============================================================================
# 5. HIEU QUA: TOC DO SUY LUAN
# ==============================================================================
def benchmark_speed(model, imgsz, device, batch_sizes=(1, 4, 8), n_warmup=10, n_iters=50):
    R.section("5. HIEU QUA - TOC DO SUY LUAN (LATENCY / THROUGHPUT)")

    model.eval()
    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    print(f"  Thiet bi: {device} | warmup={n_warmup} vong | do={n_iters} vong\n")
    print(f"  {'Batch':>6} | {'Latency TB (ms)':>16} | {'Latency P90 (ms)':>17} | {'Throughput (img/s)':>19}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*17}-+-{'-'*19}")

    for bs in batch_sizes:
        x = torch.randn(bs, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            sync()

            lat_list = []
            for _ in range(n_iters):
                sync()
                t0 = time.perf_counter()
                model(x)
                sync()
                t1 = time.perf_counter()
                lat_list.append((t1 - t0) * 1000)

        lat_list.sort()
        avg_ms = sum(lat_list) / len(lat_list)
        p90_ms = lat_list[int(0.9 * len(lat_list)) - 1]
        throughput = bs / (avg_ms / 1000)
        print(f"  {bs:>6} | {avg_ms:>16.2f} | {p90_ms:>17.2f} | {throughput:>19.1f}")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(max(batch_sizes), 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            model(x)
        sync()
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\n  Bo nho GPU dinh (peak, batch={max(batch_sizes)}): {peak_mb:.1f} MB")

        # So sanh FP16 neu co CUDA
        try:
            model_half = model.half()
            x_half = torch.randn(max(batch_sizes), 3, imgsz, imgsz, device=device).half()
            with torch.no_grad():
                for _ in range(n_warmup):
                    model_half(x_half)
                sync()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    model_half(x_half)
                sync()
                t1 = time.perf_counter()
            avg_ms_fp16 = (t1 - t0) * 1000 / n_iters
            print(f"  FP16 latency (batch={max(batch_sizes)}): {avg_ms_fp16:.2f} ms/iter")
            model.float()  # tra ve fp32 cho cac buoc sau
        except Exception as e:
            print(f"  [CANH BAO] Khong benchmark duoc FP16: {e}")
            model.float()
    else:
        print("\n  [LUU Y] Khong co CUDA -> bo qua do bo nho GPU va so sanh FP16.")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Validate NMSFreeDetector (correctness + efficiency)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="phai la boi so cua 32")
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--reg_max", type=int, default=16)
    parser.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--backbone_w", type=int, nargs="+", default=[48, 96, 192, 384, 512])
    parser.add_argument("--backbone_n", type=int, nargs="+", default=[2, 4, 4, 2])
    parser.add_argument("--neck_n", type=int, default=2)
    parser.add_argument("--skip-bench", action="store_true", help="bo qua benchmark toc do")
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    global VERBOSE_TRACEBACK
    VERBOSE_TRACEBACK = args.verbose_traceback

    assert args.imgsz % 32 == 0, "--imgsz phai la boi so cua 32 (backbone stride tong = 32)"

    device = torch.device(args.device)
    print(f"Thiet bi su dung: {device}")
    print(f"Cau hinh: imgsz={args.imgsz}, nc={args.nc}, reg_max={args.reg_max}, "
          f"strides={args.strides}, backbone_w={args.backbone_w}, "
          f"backbone_n={args.backbone_n}, neck_n={args.neck_n}")

    torch.manual_seed(0)

    # 1) Test tung block co ban
    test_basic_blocks(device)

    # 2) Test backbone / neck / head rieng le
    test_submodules(
        device, args.imgsz, tuple(args.backbone_w), tuple(args.backbone_n),
        args.neck_n, args.nc, args.reg_max, tuple(args.strides),
    )

    # 3) Test toan bo pipeline
    model, total_anchors = test_full_model(
        device, args.imgsz, tuple(args.backbone_w), tuple(args.backbone_n),
        args.neck_n, args.nc, args.reg_max, tuple(args.strides),
    )

    # 4) Hieu qua: tham so + FLOPs
    report_efficiency_params(model, args.imgsz, device)

    # 5) Hieu qua: toc do
    if not args.skip_bench:
        benchmark_speed(model, args.imgsz, device)
    else:
        R.section("5. BENCHMARK TOC DO - DA BO QUA (--skip-bench)")

    ok = R.summary()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()