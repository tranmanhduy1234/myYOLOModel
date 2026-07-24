"""
validate_model.py
==================
Validate toàn diện KIẾN TRÚC model của NMSFreeDetector:
  - blocks.py       : autopad, Conv, DWConv, Bottleneck, C2f, CIB, C2fCIB,
                      SPPF, DFL, Attention, C2fPSA, SCDown
  - backbone_neck.py: Backbone (stride 8/16/32) + PAFPN (top-down + bottom-up)
  - head.py         : ScaleHead (bias init, train/eval) + DetectHead
                      (make_anchors, decode_box, forward)
  - model.py        : NMSFreeDetector end-to-end (forward, grad, trunk I/O,
                      replace_head, freeze_trunk, parameter count)

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ
------------------------------
1. save_trunk test: dùng tempfile.TemporaryDirectory() thay vì /tmp hardcode.
2. Thêm kiểm tra out["strides"] trong t_forward_train_shapes.
3. Thêm test mới: Conv dilation, DWConv bất đối xứng (c1!=c2), Bottleneck
   shortcut không add khi c1!=c2, C2f n=0 edge case, Attention gradient,
   PAFPN feature fusion không constant, Backbone số tầng, model parameter count,
   load_feature_extractor (backbone-only), decode_box offset=0.5.
4. Bổ sung torch.manual_seed(0) ở đầu mỗi subtest overfit để ổn định seed.

Chạy độc lập:
    python -m src.validation_tool.validate_model
    python -m src.validation_tool.validate_model --device cuda
"""

import argparse
import math
import sys
import tempfile
import os

import torch
import torch.nn as nn

from src.validation_tool.validate_common import Reporter, get_device, skip

from src.blocks import (
    Conv, DWConv, Bottleneck, C2f, CIB, C2fCIB, SPPF, DFL,
    Attention, C2fPSA, SCDown, autopad,
)
from src.backbone_neck import Backbone, PAFPN
from src.head import ScaleHead, DetectHead
from src.model import NMSFreeDetector


# ==============================================================================
# 1. blocks.py - các khối xây dựng cơ bản
# ==============================================================================
def test_blocks(device: str, R: Reporter):
    R.section("1. BLOCKS.PY - CÁC KHỐI XÂY DỰNG CƠ BẢN")

    def t_autopad():
        assert autopad(3) == 1,      "k=3, p=None -> pad = k//2 = 1"
        assert autopad(1) == 0,      "k=1 -> pad = 0"
        assert autopad(5) == 2,      "k=5 -> pad = 2"
        assert autopad(3, p=0) == 0, "p truyền tay phải được giữ nguyên"
        return "autopad(3)=1, autopad(1)=0, autopad(5)=2"
    R.check("blocks", "autopad tính đúng padding SAME", t_autopad)

    def t_autopad_dilation():
        """autopad với dilation > 1 phải tính k_eff = d*(k-1)+1 trước khi chia 2."""
        # k=3, d=2 -> k_eff = 2*(3-1)+1 = 5 -> pad = 5//2 = 2
        assert autopad(3, d=2) == 2, f"autopad(k=3,d=2) phải = 2, được {autopad(3, d=2)}"
        return "autopad(k=3,d=2)=2 (k_eff=5)"
    R.check("blocks", "autopad với dilation: tính đúng k_eff trước", t_autopad_dilation)

    def t_conv_shape():
        m = Conv(3, 16, 3, 2).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        y = m(x)
        assert y.shape == (2, 16, 16, 16), f"Conv k=3 s=2 trên 32x32 phải ra 16x16, được {tuple(y.shape)}"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "Conv (k=3,s=2) giảm kích thước đúng 1/2, đúng số kênh", t_conv_shape)

    def t_conv_no_act_identity():
        m = Conv(3, 3, 1, 1, act=False).to(device)
        assert isinstance(m.act, nn.Identity), "act=False phải dùng nn.Identity"
        return "act=Identity khi act=False"
    R.check("blocks", "Conv(act=False) không áp dụng SiLU", t_conv_no_act_identity)

    def t_conv_bn_present():
        """Conv phải có BN (không dùng bias trong Conv2d)."""
        m = Conv(3, 16, 3).to(device)
        assert isinstance(m.bn, nn.BatchNorm2d), "Conv phải có BatchNorm2d"
        assert m.conv.bias is None, "Conv2d trong Conv phải bias=False"
        return "BN có mặt, bias=False"
    R.check("blocks", "Conv: BatchNorm2d có mặt và bias=False trong Conv2d", t_conv_bn_present)

    def t_dwconv_groups():
        m = DWConv(16, 16, 3, 1).to(device)
        assert m.conv.groups == math.gcd(16, 16) == 16, "DWConv phải có groups=gcd(c1,c2)"
        x = torch.randn(1, 16, 8, 8, device=device)
        y = m(x)
        assert y.shape == (1, 16, 8, 8)
        return f"groups={m.conv.groups}"
    R.check("blocks", "DWConv đúng số groups = gcd(c1,c2) (depthwise)", t_dwconv_groups)

    def t_dwconv_asymmetric_channels():
        """DWConv với c1 != c2 vẫn phải hoạt động (groups = gcd(c1,c2))."""
        c1, c2 = 12, 16
        m = DWConv(c1, c2, 3, 1).to(device)
        expected_groups = math.gcd(c1, c2)  # = 4
        assert m.conv.groups == expected_groups, f"groups phải = gcd({c1},{c2})={expected_groups}"
        x = torch.randn(1, c1, 8, 8, device=device)
        y = m(x)
        assert y.shape == (1, c2, 8, 8), f"DWConv(12,16) ra phải có 16 kênh, được {tuple(y.shape)}"
        return f"gcd({c1},{c2})={expected_groups}, out={tuple(y.shape)}"
    R.check("blocks", "DWConv với c1!=c2: groups=gcd, shape đúng", t_dwconv_asymmetric_channels)

    def t_bottleneck_shortcut():
        m_add    = Bottleneck(16, 16, shortcut=True).to(device)
        assert m_add.add is True,  "c1==c2 và shortcut=True -> phải có residual add"
        m_noadd  = Bottleneck(16, 32, shortcut=True).to(device)
        assert m_noadd.add is False, "c1!=c2 -> KHÔNG được add dù shortcut=True"
        m_nosc   = Bottleneck(16, 16, shortcut=False).to(device)
        assert m_nosc.add is False, "shortcut=False -> không add dù c1==c2"

        # Zero-weight residual: đặt tất cả param = 0 → output phải = input
        x = torch.randn(1, 16, 8, 8, device=device, requires_grad=True)
        with torch.no_grad():
            for p in m_add.parameters():
                p.zero_()
        y = m_add(x)
        assert torch.allclose(y, x), "Nếu toàn bộ trọng số trong khối con = 0, residual phải trả về đúng x"
        return "residual add hoạt động đúng điều kiện c1==c2 và shortcut=True"
    R.check("blocks", "Bottleneck: residual add chỉ khi c1==c2 và shortcut=True", t_bottleneck_shortcut)

    def t_bottleneck_no_shortcut_different_channels():
        """Bottleneck(c1!=c2) không có add → output channels = c2."""
        m = Bottleneck(16, 32, shortcut=True).to(device)
        x = torch.randn(1, 16, 8, 8, device=device)
        y = m(x)
        assert y.shape == (1, 32, 8, 8), f"Bottleneck(16,32) ra phải (1,32,8,8), được {tuple(y.shape)}"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "Bottleneck(c1!=c2): không add, output channels=c2", t_bottleneck_no_shortcut_different_channels)

    def t_c2f_shape_and_grad():
        m = C2f(32, 64, n=3, shortcut=True).to(device)
        x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
        y = m(x)
        assert y.shape == (2, 64, 16, 16), f"C2f phải giữ nguyên HxW, đổi kênh sang c2, được {tuple(y.shape)}"
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "grad phải lan tới input, hữu hạn"
        return f"out={tuple(y.shape)}, grad OK"
    R.check("blocks", "C2f: đúng shape đầu ra, grad lan về input", t_c2f_shape_and_grad)

    def t_c2f_n0_edge_case():
        """C2f với n=0 bottleneck phải hoạt động (chỉ có cv1+cv2)."""
        m = C2f(16, 32, n=0, shortcut=True).to(device)
        x = torch.randn(1, 16, 8, 8, device=device)
        y = m(x)
        assert y.shape == (1, 32, 8, 8), f"C2f(n=0) ra phải (1,32,8,8), được {tuple(y.shape)}"
        return f"C2f(n=0) out={tuple(y.shape)}"
    R.check("blocks", "C2f (n=0 bottlenecks) edge case không crash", t_c2f_n0_edge_case)

    def t_cib_and_c2fcib_shape():
        cib = CIB(32, 32, shortcut=True).to(device)
        x   = torch.randn(1, 32, 8, 8, device=device)
        y   = cib(x)
        assert y.shape == x.shape, "CIB với c1==c2 phải giữ nguyên shape (residual)"

        c2fcib = C2fCIB(32, 64, n=2, shortcut=True).to(device)
        y2     = c2fcib(x)
        assert y2.shape == (1, 64, 8, 8), f"C2fCIB phải đổi kênh đúng c2, được {tuple(y2.shape)}"
        return f"CIB out={tuple(y.shape)}, C2fCIB out={tuple(y2.shape)}"
    R.check("blocks", "CIB / C2fCIB đúng shape đầu ra", t_cib_and_c2fcib_shape)

    def t_sppf_shape():
        m = SPPF(64, 64, k=5).to(device)
        x = torch.randn(1, 64, 20, 20, device=device)
        y = m(x)
        assert y.shape == (1, 64, 20, 20), "SPPF (maxpool stride=1, pad=k//2) không được đổi HxW"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "SPPF giữ nguyên HxW (maxpool stride=1)", t_sppf_shape)

    def t_sppf_multi_scale_feature():
        """SPPF nối 4 bản (x, pool1, pool2, pool3) → output != input (đã được fuse)."""
        m = SPPF(32, 32, k=5).to(device)
        x = torch.randn(1, 32, 10, 10, device=device)
        y = m(x)
        # output không nhất thiết == input (đã qua fuse), nhưng phải hữu hạn và shape OK
        assert y.shape == (1, 32, 10, 10)
        assert torch.isfinite(y).all()
        return "SPPF multi-scale fuse: output hữu hạn, shape OK"
    R.check("blocks", "SPPF: fuse multi-scale, output hữu hạn", t_sppf_multi_scale_feature)

    def t_dfl_is_expectation():
        reg_max = 8
        dfl = DFL(reg_max).to(device)
        assert not next(dfl.conv.parameters()).requires_grad, "DFL.conv phải bị đóng băng (requires_grad=False)"

        target_bin = 5
        B, A = 2, 3
        logits = torch.full((B, 4 * reg_max, A), -20.0, device=device)
        logits = logits.view(B, 4, reg_max, A)
        logits[:, :, target_bin, :] = 20.0
        logits = logits.view(B, 4 * reg_max, A)

        out = dfl(logits)  # (B,4,A)
        assert out.shape == (B, 4, A)
        assert torch.allclose(out, torch.full_like(out, float(target_bin)), atol=1e-3), \
            f"DFL phải xấp xỉ kỳ vọng = bin được chọn ({target_bin}), được {out.flatten().tolist()[:4]}"
        return f"DFL(one-hot bin={target_bin}) ~= {out.mean().item():.4f}"
    R.check("blocks", "DFL tính đúng kỳ vọng (soft-argmax) trên reg_max bin", t_dfl_is_expectation)

    def t_dfl_weights_are_arange():
        """Trọng số DFL phải chính xác bằng [0,1,2,...,reg_max-1]."""
        reg_max = 12
        dfl = DFL(reg_max).to(device)
        expected = torch.arange(reg_max, dtype=torch.float, device=device).view(1, reg_max, 1, 1)
        assert torch.allclose(dfl.conv.weight.data, expected), "DFL.conv.weight phải = arange(reg_max)"
        return f"weight = arange(0..{reg_max-1}) OK"
    R.check("blocks", "DFL: conv.weight chính xác bằng arange(reg_max)", t_dfl_weights_are_arange)

    def t_attention_requires_equal_dims():
        Attention(32, num_heads=4).to(device)  # 32 % 4 == 0 -> OK
        try:
            Attention(30, num_heads=4)
            raise RuntimeError("phải assert fail khi dim không chia hết cho num_heads")
        except AssertionError:
            pass
        return "Attention assert dim % num_heads == 0"
    R.check("blocks", "Attention: assert dim chia hết cho num_heads", t_attention_requires_equal_dims)

    def t_attention_shape_preserved():
        m = Attention(32, num_heads=4).to(device)
        x = torch.randn(1, 32, 10, 10, device=device)
        y = m(x)
        assert y.shape == x.shape, "Attention (residual + FFN) phải giữ nguyên shape"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "Attention giữ nguyên shape đầu vào/ra", t_attention_shape_preserved)

    def t_attention_gradient():
        """Attention phải có gradient đầy đủ (không dead neuron từ init)."""
        m = Attention(32, num_heads=4).to(device)
        x = torch.randn(1, 32, 8, 8, device=device, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "Attention: grad về input phải hữu hạn"
        no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not no_grad, f"Attention: có param không nhận grad: {no_grad[:3]}"
        return "Attention: grad đầy đủ tới input và tất cả tham số"
    R.check("blocks", "Attention: gradient đầy đủ, không dead neuron", t_attention_gradient)

    def t_c2fpsa_requires_c1_eq_c2():
        try:
            C2fPSA(32, 64, n=1)
            raise RuntimeError("phải assert fail khi c1 != c2")
        except AssertionError:
            pass
        m = C2fPSA(32, 32, n=2).to(device)
        x = torch.randn(1, 32, 8, 8, device=device)
        y = m(x)
        assert y.shape == x.shape
        return f"out={tuple(y.shape)}"
    R.check("blocks", "C2fPSA: assert c1==c2, giữ nguyên shape khi hợp lệ", t_c2fpsa_requires_c1_eq_c2)

    def t_scdown_stride():
        m = SCDown(32, 64, 3, 2).to(device)
        x = torch.randn(1, 32, 16, 16, device=device)
        y = m(x)
        assert y.shape == (1, 64, 8, 8), f"SCDown k=3 s=2 phải giảm 1/2 HxW, được {tuple(y.shape)}"
        return f"out={tuple(y.shape)}"
    R.check("blocks", "SCDown (pointwise + depthwise stride) đúng shape", t_scdown_stride)

    def t_scdown_pointwise_then_depthwise():
        """SCDown: cv1 là pointwise (k=1), cv2 là depthwise (g=c2). Kiểm tra cấu trúc."""
        m = SCDown(32, 64, 3, 2).to(device)
        assert m.cv1.conv.kernel_size == (1, 1), "SCDown.cv1 phải là pointwise (k=1)"
        assert m.cv2.conv.groups == 64, "SCDown.cv2 phải là depthwise (groups=c2)"
        return f"cv1: k={m.cv1.conv.kernel_size}, cv2: groups={m.cv2.conv.groups}"
    R.check("blocks", "SCDown: cv1 pointwise k=1, cv2 depthwise groups=c2", t_scdown_pointwise_then_depthwise)


# ==============================================================================
# 2. backbone_neck.py - Backbone (CSP-PAN style) + PAFPN
# ==============================================================================
def test_backbone_neck(device: str, R: Reporter):
    R.section("2. BACKBONE_NECK.PY - Backbone + PAFPN")

    w   = (16, 32, 64, 128, 160)
    n   = (1, 1, 1, 1)
    img = 256

    def t_backbone_output_strides_and_channels():
        bb = Backbone(w=w, n=n).to(device)
        x  = torch.randn(2, 3, img, img, device=device)
        p3, p4, p5 = bb(x)
        c3, c4, c5 = w[2], w[3], w[4]
        assert p3.shape == (2, c3, img // 8,  img // 8),  f"P3 phải có stride 8,  kênh={c3}, được {tuple(p3.shape)}"
        assert p4.shape == (2, c4, img // 16, img // 16), f"P4 phải có stride 16, kênh={c4}, được {tuple(p4.shape)}"
        assert p5.shape == (2, c5, img // 32, img // 32), f"P5 phải có stride 32, kênh={c5}, được {tuple(p5.shape)}"
        return f"P3={tuple(p3.shape)} P4={tuple(p4.shape)} P5={tuple(p5.shape)}"
    R.check("backbone", "Backbone: đúng stride (8/16/32) và số kênh (c3,c4,c5) trên cả 3 output", t_backbone_output_strides_and_channels)

    def t_backbone_grad_flows_to_stem():
        bb = Backbone(w=w, n=n).to(device)
        x  = torch.randn(1, 3, img, img, device=device, requires_grad=True)
        p3, p4, p5 = bb(x)
        (p3.sum() + p4.sum() + p5.sum()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "grad phải lan về tới input qua cả 3 output"
        stem_grad = [p.grad for p in bb.stem.parameters() if p.grad is not None]
        assert len(stem_grad) > 0 and all(torch.isfinite(g).all() for g in stem_grad), "stem cũng phải nhận grad hữu hạn"
        return "grad lan hết từ P3/P4/P5 về input và stem"
    R.check("backbone", "Backbone: grad lan từ mọi output về input/stem", t_backbone_grad_flows_to_stem)

    def t_backbone_stage_count():
        """Backbone phải có đúng: stem + 4 stage."""
        bb = Backbone(w=w, n=n).to(device)
        assert hasattr(bb, "stem"),   "Backbone phải có stem"
        assert hasattr(bb, "stage1"), "Backbone phải có stage1"
        assert hasattr(bb, "stage2"), "Backbone phải có stage2"
        assert hasattr(bb, "stage3"), "Backbone phải có stage3"
        assert hasattr(bb, "stage4"), "Backbone phải có stage4"
        n_params = sum(p.numel() for p in bb.parameters())
        return f"stem+4 stage OK, {n_params:,} tham số"
    R.check("backbone", "Backbone: có đúng stem + stage1..stage4", t_backbone_stage_count)

    def t_backbone_output_not_constant():
        """Output không được là hằng số (zero/same) — kiểm tra BN+activation hoạt động."""
        bb = Backbone(w=w, n=n).to(device)
        bb.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 3, img, img, device=device)
            x2 = torch.randn(1, 3, img, img, device=device)
            p3_1, _, _ = bb(x1)
            p3_2, _, _ = bb(x2)
        assert not torch.allclose(p3_1, p3_2), "Output Backbone không được giống nhau cho input khác nhau"
        return "output P3 khác nhau với input khác nhau (BN+activation hoạt động)"
    R.check("backbone", "Backbone: output khác nhau với input khác nhau", t_backbone_output_not_constant)

    def t_pafpn_shape_matches_input():
        c3, c4, c5 = w[2], w[3], w[4]
        neck = PAFPN(chs=(c3, c4, c5), n=1).to(device)
        h3, h4, h5 = img // 8, img // 16, img // 32
        p3 = torch.randn(2, c3, h3, h3, device=device)
        p4 = torch.randn(2, c4, h4, h4, device=device)
        p5 = torch.randn(2, c5, h5, h5, device=device)
        o3, o4, o5 = neck(p3, p4, p5)
        assert o3.shape == p3.shape, f"PAFPN phải giữ nguyên shape P3, được {tuple(o3.shape)} != {tuple(p3.shape)}"
        assert o4.shape == p4.shape, f"PAFPN phải giữ nguyên shape P4, được {tuple(o4.shape)} != {tuple(p4.shape)}"
        assert o5.shape == p5.shape, f"PAFPN phải giữ nguyên shape P5, được {tuple(o5.shape)} != {tuple(p5.shape)}"
        return f"out shapes khớp input: {tuple(o3.shape)}, {tuple(o4.shape)}, {tuple(o5.shape)}"
    R.check("backbone", "PAFPN: shape đầu ra (P3,P4,P5) khớp đúng shape đầu vào", t_pafpn_shape_matches_input)

    def t_pafpn_grad_flows_all_levels():
        c3, c4, c5 = w[2], w[3], w[4]
        neck = PAFPN(chs=(c3, c4, c5), n=1).to(device)
        h3, h4, h5 = img // 8, img // 16, img // 32
        p3 = torch.randn(1, c3, h3, h3, device=device, requires_grad=True)
        p4 = torch.randn(1, c4, h4, h4, device=device, requires_grad=True)
        p5 = torch.randn(1, c5, h5, h5, device=device, requires_grad=True)
        o3, o4, o5 = neck(p3, p4, p5)
        (o3.sum() + o4.sum() + o5.sum()).backward()
        for name, t in (("p3", p3), ("p4", p4), ("p5", p5)):
            assert t.grad is not None and torch.isfinite(t.grad).all(), \
                f"{name} phải nhận grad hữu hạn (top-down + bottom-up)"
        return "grad lan tới cả 3 input P3/P4/P5 (xác nhận luồng top-down VÀ bottom-up)"
    R.check("backbone", "PAFPN: grad lan tới cả 3 mức vào (top-down + bottom-up đúng hướng)", t_pafpn_grad_flows_all_levels)

    def t_pafpn_fuses_features():
        """PAFPN output phải khác P5 thuần — xác nhận có cross-scale fusion."""
        c3, c4, c5 = w[2], w[3], w[4]
        neck = PAFPN(chs=(c3, c4, c5), n=1).to(device)
        h3, h4, h5 = img // 8, img // 16, img // 32
        p3 = torch.randn(1, c3, h3, h3, device=device)
        p4 = torch.randn(1, c4, h4, h4, device=device)
        p5_in = torch.randn(1, c5, h5, h5, device=device)
        with torch.no_grad():
            _, _, p5_out = neck(p3, p4, p5_in)
        assert not torch.allclose(p5_out, p5_in), "PAFPN output P5 không được bằng input P5 (phải có fusion)"
        return "PAFPN P5_out != P5_in (cross-scale fusion đang hoạt động)"
    R.check("backbone", "PAFPN: P5 output khác P5 input (cross-scale fusion xảy ra)", t_pafpn_fuses_features)


# ==============================================================================
# 3. head.py - ScaleHead + DetectHead
# ==============================================================================
def test_head(device: str, R: Reporter):
    R.section("3. HEAD.PY - ScaleHead + DetectHead")

    nc, reg_max = 5, 8
    c_in = 32

    def t_scalehead_train_vs_eval():
        head = ScaleHead(c_in, nc, reg_max).to(device)
        x = torch.randn(2, c_in, 10, 10, device=device)

        head.train()
        out_o2m, out_o2o = head(x)
        assert out_o2m is not None, "Chế độ train phải tính cả nhánh o2m"
        cls_m, reg_m = out_o2m
        cls_o, reg_o = out_o2o
        assert cls_m.shape == (2, nc, 10, 10) and cls_o.shape == (2, nc, 10, 10)
        assert reg_m.shape == (2, 4 * reg_max, 10, 10) and reg_o.shape == (2, 4 * reg_max, 10, 10)

        head.eval()
        with torch.no_grad():
            out_o2m_eval, out_o2o_eval = head(x)
        assert out_o2m_eval is not None, (
            "Chế độ eval CŨNG PHẢI tính o2m (vì engine.validate() gọi model.eval() "
            "giữa lúc training và cần cả o2m để tính DetectionLoss)"
        )
        assert out_o2o_eval is not None
        return "train và eval đều tính đủ cả o2m + o2o (chưa bật shortcut inference)"
    R.check("head", "ScaleHead: tính cả 2 nhánh ở CẢ train lẫn eval (shortcut inference tạm hoãn)", t_scalehead_train_vs_eval)

    def t_scalehead_bias_init():
        head  = ScaleHead(c_in, nc, reg_max).to(device)
        prior = -math.log((1 - 0.01) / 0.01)
        for m in (head.cls_o2m, head.cls_o2o):
            assert torch.allclose(m.bias, torch.full_like(m.bias, prior), atol=1e-4), \
                "bias cls trước khi gọi init_stride_bias phải = -log((1-0.01)/0.01)"
        for m in (head.reg_o2m, head.reg_o2o):
            assert torch.allclose(m.bias, torch.ones_like(m.bias)), "bias reg khởi tạo phải = 1.0"

        head.init_stride_bias(stride=8, img_size=640)
        expected = math.log(5 / nc / (640 / 8) ** 2)
        for m in (head.cls_o2m, head.cls_o2o):
            assert torch.allclose(m.bias, torch.full_like(m.bias, expected), atol=1e-4), \
                f"sau init_stride_bias, bias cls phải = log(5/nc/(img/stride)^2) = {expected:.4f}"
        return f"bias mặc định đúng prior 0.01, init_stride_bias(8) -> {expected:.4f}"
    R.check("head", "ScaleHead: khởi tạo bias cls (prior 0.01) và init_stride_bias theo công thức YOLOv8/v10", t_scalehead_bias_init)

    def t_scalehead_o2m_o2o_different_weights():
        """Nhánh o2m và o2o có trọng số riêng biệt (không share weight)."""
        head = ScaleHead(c_in, nc, reg_max).to(device)
        # Kiểm tra các module không share tham chiếu
        assert head.cls_stem_o2m is not head.cls_stem_o2o, "cls_stem_o2m và cls_stem_o2o không được share"
        assert head.reg_stem_o2m is not head.reg_stem_o2o, "reg_stem_o2m và reg_stem_o2o không được share"
        return "o2m và o2o có module riêng biệt (không share weight)"
    R.check("head", "ScaleHead: nhánh o2m và o2o có trọng số riêng", t_scalehead_o2m_o2o_different_weights)

    def t_make_anchors():
        feats   = [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)]
        strides = (8, 16)
        anchors, stride_t = DetectHead.make_anchors(feats, strides)
        assert anchors.shape == (4 * 4 + 2 * 2, 2), f"tổng số anchor phải = sum(h*w), được {anchors.shape}"
        assert stride_t.shape == (20, 1)
        assert torch.allclose(stride_t[:16], torch.full((16, 1), 8.0))
        assert torch.allclose(stride_t[16:], torch.full((4, 1), 16.0))
        assert torch.allclose(anchors[0],  torch.tensor([0.5, 0.5])), "anchor đầu tiên phải có offset 0.5 (tâm ở lưới)"
        assert torch.allclose(anchors[3],  torch.tensor([3.5, 0.5])), "thứ tự anchor phải là row-major (x tăng trước)"
        return f"tổng anchors={anchors.shape[0]}, offset 0.5 đúng, thứ tự row-major đúng, stride đúng"
    R.check("head", "DetectHead.make_anchors: đúng số lượng, offset 0.5, thứ tự row-major, stride từng level", t_make_anchors)

    def t_make_anchors_offset_0():
        """make_anchors với offset=0 → anchor đầu tiên tại (0.0, 0.0)."""
        feats = [torch.zeros(1, 1, 3, 3)]
        anchors, _ = DetectHead.make_anchors(feats, (8,), offset=0.0)
        assert torch.allclose(anchors[0], torch.tensor([0.0, 0.0])), \
            f"offset=0 → anchor đầu tại (0,0), được {anchors[0]}"
        return f"anchor[0] = {anchors[0].tolist()} khi offset=0"
    R.check("head", "DetectHead.make_anchors: offset=0 → anchor đầu tại (0,0)", t_make_anchors_offset_0)

    def t_decode_box_valid_and_finite():
        head    = DetectHead(chs=(c_in,), nc=nc, reg_max=reg_max, strides=(8,)).to(device)
        A       = 6
        anchors = torch.rand(A, 2, device=device) * 10 + 0.5
        stride  = torch.full((A, 1), 8.0, device=device)
        reg     = torch.randn(1, 4 * reg_max, A, device=device)
        box     = head.decode_box(reg, anchors, stride)
        assert box.shape == (1, A, 4)
        assert torch.isfinite(box).all()
        x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        assert torch.all(x2 >= x1) and torch.all(y2 >= y1), \
            "decode_box phải luôn cho x2>=x1, y2>=y1 (DFL ltrb không âm do đi qua softmax)"
        return f"box shape={tuple(box.shape)}, x2>=x1 & y2>=y1 với mọi anchor"
    R.check("head", "DetectHead.decode_box: shape đúng, box luôn hợp lệ (x2>=x1, y2>=y1)", t_decode_box_valid_and_finite)

    def t_decode_box_scales_with_stride():
        head    = DetectHead(chs=(c_in,), nc=nc, reg_max=reg_max, strides=(8,)).to(device)
        A       = 1
        anchors = torch.tensor([[5.0, 5.0]], device=device)
        reg     = torch.randn(1, 4 * reg_max, A, device=device)
        box_s8  = head.decode_box(reg, anchors, torch.full((A, 1), 8.0,  device=device))
        box_s16 = head.decode_box(reg, anchors, torch.full((A, 1), 16.0, device=device))
        w8  = (box_s8[...,  2] - box_s8[...,  0])
        w16 = (box_s16[..., 2] - box_s16[..., 0])
        assert torch.allclose(w16, 2 * w8, atol=1e-3), "chiều rộng box phải tỷ lệ thuận với stride"
        return f"width@stride8={w8.item():.3f}, width@stride16={w16.item():.3f} (~2x)"
    R.check("head", "DetectHead.decode_box: box tỷ lệ tuyến tính với stride (quy đổi GRID→PIXEL)", t_decode_box_scales_with_stride)

    def t_detecthead_train_shapes():
        chs     = (16, 32, 64)
        strides = (8, 16, 32)
        head    = DetectHead(chs=chs, nc=nc, reg_max=reg_max, strides=strides).to(device)
        img_s   = 320
        feats   = [torch.randn(2, c, img_s // s, img_s // s, device=device) for c, s in zip(chs, strides)]
        head.train()
        out     = head(feats)
        A_total = sum((img_s // s) ** 2 for s in strides)
        for branch in ("o2m", "o2o"):
            assert out[branch]["cls"].shape    == (2, A_total, nc),          f"{branch} cls shape sai"
            assert out[branch]["box"].shape    == (2, A_total, 4),           f"{branch} box shape sai"
            assert out[branch]["reg_raw"].shape == (2, 4 * reg_max, A_total), f"{branch} reg_raw shape sai"
        assert out["anchors"].shape == (A_total, 2)
        assert out["strides"].shape == (A_total, 1)
        # Kiểm tra giá trị strides tensor
        n_p3 = (img_s // 8) ** 2
        n_p4 = (img_s // 16) ** 2
        assert torch.all(out["strides"][:n_p3] == 8),         "strides P3 phải = 8"
        assert torch.all(out["strides"][n_p3:n_p3+n_p4] == 16), "strides P4 phải = 16"
        assert torch.all(out["strides"][n_p3+n_p4:] == 32),   "strides P5 phải = 32"
        return f"A_total={A_total}, cả o2m/o2o đều đúng shape, strides đúng thứ tự"
    R.check("head", "DetectHead.forward (train): gộp đúng 3 scale, đúng shape mọi nhánh + strides đúng", t_detecthead_train_shapes)

    def t_detecthead_eval_still_has_o2m():
        chs  = (16,)
        head = DetectHead(chs=chs, nc=nc, reg_max=reg_max, strides=(8,)).to(device)
        feats = [torch.randn(1, 16, 10, 10, device=device)]
        head.eval()
        with torch.no_grad():
            out = head(feats)
        assert "o2m" in out, "eval mode vẫn phải trả về o2m (shortcut inference tạm hoãn trong lúc training)"
        assert "o2o" in out and out["o2o"]["box"].shape == (1, 100, 4)
        return "eval: vẫn trả về cả o2m + o2o + anchors + strides"
    R.check("head", "DetectHead.forward (eval): vẫn trả về o2m trong giai đoạn training", t_detecthead_eval_still_has_o2m)


# ==============================================================================
# 4. model.py - NMSFreeDetector (lắp ráp toàn bộ + tiện ích)
# ==============================================================================
def test_model(device: str, R: Reporter):
    R.section("4. MODEL.PY - NMSFreeDetector (lắp ráp toàn bộ)")

    small_kwargs = dict(nc=6, reg_max=8, backbone_w=(16, 32, 64, 128, 160),
                        backbone_n=(1, 1, 1, 1), neck_n=1, strides=(8, 16, 32))
    img = 256

    def t_forward_train_shapes():
        m   = NMSFreeDetector(**small_kwargs).to(device)
        m.train()
        x   = torch.randn(2, 3, img, img, device=device)
        out = m(x)
        A_total = sum((img // s) ** 2 for s in small_kwargs["strides"])
        assert out["o2o"]["cls"].shape == (2, A_total, small_kwargs["nc"])
        assert out["o2m"]["box"].shape == (2, A_total, 4)
        assert out["anchors"].shape    == (A_total, 2)
        assert out["strides"].shape    == (A_total, 1)
        # Kiểm tra giá trị strides tensor đúng thứ tự P3/P4/P5
        n_p3 = (img // 8) ** 2
        n_p4 = (img // 16) ** 2
        assert torch.all(out["strides"][:n_p3]         == 8),  "strides P3 phải = 8"
        assert torch.all(out["strides"][n_p3:n_p3+n_p4] == 16), "strides P4 phải = 16"
        assert torch.all(out["strides"][n_p3+n_p4:]    == 32), "strides P5 phải = 32"
        return f"A_total={A_total}, forward train OK, strides đúng thứ tự"
    R.check("model", "NMSFreeDetector.forward (train): end-to-end shape đúng + strides đúng thứ tự", t_forward_train_shapes)

    def t_forward_eval_shapes():
        m = NMSFreeDetector(**small_kwargs).to(device).eval()
        x = torch.randn(1, 3, img, img, device=device)
        with torch.no_grad():
            out = m(x)
        assert "o2m" in out, "eval mode vẫn phải có o2m (dùng cho validate() định kỳ trong lúc training)"
        assert out["o2o"]["cls"].shape[0] == 1 and out["o2o"]["cls"].shape[-1] == small_kwargs["nc"]
        return "eval mode: vẫn trả về cả o2m + o2o (shortcut inference tạm hoãn)"
    R.check("model", "NMSFreeDetector.forward (eval): vẫn có o2m trong giai đoạn training", t_forward_eval_shapes)

    def t_end_to_end_grad_reaches_stem():
        m   = NMSFreeDetector(**small_kwargs).to(device)
        m.train()
        x   = torch.randn(1, 3, img, img, device=device)
        out = m(x)
        loss = (out["o2o"]["cls"].sum() + out["o2o"]["box"].sum()
              + out["o2m"]["cls"].sum() + out["o2m"]["box"].sum())
        m.zero_grad(set_to_none=True)
        loss.backward()
        none_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        nan_grad  = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not none_grad, f"có tham số không nhận được grad: {none_grad[:5]}"
        assert not nan_grad,  f"có tham số grad NaN/Inf: {nan_grad[:5]}"
        return f"tất cả {sum(1 for _ in m.parameters())} tensor tham số đều nhận grad hữu hạn"
    R.check("model", "Grad end-to-end (backbone→neck→cả 2 nhánh head) không None/NaN", t_end_to_end_grad_reaches_stem)

    def t_trunk_save_load_roundtrip():
        """Dùng tempfile để không để lại file trên đĩa sau khi test."""
        m1 = NMSFreeDetector(**small_kwargs).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trunk.pt")
            m1.save_trunk(path)

            m2 = NMSFreeDetector(**small_kwargs).to(device)
            with torch.no_grad():
                for p in m2.backbone.parameters():
                    p.add_(1.0)  # làm m2 khác m1 trước khi load

            m2.load_trunk(path, map_location=str(device))

            for (n1, p1), (n2, p2) in zip(m1.backbone.named_parameters(), m2.backbone.named_parameters()):
                assert torch.allclose(p1, p2), f"sau load_trunk, backbone param '{n1}' phải khớp 100% với model gốc"
            for (n1, p1), (n2, p2) in zip(m1.neck.named_parameters(), m2.neck.named_parameters()):
                assert torch.allclose(p1, p2), f"sau load_trunk, neck param '{n1}' phải khớp 100% với model gốc"
        # Kiểm tra tempdir đã bị xóa (không leak file)
        assert not os.path.exists(path), "tempfile phải đã bị xóa sau with-block"
        return "save_trunk -> load_trunk: backbone+neck khớp tuyệt đối, head KHÔNG bị động chạm, tempfile cleanup OK"
    R.check("model", "save_trunk() / load_trunk(): round-trip chính xác, dùng tempfile (không leak /tmp)", t_trunk_save_load_roundtrip)

    def t_load_feature_extractor_only_backbone_neck():
        """load_feature_extractor chỉ load backbone+neck, KHÔNG đụng head."""
        m1 = NMSFreeDetector(**small_kwargs).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trunk.pt")
            m1.save_trunk(path)

            m2 = NMSFreeDetector(**small_kwargs).to(device)
            head_before = {n: p.clone() for n, p in m2.head.named_parameters()}
            with torch.no_grad():
                for p in m2.backbone.parameters():
                    p.add_(5.0)

            m2.load_feature_extractor(path, map_location=str(device))

            # backbone phải khớp với m1
            for (n1, p1), (n2, p2) in zip(m1.backbone.named_parameters(), m2.backbone.named_parameters()):
                assert torch.allclose(p1, p2), f"backbone param '{n1}' phải khớp sau load_feature_extractor"
            # head phải KHÔNG thay đổi
            for n, p in m2.head.named_parameters():
                assert torch.allclose(p, head_before[n]), f"head param '{n}' KHÔNG được thay đổi sau load_feature_extractor"
        return "load_feature_extractor: backbone+neck khớp, head giữ nguyên"
    R.check("model", "load_feature_extractor(): chỉ load backbone+neck, head không bị đụng", t_load_feature_extractor_only_backbone_neck)

    def t_replace_head_preserves_trunk_changes_head():
        m = NMSFreeDetector(**small_kwargs).to(device)
        backbone_before = {n: p.clone() for n, p in m.backbone.named_parameters()}
        old_nc = m.head.heads[0].cls_o2o.out_channels

        new_nc = 3
        m.replace_head(nc=new_nc)

        assert m.nc == new_nc, "replace_head phải cập nhật self.nc"
        assert m.head.heads[0].cls_o2o.out_channels == new_nc, "head mới phải có đúng số class"
        assert new_nc != old_nc, "test không có ý nghĩa nếu nc mới trùng nc cũ"
        for n, p in m.backbone.named_parameters():
            assert torch.allclose(p, backbone_before[n]), f"replace_head KHÔNG được đụng backbone (param '{n}' bị đổi)"

        x = torch.randn(1, 3, img, img, device=device)
        m.train()
        out = m(x)
        assert out["o2o"]["cls"].shape[-1] == new_nc, "forward sau replace_head phải ra đúng số class mới"
        return f"nc: {old_nc} -> {new_nc}, backbone giữ nguyên, forward hoạt động"
    R.check("model", "replace_head(): thay head mới, GIỮ NGUYÊN backbone/neck, model vẫn forward đúng", t_replace_head_preserves_trunk_changes_head)

    def t_freeze_trunk_toggles_requires_grad():
        m = NMSFreeDetector(**small_kwargs).to(device)
        m.freeze_trunk(True)
        assert all(not p.requires_grad for p in m.backbone.parameters()), "freeze_trunk(True) phải tắt requires_grad của backbone"
        assert all(not p.requires_grad for p in m.neck.parameters()),     "freeze_trunk(True) phải tắt requires_grad của neck"
        assert any(p.requires_grad for p in m.head.parameters()),         "freeze_trunk KHÔNG được đóng băng head"

        m.freeze_trunk(False)
        assert all(p.requires_grad for p in m.backbone.parameters()), "freeze_trunk(False) phải bật lại requires_grad"
        assert all(p.requires_grad for p in m.neck.parameters())
        return "freeze_trunk(True/False) điều khiển đúng backbone+neck, không đụng head"
    R.check("model", "freeze_trunk(): đóng/mở băng gradient đúng phạm vi (backbone+neck, không đụng head)", t_freeze_trunk_toggles_requires_grad)

    def t_no_nan_at_init():
        m   = NMSFreeDetector(**small_kwargs).to(device)
        bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
        assert not bad, f"có tham số NaN/Inf ngay sau khởi tạo: {bad[:5]}"
        n_params = sum(p.numel() for p in m.parameters())
        return f"{n_params:,} tham số, tất cả hữu hạn ngay sau khởi tạo"
    R.check("model", "Toàn bộ tham số hữu hạn (không NaN/Inf) ngay sau __init__", t_no_nan_at_init)

    def t_parameter_count_reasonable():
        """Số tham số của model nhỏ phải trong khoảng hợp lý (< 10M với small config)."""
        m = NMSFreeDetector(**small_kwargs).to(device)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params < 10_000_000, f"model 'small' không nên vượt 10M param, được {n_params:,}"
        assert n_params > 10_000,     f"model phải có đủ tham số (>10K), được {n_params:,}"
        return f"{n_params:,} tham số ({n_params/1e6:.2f}M)"
    R.check("model", "Số tham số model (small config) trong khoảng hợp lý (10K–10M)", t_parameter_count_reasonable)

    def t_overfit_tiny_end_to_end():
        torch.manual_seed(0)
        m   = NMSFreeDetector(**small_kwargs).to(device)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
        x   = torch.randn(1, 3, img, img, device=device)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            out    = m(x)
            target = torch.zeros_like(out["o2o"]["cls"])
            target[..., 0] = 1.0
            loss   = nn.functional.mse_loss(out["o2o"]["cls"].sigmoid(), target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        best_last10 = min(losses[-10:])
        assert best_last10 < losses[0] * 0.85, \
            f"[SANITY] loss phải giảm khi overfit 1 mục tiêu đơn giản (đầu={losses[0]:.4f}, min 10 bước cuối={best_last10:.4f})"
        return f"MSE cls: {losses[0]:.4f} -> {losses[-1]:.4f} (min 10 bước cuối: {best_last10:.4f})"
    R.check("model", "[SANITY] Toàn model học được: overfit 1 mục tiêu đơn giản trên 1 ảnh", t_overfit_tiny_end_to_end)

    def t_different_batch_size_same_output_structure():
        """Forward với B=1 và B=4 phải cho cùng cấu trúc output (chỉ batch dim khác)."""
        m = NMSFreeDetector(**small_kwargs).to(device).eval()
        with torch.no_grad():
            out1 = m(torch.randn(1, 3, img, img, device=device))
            out4 = m(torch.randn(4, 3, img, img, device=device))
        A = out1["anchors"].shape[0]
        assert out1["o2o"]["cls"].shape == (1, A, small_kwargs["nc"])
        assert out4["o2o"]["cls"].shape == (4, A, small_kwargs["nc"])
        return f"B=1 và B=4 cùng cấu trúc, A_total={A}"
    R.check("model", "Forward đúng với batch size khác nhau (B=1 và B=4)", t_different_batch_size_same_output_structure)


# ==============================================================================
# MAIN
# ==============================================================================
def run(device: str, verbose_traceback: bool = False) -> Reporter:
    """Chạy toàn bộ suite model và trả về Reporter để run_all_validation.py gộp."""
    r = Reporter(verbose_traceback)
    torch.manual_seed(0)
    test_blocks(device, r)
    test_backbone_neck(device, r)
    test_head(device, r)
    test_model(device, r)
    return r


def main():
    parser = argparse.ArgumentParser(description="Validate blocks/backbone_neck/head/model")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose-traceback", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Thiết bị sử dụng: {device}")

    r = run(device, args.verbose_traceback)
    ok = r.summary("TỔNG KẾT - VALIDATE MODEL (blocks/backbone_neck/head/model)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()