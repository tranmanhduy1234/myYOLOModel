"""
init_weights.py
===============
Khởi tạo trọng số cho mô hình NMSFreeDetector (YOLOv10-style).

Quy trình 3 pha:
    1. Quét toàn bộ module → init Kaiming Normal cho Conv2d,
       Constant cho BatchNorm2d.
    2. Bỏ qua DFL (trọng số cố định [0,1,...,reg_max-1], frozen).
    3. Khôi phục bias đặc thù cho các lớp đầu ra classification,
       regression, và landmark (nếu có) — vì pha 1 đã ghi đè bias=0.

Cách dùng:
    from src.utils.init_weights import initialize_weights
    model = NMSFreeDetector()
    initialize_weights(model)
"""

import torch.nn as nn
from src.blocks import DFL


# ---------------------------------------------------------------------------
# Hàm phụ trợ: init từng loại layer
# ---------------------------------------------------------------------------

def _init_conv2d(m: nn.Conv2d):
    """Kaiming Normal (fan_out) cho Conv2d.
    
    - mode='fan_out': giữ phương sai ổn định theo chiều forward,
      phù hợp cho mạng CNN sâu có nhiều nhánh residual.
    - nonlinearity='relu': xấp xỉ tốt nhất cho SiLU/Swish
      (PyTorch chưa hỗ trợ mode='silu' riêng).
    """
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


def _init_batchnorm2d(m: nn.BatchNorm2d):
    """Constant init cho BatchNorm2d theo chuẩn YOLO.
    
    - gamma (weight) = 1, beta (bias) = 0 → BN ban đầu là identity.
    - eps = 1e-3, momentum = 0.03 → giá trị chuẩn YOLO, khác với
      mặc định PyTorch (eps=1e-5, momentum=0.1).
    """
    m.eps = 1e-3
    m.momentum = 0.03
    nn.init.constant_(m.weight, 1.0)
    nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Hàm chính: khởi tạo toàn bộ mô hình
# ---------------------------------------------------------------------------

def initialize_weights(model: nn.Module):
    """Khởi tạo trọng số cho toàn bộ mô hình NMSFreeDetector.

    Gọi SAU khi tạo model, TRƯỚC khi bắt đầu training hoặc load
    checkpoint. Nếu load checkpoint sau đó, checkpoint sẽ ghi đè
    lên các giá trị đã init ở đây (đúng hành vi mong muốn).

    Hỗ trợ cả DetectHead (head.py) và DetectHeadFaceLmk (head_tfl.py).

    Args:
        model: Instance của NMSFreeDetector (hoặc bất kỳ nn.Module nào
               có cấu trúc tương tự).
    """
    # ── Pha 1: Quét toàn bộ module ──────────────────────────────────
    for m in model.modules():
        # Bỏ qua toàn bộ DFL module (Conv2d bên trong có weight cố định)
        if isinstance(m, DFL):
            continue

        if isinstance(m, nn.Conv2d):
            # Bỏ qua Conv2d frozen (phòng trường hợp có lớp frozen khác)
            if not m.weight.requires_grad:
                continue
            _init_conv2d(m)

        elif isinstance(m, nn.BatchNorm2d):
            _init_batchnorm2d(m)

    # ── Pha 2: Khôi phục bias đặc thù cho Detection Head ───────────
    # Pha 1 đã ghi đè bias=0 cho tất cả Conv2d (bao gồm cls/reg/lmk
    # output), nên cần gọi lại hàm init đặc thù của ScaleHead.
    _reinit_head_bias(model)


def _reinit_head_bias(model: nn.Module):
    """Khôi phục bias đặc thù cho các lớp đầu ra trong DetectHead.

    Tìm thuộc tính `model.head` (DetectHead hoặc DetectHeadFaceLmk),
    duyệt qua từng ScaleHead và gọi lại:
      - _init_bias()       → prior bias cho cls & reg (& landmark nếu có)
      - init_stride_bias() → stride-aware bias cho cls

    Nếu model không có thuộc tính `head` hoặc `head.heads`, hàm này
    sẽ không làm gì (an toàn khi gọi trên model bất kỳ).
    """
    head = getattr(model, 'head', None)
    if head is None:
        return

    heads_list = getattr(head, 'heads', None)
    strides = getattr(head, 'strides', None)
    if heads_list is None or strides is None:
        return

    for scale_head, stride in zip(heads_list, strides):
        # Gọi _init_bias() → set bias cls (focal prior) & reg (1.0)
        #                      & landmark (0.0) nếu là ScaleHeadFaceLmk
        if hasattr(scale_head, '_init_bias'):
            scale_head._init_bias()

        # Gọi init_stride_bias() → ghi đè bias cls theo stride
        # (chính xác hơn prior cố định 0.01)
        if hasattr(scale_head, 'init_stride_bias'):
            scale_head.init_stride_bias(stride)
