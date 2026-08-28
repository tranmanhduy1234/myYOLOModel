import torch.nn as nn
from src.blocks import DFL

def _init_conv2d(m: nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)

def _init_batchnorm2d(m: nn.BatchNorm2d):
    m.eps = 1e-3
    m.momentum = 0.03

    if m.affine:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    if m.track_running_stats:
        m.reset_running_stats()

def _initialize_trainable_layers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, DFL):
            continue
        if isinstance(m, nn.Conv2d):
            if not m.weight.requires_grad:
                continue
            _init_conv2d(m)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                is_trainable = (
                    m.weight.requires_grad or
                    m.bias.requires_grad
                )
                if not is_trainable:
                    continue
            _init_batchnorm2d(m)

def initialize_weights(model: nn.Module):
    _initialize_trainable_layers(model)
    _reinit_head_bias(model)

def initialize_detection_head(head: nn.Module, image_size: int):
    """Chỉ khởi tạo detection head, không tác động backbone hoặc neck."""
    _initialize_trainable_layers(head)
    _reinit_scale_heads(head, image_size=image_size)

def _reinit_scale_heads(head: nn.Module, image_size=None):
    heads_list = getattr(head, 'heads', None)
    strides = getattr(head, 'strides', None)
    if heads_list is None or strides is None:
        return

    for scale_head, stride in zip(heads_list, strides):
        if hasattr(scale_head, '_init_bias'):
            scale_head._init_bias()

        if hasattr(scale_head, 'init_stride_bias'):
            if image_size is None:
                scale_head.init_stride_bias(stride)
            else:
                scale_head.init_stride_bias(stride, img_size=image_size)

def _reinit_head_bias(model: nn.Module):
    head = getattr(model, 'head', None)
    if head is None:
        return
    _reinit_scale_heads(head)