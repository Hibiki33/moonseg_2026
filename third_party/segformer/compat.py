"""Small torch-only replacements for the upstream framework infrastructure.

The MiT and SegFormerHead computations live in the original NVIDIA files.
No timm, mmcv, or mmseg package is installed or imported. DropPath follows
timm 0.3.2's per-sample Bernoulli algorithm; the ConvModule subset implements
the upstream head's Conv2d -> SyncBatchNorm -> ReLU with MMCV parameter names.
AMP is controlled by MoonSeg, rather than the old MMCV fp16 decorators.
"""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F


class _Registry:
    def register_module(self):
        return lambda cls: cls


BACKBONES = _Registry()
HEADS = _Registry()
LOSSES = _Registry()
trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg):
        super().__init__()
        if norm_cfg != dict(type='SyncBN', requires_grad=True):
            raise ValueError('Only the official SegFormer SyncBN configuration is supported')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.SyncBatchNorm(out_channels, eps=1e-5, momentum=0.1)
        self.activate = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def auto_fp16(*args, **kwargs):
    return lambda function: function


force_fp32 = auto_fp16


def build_loss(settings):
    from .cross_entropy_loss import CrossEntropyLoss
    settings = dict(settings)
    if settings.pop('type') != 'CrossEntropyLoss':
        raise ValueError('Only the official cross entropy loss is supported')
    return CrossEntropyLoss(**settings)


def build_pixel_sampler(*args, **kwargs):
    raise ValueError('The official SegFormer configuration does not use a pixel sampler')


def accuracy(pred, target):
    return pred.argmax(1).eq(target).float().mean() * 100


def get_root_logger():
    return logging.getLogger('moonseg.segformer')


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=strict)
    return checkpoint
