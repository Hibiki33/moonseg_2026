"""NVlabs' SegFormer-B5, using its original MiT and SegFormerHead definitions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import torch
from torch import nn

from ..utils import matches_github_repository


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'third_party' / 'segformer'
DEFAULT_PRETRAINED = ROOT / 'ckpts' / 'segformer' / 'mit_b5_imagenet.pth'
OFFICIAL_COMMIT = '65fa8cfa9b52b6ee7e8897a98705abf8570f9e32'


def _official_modules(source):
    source = Path(source).expanduser().resolve()
    for filename in ('mix_transformer.py', 'segformer_head.py', 'decode_head.py', 'SOURCE_MANIFEST.json'):
        if not (source / filename).is_file():
            raise FileNotFoundError(f'Official NVlabs SegFormer source missing: {source / filename}')
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
    if (not matches_github_repository(manifest.get('repository'), 'NVlabs/SegFormer')
            or manifest.get('commit') != OFFICIAL_COMMIT):
        raise ValueError('SegFormer requires the recorded official NVlabs source revision')
    package_name = '_moonseg_nvlabs_segformer'
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, source / '__init__.py',
                                                     submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    elif Path(sys.modules[package_name].__file__).resolve().parent != source:
        raise RuntimeError('Another NVlabs SegFormer source path is already imported')
    from importlib import import_module
    return (import_module(package_name + '.mix_transformer'),
            import_module(package_name + '.segformer_head'))


def convert_nvidia_mit_b5_state_dict(state):
    """Invert NVIDIA's Hugging Face tensor renames; never import Transformers.

    Query/key/value are separate in that checkpoint. The official NVlabs model
    stores key and value concatenated along dimension zero in ``attn.kv``.
    Every source tensor is consumed exactly once, except the two ImageNet-only
    classifier tensors, which the official segmentation backbone does not own.
    """
    converted, pending, consumed = {}, {}, set()
    for key, value in state.items():
        if key in ('classifier.weight', 'classifier.bias'):
            expected = (1000, 512) if key.endswith('weight') else (1000,)
            if tuple(value.shape) != expected:
                raise ValueError(f'Unexpected ImageNet classifier shape: {key}: {tuple(value.shape)}')
            continue
        match = re.fullmatch(r'segformer\.encoder\.patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)', key)
        if match:
            stage, part, suffix = match.groups()
            destination = f'patch_embed{int(stage) + 1}.{ "norm" if part == "layer_norm" else part}.{suffix}'
        else:
            match = re.fullmatch(r'segformer\.encoder\.layer_norm\.(\d+)\.(weight|bias)', key)
            if match:
                stage, suffix = match.groups()
                destination = f'norm{int(stage) + 1}.{suffix}'
            else:
                match = re.fullmatch(r'segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)', key)
                if not match:
                    raise ValueError(f'Unmapped NVIDIA MiT-B5 checkpoint key: {key}')
                stage, block, part = match.groups()
                prefix = f'block{int(stage) + 1}.{block}.'
                renames = {'layer_norm_1.': 'norm1.', 'layer_norm_2.': 'norm2.',
                           'attention.self.query.': 'attn.q.', 'attention.self.sr.': 'attn.sr.',
                           'attention.self.layer_norm.': 'attn.norm.',
                           'attention.output.dense.': 'attn.proj.', 'mlp.dense1.': 'mlp.fc1.',
                           'mlp.dense2.': 'mlp.fc2.', 'mlp.dwconv.dwconv.': 'mlp.dwconv.dwconv.'}
                kv_match = re.fullmatch(r'attention\.self\.(key|value)\.(weight|bias)', part)
                if kv_match:
                    component, suffix = kv_match.groups()
                    destination = prefix + 'attn.kv.' + suffix
                    pending.setdefault(destination, {})[component] = value
                    consumed.add(key)
                    continue
                destination = None
                for old, new in renames.items():
                    if part.startswith(old) and part[len(old):] in ('weight', 'bias'):
                        destination = prefix + new + part[len(old):]
                        break
                if destination is None:
                    raise ValueError(f'Unmapped NVIDIA MiT-B5 block key: {key}')
        if destination in converted:
            raise ValueError(f'Duplicate MiT-B5 destination key: {destination}')
        converted[destination] = value
        consumed.add(key)
    for destination, parts in pending.items():
        if set(parts) != {'key', 'value'} or destination in converted:
            raise ValueError(f'Incomplete or duplicate key/value pair: {destination}')
        converted[destination] = torch.cat((parts['key'], parts['value']), dim=0)
    if set(state) - consumed != {'classifier.weight', 'classifier.bias'}:
        raise ValueError('NVIDIA MiT-B5 checkpoint must contain exactly its two ImageNet classifier tensors')
    return converted


class _Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        return self.backbone(images)


class SegFormerB5(nn.Module):
    deformable_attention = 'not_used'
    checkpoint_full_state = True

    def __init__(self, settings, load_weights=True):
        super().__init__()
        upstream, heads = _official_modules(settings.get('segformer_repo', DEFAULT_SOURCE))
        self.num_classes = int(settings.get('num_classes', 5))
        self.backbone = _Backbone(upstream.mit_b5())
        self.decode_head = heads.SegFormerHead(
            in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32], channels=128, dropout_ratio=0.1,
            num_classes=self.num_classes, norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False, decoder_params=dict(embed_dim=768),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.decode_head.init_weights()
        # Upstream BaseDecodeHead owns conv_seg, but SegFormerHead.forward uses
        # linear_pred instead. Keep its original checkpoint tensors, exclude the
        # unused branch from optimization and DDP gradient synchronization.
        self.decode_head.conv_seg.requires_grad_(False)
        self.pretrained_load_info = None
        if load_weights:
            path = Path(settings.get('backbone_weights', settings.get('pretrained', DEFAULT_PRETRAINED))).expanduser()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint.get('state_dict', checkpoint)
            if any(key.startswith('segformer.encoder.') for key in state):
                state = convert_nvidia_mit_b5_state_dict(state)
            result = self.backbone.backbone.load_state_dict(state, strict=True)
            self.pretrained_load_info = {'path': str(path.resolve()), 'keys': len(state),
                                         'missing_keys': result.missing_keys,
                                         'unexpected_keys': result.unexpected_keys}

    def forward(self, images):
        return {'semantic_logits': self.decode_head(self.backbone(images))}


def build_segformer(settings, load_weights=True):
    return SegFormerB5(settings, load_weights=load_weights)


def validate_segformer_assets(settings):
    """CPU preflight: source provenance and strict, complete encoder loading."""
    model = build_segformer(settings, load_weights=True)
    return {'source_commit': OFFICIAL_COMMIT, 'pretrained': model.pretrained_load_info,
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad)}
