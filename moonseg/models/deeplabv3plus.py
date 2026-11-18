"""Official SMP 0.5.0 DeepLabV3+ with the ImageNet MobileNetV3-Large encoder."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from ..utils import matches_github_repository


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'third_party' / 'segmentation_models_pytorch'
DEFAULT_PRETRAINED = ROOT / 'ckpts' / 'deeplabv3plus' / 'mobilenetv3_large_100.ra_in1k.safetensors'
OFFICIAL_COMMIT = '420ce84b0c2df0286fa9bb2bd1499eea625c9b33'
ENCODER_NAME = 'tu-mobilenetv3_large_100'


def _official_smp(source=None):
    source = Path(source or DEFAULT_SOURCE).expanduser().resolve()
    manifest_path = source / 'SOURCE_MANIFEST.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Official SMP source manifest missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if (not matches_github_repository(manifest.get('repository'), 'qubvel-org/segmentation_models.pytorch')
            or manifest.get('commit') != OFFICIAL_COMMIT):
        raise ValueError('DeepLabV3+ requires the recorded official SMP v0.5.0 source revision')
    package = source / 'segmentation_models_pytorch'
    if not (package / 'decoders/deeplabv3/model.py').is_file():
        raise FileNotFoundError(f'Official SMP model source missing below {package}')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        smp = importlib.import_module('segmentation_models_pytorch')
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'Official SMP requires the baseline dependencies, including timm==1.0.17 and '
            'huggingface-hub==0.36.0. Use the project baseline environment.'
        ) from error
    if Path(smp.__file__).resolve().parent != package or smp.__version__ != '0.5.0':
        raise RuntimeError('Another SMP installation is already imported; use a fresh Python process')
    return smp


class _Backbone(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.backbone = encoder

    def forward(self, images):
        return self.backbone(images)


class DeepLabV3PlusMobileNetV3(nn.Module):
    deformable_attention = 'not_used'
    checkpoint_full_state = True

    def __init__(self, settings, load_weights=True):
        super().__init__()
        smp = _official_smp(settings.get('smp_repo'))
        encoder_name = settings.get('encoder_name', ENCODER_NAME)
        if encoder_name != ENCODER_NAME:
            raise ValueError(f'This recorded baseline uses encoder_name={ENCODER_NAME}')
        self.num_classes = int(settings.get('num_classes', 5))
        # Keep the official v0.5.0 defaults: depth 5, output stride 16, decoder
        # width 256, atrous rates (12,24,36), separable ASPP, dropout .5, upsample 4.
        native = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None,
                                  in_channels=3, classes=self.num_classes, activation=None)
        self._native_model_class = type(native)
        self.requires_divisible_input_shape = native.requires_divisible_input_shape
        # Relocate registration only. The encoder, decoder and head are the
        # actual upstream instances, and forward delegates to its original code.
        self.backbone = _Backbone(native.encoder)
        self.decoder = native.decoder
        self.segmentation_head = native.segmentation_head
        self.classification_head = native.classification_head
        self.pretrained_load_info = None
        if load_weights:
            self._load_imagenet(Path(settings.get('backbone_weights', DEFAULT_PRETRAINED)).expanduser())

    @property
    def encoder(self):
        return self.backbone.backbone

    def check_input_shape(self, images):
        return self._native_model_class.check_input_shape(self, images)

    def _load_imagenet(self, path):
        import timm
        from safetensors.torch import load_file

        state = load_file(str(path), device='cpu')
        # Verify the entire first-party ImageNet classifier checkpoint before
        # selecting its feature-extractor tensors. Only its four classifier-head
        # keys are omitted, exactly as timm's features_only model requires.
        classifier = timm.create_model('mobilenetv3_large_100', pretrained=False)
        classifier.load_state_dict(state, strict=True)
        encoder_keys = set(self.encoder.model.state_dict())
        omitted = set(state) - encoder_keys
        expected_omitted = {'conv_head.weight', 'conv_head.bias', 'classifier.weight', 'classifier.bias'}
        if omitted != expected_omitted or encoder_keys - set(state):
            raise ValueError('ImageNet MobileNetV3 checkpoint does not exactly cover the official encoder')
        result = self.encoder.model.load_state_dict({key: state[key] for key in encoder_keys}, strict=True)
        self.pretrained_load_info = {'path': str(path.resolve()), 'keys': len(encoder_keys),
                                     'omitted_head_keys': sorted(omitted),
                                     'missing_keys': result.missing_keys,
                                     'unexpected_keys': result.unexpected_keys}

    def forward(self, images):
        return {'semantic_logits': self._native_model_class.forward(self, images)}


def build_deeplabv3plus(settings, load_weights=True):
    return DeepLabV3PlusMobileNetV3(settings, load_weights=load_weights)


def validate_deeplabv3plus_assets(settings):
    """CPU preflight: pinned official source and strict ImageNet encoder loading."""
    model = build_deeplabv3plus(settings, load_weights=True)
    return {'source_commit': OFFICIAL_COMMIT, 'encoder_name': ENCODER_NAME,
            'pretrained': model.pretrained_load_info,
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad)}
