# DINOv3 + ViT-Adapter + Mask2Former

`moonseg.models.build_model(config, load_backbone=True)` constructs the original
Meta DINOv3 transformer, the official DINOv3 ViT-Adapter, the official deformable
pixel decoder, and the official multi-scale masked transformer decoder. This is
a query/mask model, with spatial-prior convolutions, four deformable extraction
interactions and convolutional FFNs in the adapter. It is not a linear/FPN-only
probe of frozen ViT features.

Input is ImageNet-normalized RGB `[B,3,H,W]`. Output is a dictionary containing
`pred_logits: [B,Q,6]` (five semantic classes plus no-object),
`pred_masks: [B,Q,h,w]` (raw logits), and `aux_outputs` from earlier predictions.
Semantic scores are `einsum('bqc,bqhw->bchw', logits.softmax(-1)[...,:5], masks.sigmoid())`.
Input dimensions are padded to multiples of 32 internally. Non-divisible sizes
return masks cropped to the original field of view; divisible sizes keep stride-4
mask logits. Minimum image size is 64 pixels per dimension.

## Configuration

All model settings belong to `config['model']`:

| Key | Default | Meaning |
| --- | --- | --- |
| `backbone` | `vitb16` | `vitb16` or `vitl16` |
| `backbone_weights` | required when loading | Local native Meta `.pth` state dictionary |
| `dinov3_repo` | repository `third_party/dinov3` | Official source directory |
| `freeze_backbone` | `true` | Freeze only the original ViT; adapter and head train |
| `hidden_dim` | 256 | Pixel/mask/decoder dimensions; multiple of 32 and head count |
| `num_classes` | 5 | Labels 0 through 4 |
| `num_queries` | 100 | Mask queries |
| `encoder_layers` | 6 | Deformable pixel encoder layers |
| `decoder_layers` | 6 | Masked transformer decoder layers |
| `decoder_heads` | 8 | Pixel encoder and mask decoder attention heads |
| `dim_feedforward` | 2048 | Pixel and transformer FFN width |
| `adapter_conv_dim` | 64 | Spatial-prior stem channels |
| `adapter_heads` | 12 / 16 | Adapter deformable heads for ViT-B / ViT-L |
| `adapter_drop_path` | 0.1 | Adapter stochastic depth |
| `gradient_checkpointing` | `true` | Adapter interaction activation checkpointing |
| `adapter_sync_batchnorm` | `false` | Retain upstream SyncBatchNorm for multi-GPU if explicitly enabled |
| `interaction_indexes` | B: `[2,5,8,11]`; L: `[4,11,17,23]` | Four ViT feature extraction layers |

Frozen backbones remain in evaluation mode even when the full model enters
training mode. With `freeze_backbone: false`, the backbone receives gradients;
the optimizer should identify original ViT parameters with
`name.startswith('backbone.backbone.')`. Matching every name containing
`backbone` would incorrectly include the adapter.

Loading uses `pretrained=False` followed by strict native state-dictionary loading,
and never starts a network download. `load_backbone=False` supports checkpoint
restoration and explicitly random-initialized smoke checks; it does not produce
pretrained features. Hugging Face Transformers safetensors require conversion
to the native Meta parameter layout before using this loader.

## Training criterion

`build_criterion(config)` returns a module accepting either semantic targets
`[B,H,W]` with `255` ignored, or a list of dictionaries containing `labels[N]`,
binary `masks[N,H,W]`, and optional `valid_mask[H,W]`. Semantic labels are grouped
into one target mask per class present in the image. Matching uses SciPy's exact
Hungarian solver with class probability, sigmoid BCE and Dice matching costs.
Each decoder prediction gets its own matching and classification, BCE and Dice
losses, including the initial query prediction. This follows Mask2Former's deep
supervision scheme.

The return value is `{'loss': differentiable_weighted_total, 'loss_ce': detached,
'loss_mask': detached, 'loss_dice': detached, ...auxiliary_terms}`. Backpropagate
only `result['loss']`; reported component terms are unweighted diagnostics.
`config['loss']` accepts `ce_weight=2`, `mask_weight=5`, `dice_weight=5`,
`no_object_weight=0.1`, `num_points=12544`, `oversample_ratio=3`,
`importance_sample_ratio=0.75`, and `ignore_index=255`.

Matching shares randomly selected valid pixel centers among all query/target
pairs. Losses emphasize uncertain points and mix in random valid points.
Target sampling is nearest-neighbor, preserving binary targets and excluding
ignored labels exactly. All-ignored images contribute zero classification and
mask loss. Mask count normalization is synchronized for distributed training;
loss/matching math runs in FP32 under mixed precision.

## Source and compatibility changes

Primary source is [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3),
revision `6876159a11b4df116f30f667f8c9888617df0751`. On 2026-10-21,
`python3 tools/download_assets.py --only source` downloaded and extracted the
complete official archive at this revision, replacing the earlier runtime subset.
All 221 upstream files (presence and byte sizes checked against the official
Git tree), original copyright notices and the
[DINOv3 license](../third_party/dinov3/LICENSE.md) are retained.
`third_party/dinov3/SOURCE_MANIFEST.json`, `SOURCE_REVISION` and `SOURCE_COMPLETE`
record the archive URL and fixed revision. This is an archive extraction rather
than a Git checkout.

The original implementation files remain unchanged. MoonSeg supplies these
explicit runtime compatibility adaptations:

- Official `MSDeformAttnFunction.forward` calls the PyTorch core but its backward
  requires a compiled CUDA extension. MoonSeg redirects the Function boundary
  to that same official `ms_deform_attn_core_pytorch` so normal PyTorch autograd
  handles both passes. Grid sampling runs in FP32. No detectron2/mmcv or custom
  CUDA build is needed. Pure PyTorch sampling can be slower and use more memory.
- A subclass uses the original Adapter components and feature flow while removing
  hard-coded CUDA autocast, preserving the outer AMP context, and supporting
  optional ViT fine tuning. The default original ViT remains frozen.
- Upstream SyncBatchNorm is converted to BatchNorm2d by default for one GPU/CPU;
  enable the explicit config option for SyncBatchNorm under CUDA DDP.
- Narrow namespace imports skip the upstream segmentation evaluation package's
  eager data/torchmetrics dependencies. Positional encoding comes directly from
  the complete official source; the former subset fallback has been removed.

Criterion design follows the original
[Mask2Former matcher](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py)
and [criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py),
with valid-pixel sampling added for semantic ignore labels. The criterion is a
local implementation and does not import detectron2.

## Verification and limits

Temporary CPU smoke runs used PyTorch 2.5.1, native official random ViT-B/16 and
the real official adapter/pixel/transformer modules. The default model has
114,357,222 parameters. A 64x96 image returned logits `[1,100,6]`, stride-4 masks
`[1,100,16,24]` and six auxiliary predictions; matching plus all losses completed
backward with finite values and nonzero deformable sampling-offset gradients.
The original frozen ViT had no gradients. Arbitrary-size inference at 65x81
returned masks `[1,100,65,81]`. Independent criterion checks covered tensor and
dictionary targets, ignored-pixel invariance, auxiliary backward and entirely
ignored images. These were temporary checks; no unit-test files were added.

On 2026-10-21, the independent uv environment (managed Python 3.12,
PyTorch 2.7.1+cu126) read the actual downloaded ModelScope ViT-B safetensors,
converted its 211 tensors into 188 native tensors and passed the official
`strict=True` key/shape check. The full model then independently loaded that
native checkpoint with `strict=True`. The adapter and segmentation head started
from random initialization; only the DINOv3 backbone used pretrained weights.

On physical GPU 0, an RTX 6000 Ada 48 GB, two consecutive synthetic batches used
the complete default model and criterion, BF16 autocast, and a frozen backbone:

| Input | Mask logits | Peak allocated | Peak reserved | Forward/backward |
| --- | --- | --- | --- | --- |
| `[1,3,128,128]` | `[1,100,32,32]` | 0.61 GiB | 0.66 GiB | Passed |
| `[1,3,512,1024]` | `[1,100,128,256]` | 3.38 GiB | 4.11 GiB | Passed |

Both produced `[1,100,6]` class logits and six auxiliary predictions. The total
loss and all 395 present trainable gradient tensors were finite. Gradients were
present for 12 deformable sampling-offset weight tensors, and the nonzero-gradient
check passed. The pretrained frozen
backbone had no gradients. The process used
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, ended normally, and took
zero optimizer steps. The detailed record is `ckpts/pretrained_model_probe.json`.
These are single-batch memory observations, excluding optimizer states; they do
not establish the memory needed for larger batches or backbone fine tuning.

No dataset training/validation was run, and no segmentation accuracy or training
throughput is claimed from these checks. DDP verification is tracked separately
with the training pipeline. The upstream project's recommended PyTorch version
is 2.7.1+; the earlier CPU smoke path also worked with PyTorch 2.5.1.
