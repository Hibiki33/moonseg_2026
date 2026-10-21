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
| `freeze_backbone` | `true` | Freeze only the original ViT; adapter and head train. `false` trains the whole ViT |
| `unfreeze_last_blocks` | 0 | With a frozen ViT, train only its last N blocks and final `norm` |
| `deformable_attention` | `auto` | `auto`, `cuda` or `torch`; see [compiled deformable attention](#compiled-deformable-attention) |
| `deformable_attention_extension_dir` | `build/ms_deform_attn` | Compiled extension directory; `MOONSEG_MSDA_DIR` also overrides |
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

The original ViT always remains in evaluation mode, also when the full model
enters training mode and also when the ViT is partly or fully fine tuned. The
hub ViT-B/16 and ViT-L/16 builders set `pos_embed_rope_rescale_coords=2`: in
training mode, RoPE multiplies the patch coordinates of every forward by a random
log-uniform factor in [1/2, 2]. This is DINOv3 pretraining augmentation and must
not run during segmentation fine tuning. The ViT contains no BatchNorm and its
hub drop-path rate is 0, so evaluation mode changes nothing else. (Before
2026-10-21, `freeze_backbone: false` put the ViT into training mode.)

`unfreeze_last_blocks: N` keeps `freeze_backbone: true` and makes only
`backbone.blocks[-N:]` and the final `norm` (applied to all four extracted layers)
trainable. Patch embedding, class, storage and mask tokens stay frozen; RoPE
periods are a buffer. Frozen leading blocks record no autograd graph because their
inputs need no gradient. N must not exceed the depth, the last interaction index
must be the final block (the default), and combining N > 0 with
`freeze_backbone: false` is rejected. CPU construction of the default ViT-B model
gave these parameter counts:

| Setting | Trainable | Trainable ViT part |
| --- | ---: | ---: |
| `unfreeze_last_blocks: 0` (default) | 28,687,590 | 0 |
| `unfreeze_last_blocks: 4` | 57,046,758 | 28,359,168 (blocks 8-11, `norm`) |
| `freeze_backbone: false` | 114,357,222 | 85,669,632 |

The optimizer identifies original ViT parameters with
`name.startswith('backbone.backbone.')` and applies `train.backbone_lr_multiplier`.
Matching every name containing `backbone` would incorrectly include the adapter.
Biases, one-dimensional tensors and embedding tables get no weight decay: names
containing `pos_embed`, `cls_token`, `storage_tokens` (DINOv3's register tokens),
`register_tokens`, `mask_token`, `level_embed`, `level_encoding`, `query_feat` or
`query_embed`. Every remaining decayed tensor with two or more dimensions is a
linear or convolution weight (including the `class_embed`/`mask_embed` heads and,
when trained, the patch-embedding convolution). Memory and speed of
`unfreeze_last_blocks: 4` have not been measured; the full-unfreeze measurement in
the [GPU capacity report](audit_2026-10-21/gpu_capacity.md) predates the
evaluation-mode change.

Loading uses `pretrained=False` followed by strict native state-dictionary loading,
and never starts a network download. Training (also on resume), `moonseg.evaluate`
and `moonseg.predict` always build with `load_backbone=True`, because checkpoints
do not contain the frozen ViT (see Checkpoints below). `load_backbone=False` is
only for explicitly random-initialized smoke checks; it does not produce
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

Matching shares one set of randomly selected valid pixel centers among all
query/target pairs of an image, as the upstream matcher shares one point set.
Loss points follow upstream `get_uncertain_point_coords_with_randomness` for each
matched mask independently: every mask draws its own `oversample_ratio x num_points`
candidates, keeps the `importance_sample_ratio` fraction with the smallest
absolute logit, and adds its own random points. Unlike upstream's continuous
coordinates, all points are centers of valid pixels, so ignored pixels never enter
matching or losses. Valid pixel indices are computed once per image per criterion
call (no per-sample `nonzero()`), and only the matched queries are cast to FP32.
Target sampling is nearest-neighbor, preserving binary targets and excluding
ignored labels exactly. All-ignored images contribute zero classification and
mask loss. Mask count normalization is synchronized for distributed training;
loss/matching math runs in FP32 under mixed precision.

The 2026-10-21 sampling change was checked on CPU (finite losses and gradients,
six auxiliary terms, zero sampled ignore pixels over every drawn coordinate, zero
mask gradient for an entirely ignored image) and on one GPU with synthetic
`[7,100,256,256]` BF16 mask logits, 1024x1024 targets and seven predictions.
There, PyTorch's CUDA sync-debug mode reported 225 instead of 407 synchronizing
operations per criterion forward, and criterion forward/backward peak memory fell
from 0.485/0.905 GiB to 0.360/0.713 GiB. Losses are not bit-identical to the
earlier shared-point sampler: over eight seeded CPU inputs the mean total loss
was 70.58 before and 70.63 after (mean relative difference 0.07%).

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
  requires a compiled CUDA extension. MoonSeg replaces that Function boundary in
  memory (process-wide) with either the official compiled
  `MultiScaleDeformableAttention` kernels for both passes, or the official
  `ms_deform_attn_core_pytorch` under normal PyTorch autograd for both passes.
  Both run in FP32. See [compiled deformable attention](#compiled-deformable-attention).
  No detectron2/mmcv is needed.
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

## Compiled deformable attention

`model.deformable_attention` selects the path when the model is built:

- `auto` (default): use the compiled extension when it imports and its
  `build_manifest.json` names the running torch version; otherwise warn and use
  the PyTorch path.
- `cuda`: require the compiled extension; raise with a build hint otherwise.
- `torch`: always the PyTorch `grid_sample` path (the earlier behavior).

The compiled path casts value, sampling locations and attention weights to FP32
with autocast disabled, because the kernels reject BF16 and FP16, and returns the
input dtype; this matches the PyTorch path's FP32 semantics. `im2col_step` is
`gcd(batch, 64)` to satisfy the kernel's divisibility check. CPU tensors always use
the PyTorch path. `build_model` prints the selected path on rank 0, and
`train.py` records it under `runtime` in `resolved_config.json`. The selection
is not part of any state dictionary, so checkpoints load under either path.
Both backward passes use atomics, so bitwise replay is not expected.

`bash scripts/build_ops.sh` builds the extension. It copies the official sources
from `third_party/dinov3/dinov3/eval/segmentation/models/utils/ops` into a
temporary directory (no vendor edits) and compiles with the project interpreter
(`uv run --no-sync python`, or `PYTHON=`), `CUDA_HOME=/usr/local/cuda-12.6` to
match the cu126 PyTorch wheels (`/usr/local/cuda` on this host points at 12.8),
and `TORCH_CUDA_ARCH_LIST="8.6;8.9"` (RTX A6000 and RTX 6000 Ada). Upstream
`setup.py` refuses to build without a visible GPU, so the script exposes
`BUILD_GPU` (default physical GPU 0). It writes
`build/ms_deform_attn/MultiScaleDeformableAttention*.so` and `build_manifest.json`
(torch, CUDA, nvcc, Python, architectures, `.so` and source SHA-256), then imports
the module as a check. `/build/` is gitignored and must be rebuilt after any torch,
CUDA or Python change. It is not installed into `.venv`, because
`uv sync --locked` removes packages that are not in the lock file.

On 2026-10-21 the build took 42 s on this host. With a random-initialized ViT-B,
a `[2,3,256,320]` FP32 input and TF32 disabled, `torch` and `cuda` modes produced
total losses 79.292198 and 79.292191 (relative difference 9.6e-8) and gradient
norms differing by 3.9e-6 relative; `auto` selected `cuda`. Kernel-level
comparisons, FP64 `gradcheck` and full-model synthetic benchmarks are in the
[GPU capacity report](audit_2026-10-21/gpu_capacity.md). At batch 7 x 2 GPUs x 5
accumulation it measured 30.85/35.37 GiB peak allocated/reserved memory per GPU
and 10.82 images/s, against 41.51/45.02 GiB and 8.29 images/s for the PyTorch
path, on synthetic inputs.

## Current 1024x1024 configuration

The current full-image configuration uses physical GPUs 0 and 7, BF16, a frozen
DINOv3 backbone, batch 7 per GPU and five gradient-accumulation microbatches:
each optimizer update receives 70 images, strictly more than 64. The configured
integer minimum is 65, so an effective batch of 64 is rejected. Incomplete
shuffled epoch tails are dropped. Whole-image inference remains the default.

Five real DDP/AdamW updates at this configuration passed with finite losses and
pre-clipping gradient norms. After two warmup updates, measured throughput was
8.33 images/second across both GPUs. Peak allocated memory was 41.46 GiB/GPU and
peak reserved memory 45.73 GiB/GPU; device-free headroom was about 0.96 GiB.
See `/home/disk02/wjj/moonseg/verification/1024_b7_a5_capacity.json`. These
numbers used the PyTorch deformable-attention path. With the compiled path, the
[GPU capacity report](audit_2026-10-21/gpu_capacity.md) measured 30.85/35.37 GiB
and 10.82 images/s at the same batch plan; that synthetic benchmark did not
include EMA (about 0.11 GiB of extra FP32 trainable parameters per GPU),
validation or checkpoint writing.

The separate [capacity report](memory_capacity.md), measured with the PyTorch
path, disables accumulation and uses fresh single-GPU processes: batch 7 passed on the actual 48GB card and
batch 8 ran out of memory; a 24GiB budget simulation passed batch 3 and failed
batch 4. The 24GiB result is not a physical 24GB-card benchmark.

Gradient accumulation extends the optimizer batch, not BatchNorm statistics.
The adapter's ordinary training BatchNorm uses seven images per rank. Mask loss
normalization uses each distributed microbatch's target-mask count, so its
average need not equal one concatenated 70-image batch when class counts differ.

The earlier full 1024 training-CLI integration report used effective batch 64
and covered tail handling, resume, metrics and packaging; it is retained as
historical evidence. Current-70 accumulation and batch-plan boundary checks were
verified separately. These are synthetic execution/capacity checks, not LuSNAR
accuracy or long-run performance measurements.

## Training-loop options

These `train` keys are filled with their defaults before training starts, so
`resolved_config.json` and every checkpoint record the values actually used.
Epoch-boundary resume requires them unchanged.

| Key | Default | Meaning |
| --- | --- | --- |
| `tf32` | `true` | `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `set_float32_matmul_precision("high")` |
| `cudnn_benchmark` | `true` | `torch.backends.cudnn.benchmark` |
| `grad_clip` | 0.1 | Global L2 clipping threshold; 0 disables clipping, the norm is still computed |
| `ema` | `{enabled: true, decay: 0.999}` | Exponential moving average of trainable parameters |
| `sampling` | `{mode: uniform, temperature: 0.01, min_pixels: 1}` | `uniform` or `rare_class` |
| `eval_stride` | 1 | Intermediate validations use every k-th validation record |

`data.label_stats` names the label-statistics JSON required by `rare_class`.

TF32 affects FP32 matrix multiplications and convolutions outside BF16 autocast,
for example the criterion's matching costs. BF16 kernels are unaffected, and so
are the FP32 deformable-attention kernels, which are not matrix multiplications.
`evaluate.py` applies the same two settings from the configuration.

Gradient norm: after unscaling, the pre-clipping global L2 norm of all gradients
(identical on every rank after DDP averaging) is logged as `grad_norm` in each
step line; `history.jsonl` records `grad_norm_mean` and `grad_norm_max` over the
epoch's applied updates.

EMA: the copy is built after DDP's rank-0 broadcast, so all ranks start from the
same weights. Frozen parameters are shared with the trained model, never copied,
and are not saved; the extra memory is the trainable parameters
(28.7M for the default model) plus buffers. After every applied optimizer step,
`ema = decay * ema + (1 - decay) * parameter` for trainable parameters, and
buffers (adapter BatchNorm statistics) are copied, again after the epoch-end
rank-0 buffer broadcast. Decay 0.999 has a time constant of about 1000 updates,
roughly 9 to 11 epochs at the 70-image effective batch, so early EMA validations
still contain a large share of the random adapter/head initialization. Each
validation evaluates both models on the same records; history stores
`validation` and `validation_ema`. Checkpoints contain `model`, `ema_model` (when
enabled), `best_miou`, `best_epoch` and `best_source` (`"model"` or
`"ema_model"`, `null` before the first validation). `best.pt` is written when the
larger of the two mIoU values improves. Resume (from `last.pt`) restores the EMA.
`moonseg.evaluate` accepts `--weights auto|model|ema_model`; `auto` uses
`best_source`. Any other consumer of `best.pt` must load
`checkpoint[checkpoint["best_source"]]` to use the selected weights.

Checkpoints: the frozen pretrained DINOv3 tensors are not stored. `model` and
`ema_model` omit every `backbone.backbone.*` tensor that is not a trainable
parameter (frozen ViT weights and the RoPE `periods`/qkv `bias_mask` buffers,
188 tensors for ViT-B/16); with `unfreeze_last_blocks: N` or
`freeze_backbone: false` the trained ViT blocks and final `norm` are kept, in
both `model` and `ema_model`. Loading always rebuilds the model from
`model.backbone_weights` first and then calls `load_partial_state_dict`: only
frozen ViT tensors may be absent; a missing trainable tensor (for example a
config with a different `unfreeze_last_blocks`), an unexpected key or a shape
mismatch is an error, checked before any tensor is copied. Older full
checkpoints still load. Every checkpoint records `frozen_backbone.weights`, the
configured weights path. Two files are kept per run: `last.pt`, the full
epoch-boundary resume state (model, EMA, optimizer, scheduler, scaler, per-rank
RNG), overwritten every epoch; and `best.pt`, inference-only (`model`,
`ema_model` and metadata, without optimizer/scheduler/scaler/RNG). `--resume
best.pt` is refused with a message pointing to `last.pt`. Periodic
`epoch_XXX.pt` snapshots are off by default (`train.save_every: 0`); a positive
value writes full-state snapshots every N epochs. A CPU smoke run on
2026-10-21 (ViT-B/16, default frozen backbone, EMA on) wrote `last.pt` of
459.7 MB and `best.pt` of 229.9 MB with 425 tensors each in `model` and
`ema_model`, against 802.8 MB for an earlier full-state `last.pt`. The
reproduction or submission package must therefore ship the exact
`model.backbone_weights` file (for ViT-B/16
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`) next to the checkpoints; a
checkpoint alone cannot be evaluated, and the code does not verify that the
weights file is the one used in training beyond the recorded path.

Rare-class sampling (`sampling.mode: rare_class`) follows DAFormer's Rare Class
Sampling (Hoyer et al., CVPR 2022). With `f_c` the share of class c among all
training label pixels, each draw picks class c with probability proportional to
`exp((1 - f_c) / temperature)`, then uniformly one image with at least
`min_pixels` pixels of c. Each epoch draws `len(dataset)` images with replacement
from a NumPy generator seeded by `(seed, epoch)`; every rank builds the same list
and takes `indices[rank::world_size]`. Per-rank length and the incomplete-update
policy equal those of `DistributedSampler`, so the batch plan is unchanged. A
training sample missing from the statistics, or a class without an eligible
image, is an error; scene 1 alone, for example, contains no crater.
`batch_plan.json` records the class pixel frequencies, the realized class
sampling probabilities, eligible image counts, and the expected per-class image
frequency next to the uniform one; resume requires them unchanged. Full images
are drawn, not crops, so a rare-class draw still trains on every class in that
image. From the training-pixel fractions in the
[data audit](audit_2026-10-21/data_audit.md), temperature 0.01 would give class
probabilities of about 0.64 crater, 0.28 rock, 0.08 mountain and 0 for regolith
and sky. This is a calculation; the effect on accuracy is untested, and
`temperature`/`min_pixels` are DAFormer-style values without LuSNAR tuning.

Validation stride: intermediate validations evaluate every `eval_stride`-th
record in `(scene, sample_id)` order; the final epoch always evaluates all
records. Each history entry records `eval_stride` and `evaluated_samples`. A
subset mIoU is not directly comparable to the full-set value, and `best.pt` may
be selected on a subset.

Data provenance: `data_provenance.json` and every checkpoint's `data_snapshot`
hold the training/validation records, the allowlist text (`null` without
`data.official_allowlist`) and `provenance`, either `official_allowlist` or
`public_development_pool` (from the manifest kind; `data.development_pool: true`
opts in). A development-pool run prints a warning at startup and records:
development only; the final submission model must be trained on the organizer's
official sample list.

On 2026-10-21 a two-GPU smoke run (physical GPUs 0 and 7) used 24 real training
images (16 from scene 1, 8 from scene 3, which contains craters) and 8 scene-5
validation images in a temporary organizer-format manifest, with batch 2 x 2
GPUs x 2 accumulation, EMA, rare-class sampling and `eval_stride: 2`. It
exercised the compiled path, step `grad_norm` logging, both validations (4
records at stride 2, all 8 in the final epoch), `best_source`, refusal to resume
with a changed epoch count, resume of an interrupted 3-epoch run at the epoch-2
boundary (the resumed first update reproduced the interrupted run's loss within
7e-5 relative, and the EMA continued rather than restarting), and
`moonseg.evaluate` on `best.pt` with both weight sets. Nine updates train nothing
meaningful; no accuracy is claimed from this run.

## Earlier verification and limits

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
