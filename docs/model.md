# DINOv3 + ViT-Adapter + Mask2Former

`moonseg.models.build_model(config, load_backbone=True)` constructs the original Meta DINOv3 transformer, the official DINOv3 ViT-Adapter, the official deformable pixel decoder, and the official multi-scale masked transformer decoder. This is a query/mask model, with spatial-prior convolutions, four deformable extraction interactions and convolutional FFNs in the adapter; it is not a linear/FPN-only probe of frozen ViT features. The training recipe that uses it is in [training.md](training.md).

Input is ImageNet-normalized RGB `[B,3,H,W]`. Output is a dictionary containing `pred_logits: [B,Q,6]` (five semantic classes plus no-object), `pred_masks: [B,Q,h,w]` (raw logits), and `aux_outputs` from earlier predictions. Semantic scores are `einsum('bqc,bqhw->bchw', logits.softmax(-1)[...,:5], masks.sigmoid())`; at inference the mask logits are upsampled before the sigmoid, as in upstream DINOv3 and Mask2Former. Input dimensions are padded to multiples of 32 internally. Non-divisible sizes return masks cropped to the original field of view; divisible sizes keep stride-4 mask logits. Minimum image size is 64 pixels per dimension.

The complete DINOv3 ViT-L + ViT-Adapter + Mask2Former has 338,785,670 parameters: 303,154,176 in the native encoder and 35,631,494 in the Adapter and segmentation head. Frozen training updates the latter; full joint fine tuning updates all parameters without changing the inference architecture or total parameter count. The [competition baseline](https://insight.onescience.cn/competition/moonseg?event_id=c2f557a58ff2e9f4265d56f112a7e01&tab=info&section=baseline) reports 4.71M trainable parameters for MobileNetV3-Large + DeepLabV3+, so our joint model has about 71.93 times as many parameters (the baseline count is rounded). This ratio is not a memory or throughput ratio.

## Configuration

All model settings belong to `config['model']`:

| Key | Default | Meaning |
| --- | --- | --- |
| `backbone` | `vitb16` | `vitb16`, `vitl16`, `vith16plus` or `da3mono_vitl14` |
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

The original ViT always remains in evaluation mode, also when the full model enters training mode and also when the ViT is partly or fully fine tuned. The hub ViT-B/16 and ViT-L/16 builders set `pos_embed_rope_rescale_coords=2`: in training mode, RoPE multiplies the patch coordinates of every forward by a random log-uniform factor in [1/2, 2]. This is DINOv3 pretraining augmentation and must not run during segmentation fine tuning. The ViT contains no BatchNorm and its hub drop-path rate is 0, so evaluation mode changes nothing else.

`unfreeze_last_blocks: N` keeps `freeze_backbone: true` and makes only `backbone.blocks[-N:]` and the final `norm` (applied to all four extracted layers) trainable. Patch embedding, class, storage and mask tokens stay frozen; RoPE periods are a buffer. Frozen leading blocks record no autograd graph because their inputs need no gradient. N must not exceed the depth, the last interaction index must be the final block (the default), and combining N > 0 with `freeze_backbone: false` is rejected. CPU construction of the default ViT-B model gave these parameter counts:

| Setting | Trainable | Trainable ViT part |
| --- | ---: | ---: |
| `unfreeze_last_blocks: 0` (default) | 28,687,590 | 0 |
| `unfreeze_last_blocks: 4` | 57,046,758 | 28,359,168 (blocks 8-11, `norm`) |
| `freeze_backbone: false` | 114,357,222 | 85,669,632 |

The optimizer identifies original ViT parameters with `name.startswith('backbone.backbone.')` and applies `train.backbone_lr_multiplier`; matching every name containing `backbone` would incorrectly include the adapter. Biases, one-dimensional tensors and embedding tables get no weight decay: names containing `pos_embed`, `cls_token`, `storage_tokens` (DINOv3's register tokens), `register_tokens`, `mask_token`, `level_embed`, `level_encoding`, `query_feat` or `query_embed`. Every remaining decayed tensor with two or more dimensions is a linear or convolution weight (including the `class_embed`/`mask_embed` heads and, when trained, the patch-embedding convolution). Memory and speed of `unfreeze_last_blocks: 4` have not been measured; full-unfreeze capacity is in [gpu_topology.md](gpu_topology.md#显存与吞吐容量).

Loading uses `pretrained=False` followed by strict native state-dictionary loading, and never starts a network download. Training (also on resume), `moonseg.evaluate` and `moonseg.predict` always build with `load_backbone=True`, because checkpoints do not contain the frozen ViT (see [Checkpoints](#checkpoints)). `load_backbone=False` is only for explicitly random-initialized smoke checks; it does not produce pretrained features. Hugging Face Transformers safetensors require conversion to the native Meta parameter layout before using this loader ([weights.md](weights.md)).

## DA3MONO encoder option

`model.backbone: da3mono_vitl14` loads only the official DA3MONO-LARGE depth-trained ViT-L encoder (24 blocks, dimension 1024, layers `[4,11,17,23]`, no alternate-view attention, QK norm or RoPE). `model.backbone_weights` points to `ckpts/da3mono-large/model.safetensors`. The strict loader selects only `model.backbone.pretrained.*`: 342 tensors / 304,367,616 parameters. No DA3 depth/camera head is built. DINOv3 defaults, state keys and historical configurations keep their existing behavior.

The native encoder keeps `patch_size=14`; `adapter_patch_size=16` describes the feature interface only. Images stay 1024²: only the encoder input receives right/bottom zero padding to 1036² (74² patches). Official normalized patch features are bilinearly sampled at Adapter pixel centers `p=(j+0.5)*16`, using grid coordinates `g=2*p/padded_extent-1` with `align_corners=False`. This yields the existing 64² feature grid without rescaling the padded field of view. The outer network still pads arbitrary TTA sizes to /32, then applies the same independent /14 encoder padding and coordinate bridge. The segmentation spatial prior and stride4/8/16/32 decoder remain unchanged.

Each image enters the native API as `[B,1,3,H,W]`; batch images are independent single views. The first DA3 run uses `freeze_backbone: true` and `unfreeze_last_blocks: 0`: all native encoder parameters have `requires_grad=False`, the encoder stays in eval mode, and its four features are computed under `torch.no_grad()`. Only ViT-Adapter and Mask2Former enter AdamW, at the baseline LR. Native block checkpointing is explicitly disabled in this frozen config; Adapter interaction checkpointing remains enabled. Frozen native tensors are omitted from all three segmentation checkpoint payloads and strictly restored from official pretrained weights before the saved Adapter/head is loaded. Real GPU update and frozen checkpoint/optimizer reload evidence is in [frozen verification](evidence/da3mono_frozen_verification_20261108.json).

For a separately configured trainable encoder, `backbone_gradient_checkpointing: true` checkpoints native blocks with `use_reentrant=False` and RNG preservation whenever gradients are enabled and block parameters train, including encoder eval mode. The encoder remains nested at `backbone.backbone.*`, so its LR multiplier and trainable-tensor checkpoint policy apply without special cases. All trainable encoder tensors enter last/best_miou/best_loss with strict missing-key rejection.

The unmodified upstream encoder and import dependencies are vendored under `third_party/depth_anything_3` at revision `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, with the Apache-2.0 license and `SOURCE_MANIFEST.json`. The complete sparse reference checkout stays on disk02. Only einops is added; the application, GUI, rendering and depth-head dependency stacks are unused. [CPU evidence](evidence/da3mono_verification_20261108.json) records exact native weight/feature parity, physical-coordinate mapping, eval-mode encoder gradients and complete simulated-learned checkpoint/optimizer restoration.

## Training criterion

`build_criterion(config)` returns a module accepting either semantic targets `[B,H,W]` with `255` ignored, or a list of dictionaries containing `labels[N]`, binary `masks[N,H,W]`, and optional `valid_mask[H,W]`. Semantic labels are grouped into one target mask per class present in the image. Matching uses SciPy's exact Hungarian solver with class probability, sigmoid BCE and Dice matching costs. Each decoder prediction gets its own matching and classification, BCE and Dice losses, including the initial query prediction, following Mask2Former's deep supervision scheme.

The return value is `{'loss': differentiable_weighted_total, 'loss_ce': detached, 'loss_mask': detached, 'loss_dice': detached, ...auxiliary_terms}`. Backpropagate only `result['loss']`; reported component terms are unweighted diagnostics. `config['loss']` accepts `ce_weight=2`, `mask_weight=5`, `dice_weight=5`, `no_object_weight=0.1`, `num_points=12544`, `oversample_ratio=3`, `importance_sample_ratio=0.75`, and `ignore_index=255`.

The mask and Dice terms are divided by the number of target masks averaged over ranks (one `all_reduce` per forward, as upstream) while `criterion.sync_num_masks` is `True`, the default and the training setting. `evaluate_model` sets it to `False` while computing the validation loss and restores it afterwards: ranks evaluate different numbers of samples (800/799 for the 1599 scene-5 frames), so a collective in every forward would deadlock, and each validation sample is normalized by its own mask count.

Matching shares one set of randomly selected valid pixel centers among all query/target pairs of an image, as the upstream matcher shares one point set. Loss points follow upstream `get_uncertain_point_coords_with_randomness` for each matched mask independently: every mask draws its own `oversample_ratio x num_points` candidates, keeps the `importance_sample_ratio` fraction with the smallest absolute logit, and adds its own random points. Unlike upstream's continuous coordinates, all points are centers of valid pixels, so ignored pixels never enter matching or losses. Valid pixel indices are computed once per image per criterion call (no per-sample `nonzero()`), and only the matched queries are cast to FP32. Target sampling is nearest-neighbor, preserving binary targets and excluding ignored labels exactly. All-ignored images contribute zero classification and mask loss. Mask count normalization is synchronized for distributed training; loss/matching math runs in FP32 under mixed precision.

Criterion design follows the original [Mask2Former matcher](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py) and [criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py), with valid-pixel sampling added for semantic ignore labels. The criterion is a local implementation and does not import detectron2.

## Source and compatibility changes

Primary source is [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), revision `6876159a11b4df116f30f667f8c9888617df0751`, downloaded with `python3 tools/download_assets.py --only source` as the complete official archive at this revision. All 221 upstream files (presence and byte sizes checked against the official Git tree), original copyright notices and the [DINOv3 license](../third_party/dinov3/LICENSE.md) are retained. `third_party/dinov3/SOURCE_MANIFEST.json`, `SOURCE_REVISION` and `SOURCE_COMPLETE` record the archive URL and fixed revision. This is an archive extraction rather than a Git checkout.

The original implementation files remain unchanged. MoonSeg supplies these explicit runtime compatibility adaptations:

- Official `MSDeformAttnFunction.forward` calls the PyTorch core but its backward requires a compiled CUDA extension. MoonSeg replaces that Function boundary in memory (process-wide) with either the official compiled `MultiScaleDeformableAttention` kernels for both passes, or the official `ms_deform_attn_core_pytorch` under normal PyTorch autograd for both passes. Both run in FP32. No detectron2/mmcv is needed.
- A subclass uses the original Adapter components and feature flow while removing hard-coded CUDA autocast, preserving the outer AMP context, and supporting optional ViT fine tuning. The default original ViT remains frozen.
- Upstream SyncBatchNorm is converted to BatchNorm2d by default; enable the explicit config option for SyncBatchNorm under CUDA DDP.
- Narrow namespace imports skip the upstream segmentation evaluation package's eager data/torchmetrics dependencies. Positional encoding comes directly from the complete official source.

## Compiled deformable attention

`model.deformable_attention` selects the path when the model is built:

- `auto` (default): use the compiled extension when it imports and its `build_manifest.json` names the running torch version; otherwise warn and use the PyTorch path.
- `cuda`: require the compiled extension; raise with a build hint otherwise.
- `torch`: always the PyTorch `grid_sample` path.

The compiled path casts value, sampling locations and attention weights to FP32 with autocast disabled, because the kernels reject BF16 and FP16, and returns the input dtype; this matches the PyTorch path's FP32 semantics. `im2col_step` is `gcd(batch, 64)` to satisfy the kernel's divisibility check. CPU tensors always use the PyTorch path. `build_model` prints the selected path on rank 0, and `train.py` records it under `runtime` in `resolved_config.json`. The selection is not part of any state dictionary, so checkpoints load under either path. Both backward passes use atomics, so bitwise replay is not expected. Kernel accuracy and the memory/speed difference between the two paths are in [gpu_topology.md](gpu_topology.md#可变形注意力算子).

`bash scripts/build_ops.sh` builds the extension. It copies the official sources from `third_party/dinov3/dinov3/eval/segmentation/models/utils/ops` into a temporary directory (no vendor edits) and compiles with the project interpreter (`uv run --no-sync python`, or `PYTHON=`), `CUDA_HOME=/usr/local/cuda-12.6` to match the cu126 PyTorch wheels (`/usr/local/cuda` on this host points at 12.8), and `TORCH_CUDA_ARCH_LIST="8.6;8.9"` (RTX A6000 and RTX 6000 Ada). Upstream `setup.py` refuses to build without a visible GPU, so the script exposes `BUILD_GPU` (default physical GPU 0). It writes `build/ms_deform_attn/MultiScaleDeformableAttention*.so` and `build_manifest.json` (torch, CUDA, nvcc, Python, architectures, `.so` and source SHA-256), then imports the module as a check. `/build/` is gitignored and must be rebuilt after any torch, CUDA or Python change. It is not installed into `.venv`, because `uv sync --locked` removes packages that are not in the lock file.

## Training-loop options

These `train` keys are filled with their defaults before training starts, so `resolved_config.json` and every checkpoint record the values actually used. Epoch-boundary resume requires them unchanged.

| Key | Default | Meaning |
| --- | --- | --- |
| `tf32` | `true` | `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `set_float32_matmul_precision("high")` |
| `cudnn_benchmark` | `true` | `torch.backends.cudnn.benchmark` |
| `grad_clip` | 0.1 | Global L2 clipping threshold; 0 disables clipping, the norm is still computed |
| `sampling` | `{mode: uniform, temperature: 0.01, min_pixels: 1}` | `uniform` or `rare_class` |
| `eval_stride` | 1 | Intermediate validations use every k-th validation record |

`data.label_stats` names the label-statistics JSON required by `rare_class`.

TF32 affects FP32 matrix multiplications and convolutions outside BF16 autocast, for example the criterion's matching costs. BF16 kernels are unaffected, and so are the FP32 deformable-attention kernels, which are not matrix multiplications. `evaluate.py` applies the same two settings from the configuration.

Gradient norm: after unscaling, the pre-clipping global L2 norm of all gradients (identical on every rank after DDP averaging) is logged as `grad_norm` in each step line; `history.jsonl` records `grad_norm_mean` and `grad_norm_max` over the epoch's applied updates.

Validation loss: with a criterion, `evaluate_model` also computes the training criterion's total loss (final and auxiliary decoder layers) for every evaluated sample at batch 1, with the training autocast mode and the model in evaluation mode, and returns `val_loss`, the mean over the evaluated samples of all ranks (sum and count reduced once at the end). When inference is a single whole-image forward (no sliding window, scales `[1.0]`, no flip, dimensions divisible by 32, as in the current recipe) the same forward gives both the prediction and the loss; otherwise the loss takes one extra plain forward. The criterion's random point sampling runs inside `torch.random.fork_rng` with a fixed seed, so `val_loss` is reproducible, each sample draws the same points in every epoch, and validation never advances the training RNG streams. Training passes its criterion every validation; `moonseg.evaluate` builds one from the config's `loss` section and reports `val_loss` next to the metrics.

EMA was removed on 2026-10-22 (reasons in [training.md](training.md#验证协议)). A config that still contains `train.ema` is rejected with an error asking to delete the block. `moonseg.evaluate --weights` and `moonseg.predict --weights-key` default to `auto`, which now always loads `model`; when a legacy checkpoint's `best_source` says `ema_model`, that is ignored with a printed note, and its `ema_model` weights load only when requested explicitly.

Validation stride: intermediate validations evaluate every `eval_stride`-th record in `(scene, sample_id)` order; the final epoch always evaluates all records. Each history entry records `eval_stride` and `evaluated_samples`. A subset mIoU or loss is not directly comparable to the full-set value, and `best_miou.pt`/`best_loss.pt` may be selected on a subset.

## Rare-class sampling

`train.sampling.mode: rare_class` follows DAFormer's Rare Class Sampling (Hoyer et al., CVPR 2022). With `f_c` the share of class c among all training label pixels, each draw picks class c with probability proportional to `exp((1 - f_c) / temperature)`, then uniformly one image with at least `min_pixels` pixels of c. Each epoch draws `len(dataset)` images with replacement from a NumPy generator seeded by `(seed, epoch)`; every rank builds the same list and takes `indices[rank::world_size]`. Per-rank length and the incomplete-update policy equal those of `DistributedSampler`, so the batch plan is unchanged. A training sample missing from the statistics, or a class without an eligible image, is an error; scene 1 alone, for example, contains no crater. `batch_plan.json` records the class pixel frequencies, the realized class sampling probabilities, eligible image counts, and the expected per-class image frequency next to the uniform one; resume requires them unchanged. Full images are drawn, not crops, so a rare-class draw still trains on every class in that image. With the full training pool's pixel fractions ([data.md](data.md#类别分布)), temperature 0.01 gives class probabilities of about 0.64 crater, 0.28 rock, 0.08 mountain and 0 for regolith and sky. This is a calculation; the effect on accuracy is untested, and `temperature`/`min_pixels` are DAFormer-style values without LuSNAR tuning. The mode is off in the current recipe.

## Checkpoints

For a new full-ViT stage, `moonseg.train --initialize-from <best_miou.pt|best_loss.pt|last.pt>` (or `train.initialize_from`) loads a frozen source under its original trainability policy, then enables all ViT gradients before constructing the optimizer. Architecture, native pretrained path, data/loss/inference settings, sample membership, provenance and source selection metadata must match. Optimizer, scheduler, scaler, RNG seed, best selection, epoch and global step start fresh. `stage_initialization.json` and every stage checkpoint record the parent path/file identity, epoch (both bases), step and selection. Later `--resume last.pt` restores this stage's full trained ViT and optimizer without reading the parent again. Explicit `--resume` and `--initialize-from` are mutually exclusive.

The frozen pretrained DINOv3 tensors are not stored. `model` omits every `backbone.backbone.*` tensor that is not a trainable parameter (frozen ViT weights and the RoPE `periods`/qkv `bias_mask` buffers, 188 tensors for ViT-B/16); with `unfreeze_last_blocks: N` the trained ViT blocks and final `norm` are kept. With `freeze_backbone: false`, all trainable ViT parameters are kept, including patch embedding and tokens: ViT-L has 343 such tensors (303,154,176 parameters). This identical model mapping is saved in `last.pt`, `best_miou.pt` and `best_loss.pt`. Loading always rebuilds the model from `model.backbone_weights` first and then calls `load_partial_state_dict`: only frozen ViT tensors may be absent; a missing trainable tensor (for example a config with a different `unfreeze_last_blocks`), an unexpected key or a shape mismatch is an error, checked before any tensor is copied. Older full checkpoints still load. Checkpoints contain no `ema_model` and no `best_source`.

Every checkpoint records `frozen_backbone.weights` (the configured weights path), `epoch` (0-based), `global_step`, `config`, `data_snapshot`, `batch_plan` and the selection state: `best_miou` and `best_miou_epoch` (highest validation mIoU so far and its 1-based epoch), `best_loss` and `best_loss_epoch` (lowest validation loss so far), and `recent_miou` (the last three validation mIoU values, from which history's `miou_ma3` is computed). All are `null` or empty before the first validation.

Files per run:

- `last.pt`: the full epoch-boundary resume state (model, optimizer, scheduler, scaler, per-rank RNG, selection state), overwritten every epoch.
- `best_miou.pt`: inference-only (`model` and metadata, without optimizer/scheduler/scaler/RNG), with `selected_by: "best_miou"`; overwritten when the validation mIoU is strictly higher than at every earlier validation.
- `best_loss.pt`: the same format with `selected_by: "best_loss"`; overwritten when the validation loss is strictly lower than at every earlier validation. Both best files may be written in the same epoch. `--resume` refuses either (and a legacy `best.pt`) with a message pointing to `last.pt`.

Resume restores the selection state from `last.pt`. A legacy `last.pt` written by the EMA code (with `ema_model`, `best_source`, `best_miou`, `best_epoch`, and `train.ema` in its recorded config) still resumes with the current config: the recorded `train.ema` is ignored by the unchanged-settings check, the EMA weights are dropped, best tracking restarts at the next validation with a printed note giving the legacy values, and an existing `best.pt` is left untouched. `moonseg.evaluate` and `moonseg.predict` record the checkpoint's `selected_by` (and, for legacy files, `best_source`) in their outputs.

`train.save_every` is 0, so no periodic snapshots are written (a positive N would add full-state `epoch_XXX.pt` files every N epochs). In the 2026-10-21 baselines, which still kept an EMA, `last.pt`/`best.pt` are about 461/231 MB for ViT-B/16 and 572/287 MB for ViT-L/16; without `ema_model` each file is one weight set smaller, so about 346/115 MB (ViT-B/16) and 430/143 MB (ViT-L/16) for `last.pt` and each best file (estimate, not yet measured). A checkpoint alone cannot be evaluated: the reproduction or submission package must ship the exact `model.backbone_weights` file (for ViT-B/16 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`) next to the checkpoints, and the code does not verify that the weights file is the one used in training beyond the recorded path.

## Validated facts

- Compiled vs PyTorch deformable attention on a random-initialized ViT-B, `[2,3,256,320]` FP32 input, TF32 disabled: total losses 79.292198 and 79.292191 (relative difference 9.6e-8), gradient norms differing by 3.9e-6 relative; `auto` selected `cuda`. The build took 42 s on this host.
- Criterion with per-mask point sampling: on CPU, losses and gradients are finite, six auxiliary terms are produced, no ignored pixel is ever sampled, and an entirely ignored image gives zero mask gradient. On one GPU with synthetic `[7,100,256,256]` BF16 mask logits and 1024x1024 targets, CUDA sync-debug mode counted 225 synchronizing operations per criterion forward (407 for the previous shared-point sampler), and criterion forward/backward peak memory was 0.360/0.713 GiB (previously 0.485/0.905 GiB).
- A 2026-10-21 two-GPU smoke run of the EMA-era code (24 real training images, 8 scene-5 validation images, batch 2 x 2 GPUs x 2 accumulation, EMA, rare-class sampling, `eval_stride: 2`) exercised the compiled path, step `grad_norm` logging, strided and final validations, `best_source`, refusal to resume with a changed epoch count, resume of an interrupted run at an epoch boundary (the resumed first update reproduced the interrupted run's loss within 7e-5 relative, and the EMA continued), and `moonseg.evaluate` on `best.pt` with both weight sets. It trained nothing meaningful; no accuracy is claimed.
- 2026-10-22 CPU-only checks of the checkpoint-selection change, with a tiny stand-in model that has the Mask2Former output contract (no GPU, no training): with `sync_num_masks = False` the criterion ran outside any process group with a finite loss and auxiliary terms. On a two-rank gloo group with 3/2 samples per rank, `evaluate_model` finished, its `val_loss` equalled the sample-weighted mean of the per-shard losses (relative difference below 1e-12) and its mIoU the single-process value, while the same uneven forward counts with the switch on left rank 0 blocked in its third `all_reduce`. Metrics with and without the loss were identical on the single-forward path (64x64) and the separate-forward path (48x48), repeated evaluations gave the same `val_loss`, and the caller's RNG was untouched. A simulated (mIoU, loss) sequence wrote `best_miou`/`best_loss` at the expected epochs (ties and NaN never count) with the expected `miou_ma3`; resuming the selection state reproduced the uninterrupted sequence; a legacy checkpoint dict resumed with reset tracking, and `auto` loaded `model` while `ema_model` loaded only explicitly.
- Strict loading of the converted pretrained weights is documented in [weights.md](weights.md#验证范围). Earlier synthetic checks of full-image geometry, metric edge cases (missing classes, all-ignored targets) and CLI integration including prediction and ZIP packaging are recorded under `/home/disk02/wjj/moonseg/verification/`; they ran on an older code revision.
