# MoonSeg 性能与显存余量报告：MSDeformAttn CUDA extension vs pure-PyTorch fallback

日期：2026-10-21。所有数字来自本主机真实 GPU 上的短程合成负载（1024×1024 随机归一化图像 + 16×16 块状 5 类标签，顶部 1/16 为 ignore=255），使用真实 `build_model` / `build_criterion` / `make_optimizer` 与 `train.py` 相同的 step 逻辑（bf16 autocast、criterion、`loss/accum`、`clip_grad_norm_(0.1)`、AdamW、GradScaler disabled、DDP `no_sync`）。**仓库内没有修改任何文件**（`git status` clean），所有构建、脚本、结果都在 `$TMPDIR/moonseg_perf/`。

- 原始 JSON：`$TMPDIR/moonseg_perf/results/`（`train_<tag>.json`、`infer_*.json`、`task1_*.json`、`task4_dataloader.json`、`task5_patch_test.jsonl`、`summary_wallclock_estimates.json`、`unfrozen_unused_parameters.json`）
- 脚本：`$TMPDIR/moonseg_perf/scripts/`（`bench_train.py`、`bench_infer.py`、`bench_data.py`、`verify_op*.py`、`msda_impl.py`、`run1.sh`/`run2.sh`/`search.sh`）
- 日志：`$TMPDIR/moonseg_perf/logs/`
- 补丁（未应用）：`$TMPDIR/moonseg_perf/use_cuda_ms_deform_attn.patch`；构建脚本建议稿：`$TMPDIR/moonseg_perf/proposed/scripts/build_ops.sh`

单卡测试均在物理 GPU 0（RTX 6000 Ada），双卡在 0,7；A6000 只用了物理 GPU 1（46068 MiB）。GPU 3 未触碰；GPU 4 在测试中途被其他用户占用，也未触碰。所有进程均已结束。

## 0. 结论速览

| 配置（2×RTX 6000 Ada，DDP） | 每卡 batch × 累积 | 每卡最大 batch | 峰值 allocated / reserved (GiB) | 总 img/s | 80 epoch / 8002 张 |
|---|---|---:|---|---:|---:|
| a 基线：pure-PyTorch、ckpt on、frozen | 7×5（eff 70） | 7（8 OOM，旧报告） | 41.51 / 45.02 | **8.29** | **21.4 h** |
| b CUDA op、ckpt on、frozen | 7×5（eff 70） | **10** | 30.85 / 35.37 | 10.82 | 16.4 h |
| b CUDA op、ckpt on、frozen（最大 batch） | 10×4（eff 80） | 10 | 43.67 / 46.65 | 10.59 | 16.8 h |
| **c CUDA op、ckpt OFF、frozen（推荐）** | 7×5（eff 70） | 7 | 39.94 / 43.68 | **11.72** | **15.1 h** |
| d CUDA op、ckpt on、**backbone unfrozen** | 7×5（eff 70） | 7 | 43.81 / 46.42 | 8.81 | 20.1 h |
| e ViT-L/16 frozen（随机权重） | 单卡实测 b7 | 9 | 33.69 / 37.64（单卡 b7） | 估 8.3 | 估 21.5 h |
| e ViT-L/16 unfrozen | 单卡实测 b4 | 4 | 40.22 / 43.41（单卡 b4） | 估 5.6 | 估 31.7 h（4×2×9） |

要点：
1. CUDA extension 构建成功（31 s），forward/backward 与 PyTorch 参考一致，fp64 `gradcheck` 全部通过。
2. 在相同 b7×2×5 设置下：显存 −10.7 GiB（41.5→30.9），吞吐 +31%（8.29→10.82 img/s）；关闭 checkpointing 后达到 11.72 img/s（+41%），显存仍低于基线。
3. 每卡吞吐对 batch 基本不敏感（ckpt on：b4 5.51、b7 5.34、b8 5.48、b10 5.38 img/s），GPU 在 b4 就已饱和。加大 batch 只会减少累积步数，不会提速。
4. 4 个 DataLoader worker/rank 实测 21 img/s，GPU 每 rank 只需 ≤5.9 img/s，余量约 3.6 倍，数据加载不是瓶颈。

## 1. CUDA extension 构建与正确性（Task 1）

### 1.1 构建

| 项目 | 结果 |
|---|---|
| 源码 | 复制 `third_party/.../utils/ops/` → `$TMPDIR/moonseg_perf/ops_build/`，`python setup.py build_ext --inplace` |
| Toolkit | **`/usr/local/cuda` 实际指向 12.8**（与 brief 不同）。改用 `CUDA_HOME=/usr/local/cuda-12.6`（nvcc 12.6.85），与 `torch.version.cuda=12.6` 一致 |
| 编译器 / arch | gcc 11.4、系统 ninja；`TORCH_CUDA_ARCH_LIST="8.6;8.9"` → cubin sm_86 + sm_89（`cuobjdump` 验证） |
| 结果 | **成功**，31 s；0 error，60 条 deprecation warning（`Tensor.data<T>()`、`.type()`），无需修改源码 |
| 产物 | `MultiScaleDeformableAttention.cpython-312-x86_64-linux-gnu.so`（11.5 MB） |

### 1.2 forward / backward 对比（`task1_op_verification.json`, `task1_op_verification_extra.json`）

| 检查 | 形状 | dtype | 结果 |
|---|---|---|---|
| (a) forward vs `ms_deform_attn_core_pytorch` | adapter extractor（N2, M12, D32, 1 level 64², Lq 21504, P4） | fp32 | max abs 6.6e-6（参考最大 2.80），`allclose(rtol1e-4, atol1e-5)` ✓ |
| (a) forward | 4-level 256²…32² stress shape, Lq 87040 | fp32 | max abs 4.3e-6 ✓ |
| (a) forward | 小奇异形状（D5, 3 levels） | fp32 / fp64 | 1.2e-7 / 2.2e-16 ✓ |
| (a)(b) fwd + 三个 grad | 上述全部 | fp64 | max abs ≤ 1.6e-12（loc grad 量级 1e3） ✓ |
| (b) `torch.autograd.gradcheck` | 官方 test.py 形状（value/loc/weights 分别 + 一起）+ N2M3D4L2P3 | fp64 | **全部 passed** |
| (b) fp32 grad | 模型真实 3-level 形状，局部采样 | fp32 vs fp64 参考 | loc grad 误差：CUDA 4.0e-4，PyTorch 4.6e-3（最大值 1630），**CUDA 更接近 fp64** |
| (b) fp32 grad | 4-level 256² stress、随机位置 | fp32 CUDA vs fp32 PyTorch | value/weights ≤ 2e-4；loc grad 少数元素 max abs 506（mean 7e-5，参考最大 1586）。推测是 fp32 下位置正好落在双线性 cell 边界 / 越界边界附近造成的跳变，但这个形状**未对 fp64 复核**；模型真实的 3-level 形状用随机位置对 fp64 复核时没有不一致元素 |
| (c) bf16 | raw kernel，bf16 / fp16 输入 | — | **拒绝**：`"ms_deform_attn_forward_cuda" not implemented for 'BFloat16'`（Half 同样） |
| (c) | value fp32 + loc bf16 | — | 拒绝：`expected scalar type Float but found BFloat16` |
| (c) | 官方 `ops/functions` Function，在 autocast 下且 value 为 bf16（模型中 value_proj 输出就是 bf16） | — | **报错**，不能直接替换 |
| (c) | fp32 cast wrapper（补丁的做法） | autocast bf16 | 可用；输出 bf16，与仓库 torch path 相差最多 1 个 bf16 ulp（max 0.00195，mean 1.4e-7） |
| (c) | 上游 DINOv3 `utils/ms_deform_attn.MSDeformAttnFunction`（`custom_fwd(cast_inputs=fp32)`，PyTorch forward + CUDA backward） | autocast bf16 | 可用（下文记为 hybrid） |

### 1.3 Op 级速度与显存（1024² 训练形状，N=7，fp32，局部采样位置）

| Op | fwd+bwd ms：CUDA / PyTorch | 峰值额外显存 GiB：CUDA / PyTorch | autograd 保存 GiB/层：CUDA / PyTorch |
|---|---|---|---|
| pixel-decoder encoder layer（3 levels 32²/64²/128², Lq 21504, M8, D32） | **10.4 / 66.0（6.3×）** | 0.90 / 6.27 | ~0 / 2.03 |
| adapter extractor（1 level 64², M12, D32） | 4.8 / 31.2（6.6×） | 0.67 / 3.50 | 0 / 0.98 |

注：`task1_op_verification.json` 里第一版 `op_level_perf` 已作废（计时循环内每次都在 CPU 上 randn 4.4e7 个元素，形状也不对），文件中已标注 INVALID。

## 2. 全模型训练 step 基准（Task 2）

每项都是新进程，3 warmup + 10 timed step。单卡每个 step = 1 个 microbatch + optimizer.step；双卡每个 step = 累积 microbatch + 1 次更新。峰值统计覆盖全过程。phase 是在阶段边界 synchronize 后测得的每 microbatch 秒数（会带来约 1% 开销，`_nosync` 版本未插入同步）。

### 2.1 单卡（物理 GPU 0，RTX 6000 Ada 48 GB，可用 47.37 GiB）

| tag | attn | ckpt | backbone | batch | 状态 | alloc GiB | reserved GiB | img/s | fwd / loss / bwd (s) |
|---|---|---|---|---:|---|---:|---:|---:|---|
| a_torch_b7 | torch | on | B frozen | 7 | pass | **41.29** | 46.51 | **4.19** | 0.623 / 0.093 / 0.945 |
| b_hybrid_b7（上游 Function 原样 + 已编译 ext） | hybrid | on | B frozen | 7 | pass | 30.63 | 33.76 | 4.74 | 0.626 / 0.097 / 0.744 |
| b_cuda_b4 | cuda | on | B frozen | 4 | pass | 17.81 | 20.54 | 5.51 | — |
| b_cuda_b7 | cuda | on | B frozen | 7 | pass | **30.64** | 35.52 | **5.34** | 0.468 / 0.117 / 0.709 |
| b_cuda_b8 | cuda | on | B frozen | 8 | pass | 34.90 | 39.19 | 5.48 | 0.535 / 0.104 / 0.812 |
| b_cuda_b10 | cuda | on | B frozen | **10** | pass | 43.44 | 46.25 | 5.38 | 0.678 / 0.131 / 1.040 |
| b_cuda_b11 / b12 | cuda | on | B frozen | 11 / 12 | **OOM** fwd step 1 | — | — | — | |
| x_expseg_cuda_b11（`expandable_segments:True`） | cuda | on | B frozen | 11 | OOM | — | — | — | |
| c_cuda_nockpt_b4 | cuda | off | B frozen | 4 | pass | 23.01 | 25.12 | 6.10 | |
| c_cuda_nockpt_b6 | cuda | off | B frozen | 6 | pass | 34.16 | 37.40 | 5.97 | |
| c_cuda_nockpt_b7 | cuda | off | B frozen | **7** | pass | 39.72 | 43.56 | **5.84** | 0.468 / 0.097 / 0.623 |
| c_cuda_nockpt_b8 | cuda | off | B frozen | 8 | **OOM** bwd step 4（前 3 step 已通过） | — | — | — | |
| x_expseg_cuda_nockpt_b8（expandable） | cuda | off | B frozen | 8 | pass，但慢 | 45.25 | 46.81 | 3.55 | 显存紧张导致明显变慢 |
| d_cuda_unfrozen_b7 | cuda | on | B **unfrozen** | **7** | pass | 42.95 | 45.99 | 4.03 | 0.486 / 0.193 / 1.010 |
| d_cuda_unfrozen_b8 | cuda | on | B unfrozen | 8 | OOM | — | — | — | |
| e_cuda_vitl_frozen_b4 / b6 / b7 / b8 | cuda | on | L frozen | 4/6/7/8 | pass | 19.93 / 29.11 / 33.69 / 38.28 | 22.02 / 32.47 / 37.64 / 42.81 | 4.33 / 4.24 / 4.17 / 4.13 | |
| e_cuda_vitl_frozen_b9 | cuda | on | L frozen | **9** | pass（已慢化） | 42.87 | 45.65 | 3.54 | |
| e_cuda_vitl_frozen_b10 | cuda | on | L frozen | 10 | OOM | — | — | — | |
| e_cuda_vitl_unfrozen_b4 | cuda | on | L unfrozen | **4** | pass | 40.22 | 43.41 | 2.83 | |
| e_cuda_vitl_unfrozen_b5 | cuda | on | L unfrozen | 5 | OOM | — | — | — | |

与旧报告对照：单卡基线 41.29 GiB / 4.19 img/s，旧报告 `capacity48_b7.json` 为 41.24 GiB / 4.15 img/s，复现成功。

### 2.2 双卡 DDP（物理 GPU 0,7）

| tag | 配置 | eff batch | alloc / reserved GiB（每卡） | 总 img/s | step s |
|---|---|---:|---|---:|---:|
| a_ddp_torch_b7_a5 / _nosync | 基线 torch，b7×a5 | 70 | 41.51 / 46.60（nosync 45.02） | 8.28 / **8.29** | 8.45 |
| README 旧值 | 同上 | 70 | 41.46 / 45.73 | 8.33 | 8.41 |
| b_ddp_cuda_b7_a5 | CUDA，ckpt on | 70 | 30.85 / 35.37 | **10.82** | 6.47 |
| b_ddp_cuda_b10_a4 | CUDA，ckpt on，最大 batch | 80 | 43.67 / 46.65 | 10.59 | 7.55 |
| c_ddp_cuda_nockpt_b7_a5 / _nosync | CUDA，ckpt off | 70 | 39.94 / 43.68 | 11.55 / **11.72** | 5.97 |
| d_ddp_cuda_unfrozen_b7_a5（nosync） | CUDA，unfrozen，0.1 lr 组 | 70 | 43.81 / 46.42 | 8.81 | 7.95 |

DDP 扩展效率约 0.99–1.0（受益于累积时 optimizer 摊销）。unfrozen 时所有 trainable 参数都有梯度（`unfrozen_unused_parameters.json` 为空），`find_unused_parameters=False` 可正常工作。

### 2.3 Checkpointing trade-off（CUDA op，b7）

| | alloc GiB | img/s 单卡 | img/s 双卡 |
|---|---:|---:|---:|
| ckpt on | 30.64 | 5.34 | 10.82 |
| ckpt off | 39.72（+9.1） | 5.84（+9%） | 11.72（+8%） |

ckpt on 时最大 batch 为 10，ckpt off 时为 7。

### 2.4 推理（Task 2f，单卡 GPU 0，bf16，`inference_mode`，含 `semantic_scores` + argmax，不含 PNG I/O）

| 模式 | torch：img/s / alloc GiB | CUDA：img/s / alloc GiB | 加速 |
|---|---|---|---:|
| batch 1，`predict_scores` 单尺度 | 11.46 / 1.83 | **15.93** / 1.36 | 1.39× |
| batch 8，`model()` + scores | 11.35 / 10.50 | 15.28 / 6.56 | 1.35× |
| TTA：scales 0.75/1.0/1.25 × flip（6 次 forward） | 1.79 / 2.57 | **2.49** / 1.85 | 1.39× |

batch 8 并不比 batch 1 快。5004 张测试图在 2 卡上的纯 GPU 时间：单尺度 torch 218 s / CUDA 157 s；TTA torch 1394 s（23 min）/ CUDA 1004 s（17 min）。另外 `predict.py` 的 PNG 解码、归一化和保存都在主线程，按单样本解码约 50 ms 估计，再加约 2–3 min（未实测）。开发配置每个 epoch 验证 1599 张（batch 1），2 卡耗时 torch 70 s / CUDA 50 s，80 epoch 合计约 +1.5 h / +1.1 h。

### 2.5 A6000（Task 2g，物理 GPU 1，RTX A6000 46068 MiB，可用约 44.99 GiB）

| tag | batch | 状态 | alloc / reserved GiB | img/s |
|---|---:|---|---|---:|
| g_a6000_cuda_b10 | 10（Ada 上的最大值） | **OOM**（fwd step 1，峰值 alloc 42.65） | — | — |
| g_a6000_cuda_b9 | 9 | pass | 39.18 / 43.41 | 3.57 |

结论：46068 MiB 的 A6000 放不下 Ada 上的最大 batch 10，最大为 9。b9 下速度约为 Ada 的 0.66 倍（Ada b8–b10 为 5.4 img/s）。b9 的 reserved 已接近上限，可能受 allocator 压力拖慢，所以 0.66 是保守值。为遵守"只用一张 A6000、跑一次对比"的约束，这里在 GPU 1 上**依次**跑了两个进程（b10 OOM 后再跑 b9），没有并发。基线 torch b7（reserved 45–46.5 GiB）按推算放不进 46068 MiB 卡，未实测。

## 3. 显存归因 ablation（Task 3，CUDA op，b7，单卡）

| 改动 | alloc GiB | Δ | img/s | 参数 total / trainable |
|---|---:|---:|---:|---|
| 基准（q100、pts 12544、enc 6、ckpt on） | 30.64 | — | 5.34 | 114,357,222 / 28,687,590 |
| `num_queries` 50 | 29.22 | −1.42 | 5.74 | 114,331,622 / 28,661,990 |
| `num_points` 4096 | 30.64 | **0.00**（峰值不在 loss） | 5.54 | 同基准 |
| `encoder_layers` 3 | **23.64** | **−7.00**（约 2.33 GiB/层，即 0.33 GiB/层/图） | 6.59 | 110,584,710 / 24,915,078 |
| ckpt off | 39.72 | +9.08 | 5.84 | 同基准 |
| pure-PyTorch deformable attn | 41.29 | +10.65 | 4.19 | 同基准 |

参数拆分（ViT-B）：ViT 85,669,632（冻结），adapter（不含 ViT）9,972,512，pixel decoder 8,991,168，transformer decoder 9,723,910。ViT-L frozen 时 total 338,785,670 / trainable 35,631,494。

每多 1 张图的边际显存：CUDA ckpt on 约 4.27 GiB，CUDA ckpt off 约 5.5 GiB，torch path 约 5.8 GiB（按旧报告 b3→b7 推算）。显存中占比最大的是 6 层 pixel-decoder encoder，其激活没有 checkpointing（约 2 GiB/图）。如果在内存中为这些层加 checkpointing，有望把 ckpt-on 的最大 batch 提到 12 以上，但这一点**未测试**。

## 4. 数据加载吞吐（Task 4，`task4_dataloader.json`）

独立 loader 复刻 `MoonDataset.__getitem__`（RGBA→RGB、`decode_label`（P-mode）、hflip 0.5、brightness 0.5、contrast 0.5、normalize、`collate_samples`，batch 7）。数据为 `data/LuSNAR/Moon_1` 的真实文件，每种 worker 数用 210 张不同文件。存储为 `/home/disk02`（rotational HDD）。

| workers | 首轮 img/s | 二次（page cache）img/s | 首 batch 延迟 s |
|---:|---:|---:|---:|
| 1 | 6.0 | 5.7 | 1.5 |
| **4** | **21.1** | 21.3 | 1.6 |
| 8 | 32.9 | 35.7 | 1.8 |
| 16 | 57.4 | 65.1 | 1.9 |

单进程每样本 157 ms，其中 `decode_label` 98 ms（62%），RGB 解码 31 ms，normalize/tensor 21 ms，augment 8 ms。

结论：最快的训练配置每 rank 需要 11.72/2 ≈ 5.9 img/s，4 worker 可提供约 21 img/s，余量约 3.6 倍，2 worker 也够用。`persistent_workers=False` 使每个 epoch 多约 1.5 s 启动时间，可以忽略。8002 张 RGB 约 13 GB，第一个 epoch 后可以完全进入 page cache（500 GB RAM）。可选优化：`decode_label` 改为 palette LUT，可再快约 10 倍，但目前没有必要。注意"首轮"未必是真正的冷读：无 root 权限无法 drop cache，文件可能已被缓存。

## 5. 补丁与可复现构建（Task 5）

- 补丁：`$TMPDIR/moonseg_perf/use_cuda_ms_deform_attn.patch`（unified diff，对 `moonseg/models/dinov3_mask2former.py`，**未应用**；`git apply --check` 通过）。
  - 新增配置项 `model.deformable_attention: auto|cuda|torch`，默认 `auto`。另有可选的 `model.deformable_attention_extension_dir`，也可用环境变量 `MOONSEG_MSDA_DIR` 指定，默认 `<repo>/build/ms_deform_attn`。
  - `auto`：能 import 编译好的 `MultiScaleDeformableAttention` 时使用官方 forward + backward kernel，否则 warning 并回退到现有 pure-PyTorch path。ABI 不匹配产生的 `ImportError`/`OSError` 也会回退。
  - `cuda`：extension 不可用时直接 `RuntimeError` 并提示运行 `scripts/build_ops.sh`。`torch`：行为与当前代码完全一致。
  - wrapper 在 autocast 关闭的状态下把输入 cast 成 fp32、输出转回 value dtype，与现有 torch path 数值语义相同。CPU tensor 自动走 PyTorch path。`im2col_step` 取 `gcd(batch, 64)`，以避开 kernel 的整除断言。
  - `model.deformable_attention` 属性记录实际使用的 backend，不进入 `state_dict`，因此 checkpoint 在 cuda 和 torch 两种模式之间可以互相加载。
- 验证（`task5_patch_test.jsonl`，在 `$TMPDIR` 的仓库副本上打补丁后运行）：

| 用例 | 选中 backend | 结果 |
|---|---|---|
| torch / auto / cuda，CUDA，bf16 | torch / cuda / cuda | loss 72.180 / 72.166 / 72.166 |
| torch vs cuda，CUDA，fp32 | torch / cuda | loss 72.17916 vs 72.17907（相对差 1e-6），grad norm 241.38 vs 241.30 |
| auto，CPU | cuda（tensor 在 CPU 上，实际走 torch） | 正常 |
| auto，extension 缺失 | torch | 发出 warning 并回退 |
| cuda，extension 缺失 | — | 按预期 RuntimeError |

- 构建脚本建议稿：`$TMPDIR/moonseg_perf/proposed/scripts/build_ops.sh`。已在副本上实测通过：34 s，生成 `.so` + `build_manifest.json`。
  - 用 uv 环境的解释器（`uv run --no-sync python`，或 `PYTHON=` 覆盖），默认 `CUDA_HOME=/usr/local/cuda-12.6`、`TORCH_CUDA_ARCH_LIST="8.6;8.9"`，并检查 nvcc 与 `torch.version.cuda` 是否匹配。
  - 把 `third_party` 的 ops 源码复制到临时目录后构建，不修改 vendor。产物用 `--build-lib` 写到 `build/ms_deform_attn/`，并写入 `build_manifest.json`，记录 `.so` sha256、源码 sha256、torch/CUDA/Python/arch。
  - 上游 `setup.py` 在 `torch.cuda.is_available()` 为假时直接拒绝构建，所以需要一张可见 GPU（`BUILD_GPU=0`）。
  - **不建议安装进 `.venv`**：`uv sync --locked` 会删除 lock 之外的包。代码包只附带脚本，不附带 `.so`；目标机运行 `bash scripts/setup_env.sh && bash scripts/build_ops.sh` 即可复现。
  - 需要在 `.gitignore` 增加 `/build/`。可选在 `pyproject` 加 `ninja`；目前用的是系统 `/usr/bin/ninja`，没有 ninja 时会退回较慢的串行编译。
  - 建议在 `train.py` 中把 `unwrap_model(model).deformable_attention` 写进 `resolved_config` 或日志，以便复现（不在本补丁范围内）。

## 6. 80 epoch 墙钟估算（8002 张，strict tail，仅训练；不含验证、每 epoch 约 5–10 s 的 checkpoint 写盘）

| 配置 | 硬件 | img/s | 张/epoch | s/epoch | 80 epoch |
|---|---|---:|---:|---:|---:|
| a 基线 torch b7×2×5 | 2×Ada（实测） | 8.29 | 7980 | 963 | **21.4 h** |
| b CUDA ckpt on b7×2×5 | 2×Ada（实测） | 10.82 | 7980 | 738 | 16.4 h |
| b CUDA ckpt on b10×2×4 | 2×Ada（实测） | 10.59 | 8000 | 755 | 16.8 h |
| **c CUDA ckpt off b7×2×5** | 2×Ada（实测） | 11.72 | 7980 | 681 | **15.1 h** |
| d CUDA unfrozen b7×2×5 | 2×Ada（实测） | 8.81 | 7980 | 906 | 20.1 h |
| e ViT-L frozen b7×2×5 | 2×Ada（单卡×2×0.99 估算） | 8.26 | 7980 | 967 | 21.5 h |
| e ViT-L unfrozen b4×2×9 | 2×Ada（估算） | 5.60 | 7992 | 1427 | 31.7 h |
| c CUDA ckpt off b7×4×3（eff 84） | 4×Ada（假设；本机只有 2 张） | 22.9 | 7980 | — | 7.7 h |
| c CUDA ckpt off b7×4×3 | 2 Ada + 2 A6000 或 4 A6000（受 A6000 速度限制） | 15.0 | 7980 | — | 11.8 h |
| b CUDA ckpt on b7×4×3 | 2 Ada + 2 A6000 | 13.8 | 7980 | — | 12.9 h |
| a 基线 torch b7×4×3 | 2 Ada + 2 A6000 | 10.8 | 7980 | — | 16.4 h（且 46068 MiB 卡放不下） |

开发配置（6403 训练 + 1599 验证）按比例为训练时间 ×0.8，再加验证时间（CUDA 约 +1.1 h，torch 约 +1.5 h）。4 卡数字都是估算，**没有实测**：混合 Ada/Ampere 的 DDP 会被最慢的卡限速，A6000 速度比 0.66 也来自一次可能受 allocator 压力影响的测量。

## 7. 局限与诚实说明

- 输入是合成的（形状固定，标签块状），不包含真实 dataloader、验证和 checkpoint 写盘，也不涉及长程 allocator 碎片和精度。真实标签只会改变 Hungarian/point sampling 的少量 CPU/GPU 时间。
- 接近上限的配置有风险。b10 ckpt on（reserved 46.25 / 47.37 GiB）和 unfrozen b7（46.42 GiB reserved）余量不到 1.5 GiB；`nockpt_b8` 跑过 3 个 step 后在第 4 个 step OOM，说明碎片化风险是真实存在的。长跑建议使用 reserved ≤ 44 GiB 的配置，例如 c（43.7）或 b7 ckpt on（35.4）。
- ViT-L 使用随机初始化（没有权重），只能说明显存和速度。ViT-L 的 DDP 未实测。
- CUDA 与 torch backward 都有 atomic/nondeterministic 行为，bitwise 复现不受影响也无法保证。
- 4-level stress shape 下 fp32 loc-grad 的少数 outlier 没有用 fp64 判定哪一方正确；模型实际使用的 3-level 形状已用 fp64 复核，CUDA 更准确。
- 吞吐测量的 phase 同步会带来约 1% 的开销（DDP 对比：11.55 vs 11.72、8.28 vs 8.29）。
