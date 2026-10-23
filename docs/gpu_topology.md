# 主机 GPU：拓扑、P2P 与显存容量

核对日期：2026-10-21。主机 vrlab-132，2 路 Intel Xeon Platinum 8350C（2 个 NUMA 节点，128 逻辑核），8 张 48 GB 级 GPU，驱动 580.126.09。所有脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，下文序号均为 PCI 序（与 `nvidia-smi` 一致）；不设该变量时 CUDA 默认按"最快优先"排序，两张 Ada 会变成 `cuda:0/1`，PCI 5/6 会变成 `cuda:6/7`。

## 卡位与 NUMA

| PCI 序号 | 型号 | PCI 总线 | NUMA 节点 | 显存 | ECC | 2026-10-21 状态 |
|---|---|---|---|---|---|---|
| 0 | RTX 6000 Ada | 4F | 0 | 49140 MiB | 关 | 本项目（ViT-L 基线） |
| 1 | RTX A6000 | 52 | 0 | 46068 MiB | **开** | 空闲 |
| 2 | RTX A6000 | 56 | 0 | 49140 MiB | 关 | 其他用户 |
| 3 | RTX A6000 | 57 | 0 | 46068 MiB | **开** | 其他用户 |
| 4 | RTX A6000 | 98 | 1 | 49140 MiB | 关 | 其他用户 |
| 5 | RTX A6000 | CE | 1 | 49140 MiB | 关 | 本项目（ViT-B 基线） |
| 6 | RTX A6000 | D1 | 1 | 49140 MiB | 关 | 本项目（ViT-B 基线） |
| 7 | RTX 6000 Ada | D6 | 1 | 49140 MiB | 关 | 本项目（ViT-L 基线） |

两张 A6000 显示 46068 MiB 是因为开启了 ECC（约占 6% 显存），不是另一种型号。NUMA 0 的 CPU 为 0–31、64–95，NUMA 1 为 32–63、96–127；GPU 0–3 挂在 NUMA 0，GPU 4–7 挂在 NUMA 1。

`nvidia-smi topo -m`（PIX = 同一 PCIe 交换芯片；PXB = 同一主桥下的多级桥；NODE = 同 NUMA 不同主桥；SYS = 跨 NUMA 经 UPI）：

```text
      0    1    2    3    4    5    6    7
0     X   PXB  PXB  PXB  SYS  SYS  SYS  SYS
1    PXB   X   PXB  PXB  SYS  SYS  SYS  SYS
2    PXB  PXB   X   PIX  SYS  SYS  SYS  SYS
3    PXB  PXB  PIX   X   SYS  SYS  SYS  SYS
4    SYS  SYS  SYS  SYS   X   NODE NODE NODE
5    SYS  SYS  SYS  SYS  NODE  X   PXB  PXB
6    SYS  SYS  SYS  SYS  NODE PXB   X   PXB
7    SYS  SYS  SYS  SYS  NODE PXB  PXB   X
```

## P2P 与 NCCL 实测

测试方法：`torch.distributed` NCCL 2.26.2，`all_reduce` 一个 28.7M 元素的 fp32 张量（114.8 MB，等于 ViT-B 每次优化器更新要同步的梯度量）和一个 1 GiB 张量，5 次预热、20 次计时，校验结果正确性；另做裸 `cudaMemcpyPeer` 拷贝校验。原始数据在 [evidence/gpu_results/p2p/](evidence/gpu_results/p2p/)（`nccl_*.json` 为 all-reduce 结果，`p2p_copy*_5_6.json` 为裸拷贝校验）。

| 卡对 | 驱动报告可 P2P | 默认 NCCL | 115 MB all-reduce | 1 GiB all-reduce | 结论 |
|---|---|---|---|---|---|
| 0 + 7（Ada + Ada，跨 NUMA） | 是 | 正常 | 9.00 GB/s | 9.49 GB/s | 直接用 |
| 5 + 6（A6000 + A6000，同 NUMA、PXB） | 是 | **初始化挂死**，超时退出码 124 | — | — | 裸 P2P 拷贝约 100% 数据未写入目标显存 |
| 5 + 6，`NCCL_P2P_DISABLE=1` | — | 正常 | 6.88 GB/s | 6.93 GB/s | **必须带此变量** |
| 5 + 7、6 + 7（A6000 + Ada） | 否（跨代不支持 P2P） | 正常，走主机内存 | 5.73 GB/s | 5.9–6.0 GB/s | 直接用 |
| 0 + 2、2 + 5、2 + 6 | — | 未测 | — | — | GPU 2 测试时被占用 |

- 同代卡之间驱动才报告 P2P 可用：Ada↔Ada 为真且工作正常（即使跨 NUMA、经 UPI）；A6000↔A6000 为真但**实际 PCIe 直连拷贝丢数据**，NCCL 信任驱动而挂死。跨代（Ada↔A6000）驱动直接报告不可 P2P，NCCL 自动走共享内存通路，反而没有问题。
- A6000 对的故障通常来自 PCIe 交换芯片的 ACS/IOMMU 设置把同域直连流量重定向后丢弃，属于 BIOS/内核层面，本项目不处理；其它 A6000 对（1+3、2+3 等）未测，默认同样加 `NCCL_P2P_DISABLE=1`。
- `NCCL_P2P_DISABLE=1` 只改变通信路径，不改变数值结果（all-reduce 校验通过）。每次更新 115 MB 的梯度同步约 17 ms，远小于一次 8×4 累积更新的数秒计算时间，带宽差异对训练时间没有可见影响。

## 启动约定

当前两条基线的卡位见 [training.md](training.md#gpu-放置与启动)，命令见 [README](../README.md#训练验证与推理)。Ada 对 PCI 0/7 是脚本默认值；A6000 对 PCI 5/6 启动时必须加 `CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1`。混合 Ada 与 A6000 组 DDP 时整体按 A6000 的速度运行。

## 显存与吞吐容量

实际运行（2026-10-21 基线，每卡 batch 8、EMA 开、真实数据，`nvidia-smi` 读数）：ViT-L 在 PCI 0/7 每卡约 45.6 GB，ViT-B 在 PCI 5/6 每卡约 40.9 GB。

其余数字都来自 2026-10-21 的短程合成负载：1024×1024 随机归一化图像加块状五类标签（顶部 1/16 为 ignore），使用真实的 `build_model`、`build_criterion`、优化器和与 `train.py` 相同的 step 逻辑（BF16 autocast、`clip_grad_norm_(0.1)`、AdamW、DDP `no_sync`），3 步预热加 10 步计时。它们不含真实读图、增强、EMA、验证和 checkpoint 写盘，也不代表精度。显存为 PyTorch 峰值 allocated / reserved（GiB）。原始结果在 [evidence/gpu_results/](evidence/gpu_results/)，训练基准的文件名为 `train_<tag>.json`。

### 双卡 DDP（PCI 0/7，RTX 6000 Ada，ViT-B/16 冻结，CUDA 扩展）

| tag | 设置 | 有效 batch | 每卡 allocated / reserved | 总吞吐 |
|---|---|---:|---|---:|
| `b_ddp_cuda_b7_a5` | 每卡 7 × 累积 5，激活检查点开 | 70 | 30.85 / 35.37 | 10.82 img/s |
| `b_ddp_cuda_b10_a4` | 每卡 10 × 累积 4，激活检查点开 | 80 | 43.67 / 46.65 | 10.59 img/s |
| `c_ddp_cuda_nockpt_b7_a5_nosync` | 每卡 7 × 累积 5，激活检查点关 | 70 | 39.94 / 43.68 | 11.72 img/s |
| `d_ddp_cuda_unfrozen_b7_a5` | 每卡 7 × 累积 5，ViT-B 全解冻 | 70 | 43.81 / 46.42 | 8.81 img/s |

这些基准用的是当时的 7×5 或 10×4 计划，每卡 8 的 DDP 没有单独合成测试，实际运行见上。编译 CUDA 扩展之前的纯 PyTorch 路径（`a_ddp_torch_b7_a5_nosync`）在同一 7×2×5 设置下为 41.51 / 45.02 GiB、8.29 img/s；当时不累积的单卡实测为 batch 7 通过、8 OOM，24 GiB 预算模拟为 batch 3 通过、4 OOM（`/home/disk02/wjj/moonseg/verification/capacity*.json`）。DDP 扩展效率约 0.99–1.0。全解冻时所有可训练参数都有梯度，`find_unused_parameters=False` 可用（`unfrozen_unused_parameters.json`）；全解冻的数字测于 ViT 固定为评估模式之前。

### 单卡最大 batch（每卡一次前向的图像数，不累积）

| 骨干与设置 | 卡 | 通过的最大 batch：allocated / reserved，吞吐 | 下一档 |
|---|---|---|---|
| ViT-B 冻结，检查点开 | Ada（可用 47.37 GiB） | 10：43.44 / 46.25，5.38 img/s | 11 OOM |
| ViT-B 冻结，检查点关 | Ada | 7：39.72 / 43.56，5.84 img/s | 8 在第 4 步 OOM |
| ViT-B 全解冻，检查点开 | Ada | 7：42.95 / 45.99，4.03 img/s | 8 OOM |
| ViT-L 冻结，检查点开（随机权重） | Ada | 9：42.87 / 45.65，3.54 img/s（已因显存紧张变慢） | 10 OOM |
| ViT-L 全解冻，检查点开（随机权重） | Ada | 4：40.22 / 43.41，2.83 img/s | 5 OOM |
| ViT-B 冻结，检查点开 | A6000 46068 MiB（PCI 1，可用 44.42 GiB） | 9：39.18 / 43.41，3.57 img/s | 10 OOM |
| ViT-H+ 冻结，检查点开（真实权重，2026-10-22） | A6000 49140 MiB（PCI 2） | 8：42.95 / 46.45，2.35 img/s；40 步 + 中途 20 张 1024² 推理未 OOM，reserved 稳定在 46.45 GiB（余量约 0.95 GiB，`bench_hplus_b8_long.json`） | 未测 9 |
| 同上 | 同上 | 6：33.16 / 37.49，2.39 img/s；4：23.38 / 26.82，2.34 img/s | — |

- H+ 的每图边际显存约 4.9 GiB，吞吐对 batch 不敏感；当前用户选定每卡 8 × 两卡 × 累积 4 = 64。短程容量证据见 [evidence/gpu_results/hplus/](evidence/gpu_results/hplus/)，不能把合成负载通过视为任意增强长跑均已验证。
- 每卡吞吐对 batch 基本不敏感：ViT-B 冻结、检查点开时 b4/b7/b8/b10 为 5.51/5.34/5.48/5.38 img/s（b8 为 34.90 / 39.19 GiB），GPU 在 b4 已饱和，加大 batch 只减少累积步数。ViT-L 冻结 b4/b6/b7/b8 为 4.33/4.24/4.17/4.13 img/s（b7 为 33.69 / 37.64 GiB，b8 为 38.28 / 42.81 GiB）。
- A6000（46068 MiB）b9 约为 Ada 速度的 0.66 倍；该测量的 reserved 已接近上限，0.66 偏保守。49140 MiB 版 A6000 没有做容量测试；当前 ViT-B 基线在 PCI 5/6 以每卡 batch 8 运行。
- 长跑宜选 reserved ≤44 GiB 的设置：检查点关、batch 8 的单卡测试跑过 3 步后在第 4 步 OOM，说明碎片化风险真实存在。

2026-10-23 的真实 H+ 整图训练以每卡 8 完成 25 轮；相同 batch 的 RRC 训练在第 1 轮验证后、第 2 轮损失采样处 OOM（allocated 41.86 GiB，reserved 未使用 4.81 GiB，申请 2 MiB 失败）。已修复跨批次和保存对象的引用生命周期，每轮结束释放空闲缓存，并在恢复进程启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；配方与 batch 未变。日志支持碎片推断，未取得原 allocator snapshot，不能断言单一根因。13:44 已完成恢复后的第 2 轮训练、验证和保存，并在第 3 轮完成 10 次更新；第 2 轮两卡峰值 allocated/reserved 均为 45.113/45.260 GiB，完整训练/验证切换通过，25 轮长跑仍在继续。原始日志为 `runs/dnv3_va_m2f_vith16plus_rrc/train.recovery_2026-10-23.log`，核对记录为 `runs/plans/2026-10-23/recovery_gpu_0_7/verification.json`。
- 激活检查点开/关（ViT-B，b7）：关闭后每卡多 9.08 GiB，单卡吞吐 5.34→5.84（+9%），双卡 10.82→11.72（+8%）。

### 显存归因（ViT-B 冻结，CUDA 扩展，单卡 b7）

| 改动 | allocated GiB | 变化 | 吞吐 img/s |
|---|---:|---:|---:|
| 基准（100 query、12544 点、encoder 6 层、检查点开） | 30.64 | — | 5.34 |
| `num_queries: 50` | 29.22 | −1.42 | 5.74 |
| `num_points: 4096` | 30.64 | 0.00（峰值不在损失） | 5.54 |
| `encoder_layers: 3` | 23.64 | −7.00（约 0.33 GiB/层/图） | 6.59 |
| 激活检查点关 | 39.72 | +9.08 | 5.84 |
| 纯 PyTorch 可变形注意力 | 41.29 | +10.65 | 4.19 |

每多 1 张图的边际显存：CUDA 扩展、检查点开约 4.27 GiB，检查点关约 5.5 GiB。最大的占用是没有激活检查点的 6 层 pixel-decoder encoder（约 2 GiB/图）。参数拆分（ViT-B）：ViT 85,669,632（冻结）、Adapter（不含 ViT）9,972,512、pixel decoder 8,991,168、transformer decoder 9,723,910，合计 114,357,222，可训练 28,687,590；ViT-L 冻结时合计 338,785,670，可训练 35,631,494。

### 可变形注意力算子

官方 `MultiScaleDeformableAttention` CUDA 扩展与 `ms_deform_attn_core_pytorch` 的前向在 FP32 下最大绝对差 ≤6.6e-6，FP64 下前向与三个梯度 ≤1.6e-12，`torch.autograd.gradcheck` 全部通过；模型真实 3-level 形状的 FP32 位置梯度相对 FP64 参考，CUDA 误差 4.0e-4、PyTorch 4.6e-3，CUDA 更准。kernel 拒绝 BF16/FP16 输入，因此包装层在 autocast 关闭时转为 FP32。1024² 训练形状（N=7）下，pixel-decoder encoder 层前向+反向 10.4 ms 对 66.0 ms，额外峰值显存 0.90 对 6.27 GiB。证据为 `task1_op_verification.json` 与 `task1_op_verification_extra.json`（前者第一版 `op_level_perf` 已标为 INVALID）。构建与选择逻辑见 [model.md](model.md#compiled-deformable-attention)。

### 推理与数据加载

- 推理（单卡 Ada，BF16，`inference_mode`，含语义分数与 argmax，不含 PNG 读写）：batch 1 单尺度 15.93 img/s（纯 PyTorch 路径 11.46），batch 8 不更快；3 尺度（0.75/1.0/1.25）× 翻转共 6 次前向为 2.49 img/s。5004 张测试图在双卡上的纯 GPU 时间：单尺度约 157 s，TTA 约 1004 s（17 min）。证据 `infer_cuda.json`、`infer_torch.json`。
- 数据加载（复刻 `MoonDataset.__getitem__`，场景 1 真实文件，存储为 `/home/disk02` 机械盘）：每 rank 1/4/8/16 个 worker 为 6.0/21.1/32.9/57.4 img/s，而最快的训练设置每 rank 只需约 5.9 img/s。该测量中标签解码占 98 ms/张，此后已改为调色板查表（7.0 ms/张，见 [data.md](data.md#文件格式与标签解码)），现在更快。证据 `task4_dataloader.json`。8002 张 RGB 约 13 GB，第一轮后可完全进入 page cache。
