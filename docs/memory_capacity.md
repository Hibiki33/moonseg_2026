# 1024×1024 训练显存容量报告（不使用梯度累积）

日期：2026-10-21。以下batch均指**每张GPU一次前向/反向处理的图像数**，`accumulation_steps=1`。在当前完整模型和精度配置下，实测48GB卡可运行batch 7；24GiB预算模拟可运行batch 3。下一档8和4分别出现OOM。

## 固定实验条件

| 项目 | 设置 |
|---|---|
| 模型 | DINOv3 ViT-B/16 + ViT-Adapter + Mask2Former |
| 分割头 | 完整配置：hidden_dim256、100 queries、6层pixel encoder、6层masked decoder |
| 输入 | 1024×1024 RGB；合成五类语义标签，包含ignore255 |
| 预训练与冻结 | 加载实际DINOv3权重，冻结原始ViT；Adapter和head可训练 |
| 参数量 | 总计114,357,222；可训练28,687,590 |
| 精度 | BF16 autocast；匹配、损失和deformable grid sampling使用FP32 |
| 优化器 | AdamW，真实执行更新，包含梯度和已初始化的优化器状态 |
| 检查点机制 | 保留当前Adapter activation checkpointing设置 |
| 测量方式 | 每个候选用全新单GPU进程，关闭梯度累积；成功候选完成5次优化器更新 |
| 环境 | uv Python3.12.15、PyTorch2.7.1+cu126、CUDA12.6、RTX6000 Ada48GB |

这里的48GB实卡测试是单GPU训练，没有DDP通信缓冲。当前项目的双卡训练另有batch7、累积5的独立验证，见文末。

## 实测结果

所有显存数字以GiB（2^30 bytes）计算，表示PyTorch记录的峰值。

| 显存条件 | 每卡batch | 结果 | 峰值allocated | 峰值reserved | 完成优化器更新 |
|---|---:|---|---:|---:|---:|
| 48GB实卡 | **7** | **通过** | **41.24 GiB** | **45.64 GiB** | 5 |
| 48GB实卡 | 8 | OOM，首次前向阶段 | — | — | 0 |
| 24GiB预算模拟 | **3** | **通过** | **18.09 GiB** | **19.94 GiB** | 5 |
| 24GiB预算模拟 | 4 | OOM，首次前向阶段 | — | — | 0 |

48GB的batch8在首次前向还需申请400MiB时失败，当时CUDA剩余约317.5MiB；因此它没有完成一次完整训练更新，不能把它列为“可训练”。24GiB预算下batch4在申请200MiB时触及进程内存上限，同样不能列为可用。

容量参考可以记为：**当前配置，48GB从每卡7开始，24GB从每卡3开始**。这是对相邻batch档位的实际验证，不是按显存大小线性估算。未通过关闭loss、缩小head、减少queries或省略优化器状态来得到这些数字。

## 24GiB模拟的边界

机器没有24GB实体卡，所以这项**不是24GB硬件实测**。使用另一张48GB卡，将PyTorch allocator上限设为23GiB，另留1GiB给CUDA上下文及非PyTorch分配，模拟24GiB总预算。成功batch3的显存占用低于该预算，batch4则失败。

真实24GB卡的架构、CUDA可见容量、cuDNN工作区和驱动开销可能不同；应在目标卡上复核。模拟运行的速度来自48GB Ada卡，不能当作24GB卡的吞吐性能。

## 与当前双卡训练配置的关系

“不考虑梯度累积”的报告用于确定每卡microbatch容量；实际训练还要满足用户要求的**全局有效batch严格大于64**。因此当前配置为：

```text
每卡7 × 2卡 × 累积5步 = 全局有效batch70
min_effective_batch = 65
```

该组合独立完成了5次实际DDP/AdamW更新：每卡峰值allocated41.46GiB、reserved45.73GiB，设备可用显存约0.96GiB，预热后双卡总吞吐约8.33张/秒。它已经接近48GB卡的容量上限。启动校验会拒绝有效batch64，epoch尾部只执行完整70张的更新。

单卡容量报告与双卡累积报告分别存放，不能将单卡batch7解释成全局batch7，也不能把梯度累积后的70解释成每卡一次前向70张。Adapter BatchNorm仍按每卡microbatch统计。

## 适用范围与证据文件

这些结果是固定形状合成输入的短程训练容量检查。它们不包含真实数据加载、增强、完整验证、检查点写盘和长程分配变化，不表示分割精度。更换ViT-L、解冻骨干、修改输入尺寸、头部结构、精度或activation checkpointing后，需要重新测量。

证据位于 `/home/disk02/wjj/moonseg/verification/`：

- `capacity48_b7.json`、`capacity48_b8.json`：48GB成功档和OOM边界。
- `capacity24sim_b3.json`、`capacity24sim_b4.json`：24GiB预算模拟成功档和OOM边界。
- `1024_b7_a5_capacity.json`：当前双卡有效batch70的累积验证。
- `memory_capacity_summary.json`：汇总元数据。
