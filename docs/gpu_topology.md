# 主机 GPU 拓扑、NUMA 与 P2P 实测

核对日期：2026-10-21。主机 vrlab-132，2 路 Intel Xeon Platinum 8350C（2 个 NUMA 节点，128 逻辑核），8 张 48 GB 级 GPU。所有脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，下文序号均为 PCI 序（与 `nvidia-smi` 一致）；不设该变量时 CUDA 默认按"最快优先"排序，两张 Ada 会变成 `cuda:0/1`，PCI 5/6 会变成 `cuda:6/7`。

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

测试方法：`torch.distributed` NCCL 2.26.2，`all_reduce` 一个 28.7M 元素的 fp32 张量（114.8 MB，等于本项目每次优化器更新要同步的梯度量）和一个 1 GiB 张量，5 次预热、20 次计时，校验结果正确性；另做裸 `cudaMemcpyPeer` 拷贝校验。原始数据在 `/home/vrlab/.cache/claude-workbench/claude-vrlab-132-nat-moon-moon/moonseg_gpu_pairs/results/`。

| 卡对 | 驱动报告可 P2P | 默认 NCCL | 115 MB all-reduce | 1 GiB all-reduce | 结论 |
|---|---|---|---|---|---|
| 0 + 7（Ada + Ada，跨 NUMA） | 是 | 正常 | 9.00 GB/s | 9.49 GB/s | 直接用 |
| 5 + 6（A6000 + A6000，同 NUMA、PXB） | 是 | **初始化挂死**，超时退出码 124 | — | — | 裸 P2P 拷贝约 100% 数据未写入目标显存 |
| 5 + 6，`NCCL_P2P_DISABLE=1` | — | 正常 | 6.88 GB/s | 6.93 GB/s | **必须带此变量** |
| 5 + 7、6 + 7（A6000 + Ada） | 否（跨代不支持 P2P） | 正常，走主机内存 | 5.73 GB/s | 5.9–6.0 GB/s | 直接用 |
| 0 + 2、2 + 5、2 + 6 | — | 未测 | — | — | GPU 2 测试时被占用 |

关键事实：

- 同代卡之间驱动才报告 P2P 可用：Ada↔Ada 为真且工作正常（即使跨 NUMA、经 UPI）；A6000↔A6000 为真但 **实际 PCIe 直连拷贝丢数据**，NCCL 信任驱动而挂死。跨代（Ada↔A6000）驱动直接报告不可 P2P，NCCL 自动走共享内存通路，反而没有问题。
- A6000 对的故障通常来自 PCIe 交换芯片的 ACS/IOMMU 设置把同域直连流量重定向后丢弃，属于 BIOS/内核层面，本项目不处理；其它 A6000 对（1+3、2+3 等）未测，默认按同样方式加 `NCCL_P2P_DISABLE=1`。
- `NCCL_P2P_DISABLE=1` 只改变通信路径，不改变数值结果（all-reduce 校验通过）。对本项目每次更新 115 MB 的梯度同步，耗时约 17 ms，远小于一次 8×4 累积更新的数秒计算时间，因此带宽差异对训练时间没有可见影响。

## 启动约定

```bash
# Ada 对（默认）：PCI 0,7
PYTHON=.venv/bin/python bash scripts/train.sh <config>

# A6000 对：PCI 5,6，必须关闭 NCCL P2P
CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 PYTHON=.venv/bin/python bash scripts/train.sh <config>
```

GPU 1、3 的 ECC 版显存少 3 GiB，1024² 配置的每卡 batch 上限相应下降（A6000 49140 MiB 型号实测可放 batch 9，46068 MiB 型号实测 batch 9 通过、10 OOM，见 `audit_2026-10-21/gpu_capacity.md`）。混合 Ada 与 A6000 组 DDP 时整体按 A6000 的速度运行。

2026-10-21 实际启动：ViT-L 基线在 PCI 0/7（每卡约 45.6 GB），ViT-B 基线在 PCI 5/6（每卡约 40.9 GB，P2P 关闭）。
