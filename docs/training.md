# 训练配方

更新：2026-11-08。后续新训练以 [`configs/dnv3_va_m2f_vitl16_default.yaml`](../configs/dnv3_va_m2f_vitl16_default.yaml) 为模板：冻结ViT-L、RRC、梯度裁剪0.01、decoder6层/hidden_dim256，五场景训练、场景5验证、有效batch64。复制模板时设置独立输出目录；25轮只是已测参考预算，最合适轮数尚待校准。当前保留独立场景5验证集，可以按验证mIoU或验证loss选择best权重；后续场景5并入训练、没有独立验证集时，使用末轮last.pt。best_loss.pt指最低验证loss，不按训练loss选择模型。完整跑完训练预算与保存、选择验证best并不冲突；10-29的末轮对照是已执行的一组实验，不是所有开发实验的硬性选权重规则。现有结果见 [roadmap.md](roadmap.md#实验记录)。历史B/L的80轮、warmup/EMA配方及其YAML保留用于追溯，不能再当成新训练默认。

## 配方与理由

| 项目 | 当前值 | 理由 |
|---|---|---|
| 模型 | 冻结 DINOv3 ViT-L/16（LVD-1689M）+ 官方 ViT-Adapter + Mask2Former；100 query、hidden 256、pixel encoder 6 层、decoder 6 层 | 当前主线；decoder9的末轮对照未提升，结构见 [model.md](model.md) |
| 骨干 | `freeze_backbone: true`、`unfreeze_last_blocks: 0` | 先把冻结方案训到收敛；部分解冻是候选 |
| 输入 | `train_mode: random_resize_crop`，1024×1024 | 沿用L1基线；未加入场景5训练 |
| 几何增强 | resize_range 0.5–2.0，1024裁剪，cat_max_ratio0.75，水平翻转p=0.5 | 已完成L0/L1/L2比较，详细类别取舍见roadmap |
| 光度增强 | 亮度加性 U[−32,32]、对比度乘性 U[0.5,1.5]，各自独立 p=0.5；随后 `pixel_floor: 22` | DINOv3/MMSeg PhotoMetricDistortion 的标准强度；下限等于渲染器黑位 22 |
| 其余增强 | gamma、阴影、尺度抖动的概率为 0 | 已实现，默认关闭（[augmentation_research.md](augmentation_research.md)） |
| 归一化 | ImageNet mean/std | DINOv3 预训练输入约定 |
| 损失 | Hungarian 匹配；CE / mask BCE / Dice = 2 / 5 / 5，no-object 0.1，12544 个采样点，每个解码层深监督 | Mask2Former 原始配方 |
| 有效 batch | 每卡 8 × 2 卡 × 累积 4 = 64；`min_effective_batch: 64` | 用户要求不低于 64；每卡 8 在 48 GB 卡上可运行，ViT-L 实际每卡约 45.6 GB（[gpu_topology.md](gpu_topology.md#显存与吞吐容量)） |
| 优化器 | AdamW，lr 2e-4，weight decay 0.05（偏置、一维张量和嵌入表不衰减）；`backbone_lr_multiplier: 0.1` 只在 ViT 可训练时生效 | Mask2Former 原配方 batch 16 / lr 1e-4，按 √(64/16) 缩放到 2e-4 |
| 调度 | 无warmup，poly0.9按预设总更新数降到0 | 不同训练轮数的学习率轨迹不同；完整短训练不能由长训练中途快照替代 |
| 梯度裁剪 | 全局 L2 **0.01**，日志记录裁剪前范数 | 用户10-29指定后续默认；省略grad_clip时解析为0.01，显式0禁用，历史显式0.1保持 |
| 数值 | BF16 autocast + TF32，`cudnn_benchmark: true`；匹配、损失与可变形注意力在 FP32 | 速度；数值语义见 model.md |
| 可变形注意力 | `deformable_attention: auto`，选中已编译的官方 CUDA 扩展 | 比纯 PyTorch 路径省约 10.7 GiB、快约 31%（[gpu_topology.md](gpu_topology.md#显存与吞吐容量)） |
| 激活检查点 | `gradient_checkpointing: true` | 关闭只快约 8%，但每卡多约 9 GiB |
| EMA | 不使用（2026-10-22 移除；配置含 `train.ema` 会报错） | 所查官方分割配方都不用 EMA；两条基线中它只平滑了漂移的轨迹（见[验证协议](#验证协议)） |
| 训练长度 | 模板25轮，五场景2500次更新；最优预算未定 | 用户暂缓12/18轮搜索，当前先做width384的25轮单变量容量消融 |
| 采样 | `sampling.mode: uniform`，每轮重新打乱 | 稀有类采样已实现，作为候选 |
| 数据 | 当前配置仍为开发池 `dev_train.json`（场景1/3/6/7/8，6403帧）训练，`dev_val.json`（场景5，1599帧）验证 | 11-08已核对官方包内容完全一致；用户确认沿用当前数据与划分，已有训练测试结论有效，见 [data.md](data.md#官方数据一致性核对) |
| 验证 | 每轮验证 mIoU 与验证损失；中间轮 `eval_stride: 2`（800 帧），最后一轮全量 1599 帧 | 隔帧抽样的类别占比偏差 ≤0.4%（[data.md](data.md#时间冗余)） |
| 推理 | 模板为整图单尺度；公开评估另测0.9/1.0/1.1+翻转TTA | TTA已有正向证据；比较必须使用相同推理设置和权重选择口径 |
| checkpoint | 每轮覆盖 `last.pt`；验证 mIoU 新高时覆盖 `best_miou.pt`，验证损失新低时覆盖 `best_loss.pt`；`save_every: 0` | 不存冻结骨干；格式见 [model.md](model.md#checkpoints) |
| 其他 | seed 42，`num_workers: 4`，`log_every: 10` | 4 个 worker 每 rank 约 21 img/s，远高于训练需求（[gpu_topology.md](gpu_topology.md#显存与吞吐容量)） |

## Batch 计划

| 训练清单 | 样本数 | 每轮使用 / 跳过 | 每卡 microbatch / 轮 | 更新 / 轮 | 80 epoch 更新数 |
|---|---:|---:|---:|---:|---:|
| `dev_train`（场景 1/3/6/7/8） | 6403 | 6400 / 3 | 400 | 100 | 8000 |
| 全训练池（场景 1/3/5/6/7/8） | 8002 | 8000 / 2 | 500 | 125 | 10000 |

只执行完整的 64 张累积更新：每轮重新打乱后跳过尾部不足一组的样本，不让最后一次更新变成小 batch；实际数量写入 `batch_plan.json`。DDP 在累积组的最后一次反向传播时同步梯度，再统一 unscale、裁剪和更新；调度器总步数为 epoch 数 × 每轮更新数。恢复训练要求 batch 计划不变。

累积 64 只表示每次优化器更新的图像数，不等价于一次前向 64 张：Adapter 的普通 BatchNorm 按每卡 microbatch（8 张）统计，mask 损失的归一化也按当前分布式 microbatch 的目标数计算。

## 验证协议

- 训练启动时检查训练与验证的样本 ID、场景互不重叠。
- 每轮在场景 5 上验证；中间轮按 `(scene, sample_id)` 顺序每 2 帧取 1 帧（800 帧），最后一轮评估全部 1599 帧。`history.jsonl` 记录每轮的 `eval_stride`、`evaluated_samples`、`val_loss`、`miou_ma3`、`best_miou`/`best_miou_epoch`、`best_loss`/`best_loss_epoch`、`saved_best`（本轮写入的最佳文件）与完整的 `validation`（含 `val_loss`）。
- 验证损失 `val_loss`：与训练相同的 Mask2Former 总损失（最终层加各辅助层），每帧 batch 1、与训练相同的 autocast，模型处于 eval 模式，对所有 rank 上评估的帧取平均。当前整图单尺度推理下与 mIoU 共用同一次前向。点采样在 `fork_rng` 内用固定种子，所以结果可复现、同一帧各轮抽到相同的点，且不推进训练的随机数流。各 rank 的帧数可能不同（全量 1599 帧时为 800/799），评估期间关闭损失里按 rank 同步 mask 数的 `all_reduce`（否则会死锁），每帧用自己的 mask 数归一化。因此 `val_loss` 与 `train_loss` 量级相近但不严格可比：后者是 train 模式、每卡 8 张、跨 rank 平均归一化，且带数据增强。
- 最佳文件：验证 mIoU 严格高于此前所有验证时写 `best_miou.pt`，验证损失严格低于此前所有验证时写 `best_loss.pt`，两者可在同一轮写入，持平不覆盖；`last.pt` 每轮覆盖。`miou_ma3` 是最近 3 次验证 mIoU 的均值，只作参考，不参与选择。最佳文件可能在抽样子集上选出，子集分数不能直接与全量比较；需要时用 `moonseg.evaluate` 在全量上分别复评两个文件（命令见 README）。
- 场景5加入训练后设置 `validate: false`，不再产生验证best_miou/best_loss，最终使用 `last.pt`。`train_loss`仍记录用于观察优化过程，不用于挑选best模型。
- 移除 EMA（2026-10-22）的理由：所查官方分割配方（Mask2Former、DINOv3、ViT-Adapter，见[参考配方与证据](#参考配方与证据)）都不用 EMA；两条基线截至 2026-10-22 12:20（ViT-B 64 轮、ViT-L 67 轮，800 帧子集）的 `history.jsonl` 中，原始权重 mIoU 在第 12/15 轮达到峰值 0.871/0.920 后漂移回落，EMA 最高只有 0.829/0.909，`best_source` 始终是 `model`；近 20 轮 EMA 只把轮间标准差从 0.009/0.010 压到 0.003，均值比原始权重高 0.003/0.009，仍低于早期峰值。也就是说 EMA 只平滑了一条漂移的轨迹，却让每轮验证时间翻倍。验证损失给出与 mIoU 互补的第二条选择依据。EMA 时代的 `last.pt` 仍可用新配置恢复，EMA 权重被丢弃、最佳记录重新开始；旧 checkpoint 的 `ema_model` 只在显式指定 `--weights ema_model`/`--weights-key ema_model` 时加载，`auto` 一律用 `model`（[model.md](model.md#checkpoints)）。
- 指标为按场景累积混淆矩阵的 mIoU（只平均参考标签中出现的类别）、各类 IoU、mAcc 等，定义见 [metrics.md](metrics.md)。场景 5 分数不是排行榜成绩。
- 第二验证场景、暗帧子集和 block bootstrap 置信区间是候选（[roadmap.md](roadmap.md) #7，依据见 [data.md](data.md#验证场景)）。

## GPU 放置与启动

11-08两条全解冻训练已获用户确认并启动：`dnv3_va_m2f_vitl16_rrc_fullft`在0/7从原生DINOv3开始联合训练25轮/2500更新，Adapter/head与骨干初始lr为2e-4/2e-5；`dnv3_va_m2f_vitl16_rrc_clip001_stage2_fullft`在4/5从clip001的第15轮`best_miou.pt`权重初始化，完整解冻后以新优化器训练5轮/500更新，lr为2e-5/2e-6。均为每卡4×2卡×累积8=64、1024²/RRC/width256/decoder6/clip0.01；4/5关闭NCCL P2P，0/7保持已验证P2P路径。`--initialize-from`不同于`--resume`，阶段加载约定见[model.md](model.md#checkpoints)。

本批使用单计划启动器`scripts/run_finetuning_plan.py`，计划与启动状态在`runs/plans/2026-11-08/fullft/`中的`joint/stage2.json`与对应`.status.json`，训练日志在各运行目录的`train.log`。启动前核对物理UUID/进程占用，同时持有既有训练与评估GPU锁，并向torchrun继承锁描述符；历史队列和配置不改。训练完成后校验完整history、全量末轮验证及三个checkpoint，再写`queue_completed.json`。公开评估计划为同目录`public_test.json`，状态在`runs/public_development_fullft_2026-11-08/`；只等这两组卡释放，补评父权重及A/B的验证`best_miou.pt`，每项覆盖5004张单尺度与窄尺度翻转，不增第三组卡。

| 运行目录 | 配置 | 物理 GPU（PCI 序） | 启动时必需的环境变量 | 实际每卡显存 |
|---|---|---|---|---:|
| `runs/dnv3_va_m2f_vitl16` | `dnv3_va_m2f_vitl16.yaml` | 0、7（RTX 6000 Ada） | 无（脚本默认 `CUDA_VISIBLE_DEVICES=0,7`） | 约 45.6 GB |
| `runs/dnv3_va_m2f_vitb16` | `dnv3_va_m2f_vitb16.yaml` | 5、6（RTX A6000） | `CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1` | 约 40.9 GB |

启动、恢复、评估和预测命令见 [README](../README.md#训练验证与推理)。启动脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；PCI 5/6 卡对的 PCIe P2P 会静默丢数据，必须关闭 NCCL P2P，原因见 [gpu_topology.md](gpu_topology.md)。`runs` 是指向本地盘 `/home/disk02/wjj/moonseg/runs` 的软链接，不写 NAS。只能从 `last.pt` 恢复，且配置、数据清单和进程数必须与原运行一致；唯一例外是 EMA 时代的 `last.pt` 记录的 `train.ema`，恢复时被忽略。

## 预期墙钟时间

- 实测（EMA 时代两条运行前两轮的 `history.jsonl` `seconds`，含每轮两次 800 帧验证）：ViT-L 在 PCI 0/7 每轮 790–796 s，80 轮约 17.6 h；ViT-B 在 PCI 5/6 每轮 828–841 s，80 轮约 18.5 h。最后一轮全量验证再多几分钟。
- 与合成基准一致：ViT-L 在 Ada 上每卡 batch 8 为 4.13 img/s，6400 张约 780 s（不含验证）；ViT-B 在 A6000 上按 46068 MiB 卡测得的 0.66×Ada 估计约 900 s，实测更快，说明该比例偏保守。
- 移除 EMA 后每轮只验证一次，验证损失与 mIoU 共用同一次前向，只多出损失计算，新运行每轮应略短于上面的实测，尚未实测。
- 最终在全训练池重训（`validate: false`）时，每轮训练量为开发池的 1.25 倍且没有验证。
- 预测 5004 张测试图：双卡单尺度纯 GPU 时间约 157 s，3 尺度 × 翻转 TTA 约 17 min，PNG 读写另计（[gpu_topology.md](gpu_topology.md#显存与吞吐容量)）。

## 结果看什么

- 场景 5 的 mIoU 与验证损失、岩石和撞击坑 IoU、山体 IoU（场景 5 只有 48.3% 的帧有山体）、mAcc 与各类支持数。
- 曲线：`train_loss`、`val_loss`、`grad_norm_mean`/`grad_norm_max`（裁剪前范数）和每轮验证 mIoU（及 `miou_ma3`）是否在 80 轮内饱和或回落；`val_loss` 回升而 `train_loss` 继续下降是过拟合信号。
- 过拟合信号：训练损失持续下降而撞击坑、岩石 IoU 先升后降；暗区被系统性预测为撞击坑；加入第二验证场景后两者走势分歧。
- B 与 L 同配方并排比较，回答骨干容量是否重要；DINOv3 线性探测从 B 到 L 提升 3.1（见下），可作先验参照。
- 记录 `best_miou_epoch`、`best_loss_epoch`，在全量上复评 `best_miou.pt` 与 `best_loss.pt` 后填入 [roadmap.md](roadmap.md#实验记录)。

## 参考配方与证据

本节是方法与竞赛实践文献的唯一出处（2026-10-21 调研）。[weak] 表示只看到搜索摘要；其余条目当时直接读取了所列页面、源码或 API。增强相关文献（ShadowCast、HRDA、DAFormer）见 [augmentation_research.md](augmentation_research.md#文献证据)，LuSNAR 论文本身见 [paper_alignment.md](paper_alignment.md)。

### Mask2Former 官方配置

- ADE20K 语义分割 base：IMS_PER_BATCH 16，AdamW，BASE_LR 1e-4，WEIGHT_DECAY 0.05，BACKBONE_MULTIPLIER 0.1；WarmupPolyLR，WARMUP_ITERS 0，MAX_ITER 160k；全模型 L2 梯度裁剪 0.01，开启 AMP；crop 512²，短边 {0.5…2.0}×512、长边 ≤2048，COLOR_AUG_SSD，SINGLE_CATEGORY_MAX_AREA 1.0；TTA 默认关闭。[配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml)；Swin-L 版 crop 640²、drop_path 0.3（[配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml)）。
- Cityscapes base：90k iter，batch 16，lr 1e-4，同样的 0.01 全模型裁剪，SyncBN，crop 512×1024。[配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/cityscapes/semantic-segmentation/Base-Cityscapes-SemanticSegmentation.yaml)
- [Model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) 单尺度 → 多尺度+翻转：ADE20K R50 47.2→49.2、Swin-L 56.1→57.3；Cityscapes R50 79.4→82.2、Swin-B 83.3→84.5、Swin-L 83.3→84.3，即 TTA +1.0–2.8。
- 损失的不确定点采样（oversample 3、importance 0.75，本项目已实现）本身就是一种难例挖掘，额外的 boundary loss/OHEM 优先级低。

### DINOv3

- 官方 Mask2Former 配方（ViT-7B 冻结，[arXiv 2508.10104](https://arxiv.org/html/2508.10104v1) §6.3.2、Appendix D.10）：ViT-Adapter 去掉 injector，在第 [10,20,30,40] 层特征上加 final LayerNorm 与可学习 BN，Mask2Former 维度增到 2048，可训练 decoder 927M 参数。全部阶段 batch 16、AdamW、cosine：COCO-Stuff 80k iter（lr 1.5e-5、warmup 6k、分辨率 1280）→ Hypersim 10k（2.5e-5、warmup 1.5k）→ ADE20K 20k（3e-5、warmup 1.5k、896）。推理 896 滑窗、stride 596，TTA [0.9,0.95,1.0,1.05,1.1]×hflip；ADE20K 单尺度 62.6、TTA 63.0，Cityscapes 86.1/86.7。用 COCO-Stuff/Hypersim 预训练 decoder 属于额外分割数据，本比赛不允许（[roadmap.md](roadmap.md#红线)）。
- 官方仓库在本项目 vendor 的 revision `6876159` 只发布了 Mask2Former [推理配置](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml)和线性训练配置，没有 Mask2Former 训练配置。
- ADE20K 线性探测 @512（同一论文 Table 3/14/15）：ViT-S 47.0、S+ 48.8、B 51.8、L 54.9、H+ 54.8、7B 55.9（DINOv2 B 48.4、L 48.8）；ConvNeXt-DINOv3 T/S/B/L 为 42.7/44.8/46.3/47.8。ViT-L 是性价比拐点。论文只有冻结结果，没有 ViT-S/B/L/H+ 接 Mask2Former 的数字，也没有分割微调对照。

### 冻结与微调

- [Rein（CVPR 2024）](https://arxiv.org/html/2312.04265)，GTAV（合成）→ 真实域，DINOv2-L + Mask2Former：全量微调 63.7、冻结 61.1、Rein 66.4；EVA02-L 为 62.2/56.5/65.3。冻结 VFM 在 Cityscapes 上：DINOv2-L 63.3、SAM-H 57.0、EVA02-L 56.5、CLIP-L 53.7、MAE-L 43.3。训练为 AdamW，backbone lr 1e-5 / head 1e-4，40k iter，batch 4，crop 512。结论：域偏移下"低 lr 的少量可训练参数"优于两个极端。
- DINOv2-g + ViT-Adapter + Mask2Former（冻结）ADE20K 60.2（[arXiv 2304.07193v2](https://arxiv.org/html/2304.07193v2)）。
- [DINO Soars](https://arxiv.org/html/2605.03175v1) §4.4.1：只解冻最后 2 个 block，比完全冻结最差情况高 11.9 mIoU（开放词汇遥感分割，迁移性有限）。
- [DINOv3 ConvNeXt 工业检测研究](https://arxiv.org/abs/2605.23472)（摘要）：RGB 数据上冻结没有明显优势，全量微调后初始化更强。
- ViT-Adapter BEiTv2-L + Mask2Former：AdamW lr 2e-5、layer decay 0.90、wd 0.05、drop_path 0.3、1500 步线性 warmup、poly(1.0)、80k iter、crop 896、从 COCO-Stuff 的 Mask2Former 初始化，ADE20K 61.2/61.5（[配置](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py)、[README](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/README.md)）。

### 更新次数

本配方 80 epoch 为 8000（开发池）或 10000（全训练池）次 64 张更新，共见 51.2 万（开发池）张图。参照：DINOv3 ADE20K 阶段 20k（batch 16）、Rein 40k（batch 4）、ViT-Adapter 80k、Mask2Former 90k–160k（batch 16，ADE20K 160k×16 约 256 万张）。冻结骨干时可训练的 Adapter 与头部从随机初始化起步，匹配式训练一般需要较多更新才能稳定类别 logit；因此以验证曲线判断是否加长训练或提高 lr（候选 #6）。

### 竞赛常用技巧

- 多尺度+翻转 TTA：Mask2Former +1.0–2.8；DINOv3 只用 0.9–1.1 的窄比例也有 +0.4；ViT-Adapter BEiTv2 +0.3。
- EMA：HuBMAP-2023 第 1 名把 EMA 列为最关键技巧 [weak: [write-up](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/tascj-1st-place-solution)，页面被拦截]。本项目已于 2026-10-22 移除 EMA，理由见[验证协议](#验证协议)。
- logit 集成：Contrails 第 1 名用 unet1024 与 vit 集成 [weak: 搜索摘要]。
- Copy-paste：只有通用证据，没有找到行星场景的定量证据。

### 行星表面分割

- AI4Mars（经 [MarsSeg](https://arxiv.org/html/2404.04155) Table III）：soil/bedrock/sand IoU 85–94，big rock 只有 46.28（DeepLabv3+）/ 50.50（MarsSeg），big rock 约占训练样本 2%；MSL-Seg 上 rocks 从 34.21 提到 52.81。小而稀的岩石类在所有行星数据集上都是瓶颈。
- Synthetic Moon / Kaggle Artificial Lunar Landscape（[arXiv 2504.15991](https://arxiv.org/html/2504.15991)）：9,766 张 480×720 渲染图加 36 张真实月面照片，类别为天空/小石/大石；真实月面上最佳做法是从合成模型全量微调（adapter 约低 3%，p=0.02），增强网格搜索中 random crop（p=0.5）最有效。结论来自小 CNN 与小数据，迁移价值有限。
- Light4Mars / SynMars（ISPRS J. 2024）用合成火星图像训练、真实 MarsScapes 评测 [weak: [摘要](https://www.sciencedirect.com/science/article/abs/pii/S0924271624002466)]。

### 候选替代骨干（均未采用）

| 模型 | 公开分割结果 | 预训练数据 | 对本项目 |
|---|---|---|---|
| DINOv3 ViT-H+/16 | ADE20K 线性 54.8 | LVD-1689M | 与 ViT-L 持平，不值得 |
| DINOv3 7B + Mask2Former | ADE20K 62.6/63.0 @896 | LVD-1689M；head 用了额外分割数据 | 2×48 GB 不可行，官方 head 不合规 |
| DINOv3 ConvNeXt-L | ADE20K 线性 47.8 | LVD-1689M 蒸馏 | 可作异构集成成员 |
| SAM-H（冻结 + Mask2Former） | GTAV→Cityscapes 57.0，低于 DINOv2-L 的 63.3（Rein Table 1） | SA-1B | 不优先 |
| SegFormer-B5 | Cityscapes 84.0（[arXiv 2105.15203](https://arxiv.org/abs/2105.15203)） | ImageNet-1k | 轻量异构成员候选 |

InternImage-H、EVA-02-L、BEiT-3 等更大或预训练来源链更复杂的模型在 2×48 GB 上吃力、来源声明复杂，不考虑。
