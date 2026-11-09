# 训练配方

更新：2026-11-09。本批全六场景预算对照见下文；原五场景配方以 [`configs/dnv3_va_m2f_vitl16_default.yaml`](../configs/dnv3_va_m2f_vitl16_default.yaml) 为模板：冻结ViT-L、RRC、梯度裁剪0.01、decoder6层/hidden_dim256，五场景训练、场景5验证、有效batch64。复制模板时设置独立输出目录；25轮只是已测参考预算，最合适轮数尚待校准。当前保留独立场景5验证集，可以按验证mIoU或验证loss选择best权重；后续场景5并入训练、没有独立验证集时，使用末轮last.pt。best_loss.pt指最低验证loss，不按训练loss选择模型。完整跑完训练预算与保存、选择验证best并不冲突；10-29的末轮对照是已执行的一组实验，不是所有开发实验的硬性选权重规则。现有结果见 [roadmap.md](roadmap.md#实验记录)。历史B/L的80轮、warmup/EMA配方及其YAML保留用于追溯，不能再当成新训练默认。

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

## 全六场景训练与末轮测试

11-09用户确认将场景5加入训练，四个独立预算按两条双卡队列接续；仅使用物理GPU0/7和2/3。原训练继续，GPU1上的阶段last5 TTA已完成，后续不再占用1/4/5/6。每条新队列等原卡对的训练与公开测试全部完成，再执行下表任务；一项训练结束后，同卡对各跑一个last普通/TTA，完成后才进入下一项。

| 卡对 | 顺序 | 配置 | 完整预算 |
|---|---:|---|---:|
| 0/7 | 1 | [fulltrain e10](../configs/dnv3_va_m2f_vitl16_fulltrain_clip001_e10.yaml) | 10轮，1250更新 |
| 0/7 | 2 | [fulltrain e25](../configs/dnv3_va_m2f_vitl16_fulltrain_clip001_e25.yaml) | 25轮，3125更新 |
| 2/3 | 1 | [fulltrain e15](../configs/dnv3_va_m2f_vitl16_fulltrain_clip001_e15.yaml) | 15轮，1875更新 |
| 2/3 | 2 | [fulltrain e20](../configs/dnv3_va_m2f_vitl16_fulltrain_clip001_e20.yaml) | 20轮，2500更新 |

全部采用冻结DINOv3 ViT-L + ViT-Adapter + Mask2Former、1024² RRC、decoder6/width256、clip0.01、AdamW lr2e-4/wd0.05、poly0.9、无warmup、seed42。每卡8×2卡×累积4=64。每个预算独立从原始预训练骨干与新Adapter/head开始，不从其他预算的中途权重续训；每次都完整走到本次学习率计划终点。

沿用本地数据实体与`dev_final_train.json`，训练场景1/3/5/6/7/8共8002张；其样本成员已核对等于原6403张加场景5的1599张，并与官方全训练清单完全一致。`validate: false`、`val_manifest: null`、`val_scenes: []`，不再用场景5选best，不按训练loss选权重。每轮125次更新、8000张，随机打乱后跳过不足完整batch的2张；完整运行后只测最终`last.pt`，公开2/4/9全部5004张分别做single和narrowflip。比较训练预算及后续路线只依据这些测试结果。

计划为`runs/plans/2026-11-09/fulltrain/lane_07.json`与`lane_23.json`，每lane由`scripts/run_fulltrain_plan.py --plan <path>`管理。代码、解释器、CUDA扩展、数据和输出均走disk02本地路径；共享训练/评测锁、GPU UUID与占用检查保留，2/3训练关闭NCCL P2P，两组单卡测试均关闭P2P。新队列不继承已撤销的临时额外卡授权。数据及配置核对见[evidence/fulltrain_config_verification_20261109.json](evidence/fulltrain_config_verification_20261109.json)。 两条队列已于11-10 01:15全部完成四个预算和8项末轮测试，结果见[evidence/fulltrain_frozen_20261110/summary.json](evidence/fulltrain_frozen_20261110/summary.json)；状态分别在`lane_07.state/status.json`与`lane_23.state/status.json`；启动记录见[evidence/fulltrain_queue_launch_20261109.json](evidence/fulltrain_queue_launch_20261109.json)。 无验证集的末轮校验、旧checkpoint兼容和调度恢复检查见[evidence/fulltrain_no_val_verification_20261109.json](evidence/fulltrain_no_val_verification_20261109.json)。

### 全六场景联合微调（11-09晚追加）

用户根据已完成公开测试的收益，追加同样10/15/20/25轮的联合微调预算。原2/3上的冻结20轮及其测试已于11-10 01:15完成；新任务使用不冲突的0/7、4/5，每条lane依次训练、并行测试last普通/TTA、再进入下一预算。该授权更新此前仅四张卡的范围，不使用1/6。

| 卡对 | 顺序 | 配置 | 完整预算 |
|---|---:|---|---:|
| 0/7 | 1 | [joint e10](../configs/dnv3_va_m2f_vitl16_fulltrain_joint_e10.yaml) | 10轮，1250更新 |
| 0/7 | 2 | [joint e25](../configs/dnv3_va_m2f_vitl16_fulltrain_joint_e25.yaml) | 25轮，3125更新 |
| 4/5 | 1 | [joint e15](../configs/dnv3_va_m2f_vitl16_fulltrain_joint_e15.yaml) | 15轮，1875更新 |
| 4/5 | 2 | [joint e20](../configs/dnv3_va_m2f_vitl16_fulltrain_joint_e20.yaml) | 20轮，2500更新 |

每个预算独立从原始DINOv3预训练权重和新Adapter/head开始，从第一步训练整个ViT；不从冻结模型或其他预算续训。配方沿用已测联合微调：每卡4×2卡×累积8=64，Adapter/head lr2e-4、骨干lr2e-5、AdamW wd0.05、clip0.01、poly0.9、无warmup、bf16、seed42，1024² RRC与原亮度/对比度增强，decoder6/width256。训练全部8002张，不建立val、不生成best，`save_every: 0`不保存独立epoch权重；每轮原子覆盖一个`last.pt`，保留全部可训练骨干/分割头和优化器等恢复状态，只有完整预算结束后的last进入测试。

新计划位于`runs/plans/2026-11-09/fulltrain_joint/lane_07.json`与`lane_45.json`，显式`recipe: joint_fullft`、`predecessor: null`；仍先查实时空闲和UUID并取得共享训练/测试锁。4/5训练关闭NCCL P2P，0/7保留已验证P2P；所有单卡评测关闭P2P。旧冻结计划仍保留原卡对、配方与依赖约束。配置及验证见[evidence/joint_fulltrain_configs_20261109.json](evidence/joint_fulltrain_configs_20261109.json)、[调度与保存检查](evidence/joint_fulltrain_verification_20261109.json)，实际启动状态见[启动记录](evidence/joint_fulltrain_launch_20261109.json)。

## 基线准备与SAM3推理

按用户最新安排，仅SAM3排公开测试，SegFormer-B5与官方轻量基线暂不训练。两个[准备配置](../configs/segformer_b5_fulltrain.yaml)及[轻量配置](../configs/deeplabv3plus_mobilenetv3_fulltrain.yaml)的`train.epochs`均为null，直接启动会在初始化CUDA前报错；没有默认25轮或80轮，也没有新增训练计划。训练长度确定后再复制配置、设置唯一输出目录并测实际显存；模板每卡4×双卡×累积8=64只是预置值，尚未做GPU容量测试。

B5保留作者的编码器lr6e-5、head lr6e-4、AdamW衰减0.01与poly1；作者1500步warmup按batch16→64的样本数折为375步。数据增强沿用本项目1024RRC与光度方案，裁剪仍为项目默认0.01。轻量基线准备使用CE1+Dice1、不设类别权重，lr2e-4、衰减0.01；这些系数与学习率并非官方已公开配方。二者均预置全部8002张、无val、最终last，与官方轻量演示使用2400张×10轮、batch16及400张验证不同，未来分数不能称作该演示的严格复现。B5原版SyncBN与轻量原版BN均保留。

SAM3配置为[`sam3_pretrained.yaml`](../configs/sam3_pretrained.yaml)，由`moonseg.evaluate_pretrained`读取官方预训练权重，固定五类文本、单尺度覆盖公开场景2/4/9全部5004张。输出标为`pretrained_only`，记录源码、权重文件身份、提示词、官方内部1008预处理及混淆矩阵；不伪造best、last或训练轮数。它使用独立uv基线环境，在物理GPU2单卡执行，使用共享GPU锁、物理UUID和`NCCL_P2P_DISABLE=1`。11-10用户要求立即使用已空闲的GPU2后，启动条件改为该卡最后的普通测试成功完成；GPU3末次TTA独立继续，不再等待整组结束。0/7与4/5的现有DINO联合微调照常，1/6不使用。本次计划为`runs/plans/2026-11-10/baselines/sam3.json`，GPU2交接记录在同目录`gpu2_handoff.json`。11-10 01:01:40评估正常退出，完整覆盖5004张，监督进程已确认完成并释放GPU2；总耗时23分45秒（含加载与单图检查），均值3.51张/秒。结果保存在同目录`sam3.state/scores/scores.json`，并[归档到Git](evidence/sam3_public_test_20261110.json)；场景等权mIoU68.486%、mAcc76.051%。按用户决定不追加批量提速实验。随后正确性审计确认dense后处理遗漏presence，且未采用Meta官方论文的实例掩码语义评测流程；上述全量分数只代表旧自定义适配，不作为正确接入SAM3的性能结论。18张固定样本诊断见[协议审计](evidence/sam3_protocol_audit_20261110.json)，本次未修改生产模型、提示词配置或重跑全量。

## GPU 放置与启动

11-09结果已完整归档：联合微调DINOv3在best与last两种选择口径下均提高公开场景2/4/9；冻结DA3当前方案未显示整体优势。逐场景单尺度/TTA、mAcc及阶段微调取舍见[roadmap.md](roadmap.md#实验记录)和[evidence/public_development_20261109/summary.json](evidence/public_development_20261109/summary.json)。已排全六场景任务仍按冻结10/15/20/25轮执行，本次记录不改变训练计划。

11-08首组DA3MONO采用冻结骨干对照：`da3mono_va_m2f_vitl14_rrc_frozen`在PCI2/3从官方DA3MONO-LARGE encoder与新Adapter/head开始，完整冻结encoder（eval、no_grad、无encoder优化器参数）。严格复制冻结`dnv3_va_m2f_vitl16_rrc_clip001`：25轮/2500更新、1024² RRC0.5–2、每卡8×2卡×累积4=64、AdamW lr2e-4/wd0.05、poly0.9/无warmup、clip0.01、decoder6/width256、seed42、原亮度/对比度增强及uniform采样；dev_train6403与dev_val1599、场景5独立验证保持不变。保留每卡batch8以对齐Adapter普通BatchNorm，不能仅用有效batch相等代替。原生patch14仅右下补到1036²，再按像素坐标桥接64²的stride16特征；输入未缩小。独立计划/状态在`runs/plans/2026-11-08/da3mono_frozen/`，从disk02本地代码、环境与CUDA扩展运行；独立队列`runs/public_development_da3mono_frozen_2026-11-08/`已完成`best_miou.pt`与`last.pt`的公开2/4/9全5004张single/narrowflip。best与原A/B队列中的clip001验证best比较，25轮last与已有clip001末轮结果比较。

此前误启动的`da3mono_va_m2f_vitl14_rrc_fullft`因路线纠正在完成第1轮后、第2轮最后日志step114处停止；全部日志、三个checkpoint与旧计划保留，旧训练/评测计划已禁用。它不用于性能淘汰结论，也不向冻结运行提供初始化。2/3沿用PCI_BUS_ID、UUID检查、共享训练/评估双锁和`NCCL_P2P_DISABLE=1`。冻结实测与checkpoint回载证据见[evidence/da3mono_frozen_verification_20261108.json](evidence/da3mono_frozen_verification_20261108.json)，配置逐项差异、disk02进程与正式DDP更新见[启动证据](evidence/da3mono_frozen_launch_20261108.json)。

11-08两条全解冻训练已于11-09完成训练和全部公开测试；原配方为：`dnv3_va_m2f_vitl16_rrc_fullft`在0/7从原生DINOv3开始联合训练25轮/2500更新，Adapter/head与骨干初始lr为2e-4/2e-5；`dnv3_va_m2f_vitl16_rrc_clip001_stage2_fullft`在4/5从clip001的第15轮`best_miou.pt`权重初始化，完整解冻后以新优化器训练5轮/500更新，lr为2e-5/2e-6。均为每卡4×2卡×累积8=64、1024²/RRC/width256/decoder6/clip0.01；4/5关闭NCCL P2P，0/7保持已验证P2P路径。`--initialize-from`不同于`--resume`，阶段加载约定见[model.md](model.md#checkpoints)。

本批使用单计划启动器`scripts/run_finetuning_plan.py`，计划与启动状态在`runs/plans/2026-11-08/fullft/`中的`joint/stage2.json`与对应`.status.json`，训练日志在各运行目录的`train.log`。启动前核对物理UUID/进程占用，同时持有既有训练与评估GPU锁，并向torchrun继承锁描述符；历史队列和配置不改。训练完成后校验完整history、全量末轮验证及三个checkpoint，再写`queue_completed.json`。公开评估计划为同目录`public_test.json`，状态在`runs/public_development_fullft_2026-11-08/`；只等这两组卡释放，补评父权重的验证`best_miou.pt`及A/B各自的`best_miou.pt`、`last.pt`，共10项；DA3MONO原独立2/3计划扩为best+last共4项。每队列默认最多2个单卡任务；11-09曾临时授权本次fullft测试增加空卡1/6（随后已撤销，仅已有GPU1任务允许结束）；计划显式声明`temporary_extra_gpus: [1, 6]`时最多4个单卡任务，并发不超过计划GPU数，UUID、锁、占用检查及单卡P2P禁用保持。每个checkpoint先单尺度再窄尺度翻转，各覆盖5004张，父权重优先。

本批best对best、last对last分别报告。clip001现成25轮/2500更新last的[单尺度](evidence/public_development_last_20261029/rrc_clip001_single.json)与[TTA](evidence/public_development_last_20261029/rrc_clip001_narrowflip.json)证据直接复用，不重复排父last；只给A、B、DA3各追加两项last评估。B与DA3的last是25轮/2500更新；A的last是从clip001第15轮best初始化后微调5轮/500更新，初始化和预算不同，只作阶段诊断，不称为相同总预算。

混合队列顶层标记`checkpoint_policy: mixed`，单项只允许`validation_selected`或`final_epoch`；run可用`checkpoints: [{"checkpoint": "best_miou.pt", "checkpoint_policy": "validation_selected"}, {"checkpoint": "last.pt", "checkpoint_policy": "final_epoch"}]`，旧单checkpoint计划保持兼容。准备、命令和分数来源核对按各checkpoint处理，状态与汇总逐项记录选择策略和计划轮数。last严格核对计划5/25轮、resolved config、完整history、末轮1599张验证、global_step及保存的更新预算；完成marker、物理UUID、共享GPU锁与单卡评测P2P禁用保留。计划变更仍拒绝静默配置漂移，迁移须先备份并显式更新原status的config，保留已有jobs和结果；同一run的每个checkpoint连续安排single/narrowflip，不增加队列，并发遵循上述本次临时授权。 11-09阶段last评估曾在配置检查阶段退出：resolved JSON中的lr=2e-05被YAML读成字符串；现按JSON格式读取，未放松配置一致性校验。两项失败日志保留并显式重试，正在运行的best TTA和训练未停止，见[evidence/last_evaluation_json_fix_20261109.json](evidence/last_evaluation_json_fix_20261109.json)。

| 运行目录 | 配置 | 物理 GPU（PCI 序） | 启动时必需的环境变量 | 实际每卡显存 |
|---|---|---|---|---:|
| `runs/dnv3_va_m2f_vitl16` | `dnv3_va_m2f_vitl16.yaml` | 0、7（RTX 6000 Ada） | 无（脚本默认 `CUDA_VISIBLE_DEVICES=0,7`） | 约 45.6 GB |
| `runs/dnv3_va_m2f_vitb16` | `dnv3_va_m2f_vitb16.yaml` | 5、6（RTX A6000） | `CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1` | 约 40.9 GB |

启动、恢复、评估和预测命令见 [README](../README.md#训练验证与推理)。后续统一从本地完整checkout启动，本机为 `/home/disk02/wjj/moonseg/code`；旧NAS入口不会自动迁移。脚本先切物理代码目录，直接使用本地venv；解释器本体/标准库、源码、`third_party`和CUDA扩展均须在本地，不能只移动数据和输出。已启动的原A/B训练暂不停止或重启，其现有路径依赖保留。数据根目录与训练输出仅在运行时解析成物理路径，不改原配置、清单、来源记录或恢复比较。

启动脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；PCI 5/6 卡对的 PCIe P2P 会静默丢数据，必须关闭 NCCL P2P，原因见 [gpu_topology.md](gpu_topology.md)。`runs` 是指向本地盘 `/home/disk02/wjj/moonseg/runs` 的软链接，不写 NAS。只能从 `last.pt` 恢复，且配置、数据清单和进程数必须与原运行一致；唯一例外是 EMA 时代的 `last.pt` 记录的 `train.ema`，恢复时被忽略。

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
