# 交接文档：路线、设计与操作手册

更新：2026-10-22 23:30（UTC+8）。本文供接手的人或 agent 一次读完就能继续工作：现状、已定决策、实验结果、保守消融阶梯、大胆创新设计、决策规则、最终流程、操作手册与流程规则。候选项的完整清单见 [roadmap.md](roadmap.md)，当前配方与理由见 [training.md](training.md)，数据事实见 [data.md](data.md)。

## 1. 现状一句话

冻结 DINOv3 ViT-L/16 + ViT-Adapter + Mask2Former 在场景 5 上单尺度约 0.92、加 TTA 约 0.93；随机缩放裁剪（L1）消除了过拟合并把末轮抬高 1.4 个点；ViT-H+ 在同几何下再高约 0.9 个点；两者尚未组合，H+ 正在训练，L2 排队中。

## 2. 已定决策

| 项目 | 决定 | 来源 |
|---|---|---|
| 骨干 | 冻结 DINOv3，ViT-B 退役（比 L 低 7.5 点），L 为主线，H+ 已就绪 | 80 轮对比；用户 |
| 输入 | 1024×1024，训练几何增强以 L0/L1/L2 消融决定（L1 目前领先） | 本页第 3 节 |
| 有效 batch | 每卡 8 × 2 卡 × 累积 4 = 64（要求不低于 64）；H+ 同样 8×4 | 用户；显存实测 |
| 优化 | AdamW、lr 2e-4、wd 0.05、poly 0.9、无 warmup、梯度裁剪 0.1、bf16 + TF32、25 轮 | 用户；80 轮在第 12–15 轮过拟合 |
| EMA | 不用（官方分割配方均不用；我们的实验里只平滑了下滑的轨迹） | 用户 |
| checkpoint | `last.pt`（完整恢复）、`best_miou.pt`、`best_loss.pt`（仅权重），不存冻结骨干 | 用户 |
| 验证 | 场景 5，每轮隔 2 帧、末轮全量；同时记录验证损失与 3 轮滑动均值 | — |
| TTA | 尺度 0.9/1.0/1.1 + 水平翻转，概率平均；B/L 上阶梯形状一致，不必逐实验重测 | 第 3 节 |
| 数据 | 官方清单到位前用公开训练场景开发池；最终模型在官方清单上从骨干重训 | [data.md](data.md) |
| GPU | 0/7（Ada）与 5/6（A6000，须 `NCCL_P2P_DISABLE=1`）各为一对；不占 1–4 做训练，空卡只跑推理 | 用户；[gpu_topology.md](gpu_topology.md) |

## 3. 实验结果汇总（场景 5；mIoU 为隔 2 帧 800 帧，标注"全量"者为 1599 帧）

| 运行 | 配置差异 | best | 末轮 | 每类（best 轮：岩石 / 撞击坑 / 山体） | 结论 |
|---|---|---|---|---|---|
| `dnv3_va_m2f_vitb16`（80 轮，旧配方：warmup 250、EMA） | ViT-B 整图 | 0.871（第 12） | 全量 0.827 | 0.776 / 0.813 / 0.787（全量复核） | 第 12 轮后持续过拟合 |
| `dnv3_va_m2f_vitl16`（80 轮，旧配方） | ViT-L 整图 | 0.920（第 15） | 全量 0.885 | 0.851 / 0.858 / 0.904 | 第 15 轮后下滑，poly 尾段无回升 |
| `dnv3_va_m2f_vitl16_e25`（L0） | ViT-L 整图，25 轮、无 warmup、无 EMA | 0.9147（第 9） | 第 23 轮 0.905（进行中） | 0.841 / 0.828 / 0.917 | 验证损失 5.0 → 6.3，仍过拟合 |
| `dnv3_va_m2f_vitl16_rrc`（L1） | L0 + 随机缩放 0.5–2.0 + 1024 裁剪 | **0.9239（第 11）** | **全量 0.9188** | 0.857 / 0.853 / 0.922 | 验证损失 4.1 → 4.6 持平，过拟合消失 |
| `dnv3_va_m2f_vith16plus_e25` | H+ 整图，8×4 | 0.9232（第 7，进行中） | — | 0.850 / 0.856 / 0.923 | 同几何比 L0 高约 0.9；每卡 46.6 GB |
| TTA（best.pt，全量） | 单尺度 → +翻转 → 0.75/1/1.25 → 0.9/1/1.1 | L：0.9200 → 0.9242 → 0.9291 → 0.9300；B：0.8713 → 0.8735 → 0.8829 → 0.8858 | — | 山体受益最大 | 设置定为窄尺度 + 翻转 |

原始数据：`runs/<run>/history.jsonl`（逐轮全部指标与混淆矩阵）、`docs/evidence/tta/`、`docs/evidence/gpu_results/`。

## 4. 正在跑与排队（2026-10-22 23:30）

| 卡对 | 当前 | 之后 | 预计 |
|---|---|---|---|
| PCI 0/7 | `dnv3_va_m2f_vith16plus_e25`（第 8 轮） | 无 | 约 03:50 结束 |
| PCI 5/6 | `dnv3_va_m2f_vitl16_e25`（L0，第 23 轮） | `dnv3_va_m2f_vitl16_sj`（L2，温和抖动 0.75–1.25） | L0 约 00:00 结束，L2 约 07:30 结束 |

队列由 `build/queue_2026-10-22.sh` 驱动（见第 9 节），事件写在 `runs/queue_2026-10-22.log`。

## 5. 保守路线：逐步消融阶梯

原则：一次只改一项；25 轮预算；判据见第 7 节；每一步的赢家成为下一步的基线。

| 步 | 改动 | 配置做法 | 预算 | 预期 | 状态 |
|---|---|---|---|---|---|
| 1 | 几何增强：L0 整图 vs L1 随机缩放裁剪 vs L2 温和抖动 | 已有三份配置 | 3 × 5.5–8 h | L1 领先 | L1 完成，L0/L2 进行中 |
| 2 | 骨干容量：H+ vs L，同几何 | `dnv3_va_m2f_vith16plus.yaml` | 7.5 h（Ada） | +0.5–1 | 进行中 |
| 3 | 组合：H+ + 步 1 赢家（预期 H+ + rrc） | 新配置 `dnv3_va_m2f_vith16plus_rrc.yaml`：H+ 配置改 `train_mode: random_resize_crop` 等五个键 | 7.5 h | 两者增益叠加 | 待开 |
| 4 | 光度增强：+ gamma 0.7–1.5；+ 按标签合成阴影（非天空） | `gamma_probability: 0.5`；`shadow_probability: 0.5`（其余键默认） | 各一次 | 针对场景 9 阴影；场景 5 上可能只持平 | 候选 |
| 5 | 稀有类采样 | `train.sampling: {mode: rare_class, temperature: 0.01}` + `data.label_stats` | 一次 | 岩石/撞击坑 +0.5–2 | 候选 |
| 6 | 损失采样点 12544 → 50176 | `loss.num_points` | 一次（显存几乎不变） | 小岩石边界 | 候选 |
| 7 | 预算与 lr：25 轮 vs 40 轮（有 L1 正则后可能不再过拟合）；lr 1e-4 | `train.epochs`、`train.lr` | 两次 | 在 L1 基线上重新确认长度 | 候选 |
| 8 | 头部：query 100 → 50；decoder 6 → 9 层；adapter drop path 0.1 → 0.3 | `model.*` | 各一次 | 小 | 低优先级 |
| 9 | 第二验证场景确认：对前两名配方用 `dev_val6_train`/`dev_val6` 重训一次 | 配置里换清单与场景 | 两次 | 排除只在场景 5 有效的方案 | 定方案前必做 |
| 10 | 部分解冻（最后 4–8 个 block，lr ×0.1） | `model.unfreeze_last_blocks: 4` | 一次，显存更紧 | −1 到 +2 | 用户定位为候选，不是当前目标 |

## 6. 大胆路线：创新设计

按"预期收益 ÷ 工程量"排序。每条给出动机、机制、实现草图、代价、风险与合规。

### B1. 几何双流：DINOv3 语义 token + DA3/VGGT 几何 token

动机：撞击坑是凹陷、岩石是凸起、山体是远处起伏、天空在无穷远，而阴影只改亮度；终稿论文把撞击坑错误归因于"外观太像月壤"，几何特征正好补这一点。机制：Depth Anything 3 / VGGT 的 ViT 以 DINOv2 初始化、经大规模多视角几何监督，patch token 编码深度与表面朝向。实现草图由省到贵：(0) 冻结 DA 推伪深度/法线，作辅助头目标，推理不变；(A) 在 4 个交互层位把两路 token 拼通道、线性降维后进现有 Adapter；(B) 每层位加 cross-attention，DINO 作 query、DA3 作 key/value；(C) 两路 token 拼成 8192 长序列过 2–4 个联合自注意力块。网格对齐：DA3 输入缩到 896²（patch 14 → 64×64）与 DINOv3 1024²（patch 16 → 64×64）一致。代价：多一个冻结 ViT-L 前向，每卡 batch 约降到 5–6。风险：从头训练的融合层在 6.4k 张图上过拟合（C 最甚）。合规：须核对 DA3/VGGT 训练数据无 LuSNAR 测试场景（DA3 晚于 LuSNAR 公开，不能靠时间排除）、权重许可（VGGT 为 CC-BY-NC）并申报；LuMon 系列深度模型因训练含 LuSNAR 禁用；公开包里的深度/双目/LiDAR 禁用。先做 (0) 或 (A)。

### B2. 地平线归一化的位置先验

动机：标签与绝对行号强相关（天空从不低于 365 行，512 行以下 99% 是月壤），模型会学到"离顶边多远"这个捷径，而场景 9 的地平线起伏是其他场景的 1.5–3 倍，正是捷径失效的地方；L1 用裁剪打掉捷径有效，说明这个方向对。机制：显式估计地平线（天空下边界，可由当前模型自身预测或简单的行均值亮度阶跃得到），把"相对地平线的行偏移"作为额外输入通道或位置编码偏置，让不同起伏的场景在位置坐标上对齐。实现：数据集层加一个通道（或 Adapter 的 SPM 输入加一通道），训练时用标签算地平线、推理时用模型第一遍预测算；两遍推理。代价小。风险：地平线估计错误会传播；需在场景 5 的高起伏帧上验证。合规：无问题（只用官方图像与标签）。

### B3. 参数高效地解冻骨干（LoRA / Rein 式 token）

动机：Rein 在合成→真实的域偏移上给出冻结 61.1、全微调 63.7、参数高效微调 66.4，说明"少量可学参数进骨干"优于两端。机制：在 DINOv3-L 的 q/v 投影加 LoRA（rank 8–16），或 Rein 的可学习 token；骨干主权重仍冻结，checkpoint 只多几 MB。实现：现有 `unfreeze_last_blocks` 是粗版本；LoRA 需要在 `dinov3_mask2former.py` 里包一层线性替换，训练时 lr 1e-5 到 1e-4 单独分组。代价：显存增加有限（LoRA 梯度小，但骨干激活需保留，批量约降一半）。风险：在 2–3.6k 有效帧上过拟合训练场景光照；必须看场景 6 确认。合规：无。用户定位：候选，不是当前目标。

### B4. 类别条件化 query 与稠密小目标损失

动机：5 类语义目标下 100 个 query 有 95 个永远匹配"无物体"，匈牙利匹配带来逐轮抖动；小岩石（中位 63 像素）在 12544 个采样点下只分到几个点。机制：每类固定 K 个 query（如 5×8），类别嵌入初始化，匹配只在同类 query 内做或直接固定分配；对面积小于阈值的目标 mask 用稠密 BCE/Dice 替代点采样。实现：改 criterion 的匹配与采样，decoder 的 query 初始化；改动集中在两个文件。代价：无额外推理成本。风险：偏离原版 Mask2Former，需要对照实验证明。合规：无。

### B5. 按标签合成硬边阴影与阴影域随机化

动机：场景 9 非天空区的暗像素占 11.2%，训练场景只有 5.8–6.7%；真实月面阴影边缘很硬（无大气散射），现有实现默认 8–48 px 模糊偏软。机制：在非天空区投射随机多边形硬阴影（模糊 0–4 px）、按线性光变暗后钳到像素下限 22；配合 gamma 与整体变暗，把训练光照分布向场景 9 推。实现：已有 `shadow_*` 键，只需放开模糊下限并加线性光变暗选项。代价：零。风险：场景 5 可能不涨，收益只在场景 9 上体现而我们看不到，需要用暗帧子集代理。合规：规则明文允许颜色扰动；方法文档要申报。

### B6. 多骨干集成与蒸馏

动机：L 与 H+ 的错误不完全重合，logit 平均通常稳赚 0.5–1.5；若推理时限不允许双模型，可把集成蒸馏回单模型（AM-RADIO 思路）。机制：对 L-rrc 与 H+-rrc 的五类概率做加权平均再 argmax；蒸馏版用集成的软标签作为额外监督训练单模型。实现：集成只需在 predict 里加多 checkpoint 平均（新代码约 50 行）；蒸馏需要新训练流程。代价：集成推理 2 倍。风险：低。合规：两份权重来源均可核验；蒸馏只用官方训练图像。

### B7. 对象带局部放大的 TTA

动机：岩石和撞击坑集中在 256–512 行，62.5% 的岩石像素属于小于 128² 的块。机制：除全局 0.9/1/1.1 尺度外，对 192–576 行的横带以 1.5× 放大单独推理，把概率贴回对应区域再平均。实现：predict_scores 加一个"带状放大"分支，约 30 行。代价：多一次前向。风险：拼接边界处概率不连续，需在带边缘做线性过渡。合规：无。

### B8. 2× 分辨率训练小目标

动机：输出 stride 4 下 63 像素的岩石只有约 4 个 mask 单元。机制：训练时随机把图像放大到 1536–2048 再裁 1024（L1 的 2.0 上限已部分覆盖），推理时对全图以 1.25–1.5 倍尺度做 TTA（已有）。这条实际上已被 L1 + TTA 覆盖一大半，列出是为了说明不必再单独做"高分辨率训练"。

### 不做的方向

7B 骨干（算力 15 倍以上）；测试图伪标签、自训练、TENT/AdaBN；公开包的深度/双目/LiDAR；COCO-Stuff 等额外分割数据预训练头部；SAM 系编码器作主骨干（Rein 证据显示其冻结特征在域偏移下明显弱于 DINO 系）；时序模型（比赛规定单张输入）。

## 7. 决策规则与验证协议

每个改动只在"best_miou.pt 在场景 5 全量 1599 帧上的 TTA 分数"上比较，同时看岩石、撞击坑、山体三类 IoU 与验证损失走势；单轮抖动约 ±0.5（lr 高时 ±2），差距小于 0.3 视为持平；只有稳定增益才采纳。方案冻结前，对前两名配方用场景 6 留出（`dev_val6_train`/`dev_val6`）重训一次确认，并报告场景 5 暗帧子集（定义见 [data.md](data.md#验证场景)）。候选的 25 轮预算在 L1 正则下可能偏短，步 7 负责重新确认。

## 8. 最终流程与截止

1. 官方训练包到位（10-22 起发放）：`tools/prepare_data.py --root data/LuSNAR --official-allowlist … --official-test-allowlist … --validate-labels`，重新生成 `label_stats`，与开发池清单比对差异。
2. 用冻结的配方在全部 6 个训练场景上从骨干重训（新配置 `dnv3_va_m2f_<骨干>_final.yaml`，`validate: false`，轮数按开发曲线定，`development_pool: false`）。
3. 测试输入到位（11-02）：`scripts/predict.sh … --weights-key auto`，配置 `inference` 段写死 `scales: [0.9, 1.0, 1.1]`、`flip: true`；`tools/package_submission.py` 校验并打包。
4. 11-20 23:59 前一次性提交；同时准备 train.py / predict.py / 权重 / 配置 / 环境 / README / 训练日志 / 来源说明和 ≤6 页方法文档（模型来源：DINOv3 LVD-1689M，ModelScope 镜像，sha256 见 [weights.md](weights.md)）。

## 9. 操作手册

### 环境与路径

- 代码 `/home/vrlab-132-nat-moon/moon`（在训练主机上表现为 `/Users/jiajunwang/ClaudeWork/vrlab-132-nat-moon/moon` 的挂载，NAS，不要往里写大文件）；数据、权重、运行输出、引用 PDF 都在本地盘 `/home/disk02/wjj/moonseg/` 下，通过软链 `data/`、`ckpts/`、`runs/`、`references/` 访问。
- 解释器 `.venv/bin/python`（uv 管理，`uv` 不在 PATH 时用 `PYTHON=.venv/bin/python` 传给脚本）。
- CUDA 扩展：`bash scripts/build_ops.sh` 一次；`model.deformable_attention: auto` 自动选用，日志首行会打印 `deformable attention: cuda`。

### 启动训练

```bash
# Ada 对（脚本默认 0,7）
PYTHON=.venv/bin/python bash scripts/train.sh configs/<cfg>.yaml
# A6000 对 5/6：必须关闭 NCCL P2P
CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 PYTHON=.venv/bin/python bash scripts/train.sh configs/<cfg>.yaml
```

从命令行交互式启动时用 `nohup setsid … > runs/<out>/train.log 2>&1 < /dev/null &`，否则远程命令通道会把前台进程挂住或超时。输出目录由配置 `train.output_dir` 决定，先 `mkdir -p`。

### 队列脚本

`build/queue_2026-10-22.sh` 是本次用的排队器：两个子 shell 各自 `kill -0 <launcher PID>` 轮询等待某对卡上的 torchrun 退出，然后顺序 `run <gpus> <config> <outdir> [env]`，每次启动/结束写 `runs/queue_2026-10-22.log`。复用时只需改 `PID_L`/`PID_B` 和 `run` 行。**PID 必须写死**：用 `pgrep -f` 取 PID 会匹配到当前 shell 自己的命令行（命令文本里含有同样的字符串），这个坑今天踩过一次。要杀进程时用 `pgrep -f "pattern[.]yaml"` 这类带方括号的正则，避免自杀。

### 看进度

```bash
cat runs/queue_2026-10-22.log
pgrep -af "torch[.]distributed[.]run" | grep -v pgrep        # 当前 launcher
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
grep -c global_step runs/<run>/history.jsonl                   # 已完成轮数
tail -n 1 runs/<run>/history.jsonl | python -m json.tool | head -40
```

`history.jsonl` 每轮一行：`train_loss`、`grad_norm_mean/max`、`val_loss`、`miou_ma3`、`best_miou(_epoch)`、`best_loss(_epoch)`、`saved_best`、`validation`（mIoU、每类 IoU、mAcc、混淆矩阵、`evaluated_samples`）。

### 评估与 TTA

```bash
# 单进程全量评估（空卡上），inference 段决定是否 TTA
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m moonseg.evaluate \
  --config <cfg 或 build/tta/<setting>.yaml> --checkpoint runs/<run>/best_miou.pt --weights model --output <json>
```

`build/tta/` 里有四档设置的配置模板（`single`/`flip`/`ms_wide_flip`/`ms_narrow_flip`）和批量脚本 `run_tta.sh`、`run_tta_vitb16.sh`；全量 1599 帧单尺度约 5 分钟、六次前向的 TTA 约 30 分钟（A6000）。

### 显存/吞吐基准

`build/bench_hplus.py --config <cfg> --batch N [--accum K] [--steps 40] [--eval-images 20]`：合成输入、真实模型与损失、真实优化器，报告峰值 allocated/reserved 与 img/s，可在中途插入一段 1024² 推理检查碎片；结果 JSON 在 `docs/evidence/gpu_results/`。每卡吞吐对 batch 基本不敏感，加大 batch 不省时间。

### 远程命令通道的坑

- 同一命令偶发 `curl: (22) … 500`，与内容无关，重试即可；含 `\|` 的 grep 曾连续失败，用 `grep -E "a|b"` 代替。
- 带中文全角引号的 heredoc 曾失败，改用"先 Write 一个脚本文件到仓库目录再执行"的方式更稳；Write 工具只能写进仓库挂载路径，写到 `$TMPDIR` 或软链目录会失败或落到别处。
- Monitor 工具（tail -F 监视日志）多次到期不报事件，不可依赖；到点直接读日志。
- `sleep` 超过几十秒会被拦截；长等待用后台命令或 `until` 循环。

### GPU 约定

PCI 序号即 `nvidia-smi` 序号（脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`）；0/7 为 Ada，1–6 为 A6000（1/3 开了 ECC，少 3 GiB）；5/6 之间 P2P 坏，必须 `NCCL_P2P_DISABLE=1`；不占别人的卡，启动前 `nvidia-smi` 确认空闲；空卡只跑推理与基准。详见 [gpu_topology.md](gpu_topology.md)。

## 10. 流程规则（用户要求）

- 任何训练、下载、改 provenance、占 GPU 的动作先汇报方案，得到明确同意再做；"去跑"之前的汇报不算许可。
- 不自行改代码，代码改动派子 agent；子 agent 不得再派子 agent；不做冒烟测试训练，验证用静态审查与 CPU 单元检查。
- 一次只改一项；候选写进 roadmap 而不是直接进配置；结果写进本文第 3 节和 roadmap 的实验记录。
- 不往 NAS 写训练输出；只保留 `last.pt`、`best_miou.pt`、`best_loss.pt`。
- 红线：不读测试标签；不用测试图做任何自适应；不引入外部图像或标注；预训练权重只用来源可核验的 DINOv3 并申报。

## 11. 待办

- H+ 与 L2 结束后：更新第 3 节与 roadmap；启动 H+ + rrc 组合（步 3）。
- H+ 的 SwiGLU gate/up 映射尚未与 HF 参考实现做数值比对（机器上另有带 transformers 的环境可用；H+ 第 7 轮已达 0.923，说明映射大概率正确，但仍建议闭合）。
- 官方训练包到位后切换清单并复核与开发池的差异。
- 测试输入到位后用 `tools/package_submission.py` 做一次完整打包演练。
- `configs/dnv3_va_m2f_vitb16.yaml` 保留作 80 轮记录，不再使用。
