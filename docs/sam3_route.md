# SAM3 语义分割接入方案

核对及接入日期：2026-11-10。本文细化 [roadmap 的 SAM3 诊断](roadmap.md)。第一阶段已接入现有Meta模型和五个原词，共享每图五次grounding，同时计算SegEarth双头等5组完整五类结果与2组实例拒识诊断；CPU边界、GPU2固定18图及两张真实图的重构前旧适配/作者公式数值对照已通过，[检查证据](evidence/sam3_multimode_verification_20261110.json)已归档。主agent验收后于11-10 11:41在GPU2启动新一轮5004张全量，结果尚未完成；不能将小集结果当作有效全量结论。DINOv3联合微调仍是主线。当前无独立验证集，后续方法选择依据公开场景2/4/9全量结果，统一标记为公开开发评估。

## 现有证据及问题边界

旧适配对五类独立 `semantic_seg` 概率 argmax，公开测试 5004 张的场景等权 mIoU 为 68.485789%；这是已完成的自定义适配测量，不能叫 Meta 官方语义评测。Meta [SAM3 §C.2、§F.1](https://arxiv.org/html/2511.16719v1)分别说明 dense 头的 presence 概率分解，以及按类别生成实例后结合 presence、mask 分数和面积合成语义预测。论文没有给出足以逐项还原的阈值及冲突处理；本地 Meta 源码是推理依赖子集，不含 eval 目录，不能据此声称官方未发布任何评测代码，也不能把下述 SegEarth 流程冒称为 Meta F.1 的严格复现。

固定分位点 18 张诊断中，11 张有坑；旧 dense 等权 mIoU 69.168%，补 presence 后 67.815%，坑 IoU 从 4.167% 降到 0。`impact crater` 平均 presence 约 0.03314，默认 0.5 阈值没有保留任何坑实例。已有直接调用一致性、归一化、文本顺序及 raw-logit/sigmoid argmax 检查暂不支持这些是主因。月壤改词的子集改善不能外推为全量收益。[全量旧结果](evidence/sam3_public_test_20261110.json)和[18 张审计](evidence/sam3_protocol_audit_20261110.json)保持原样。

LuSNAR TGRS 表 V 的 SAM3 91.48 mIoU / 95.50 mAcc 没有披露提示词、初始化/微调和五类合并方法，且训练场景 1/2/4/6/8/9、测试 3/5/7 与比赛不同；不能设为零样本复现目标，详见 [paper_alignment](paper_alignment.md)。

## 第一阶段：按 SegEarth 源码建立基线

采用 [SegEarth-OV-3 v2](https://arxiv.org/html/2512.08730v2) 的可读实现，固定作者提交 [`87e5cc18`](https://github.com/earth-insights/SegEarth-OV-3/tree/87e5cc18c8bc9c0b4713b975587281ac1c97c79a)。关键证据是 [segmentor L51–95、L178–190、L220–232](https://github.com/earth-insights/SegEarth-OV-3/blob/87e5cc18c8bc9c0b4713b975587281ac1c97c79a/segearthov3_segmentor.py#L51)及其[修改后的 processor L195–235](https://github.com/earth-insights/SegEarth-OV-3/blob/87e5cc18c8bc9c0b4713b975587281ac1c97c79a/sam3/model/sam3_image_processor.py#L195)。论文是双头/概念存在性的依据，**精确乘权和默认参数以该提交代码为准**。

对类别的每个别名 prompt `a`，令 `R` 为原图大小的双线性插值（`align_corners=False`），`x` 为原图像素位置：

- `p_a = sigmoid(presence_logit_dec)`：整图概念存在概率，不是逐像素概率。
- `r_ai = sigmoid(pred_logits_i)`：实例 query 的条件分类分数，不是 mask 的平均置信度，也不是 IoU 预测。
- `q_ai = r_ai * p_a`：processor 的 `scores`，也是 SegEarth 新增的 `object_score`；已经包含一次 presence。
- `M_ai(x) = sigmoid(R(pred_masks_i))(x)`，`D_a(x) = sigmoid(R(semantic_seg))(x)`：分别是实例和 dense 概率。字段名 `masks_logits`、`semantic_mask_logits` 在该 processor 中实际保存的都是 sigmoid 后概率，不能再次 sigmoid。
- 保留集合 `K_a(t) = {i: q_ai > t}`，严格大于；`I_a(x) = max_{i∈K_a} q_ai*M_ai(x)`，空集合为零图。
- 作者完整融合 `F_a(x) = p_a * max(D_a(x), I_a(x))`；一个类别有多个别名时 `F_c(x)=max_{a∈aliases(c)}F_a(x)`；最后一次五类 argmax。

因此作者实例路径是 `p_a²*r_ai*M_ai`，dense 路径是 `p_a*D_a`。不能将 processor 的分数当成不含 presence 的 `r`，也不能先“修正”为单次 presence 后称为作者复现。`use_presence_score=False` 只关闭**最后一次**乘权，实例筛选/加权中的 `q` 仍含 presence。作者没有在这条流程追加面积过滤或 NMS；Meta F.1 提到面积不等于应自行给 SegEarth 加面积阈值。

下面伪代码对应已实现的[共享grounding接口](../moonseg/models/sam3.py)、[原生后处理](../moonseg/models/sam3_semantic.py)和[专用评估入口](../moonseg/evaluate_sam3_multimode.py)：

```text
base_state = official_processor.set_image(restored_uint8_rgb)
for prompt in class_id_order:
    state = reset_prompts(copy_image_features(base_state))
    out = official_model.forward_grounding(text(prompt), empty_geometry)
    p = sigmoid(out.presence_logit_dec)
    r = sigmoid(out.pred_logits)
    q = r * p
    D = sigmoid(resize_raw(out.semantic_seg, original_HW))
    emit_dense(D), emit_dense_presence(D * p)
    for t in [0.5, 0.3]:
        K = q > t
        I = zeros(original_HW)
        for i in K: I = maximum(I, sigmoid(resize_raw(out.pred_masks[i])) * q[i])
        emit_instance_diagnostics(K, q, raw_masks)
        emit_segearth(t, maximum(D, I) * p)
        if t == 0.5: emit_ablation(t, maximum(D * p, I))
aggregate_aliases_by_class_max_if_enabled()
argmax_once_across_exactly_five_classes()
```

最后的 `max(pD,I)` 是**项目单次 presence 消融**，首轮固定主阈值 `t=0.5`，不是作者算法；它保持 `q>t` 的筛选不变，因而只检验第二次乘权的影响，不能修复因 presence 低而已被筛掉的实例。若要检验“去掉筛选中的 presence”，只在故障统计中比较 `r>t` 与 `q>t`，不把它混入第一版基线。

参数和移植边界：

| 项目 | 作者实际代码 | Moon 首轮预登记 |
|---|---|---|
| 实例阈值 | 构造器/processor 默认 0.5；ADE20K、Cityscapes 配置为 0.3，LoveDA 为 0.5 | 0.5 为主复现项，0.3 为已有作者配置对照；同一次原始输出计算，均完整报告 |
| 像素 mask | processor `M>0.5` 产生二值 mask；融合使用连续 `M` | 融合保持连续；二值阈值仅用于覆盖率及定位诊断 |
| 双头开关 | dense / instance / 最终 presence 默认均开启 | 完整融合为主；保留旧 dense、dense×presence 和单次乘权对照 |
| 图像/尺度 | processor 默认 1008；滑窗默认关闭（crop/stride=0），不同数据配置可覆盖 | 原图 1024→官方 1008→原图；whole、scale 1.0、无翻转，不追加滑窗/TTA |
| 类别/别名 | 文本文件每行一类，逗号分别名，逐别名解码后类内 max | 首轮只用原五词，每类一词、ID 0–4 不变；别名第二阶段再开 |
| 背景阈值 | `prob_thd=0.0`、`bg_idx=0` 默认；如 LoveDA 配置覆盖为 0.5 | 明确无额外背景、无低置信背景回填；月壤 0 是有效类，不能用来吸收空洞 |
| 数值 | 外层 BF16；processor 在插值/激活前无显式 `.float()`；累计图为 FP32 | 已保留作者操作顺序及逐实例标量乘权。实测raw dense/mask、p/r/q为BF16，当前torch autocast中的resize/sigmoid结果为FP32；旧dense仍显式先转FP32。实际dtype逐图逐词写入结果 |

移植只需要现有 Meta `forward_grounding` 的四个原始输出和项目评估器，不需要采用作者的 MMSeg/MMEngine 外壳或 Transformers 包装。保留现有 Meta 提交 `f6e51f59500a87c576c2df2323ce56b9fd7a12de`、ModelScope revision `f8ee5d16d5c30d0e87b1d22cbd358ce97761df56` 和同一个 3,450,062,241 字节 `sam3.pt`；不能同时升级骨干源码/权重再把变化归给融合。这是作者融合算法在Moon既有Meta运行时上的移植，不等于复现作者整个依赖/数值环境。两份processor逐行对照只新增dense resize、presence和object_score输出，实例处理主流程一致；还不能据此断言作者所有backbone文件均与本地完全一致。旧`forward`仍返回原dense分数，专用入口一次前向计算5+2模式；新结果记录模式、阈值、词表和版本。缓存键现已包含完整prompt元组、设备与autocast启用/精度；每类使用独立空几何提示，不继承前一类预测。

### 实例路径如何单独诊断

纯实例并不保证整图覆盖。即使连续 sigmoid 在远处非零，不能把它视为存在有效掩码。诊断以保留实例 `M>0.5` 的并集作为覆盖区，仅在覆盖区内对参与覆盖的实例加权类别图做 argmax；未覆盖区保留为 `reject`，不自动归月壤。空类别返回零图，五类全空图返回整图 reject，并记录全空图数。该覆盖约定是 **Moon 诊断定义**，不是声称还原 Meta F.1。

分别报告每类二值实例并集的 precision/recall/IoU、总覆盖率、各 GT 类覆盖率、重叠率与拒识率。需汇总时使用 `5×6` 混淆表（第六列 reject），`IoU_c=TP_c/(GT_c+Pred_c-TP_c)` 中 `GT_c` 包含被拒识的真实像素，reject 全计为漏检；命名为“含拒识的诊断 IoU”，不送入当前只接受 0–4 的五类官方口径评估器。只算覆盖区的条件准确率须同时给覆盖率，不能冒充全图 mIoU。双头融合则有 dense 覆盖，才进入正常五类混淆矩阵；若任一像素五类全零或非有限，标记该模式评测失败并留诊断，不静默用 ID 0 生成正式分数。有效同分按固定类别 ID 顺序决定并报告同分像素数。

## 故障分解与全量验证

“没有坑实例”至少有三种原因，不能只归因于提示词。固定模型/图像/词之后，按下表独立记录；GT 仅在推理输出固定后用于计算这些统计，绝不产生 query 提示、类别清单或点框。

| 环节 | 要记录的可复核量 | 能回答的问题 |
|---|---|---|
| 原始定位 | 所有 decoder query 的 `M>0.5` 面积；对 GT 坑区域的最佳单 mask IoU、全 query 并集覆盖；再按 `r>t` 分层 | 是否连原始 query 都缺少有用定位；“最佳 IoU”是 GT 事后 oracle 上界，不能当预测分数 |
| 条件分类 / presence | 每图 p、r/q 分布、`r>t` 和 `q>t` 的数量及对应 GT 覆盖；GT 有/无坑分别累计 | 有定位但 r 低，还是乘 p 后被删；`p≤t` 时必然没有 `q>t`，降阈值能否救回不可只看平均 p |
| 合并与竞争 | GT 坑像素上 `D_crater`、`pD_crater`、`I_crater`、`F_crater` 对最大其他类的胜负率与差值；被月壤/岩石/山体覆盖的比例 | 坑证据是否已经存在，但类间分数不均衡而在最后 argmax 丢失 |
| 缺失类误报 | GT 不含坑图上的坑像素数/面积、p 与保留实例数 | 放宽实例阈值是否只是扩大误报；不能只展示含坑图召回 |

原始 query 总是固定数量的预测槽，所谓“proposal 不存在”在这里严格指**没有可用定位**，不表示模型完全没有输出槽。最佳单 mask IoU 与并集覆盖都要看：前者会受 Moon 语义标签多坑合并影响，后者又可能被大面积假阳性抬高。报告按场景分开，坑组件大小仅作语义连通域诊断，不冒充实例标注。

有限检查已完成。CPU核对了严格阈值、无保留实例、全空reject、含拒识FN、全零/NaN失败、同分最低ID、FP32/BF16分块一致、混合dtype下逐实例标量乘权，以及两张非方形输入的旧适配一致性/改词缓存/类别换序。GPU2按旧审计固定18个ID处理，恰好90次grounding，三场景各6张；另在固定首尾两图上对比重构前旧适配与现有`forward`，最大分数差和标签差均为0，并对10个prompt、两个阈值独立按作者processor/segmentor计算keep、I、F，最大差均为0。没有使用GT生成提示或筛类别。GPU2身份、空闲、共享训练/测试锁均核对，结束后无计算进程且两把锁可重新取得。

固定18图的场景等权mIoU仅作回归诊断：legacy dense 69.170%、dense_presence 67.859%、SegEarth t0.5 67.674%、t0.3 65.211%、单次presence t0.5 67.656%；两个实例诊断覆盖27.568%/29.652%，坑实例两档均为0。它们不证明全量收益或方法胜负；旧18图审计的69.168%与本次不是同一次数值运行，回归依据是本轮同进程重构前/后的逐像素零差，而不是用四舍五入旧小集分数替代核对。全量结果尚无。固定18图峰值已分配显存约3.90GiB，另行数值对照约4.16GiB，这不是全量吞吐优化实验。

运行使用现有[`sam3_pretrained.yaml`](../configs/sam3_pretrained.yaml)，不改模型权重、五词、单尺度或DINO配置。完整命令如下；入口自动核对物理GPU2 UUID、实时空闲并持有两套共享锁，进程内只暴露该UUID，设置`NCCL_P2P_DISABLE=1`，输出不得覆盖旧文件：

```sh
cd /home/disk02/wjj/moonseg/code
/home/disk02/wjj/moonseg/.venv-baselines/bin/python -m moonseg.evaluate_sam3_multimode \
  --config configs/sam3_pretrained.yaml --profile single --mask-chunk-size 8 \
  --data-root data/research_test/LuSNAR --manifest data/research_test/manifest.json \
  --output runs/sam3_protocol_20261110/full.scores.json
```

加`--subset audit18`只运行既有固定18图，输出明确写`evaluation_scope=subset_diagnostic`，不能伪装全量。已运行的完整小集结果为`runs/verification/sam3_multimode_20261110/audit18.scores.json`；详细作者/旧适配对照为同目录`parity.json`。全量输出是一个combined JSON：`modes`有5组完整五类混淆、每场景指标及scene_equal_macro，`instance_diagnostics`有2组5×6拒识混淆和二值覆盖，`per_image_diagnostics`保存p、r/q筛选计数、坑原始query最佳IoU/覆盖及各模式GT坑被其他类别覆盖的计数。全query坑mask按8个分块还原；不保存整套全量mask张量。结果包含预训练权重/Meta/作者提交/代码身份、原词、输入规范、dtype、profile及样本计数；异常输出独立failure状态，不产生伪完整分数。

之后**一次遍历公开测试 5004 张**，每图每个原词只做一次 grounding，流式从同一份原始输出计算以下结果；这只是共享前向的评测组织，不另做批量、缓存、编译等性能工程，也不保存 5004 张全部高分辨实例张量：

1. legacy dense：必须重现旧结果，作为输入、权重和数据口径的锚点。
2. dense×presence：定位 presence 的纯影响，不能称完整修复。
3. instance 诊断：`t=0.5/0.3`，保留 reject、覆盖及二值并集统计，不与完整五类分数直接排名。
4. SegEarth 双头：`t=0.5/0.3`，主比较；另计算主阈值 `t=0.5` 的 `max(pD,I)` 单次 presence 消融。连同前两项共 5 组完整五类结果，另有 2 组实例诊断；后续参数扫描逐项登记，保持因素可区分。

完整五类输出记录每场景 1597/1320/2087 张、五类 IoU/召回、混淆矩阵、场景等权 mIoU/mAcc及 pooled 指标。1:1:2 如展示仍只是估算；接入和上述有限检查已执行，这轮5004张全量已在GPU2运行，状态见 `runs/sam3_protocol_20261110/full.receipt.json`，日志为同目录 `full.log`，完成后写 `full.scores.json`。各后处理失败也要保存状态，不能只留下最高分。原词对照可在一次遍历中完成；新增提示词需要对应的新前向，不能假称可仅靠旧输出复用。小集用于工程核对，改进是否成立以及词表/参数如何选择均以公开 2/4/9 全量开发评估为依据，完整记录扫描范围和各配置结果。

## 第二阶段：先文本，再训练集视觉参考

如果第一阶段表明是类别竞争主导，应优先核实竞争解码；如果原始坑定位或 presence 仍弱，文本扩展/视觉参考才对准故障。这里是条件优先级，不预设双头必然涨分。

**词表是最小新增工作。** SegEarth 已实现“逐别名推理、类内最大”的明确规则；可保留原词基线，新增月壤 `ground` / `lunar soil`，坑 `crater` / `lunar crater` / `pit` 的候选，但这些只是待筛选词，已有 18 张没有证明坑改词有效。当前无独立 val，训练池小集仅可选作工程调试，不新设场景 5 开发门槛。按用户既定公开开发评估方式，在公开 2/4/9 全 5004 张上比较词表、别名及阈值，扫描范围和全部结果留档；不同别名数同时改变推理量与误报机会，记录每类别名和解码次数。query GT 只用于输出后的指标与故障统计，不传入模型，也不用于生成点框或确定某张测试图要查询哪些类。

视觉参考的共同前提是只取官方训练池 1/3/5/6/7/8，单列为**有标签参考、冻结模型**，不与无标签 text-only 都写成同一个零样本方案。参考可取完整合法训练池；训练池小集可选作坐标和缓存调试，不重新留出场景 5。支持清单、随机种子和 1/5 shot 数量逐配置记录，完整方案以公开 2/4/9 全量结果比较选择。不得把测试帧或其 GT 加入支持库，也不得用 query GT 生成点框或存在类门控；支持清单变更及对应全量结果须可追溯。

| 路线及一手证据 | 已核实的可接入部分 | 与 Moon 的差异 / 优先级 |
|---|---|---|
| [FSS-SAM3 论文](https://arxiv.org/html/2604.05433v1)，[作者 b82ba838](https://github.com/WongKinYiu/FSS-SAM3/tree/b82ba838678ae8c2f27f618d9500a50c171e223e) | `evaluate.py` 正框 canvas 推理直接调用 Meta processor；1-shot 默认 1008 canvas、vertical、support 比例 0.6、text 开启，COCO 默认 swap 支持在下；1/5-shot 都把保留二值实例取并集，裁出 query 后最近邻还原 | 工程最短的视觉定位诊断，可先做 1-shot 含坑/无坑开发对照；完整五类方案优先级低于 SegRAG。query 只占 1008×404，原 1024² 非等比压缩；小坑信息损失和形变是实在代价，不先发明多尺度 canvas 称作者复现 |
| [SegRAG 论文](https://arxiv.org/html/2605.17630v1)，[作者 6e9584be](https://github.com/boudiafA/SegRAG/tree/6e9584bef340e8e3b065897dfa41c62959537aff) | 原生 DINOv3 ViT-L/16 建类索引 bank，ICCD、TSG points 与 text 同一次 SAM3 grounding；现有 Meta prompt 支持 append_points，代码可接 | 完整视觉路线首选研究候选；query 不需与支持图拼 canvas，但增加 DINO 特征/存储/预处理及 GT 门控移除工作。先验收全五类评测，再考虑 full test，不直接照搬论文 mIoU |
| [ActiveSAM v2](https://arxiv.org/html/2606.16996v2)，[作者 25b84d85](https://github.com/VILA-Lab/ActiveSAM/tree/25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe) | CPE、EPG、ECD 均有源码；融合也在 q 和最终图各用一次 presence | 五类无需大词表 preview 剪枝/加速，且坑 p 已弱；默认 COCO 10k 无标签 signature 池不是现成 Moon 工具。竞争主导时才研究 ECD，校准限定合法训练 RGB，并明确这是改校准域，不是原作者默认复现 |
| [CoCo-SAM3](https://arxiv.org/html/2604.19648v1) | 论文用跨类语义 prior 校准 mask/presence；同义词只在 prior 分支聚合，与 SegEarth 逐词结构解码不同 | 本次未找到可核实作者实现，原文未链仓库；中间层、特征投影和精确接口尚不足以直接接入。保留竞争问题依据，暂缓实现，不能自行拼 softmax prior 冒充复现 |
| [DCR-SAM](https://arxiv.org/html/2610.27960v1)，[作者 3c134aa0](https://github.com/set-path/DCR-SAM/tree/3c134aa0e1849ea1bfbfe166eaa0d1309bc24cb2) | 训练 prompt/跨模态与 dense competitive head，是监督新头路线，不是 LoRA，也不是纯后处理 | 与本轮纯推理问题分开，暂缓；README 明示评估导入的 `vis.visualizer` 未包含，完整可跑性未验收。不得影响 DINO 主线或直接安排训练 |

### FSS 和 SegRAG 还缺什么

FSS 的[标准 `evaluate.py`](https://github.com/WongKinYiu/FSS-SAM3/blob/b82ba838678ae8c2f27f618d9500a50c171e223e/model/evaluate.py#L217)对单一目标类别做 foreground/background 分割，各实例二值并集没有跨类别冲突处理。Moon 要输入训练支持的 tight bbox；标准脚本实际导入的是 [`data/dataset_tool.py`](https://github.com/WongKinYiu/FSS-SAM3/blob/b82ba838678ae8c2f27f618d9500a50c171e223e/data/dataset_tool.py#L127)，COCO随机选择面积>1000的非crowd实例原bbox；PASCAL从实例图中选与目标语义相交的最大对象再取外接框，而非整个类别大框。仓库另一份`data/dataset.py`虽有8连通域取框，但不是标准脚本导入路径，不能误用来证明标准协议。Moon标签是语义而非实例，把连通域或整类外接框当实例会改变原协议，须单独登记Moon框来源，不能把测试GT框输入query。它也没有现成可比的五类连续分数融合规则：先作为坑定位/召回诊断；若推进完整五类，须单独核实如何保留类别概率和处理重叠/未覆盖，不能二值类别顺序覆盖或默默回填月壤。`evaluate_neg.py` 是独立负提示实验：support 先以 threshold 0 推理，`score>0.5` 为正、`0.1<score≤0.5` 为负，NMS 0.5、默认最多 1 负框，query 阈值再设回 0.5。README 的“refine”措辞不等于已证实增益；[论文 §4.3/§5.3](https://arxiv.org/html/2604.05433v1)报告负提示退化，因此首轮只用标准正提示。

SegRAG 的[联合提示实现](https://github.com/boudiafA/SegRAG/blob/6e9584bef340e8e3b065897dfa41c62959537aff/src/segrag/modeling/sam3_text_points.py#L104)逐类 clone/reset state，将 point 的 x/y 分别除 W/H，append_points 后一次 `_forward_grounding`；没点退回 text-only，异常回退单独计数。当前构造器 confidence 为 **0.4**，不是 Meta 默认 0.5。作者 [Cityscapes 固定配置](https://github.com/boudiafA/SegRAG/blob/6e9584bef340e8e3b065897dfa41c62959537aff/configs/paper/cityscapes.yaml)可作为可执行参数起点：DINO 输入 1536、patch16、foreground occupancy 0.90，ICCD `clip(0.90*q75,0.65,0.82)`、top-K 10000，TSG similarity/loose/validation 0.80、peak 0.20、min component 4、min peak distance 10、每类最多 10 points。单参考直接保留前景特征，不执行多参考 ICCD；这些参数是作者通用数据集配置，未证明适合 Moon，小坑可能在 bank occupancy/组件阈值前消失，应记录中间存活数；参数扫描需连同全量公开开发结果登记，不以小集猜测代替有效结论。

更关键的是其[记录生成](https://github.com/boudiafA/SegRAG/blob/6e9584bef340e8e3b065897dfa41c62959537aff/src/segrag/modeling/sam3_points.py#L43)从 query 注释构造 `valid_cats=cats_in_img-not_exhaustive`，[运行循环](https://github.com/boudiafA/SegRAG/blob/6e9584bef340e8e3b065897dfa41c62959537aff/src/segrag/modeling/sam3_text_points.py#L274)只对这些存在且有 GT 实例的类预测；[指标](https://github.com/boudiafA/SegRAG/blob/6e9584bef340e8e3b065897dfa41c62959537aff/src/segrag/utils/metrics.py#L46)各类独立并集，没有互斥 argmax，也未惩罚对未查询缺失类的误报。这是其当前 query-class 评测协议，不应直接等同于 Moon 全词表五类语义分割。接入时必须从输入侧去掉 GT 门控：bank 仅训练标签参与，query 的点生成和 grounding 固定全五类；推理完成之后才接 GT 指标。原生二值 union 指标可保留为定位诊断，完整五类输出另用第一阶段冻结的 dense/instance 融合，标为“SegRAG 提示 + Moon 五类适配”；这个组合尚未测，不声称作者提供了该完整工具。先用相同 text-only / text+points 下游融合做控制，避免同时变 prompts、阈值和解码再归因给检索。

## 工作分块与验收

| 工作块 | 交付及验收标准 | 晋级条件 |
|---|---|---|
| 0：源码审查，已完成 | 本文、原始小源码/树和来源清单；公式逐项对应pinned源码；区分Meta原生、作者改动及Moon诊断/适配 | 作为实现的来源依据 |
| 1：接入与有限核对，已完成 | 共享前向、prompt缓存键、空集合/reject、原生dtype、CPU边界及固定18张；两图旧适配和作者公式最大差0 | 主agent验收代码及有限检查；不能用涨分替代正确性验收 |
| 2：一次原词全量，GPU2运行中 | 同 checkpoint/源码/原五词，5004 张同时给出上述所有后处理；完整每场景混淆及失败层级统计；旧 dense 复现，双头来源/数值模式清楚 | 确定主要瓶颈。性能没有先设涨分保证；若坑定位依旧缺失，不继续堆后处理 |
| 3：词表与参数扫描 | 在公开2/4/9全5004张比较候选prompt、阈值及别名机制，标明公开开发评估；记录全部配置、五类指标及无坑图误报 | 依据全量结果选择改进；对比同时看三场景和五类，坑召回增益不得隐去其它类退化；小集仅可选工程调试 |
| 4：视觉参考可行性 | 核对合法训练支持清单及公开query清单，FSS正框诊断、SegRAG全五类无GT门控原型；支持及query严格分离，text-only对照共享解码；本轮未执行 | 五类互斥输出及输入隔离正确后，以1-shot/5-shot完整公开开发评估判断收益，逐配置记录支持清单、参数与结果 |

仍未核实的项目是：Meta F.1 全部语义阈值/面积/冲突配置；FSS 已核实的实例框约定向 Moon 语义标签的映射；SegRAG 在全五类、无 query GT 门控时的收益、点缓存上游全部路径的输入隔离及不同 SAM3 源码版本的兼容性；CoCo 作者源码和 DCR 完整运行性。这些缺口都有明确检查位置，不以“可能有效”充当已完成集成。

本次原始小文件放在 `references/sam3_route_20261110/`（disk02，仓库 `references` 软链，未入 Git）；[`source_index.json`](../references/sam3_route_20261110/source_index.json)记录仓库、提交、抓取时间、路径及字节数。未下载新模型或大型仓库，没有提交测试脚本、commit或push；仅GPU2执行上述有限检查，未启动新全量或训练任务。
