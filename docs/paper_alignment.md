# LuSNAR 论文与当前实现的对照

核对日期：2026-10-22（UTC+8）。本文记录 LuSNAR 论文中可核实的事实、公开渠道能找到的第三方结果，以及论文、比赛规则和本项目工程选择三者的边界。1024×1024 整图、补充 mAcc、按现有分割实现选择增强、用梯度累积使有效 batch 不低于 64，都是用户确认的本项目决定，不写成论文作者的做法。

## 论文版本与可用证据

主要依据是 TGRS 期刊终稿：**LuSNAR: A Lunar Segmentation, Navigation, and Reconstruction Dataset Based on Multisensor for Autonomous Exploration**，Jiayi Liu 等 12 位作者，IEEE TGRS vol. 64，2026，article 4706723，DOI [10.1109/TGRS.2026.3720373](https://doi.org/10.1109/TGRS.2026.3720373)（[出版社记录](https://ieeexplore.ieee.org/document/11643283/)）。论文 2025-10-23 收稿，2026-03-05 与 2026-05-28 两次修改，2026-07-31 接收，2026-08-05 在线发表，当前版本日期为 2026-09-10；全文 23 页，参考文献 52 条。

本地副本是用户提供的出版社 PDF（IEEE Xplore 机构授权下载，页脚注明下载时间为 2026-10-22 04:06:02 UTC），文件名、sha256 和使用限制见 [references.md](references.md#存档方式)。该 PDF 受 IEEE 版权约束，只放在本地 `references/` 中用于核对，不入库，也不转发。下文的章节号、表号和图号都指这份终稿。此前"终稿正文无法取得"的说法已经作废。

arXiv 共有 v1（2024-07-09）、v2（2024-09-05）、v3（2024-09-26）三版（[版本页](https://arxiv.org/abs/2407.06512)）。**arXiv:2407.06512v3** 只在终稿没有涉及的地方作补充来源，并用于下面的版本差异核对；两者冲突时以终稿为准。

### 终稿相对 arXiv v3 的变化

逐节对照两版全文后，与本项目有关的变化如下：

- 2D 分割基线从 5 个增加到 7 个。表 V 新增 SAM2 和 SAM3：SAM3 最佳（mIoU 91.48、mAcc 95.50），SAM2 为 90.15，都高于 SegFormer 的 89.96；原有五个模型的数字没有变。§IV-A 还新写明 2D 实验 "are all based on the MMSegmentation platform"。
- 表 V 的文字分析改写了。v3 说 PointRend 在月壤和撞击坑上最好，终稿改为月壤和天空（"due to its efficient edge refinement in large continuous regions"）。v3 的 "Both Deeplabv3+ and Unet have errors in the category of impact craters, which may be due to the presence of shadow areas within the craters" 一句已删除，换成对月壤与撞击坑、岩石与山体之间混淆的描述（见[论文的误差分析](#论文的误差分析)）。
- 3D 与 2D 的差距按新的最佳 2D 结果（SAM3）重算：Point Transformer 在岩石上高 14.71、在撞击坑上高 3.55（v3 为 16.47 与 6.22）；表 VI 的数值没有变。
- 泛化实验的表 VII、表 VIII 数值没有变。正文的分组均值改为场景 4–6 是 86.53、场景 7–9 是 83.27，与表 VII 自洽；v3 正文写的 85.30 与 80.87 和它自己的表对不上。终稿说明选 SegFormer 和 Point Transformer 是 "considering the limitation of the onboard computing resources"，SAM2 和 SAM3 没有做泛化实验。
- 新增嫦娥实拍图像实验（§IV-A 末段、表 IX、图 10–11），摘要中 "sim-to-real transfer potential" 一句指的就是这个实验。
- 场景设计：§III-B 改写为 "systematic orthogonal design"，两个维度各分低、中、高三级，3 × 3 恰为九个场景。v3 中 "Scenes 1, 4, and 6 … 2, 5, and 7" 的笔误已改为 "Scenes 1, 4, and 7 exhibit relatively flat terrain, while Scenes 2, 5, and 8 feature minor undulations … Scenes 3, 6, and 9 display significant terrain variations"，图 2(b) 也加了场景编号。终稿还增加了地貌类比：mare-like（"e.g., Scenes 1 and 4"）、transition（场景 2、5、8）、highland-like（场景 3、6、9，"similar to the environment at the lunar south pole"）。
- 表 III 的数值与 v3 相同，但正文从 v3 的 "all impact craters" 改为 "the major impact craters"。v3 中参照 Apollo 16 撞击坑（约 10–100 m）的一句被删除，只保留 CE-5 着陆区的统计。
- 数据描述：v3 的 "13,006 sequences" 改为 "a collection of nine sequences"，每个场景一条 1000–2000 帧的序列。标签说明从 v3 的 "an ID of the represented object is encoded" 改为 "a different color represents a different type of object"。新增内容有：传感器建模与局限（理想针孔相机，不加畸变和暗角，"auto-exposure with a simplified photometric model"）；标签由 G-Buffer 直接生成，称为 "pixel-perfect"；按 5% 抽检 650 组样本、全部通过的数据核验；未模拟 opposition surge、月尘等仿真局限。
- 立体匹配增加了 FoundationStereo（共 6 种方法），SLAM 部分的方法没有变。作者从 10 人增至 12 人（新增 Chenming Ye、Zhizhong Kang）。终稿参考文献 52 条，包括 SAM2、SAM3 和 FoundationStereo；此前 OpenAlex 只解析出 39 条，并不完整，据此推测"终稿仍是 CNN/SegFormer 一类基线"并不成立。
- 摘要说评测了 "25 state-of-the-art algorithms"，按表计数可以凑齐：2D 分割 7 个、3D 分割 5 个、表 X 的 7 个 SLAM 配置（ORB-SLAM3 的 stereo 与 stereo-inertial 分开计）、立体匹配 6 个。这是本文的计数，论文没有列出清单。

## 已明确的实验内容与当前实现

| 核对项 | TGRS 终稿的明确内容与位置 | 当前实现 | 对齐结论 |
|---|---|---|---|
| 原始数据 | §III-C、表 IV、§IV-A：9 个场景各一条序列，每条 1000–2000 帧，"eight short sequences and one long sequence"，共 13,006 张图像；RGB 与 2D 标签为 1024×1024 PNG；相机 10 Hz、80°×80° FOV、基线 310 mm | 原始池整理出 13,006 张 RGB，按场景保留 | 数据规格相符；这不能证明作者的训练输入尺寸 |
| 场景设计 | §III-B、图 1、图 2(a)(b)(d)、表 II、表 III：起伏 × 地物分布的 3×3 正交网格，以及各场景主要撞击坑的尺寸 | 网格、帧数与撞击坑见 [data.md](data.md#场景总览)，用于选择验证场景 | 属于场景属性，不是训练配方 |
| 标注视角 | 图 1 图注：每个场景展示 "the left-eye image, semantic labels, depth map, and LiDAR point cloud"；v3 §III-C 明确写了 2D 标签对应左侧图像 | 使用原始包中带标注的 `image0/color` 与 `image0/label` | 按时间戳配对；比赛进一步限定为单张左 RGB |
| 2D 语义类别与编码 | §III-B、§III-C、§IV-A、表 V、图 6(a) 图例：月壤、岩石、撞击坑、山体、天空；每个像素用颜色区分类别，标签由 G-Buffer 直接生成 | 五类全部保留，比赛 ID 0–4，按调色板 RGB 解码 | 类别语义相符；论文没有给出颜色数值或类别 ID，这两项另据官方数据说明和比赛规则 |
| 主实验划分 | §IV-A 首段：训练 1/2/4/6/8/9，测试 3/5/7（与 v3 相同） | 比赛训练池 1/3/5/6/7/8，测试 2/4/9；开发期留场景 5 验证 | 必须按比赛划分，不能照抄论文划分 |
| 泛化实验划分 | §IV-A、表 VII 之前：训练 1–3，测试 4–9 | 不使用 | 是另一项论文实验，不是本比赛或开发验证的划分 |
| 比较模型 | §IV-A、表 V：在 MMSegmentation 上评测 DeepLabv3+、MobileNetV3、U-Net、PointRend、SegFormer、SAM2、SAM3 | DINOv3 + ViT-Adapter + Mask2Former | 是用户确定的新方法路线，不是复现论文中的模型 |
| 报告指标 | §IV-A：每类 IoU、mIoU、mAcc | 每类 IoU、mIoU、mAcc，每类 accuracy/recall/support，pixel accuracy 及混淆矩阵（[metrics.md](metrics.md)） | mAcc 采用公开分割实现的宏平均召回率；缺类处理按比赛口径，不声称还原了作者未公开的统计细节 |
| 原论文基线分数 | 表 V、表 VII、表 IX（见下） | 开发池基线 2026-10-21 启动，结果待填（[roadmap.md](roadmap.md#实验记录)） | 模型和划分都不同，论文分数不能作为比赛的复现目标 |
| 仿真到真实 | §IV-A 末段、表 IX：用 200 张 CE-3 实拍图训练、20 张 CE-4 图测试，并与加入 LuSNAR 联合训练的结果对比 | 不使用任何实拍或外部数据 | 比赛只允许使用官方训练清单；该实验只说明数据集的用途，不是可用的训练来源 |
| 实验观察 | §IV-A 讨论颜色和纹理相近造成的混淆，以及阴影、地形起伏、地物分布与分割表现的关系；§IV-C 讨论场景 9 的阴影对立体匹配的影响 | 水平翻转与标准强度的亮度/对比度；gamma、阴影增强关闭 | 论文的误差分析不能当作增强配方或增益证据（[augmentation_research.md](augmentation_research.md)） |

终稿仍然没有交代三件事：指标如何聚合（逐图平均、全测试集像素累计还是逐场景累计）、无有效类别时怎么处理、mAcc 如何实现。表 VII 按场景报告，但没有说明场景内怎样汇总。表 V、表 VII 中各类 IoU 都不为零；场景 1 没有撞击坑，而它从未作为测试场景，所以论文结果不受缺类口径影响，也无法据此反推作者的口径。当前按场景累计像素计数、只平均参考标签中出现的类别，这是**比赛官网**给出的口径，不应归为论文的实现。

### 表 V：主实验（训练 1/2/4/6/8/9，测试 3/5/7）

设置方面，终稿只写明实验在 MMSegmentation 上进行、分五类。输入尺寸、迭代次数、优化器、增强、预训练权重、SegFormer 型号都没有报告，SAM2/SAM3 用的哪个型号、是否微调、如何改造成五类语义分割也没有说明。

| 模型 | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 | mAcc | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLabv3+ | 98.01 | 75.64 | 79.62 | 83.11 | 99.68 | 92.90 | 87.39 |
| MobileNetV3 | 98.72 | 75.16 | 76.96 | 82.98 | 99.57 | 91.99 | 86.68 |
| U-Net | 98.70 | 73.50 | 73.09 | 79.46 | 99.34 | 91.01 | 84.82 |
| PointRend | **99.07** | 76.17 | 82.45 | 84.00 | **99.70** | 93.29 | 88.28 |
| SegFormer | 99.05 | 81.69 | 80.37 | 89.16 | 99.54 | 94.22 | 89.96 |
| SAM2 | 98.90 | 81.12 | 83.56 | 88.42 | 99.62 | 94.32 | 90.15 |
| SAM3 | 98.98 | **83.45** | **85.12** | **90.28** | 99.65 | **95.50** | **91.48** |

前五行与 v3 相同。v3 中由 SegFormer（岩石、山体、mAcc、mIoU）和 PointRend（撞击坑）保持的最好成绩，在终稿中都被 SAM3 超过；PointRend 仍在月壤和天空上最高。

### 表 VII：泛化实验（SegFormer，只用场景 1–3 训练，逐场景测试）

| 测试场景 | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 | mAcc | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 98.75 | 87.07 | 66.81 | 96.93 | 99.61 | 92.19 | 89.83 |
| 5 | 98.22 | 70.68 | 60.48 | 74.95 | 99.53 | 89.19 | 80.77 |
| 6 | 98.75 | 85.18 | 68.05 | 93.72 | 99.28 | 92.90 | 89.00 |
| 7 | 98.43 | 86.73 | 66.36 | 93.75 | 99.25 | 92.59 | 88.90 |
| 8 | 98.60 | 83.06 | 73.86 | 86.18 | 98.52 | 92.36 | 88.05 |
| **9** | 96.20 | 76.20 | **19.19** | 76.23 | 96.43 | 78.47 | **72.85** |

六个测试场景的平均 mIoU 为 84.90，其中场景 4–6 平均 86.53，场景 7–9 平均 83.27。同一实验中的 Point Transformer（表 VIII，只分三类的 3D 点云分割）平均 93.27，也在场景 9 最低（86.41，撞击坑 65.06）。

### 表 IX：嫦娥实拍图像（SegFormer，在 20 张 CE-4 图像上测试）

| 训练集 | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 | mAcc | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE-3（200 张） | 95.43 | 48.80 | 50.99 | 0.00 | 98.79 | 72.19 | 58.80 |
| CE-3 + LuSNAR | 96.42 | 45.47 | 54.74 | 44.41 | 99.38 | 81.33 | 68.08 |

实验做法：CE-3、CE-4 巡视器图像没有标注，作者用 labelme 人工标了五类，其中岩石和撞击坑 "only regions that pose potential traversal hazards were annotated"，所有图像缩放到 1024×1024。CE-3 区域有月壤、岩石、天空和少量撞击坑，没有山体；CE-4 区域有月壤、撞击坑、天空和少量岩石、山体。第二组把 "nine scene images from the LuSNAR dataset" 加入 CE-3 训练集，但没有说明用了多少帧、哪些场景、是否加权。两组实验 "with all parameters kept identical"，选 SegFormer 的理由是机载算力有限。

论文的结论是 "LuSNAR effectively compensates for the scarcity of labeled real data"，以及 "the visual and geometric patterns learned from LuSNAR are highly transferable"。按表 IX 核算：mIoU 的 +9.28 中，山体从 0 升到 44.41 一项就贡献了 8.88（44.41/5）；不计山体，其余四类的平均只从 73.50 升到 74.00，岩石 IoU 反而下降 3.33。撞击坑提高了 3.75，正文写成 "a 3.75% increase in mIoU"，实际是该类 IoU 的增量。图 11 定性显示，只用 CE-3 训练的模型 "incorrectly classified the shadows around rocks as craters"，加入 LuSNAR 后有所改善。

对本项目的含义（推断）：

- 论文没有报告只用 LuSNAR 训练、直接在实拍图像上测试的结果，因此不能推断本比赛中只用 LuSNAR 训练的模型可以零样本用于嫦娥或载人登月影像。现有证据只支持把 LuSNAR 当作少量实拍标注的补充数据。
- 增益主要来自补上实拍训练集中缺失的山体类，这说明的是类别覆盖，不是外观迁移。
- 实拍标注与 LuSNAR 的口径不同：只标有通行风险的岩石和撞击坑，测试集只有 20 张，相机和成像条件也不同。比赛提到的真实影像应用验证如果进行，需要另外适配；本项目不以此为目标，比赛训练中也不能引入任何实拍数据。

### 论文的误差分析

- §IV-A（表 V）："minor errors in distinguishing between lunar regolith and impact craters, as well as between rocks and mountains, still persist. This difficulty arises due to minimal variation in the color and texture of the lunar surface …, although SAM-based models demonstrate stronger robustness in these scenarios."
- §IV-A（泛化）："Overall, segmentation performance is highest in Scene 4 and lowest in Scene 9." 场景 5、6 的 SegFormer mIoU 都低于场景 4，"with Scene 5 showing the lowest accuracy, likely due to the presence of extensive shadow areas"；在场景 7–9 中，mIoU 随起伏增大而下降。场景 4–6 内部并不单调（场景 5 为 80.77，场景 6 为 89.00）。v3 中 "scenes 4, 5, and 6 roughly show a decreasing trend in mIoU as the slope steepness rises" 这一笼统说法终稿已删去，"similar decreasing trend in Scenes 4–6, as well as in Scenes 7–9" 只用来描述 Point Transformer。
- 地物分布：SegFormer 在场景 4–6 平均 86.53，到场景 7–9 降为 83.27，"as the richness of lunar surface objects increases, segmentation performance declines"。
- 2D 与 3D：Point Transformer 在岩石上比最佳 2D 方法（SAM3）高 14.71，在撞击坑上高 3.55，2D 的短板集中在岩石和撞击坑。
- §III-A：相机 "may struggle to detect obstacles in shadows"。
- v3 把 DeepLabv3+ 和 U-Net 的撞击坑误差归因于坑内阴影，这句终稿已删除，不能再作为终稿结论引用。

### 场景 9 与阴影

比赛概述称场景 9 "地形起伏最为显著，岩石与撞击坑分布密集，并存在大面积阴影区域，是九个场景中感知难度最高的场景"。终稿中的直接证据如下：

- 设计位置：在图 2(a) 中，场景 9 是高起伏、地物丰富的那个角；§III-B 称复杂度 "gradually increases from Scenes 1 to 9"，场景 7–9 地物最丰富。图 2(c) 中场景 9 的岩石总量（圆面积）远大于其他场景；图 2(d) 中场景 9 有 6 个撞击坑，与场景 8 并列最多，而表 III 只列出其中 4 个主要撞击坑。
- 分割：表 VII 中场景 9 的 SegFormer mIoU 为 72.85，是六个测试场景中最低的，撞击坑 IoU 只有 19.19，mAcc 78.47；表 VIII 中 Point Transformer 也在场景 9 最低。
- 阴影：§IV-C 称场景 9 "characterized by highly complex terrain and extensive shadowed regions"，并写道 "the large shadowed area in Scene 9 leads to severe matching errors, possibly due to confusion between dark terrain regions and the black sky"。表 XI 中学习型立体匹配方法在场景 9 的误匹配率为 68.20–98.16%（FoundationStereo 在其他场景为 6.40–26.98%），只有 BM 较低（64.62%）。表 XII 中场景 9 在 5 m 范围内的重建误差为 0.6146 m，其他场景都不超过 0.0081 m。
- 在分割部分，论文没有把场景 9 的低分直接归因于阴影。针对具体 LuSNAR 场景的阴影归因只有两处：场景 5 的分割和场景 9 的立体匹配；此外 §III-A 有一句泛泛的相机局限，嫦娥实验中也提到阴影被误分为撞击坑。
- 推断：论文说的"暗地形与黑色天空混淆"与本项目的实测一致。深阴影和天空都截断在 22/28 灰度级（[data.md](data.md#像素值与下限-22)），场景 9 物体带中的暗像素占 19.6%，训练场景只有 3.1–9.5%。泛化实验的训练集（场景 1–3）按表 III 只有 2 个撞击坑，而比赛训练池中仅场景 5/6/7/8 就有 15 个（按图 2(d) 是 17 个），所以场景 9 撞击坑 IoU 19.19 是一个悲观的下界。即便如此，撞击坑仍是场景 9 上最差的类别，岩石和山体（约 76）次之。场景 9 有两个 d/D 只有 0.05 的浅坑，而 w9 权重最高，所以场景 9 的撞击坑 IoU 是最大的提分杠杆。

### 深度、SLAM 与 3D 分割（非本任务）

- SLAM（§IV-B、表 X）：在九个场景上评测了 ORB-SLAM3（stereo 与 stereo-inertial）、VINS-Mono、VINS-Fusion、A-LOAM、LeGO-LOAM 和 FAST-LIO，视觉 SLAM 整体优于 LiDAR SLAM（视觉 SLAM 的最大 RMSE_ATE 为 0.4632%，LiDAR SLAM 平均 1.9958%）。
- 立体匹配与重建（§IV-C、表 XI–XII）：6 种方法零样本评测，FoundationStereo 的平均误匹配率最低（22.56%）；用 CREStereo 做单站重建，误差随范围增大，场景 9 最大。
- 3D 分割（表 VI、表 VIII）：月壤、岩石、撞击坑三类，Point Transformer 的 mIoU 最高（95.48）。

## 公开代码与第三方结果

公开渠道**没有按场景划分训练的 LuSNAR 2D 分割配方，也没有排行榜**。论文表 V 和表 VII 是唯一同源的参照，但划分不同，不能直接比较。

- [官方 GitHub](https://github.com/zqyu9/LuSNAR-dataset)（终稿摘要给出的数据集地址）：最新提交为 2026-09-15 的"添加 LuSNAR 发表在 TGRS 的信息"，[根目录](https://api.github.com/repos/zqyu9/LuSNAR-dataset/contents/)只有 `README.md`、`index.html`、`static/`，没有训练或评测代码；README 中的"code released"不能当作训练入口。
- [README](https://api.github.com/repos/zqyu9/LuSNAR-dataset/readme) 的"Baseline Results"表（FCN 76.3、SegNet 78.9、PSPNet 82.1、DeepLabV3+ 84.6，以及 3D 重建的"Ours 0.082 m RMSE"、COLMAP、MVSNet）与论文矛盾：论文表 V 中 DeepLabv3+ 为 87.39，终稿和 v3 都没有评测 FCN、SegNet、PSPNet。多个聚合站在转载这组数字，**不能当作目标或基线**。README 写的 HF 数据集 id 是 `JeremyLuo/LuSNAR`，但 HF 搜索实际只返回 [`xuboluo2001/LuSNAR`](https://huggingface.co/api/datasets?search=lusnar)。
- [官方仓库 issue #18](https://github.com/zqyu9/LuSNAR-dataset/issues/18) 询问 SegFormer B0/B2 的复现，作者没有回复；终稿也没有写明型号。这是读者的提问，不能据此断定具体型号。
- [GitHub 仓库搜索](https://api.github.com/search/repositories?q=LuSNAR)只找到 3 个仓库：官方数据集仓库，以及 `ArturPrentki/orb-slam3-lusnar`、`henrylanday/lunar-lidar-slam` 两个 SLAM 项目。[HF 模型搜索](https://huggingface.co/api/models?search=lusnar)只有深度模型 `LuMonDepth/DAv2-FT-LuSNAR`。[Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2407.06512/citations) 列出的 12 篇引用中，没有一篇给出按场景划分的 2D 分割配方或成绩；Papers with Code 已停止服务，没有找到排行榜。
- [Stanford CS231N 2025 报告](https://cs231n.stanford.edu/2025/papers/CS231N_Final_Project_Report%20-%20Brian%20Y%20Wu.pdf)（Brian Y. Wu）：80/20 划分，约 10k 训练、2k 验证。报告只写了 "suggested 80-20 split"，没有说明是否按场景划分；10,919/2,087 恰好也等于留出场景 9 时的帧数（见 [references.md](references.md)）。增强为水平翻转、±10° 旋转和亮度/对比度；5 epoch，Adam，SegFormer lr 5e-5，MaskFormer lr 1e-5，batch 4，单卡 3090；mIoU 为 SegFormer 0.9686、MaskFormer 0.8955、UNet++ 0.7681、SAM-based 0.7766。这些分数比论文的跨场景结果高约 10–25 个点，说明随机按帧划分会因相邻帧泄漏而严重高估，必须坚持按场景验证。
- [Frontiers in Robotics & AI 2026 的视觉里程计论文](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1819714/full)只用 Moon_1/2/3 做 VO，没有 IoU。定性观察为 "the Segment Anything Model is robust to lunar lighting conditions; it successfully segments rocks even when they are partially obscured by deep shadows"，并指出 Moon_2 的地标稀疏可能来自地形自遮挡、深阴影或岩石稀少。
- 使用 LuSNAR 的第三方工作（包括唯一一份按场景留出的 2D 分割结果），以及方法论文的本地存档、逐项核对和可比性结论，见 [references.md](references.md)。

## 训练与推理配方：终稿没有给出的内容

以下逐项查阅了终稿 §III-C 与 §IV-A 全文，并对照了 v3。这里只登记现有配置的来源，不用另一组参数去填补证据空缺；取值理由见 [training.md](training.md)。

| 项目 | TGRS 终稿能否确定 | 当前值或做法 | 依据 |
|---|---|---|---|
| 实验框架 | 只写明 2D 实验基于 MMSegmentation，没有给出版本或配置文件 | 本仓库 `moonseg` 训练代码，模型组件取自 DINOv3 官方实现（[model.md](model.md)），不使用 MMSegmentation | 由所选模型的官方实现决定 |
| SegFormer、SAM2/SAM3 的型号与初始化 | 不能确定 SegFormer 是 B0–B5 中的哪一个，也不知道预训练来源和微调设置；SAM2/SAM3 的型号、是否微调、如何输出五类都没有说明 | DINOv3 ViT-B/16 与 ViT-L/16，LVD-1689M | 用户选定的路线和可核验的权重来源 |
| 训练 crop、resize 及长宽比 | 未报告；只说明嫦娥实拍图像缩放到 1024×1024 | 1024×1024 整图，不做随机缩放或裁剪 | 用户确认的基线，不能反推作者也用了同样的预处理 |
| 骨干冻结 | 未报告 | 冻结 DINOv3，只训练 Adapter 和头部 | DINOv3 官方适配实现的做法 |
| Batch、梯度累积 | 未报告 | 每卡 8 × 2 卡 × 累积 4 = 64，只在累积组完整时更新 | 用户要求有效 batch 不低于 64 |
| 训练 epoch 或 iteration | 未报告 | 80 epoch（开发池 8000 次更新） | 本项目初始预算 |
| 优化器与初始 lr | 未报告 | AdamW，2e-4 | Mask2Former 原配方按 batch 平方根缩放 |
| Weight decay 与梯度裁剪 | 未报告 | 0.05，全局 L2 0.1 | 本项目选择 |
| 学习率调度和 warmup | 未报告 | warmup 250 次更新（从 0.001× 起），poly 0.9 | 本项目选择 |
| 损失及权重 | 未报告可复现的配方 | Mask2Former 匹配，CE/BCE/Dice = 2/5/5，no-object 0.1，辅助层监督 | 由选定模型及其原始实现决定 |
| 标签采样点数 | 未报告 | 12,544 点，含不确定性采样 | Mask2Former 实现 |
| 类别比例约束 crop | 未报告 | 整图模式不使用，只存在于可选的 `random_resize_crop` 模式中 | 通用分割 pipeline，属于单独的视野变化实验 |
| 水平翻转 | 未报告 | 概率 0.5 | 通用分割做法，不能因为"常用"就声称论文也用了 |
| 颜色、亮度、gamma 等增强 | 未报告；§III-C 只说明渲染采用 "auto-exposure with a simplified photometric model" | 亮度 ±32、对比度 0.5–1.5，各自独立 p=0.5，像素下限 22；gamma、阴影、尺度抖动关闭 | 算子和强度取自 DINOv3/MMSeg 分割实现，不是 LuSNAR 的设定 |
| 归一化 | 未报告 | ImageNet mean/std | DINOv3 的输入约定 |
| 混合精度及 GPU 环境 | 未报告 | BF16 + TF32；ViT-L 在两张 RTX 6000 Ada 上，ViT-B 在两张 RTX A6000 上 | 本项目的硬件与数值选择 |
| 推理方式与 TTA | 未报告 | 整图、单尺度、无翻转 | 与整图基线一致 |
| 最优模型选择、seed、多次运行 | 未报告 | 按场景 5 的验证 mIoU 选 `best.pt`，seed 42，单次运行 | 场景 5 来自比赛允许的开发划分 |

## 当前可确定的边界

- 数据模态、五类语义、原始尺寸、场景组织和配对关系都有终稿依据，可以据此核查实现。
- 比赛的训练/测试划分和评分/提交口径优先于论文的研究实验划分，不能为了"对齐论文"而使用场景 2/4/9 的标签。
- DINOv3 路线、1024 整图和不低于 64 的有效 batch 由用户确认；不会因为论文比较了 SegFormer 或 SAM3 就换模型。
- 当前的增强与优化参数是有出处、记录了取舍的工程选择。论文没有报告某一项，既不能推断作者用了默认值，也不能推断作者没有用。
- 嫦娥实拍实验用到了比赛范围以外的真实图像，不能据此在比赛训练中引入外部数据；这个实验也没有证明只用 LuSNAR 训练的模型可以零样本用于实拍影像。
- 运行检查和显存/吞吐数据不等于论文精度的复现证据；正式比赛训练仍需要组织方的清单。
