# LuSNAR 论文与当前实现的对照

核对日期：2026-10-21。本文记录 LuSNAR 论文可核实的事实、公开渠道能找到的第三方结果，以及论文、比赛规则与本项目工程选择之间的边界。1024×1024 整图、补充 mAcc、按现有分割实现选择增强、梯度累积使有效 batch 不低于 64，都是用户确认的本项目决定，不写成论文作者的做法。

## 论文版本与可用证据

优先依据为 TGRS 期刊终稿：**LuSNAR: A Lunar Segmentation, Navigation, and Reconstruction Dataset Based on Multisensor for Autonomous Exploration**，IEEE TGRS vol. 64，article 4706723，2026，DOI [10.1109/TGRS.2026.3720373](https://doi.org/10.1109/TGRS.2026.3720373)，[出版社记录](https://ieeexplore.ieee.org/document/11643283/)，[Crossref 元数据](https://api.crossref.org/works/10.1109/TGRS.2026.3720373)，[OpenAlex 记录](https://api.openalex.org/works/https://doi.org/10.1109/TGRS.2026.3720373)。

终稿正文没有取得：出版社页面返回空的 HTTP 202，正文、REST、PDF、stamp 及 Crossref 提供的 PDF 地址返回 HTTP 418；Semantic Scholar 标注 openAccessPdf 为 CLOSED，OpenAlex 也没有公开全文 URL；[作者项目页](https://luoxubo.github.io/LuSNAR-dataset/)的 Paper 链接仍指向 arXiv。这些响应和元数据不是论文内容。

终稿与 arXiv 版确有不同。[Semantic Scholar 的终稿摘要](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TGRS.2026.3720373)写有 "we evaluated 25 state-of-the-art algorithms across semantic segmentation, autonomous navigation, and 3-D reconstruction" 和 "experiments based on real imagery from the Chang'e missions demonstrate the sim-to-real transfer potential of LuSNAR for semantic perception tasks"，v3 摘要没有这两句。OpenAlex 解析出 39 条参考文献，比 v3 新增 FoundationStereo（2025）、Light4Mars（2024）、AI4Mars（2021）、Chang'e-4 概述和 3-D Semantic Terrain Reconstruction of Martian close-up images（2024），没有 Mask2Former、SAM 或 DINO 系列；据此推测终稿的 2D 分割基线可能仍是 CNN/SegFormer 一类，但 OpenAlex 的引用列表常不完整，不能作定论。

arXiv 共有 v1（2024-07-09）、v2（2024-09-05）、v3（2024-09-26），没有 v4（[版本页](https://arxiv.org/abs/2407.06512)）。用户已明确允许取不到终稿时使用 arXiv，因此以下内容以 **arXiv:2407.06512v3** 为依据（[全文](https://arxiv.org/html/2407.06512v3)）。所有"论文未说明"仅指这个已读版本，**不能据此断言 TGRS 终稿也未说明**；取得终稿后需重新核对下表及表号。

## 已明确的实验内容与当前实现

| 核对项 | arXiv v3 的明确内容与位置 | 当前实现 | 对齐结论 |
|---|---|---|---|
| 原始数据 | III-C、表 IV：9 个场景，13,006 组序列；RGB 与 2D 标签为 1024×1024 | 原始池整理出 13,006 张 RGB，按场景保留 | 数据规格相符；这不是训练输入尺寸的证据 |
| 场景设计 | §III-B、立体匹配结果段、Table III：起伏 × 地物丰富度的 3×3 网格，各场景撞击坑尺寸 | 网格、帧数与撞击坑见 [data.md](data.md#场景总览)，用于选择验证场景 | 场景属性，不是训练配方 |
| 标注视角 | III-C 数据列表：2D 标签对应左侧图像 | 使用原始包带标注的 `image0/color` 与 `image0/label` | 按时间戳配对；比赛进一步限定单张左 RGB |
| 2D 语义类别 | IV-A、表 V：月壤、岩石、撞击坑、山体、天空 | 五类均保留，比赛 ID 0–4 | 类别语义相符；颜色与 ID 另据官方数据说明和比赛规则 |
| 主实验划分 | IV-A 首段：训练 1/2/4/6/8/9，测试 3/5/7 | 比赛训练池 1/3/5/6/7/8，测试 2/4/9；开发期留场景 5 验证 | 必须按比赛划分，不能照抄论文划分 |
| 泛化实验划分 | IV-A 表 VII 前：训练 1/2/3，测试 4–9 | 不使用 | 另一项论文实验，不是本比赛或开发验证划分 |
| 比较模型 | IV-A、表 V：DeepLabV3+、MobileNetV3、U-Net、PointRend、SegFormer | DINOv3 + ViT-Adapter + Mask2Former | 用户确定的新方法路线，不是复现论文中的模型 |
| 报告指标 | IV-A：每类 IoU、mIoU、mAcc | 每类 IoU、mIoU、mAcc、每类 accuracy/recall/support、pixel accuracy 及混淆矩阵（[metrics.md](metrics.md)） | mAcc 采用公开分割实现的宏平均召回率；缺类处理按比赛口径，不冒称还原了作者未公开的统计细节 |
| 原论文基线分数 | 表 V、表 VII（见下） | 开发池基线 2026-10-21 启动，结果待填（[roadmap.md](roadmap.md#实验记录)） | 模型与划分不同，论文分数不能作为比赛复现目标 |
| 实验观察 | IV-A 讨论阴影、地形起伏、地物密度与分割表现的关系 | 水平翻转与标准强度亮度/对比度；gamma、阴影关闭 | 论文误差分析不能当作增强配方或增益证据（[augmentation_research.md](augmentation_research.md)） |

论文未明确交代逐图/全测试集/逐场景的指标聚合、无有效类别的处理以及 mAcc 的实现细节。当前按场景累计像素计数、只平均参考标签中存在的类别，是**比赛官网**给出的口径，不应归为论文实现。

### 表 V：主实验（训练 1/2/4/6/8/9，测试 3/5/7）

| 模型 | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 | mAcc | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLabv3+ | 98.01 | 75.64 | 79.62 | 83.11 | 99.68 | 92.90 | 87.39 |
| MobileNetV3 | 98.72 | 75.16 | 76.96 | 82.98 | 99.57 | 91.99 | 86.68 |
| U-Net | 98.70 | 73.50 | 73.09 | 79.46 | 99.34 | 91.01 | 84.82 |
| PointRend | 99.07 | 76.17 | **82.45** | 84.00 | 99.70 | 93.29 | 88.28 |
| SegFormer | 99.05 | **81.69** | 80.37 | **89.16** | 99.54 | **94.22** | **89.96** |

### 表 VII：泛化实验（SegFormer，只用场景 1–3 训练，逐场景测试）

| 测试场景 | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 | mAcc | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 98.75 | 87.07 | 66.81 | 96.93 | 99.61 | 92.19 | 89.83 |
| 5 | 98.22 | 70.68 | 60.48 | 74.95 | 99.53 | 89.19 | 80.77 |
| 6 | 98.75 | 85.18 | 68.05 | 93.72 | 99.28 | 92.90 | 89.00 |
| 7 | 98.43 | 86.73 | 66.36 | 93.75 | 99.25 | 92.59 | 88.90 |
| 8 | 98.60 | 83.06 | 73.86 | 86.18 | 98.52 | 92.36 | 88.05 |
| **9** | 96.20 | 76.20 | **19.19** | 76.23 | 96.43 | 78.47 | **72.85** |

六个测试场景的平均 mIoU 为 84.90。

### 论文的误差分析

- "Both Deeplabv3+ and Unet have errors in the category of impact craters, which may be due to the presence of shadow areas within the craters."
- 场景 5 精度最低，"may be due to the presence of extensive shadow areas in scene 5"。
- 场景 4–6 的平均 mIoU 为 85.30，场景 7–9 降到 80.87："with the increase of lunar surface objects richness, the segmentation performance decreases"；在 4/5/6 和 7/8/9 内部，mIoU 都随起伏增大而下降。
- 3D 的 Point Transformer 在岩石上比最佳 2D 方法高 16.47、撞击坑上高 6.22，2D 的短板集中在岩石和撞击坑。
- 推断：论文泛化实验的训练集（场景 1–3）只有 2 个撞击坑，比赛训练池仅场景 5/6/7/8 就有 15 个，所以场景 9 撞击坑 IoU 19.19% 是悲观下界。即便如此，撞击坑和山体仍是场景 9 上最不稳定的类别，阴影是第一嫌疑；场景 9 有两个 d/D 只有 0.05 的浅坑，在 w9 权重最高时，场景 9 的撞击坑 IoU 是最大的提分杠杆。

## 公开代码与第三方结果

公开渠道**没有按场景划分训练的 LuSNAR 2D 分割配方或排行榜**；论文表 V 和表 VII 是唯一同源参照，但划分不同，不能直接比较。

- [官方 GitHub](https://github.com/zqyu9/LuSNAR-dataset)：最新提交为 2026-09-15"添加 LuSNAR 发表在 TGRS 的信息"，[根目录](https://api.github.com/repos/zqyu9/LuSNAR-dataset/contents/)只有 `README.md`、`index.html`、`static/`，没有训练或评测代码；README 的"code released"文字不能当作训练入口。
- [README](https://api.github.com/repos/zqyu9/LuSNAR-dataset/readme) 的"Baseline Results"表（FCN 76.3、SegNet 78.9、PSPNet 82.1、DeepLabV3+ 84.6，以及 3D 重建"Ours 0.082 m RMSE"、COLMAP、MVSNet）与论文矛盾：论文表 V 的 DeepLabv3+ 为 87.39，也从未评测 FCN、SegNet、PSPNet。多个聚合站在转载这组数字，**不能当作目标或基线**。README 写的 HF 数据集 id 是 `JeremyLuo/LuSNAR`，HF 搜索实际只返回 [`xuboluo2001/LuSNAR`](https://huggingface.co/api/datasets?search=lusnar)。
- [官方仓库 issue #18](https://github.com/zqyu9/LuSNAR-dataset/issues/18) 提问 SegFormer B0/B2 复现，无作者回复；这是读者提问，不能据此断言具体型号。
- [GitHub 仓库搜索](https://api.github.com/search/repositories?q=LuSNAR)只有 3 个仓库：官方数据集仓库，以及 `ArturPrentki/orb-slam3-lusnar`、`henrylanday/lunar-lidar-slam` 两个 SLAM 项目。[HF 模型搜索](https://huggingface.co/api/models?search=lusnar)只有深度模型 `LuMonDepth/DAv2-FT-LuSNAR`。[Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2407.06512/citations) 的 12 篇引用中没有一篇给出场景划分下的 2D 分割配方或成绩；Papers with Code 已停止服务，未找到排行榜。
- [Stanford CS231N 2025 报告](https://cs231n.stanford.edu/2025/papers/CS231N_Final_Project_Report%20-%20Brian%20Y%20Wu.pdf)（Brian Y. Wu）：随机 80/20 帧划分（约 10k 训练 / 2k 验证，**不按场景**），增强为水平翻转、±10° 旋转、亮度/对比度，5 epoch、Adam、SegFormer lr 5e-5、MaskFormer 1e-5、batch 4、单卡 3090；SegFormer 0.9686、MaskFormer 0.8955、UNet++ 0.7681、SAM-based 0.7766。比论文跨场景结果高约 10–25 个点，说明随机帧划分因近邻帧泄漏而严重高估，必须坚持按场景验证。
- [Frontiers in Robotics & AI 2026 视觉里程计论文](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1819714/full)只用 Moon_1/2/3 做 VO，没有 IoU；定性观察为 "the Segment Anything Model is robust to lunar lighting conditions; it successfully segments rocks even when they are partially obscured by deep shadows"，并指出 Moon_2 的稀疏地标可能来自地形自遮挡、深阴影或稀疏岩石。

## 训练与推理配方：公开版本没有给出的内容

以下逐项查阅 III-C 和完整 IV-A，并检索全文核对。这里只登记现有配置的来源，不用另一组参数填补证据空缺；取值理由见 [training.md](training.md)。

| 项目 | arXiv v3 能否确定 | 当前值或做法 | 依据 |
|---|---|---|---|
| SegFormer 具体型号、初始化 | 不能确定 B0–B5、预训练来源与微调设置 | DINOv3 ViT-B/16 与 ViT-L/16，LVD-1689M | 用户选定的路线和可核验权重来源 |
| 训练 crop、resize 及长宽比 | 未报告 | 1024×1024 整图，不随机缩放或裁剪 | 用户确认的基线，不能反推作者也用了同一预处理 |
| 骨干冻结 | 未报告 | 冻结 DINOv3，训练 Adapter 和头部 | DINOv3 官方适配实现的做法 |
| Batch、梯度累积 | 未报告 | 每卡 8 × 2 卡 × 累积 4 = 64，只做完整累积组更新 | 用户要求有效 batch 不低于 64 |
| 训练 epoch 或 iteration | 未报告 | 80 epoch（开发池 8000 次更新） | 本项目初始预算 |
| 优化器与初始 lr | 未报告 | AdamW，2e-4 | Mask2Former 原配方按 batch 平方根缩放 |
| Weight decay 与梯度裁剪 | 未报告 | 0.05，全局 L2 0.1 | 本项目选择 |
| 学习率调度和 warmup | 未报告 | warmup 250 次更新（起点 0.001×），poly 0.9 | 本项目选择 |
| 损失及权重 | 未报告可复现配方 | Mask2Former 匹配，CE/BCE/Dice = 2/5/5，no-object 0.1，辅助层监督 | 由选定模型及其原始实现决定 |
| 标签采样点数 | 未报告 | 12,544 点，含不确定性采样 | Mask2Former 实现 |
| 类别比例约束 crop | 未报告 | 整图模式不使用；只在可选 `random_resize_crop` 模式中存在 | 通用分割 pipeline，属独立视野变化实验 |
| 水平翻转 | 未报告 | 概率 0.5 | 通用分割做法，不能据"常用"声称论文采用 |
| 颜色、亮度、gamma 等增强 | 未报告 | 亮度 ±32、对比度 0.5–1.5，各自独立 p=0.5，像素下限 22；gamma、阴影、尺度抖动关闭 | 算子与强度取自 DINOv3/MMSeg 分割实现，不是 LuSNAR 设定 |
| 归一化 | 未报告 | ImageNet mean/std | DINOv3 输入约定 |
| 混合精度及 GPU 环境 | 未报告 | BF16 + TF32；ViT-L 在两张 RTX 6000 Ada，ViT-B 在两张 RTX A6000 | 本项目硬件与数值选择 |
| 推理方式与 TTA | 未报告 | 整图、单尺度、无翻转 | 与整图基线一致 |
| 最优模型选择、seed、多次运行 | 未报告 | 场景 5 上原始权重与 EMA 中较高的 mIoU 选 `best.pt`，seed 42，单次运行 | 场景 5 来自比赛允许的开发划分 |

## 当前可确定的边界

- 数据模态、五类语义、原始尺寸、场景组织和配对关系有论文依据，可以据此核查实现。
- 比赛训练/测试划分及评分/提交口径优先于论文的研究实验划分，不能把场景 2/4/9 的标签用于"对齐论文"。
- DINOv3 路线、1024 整图和有效 batch 不低于 64 由用户确认；不因为论文比较了 SegFormer 就换模型。
- 当前增强与优化参数是有出处、记录了取舍的工程选择。论文没有报告某项，既不能推断作者用了默认值，也不能推断作者禁用了它。
- 运行检查和显存/吞吐数据不等价于论文精度复现证据；正式比赛训练仍需组织方清单。
