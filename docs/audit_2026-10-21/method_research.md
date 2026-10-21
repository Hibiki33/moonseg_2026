# 月面语义分割竞赛：文献与公开代码调研报告

- 核对日期：2026-10-21。项目：DINOv3 ViT-B/16（LVD-1689M，冻结）+ 官方 DINOv3 ViT-Adapter + Mask2Former，1024×1024 整图训练。
- 标注规则：**[verified: URL]** 表示本次实际读取了该页面、源码或 API 响应。**[inference]** 表示本报告的推断或经验估计。**[weak]** 表示只看到了搜索摘要，原页面被拦截。
- 范围：本报告不重复 `docs/augmentation_research.md`、`docs/paper_alignment.md`、`docs/model.md` 已核对的内容，包括各仓库的增强源码、arXiv v3 未报告的训练超参、Table V 总分、HF zip 结构和场景计数。只在需要时引用这些结论。
- 合规：本次没有访问任何测试场景标签，也没有修改仓库文件。临时文件只写在 `$TMPDIR/moonseg_research/`。

---

## 0. 关键结论速览

1. **LuSNAR 场景网格按行主序排列，已核实** [verified: https://arxiv.org/html/2407.06512v3]：
   - 起伏：1/4/7 低，2/5/8 中，3/6/9 高。
   - 丰富度：1/2/3 稀疏，4/5/6 中等，7/8/9 丰富。
   - 因此测试场景 9（高起伏 × 高丰富）位于网格的**外推角**。训练集中离它最近的是场景 6（高起伏、中等丰富）和场景 8（中起伏、丰富）。
   - 官方建议的验证场景 5 是网格**正中心**，论文还指出它的阴影面积大。
2. **在论文的跨场景实验中，场景 9 的撞击坑 IoU 只有 19.19%** [verified: 同上 Table VII]。
   - 该实验只用场景 1–3 训练，SegFormer 在场景 9 上 mIoU 72.85，6 个场景平均 84.90。
   - 场景 9 有两个撞击坑的 d/D 只有 0.05（很浅）[verified: Table III]。
   - 在 w9 权重最高的情况下，**场景 9 的 crater IoU 是最大的提分杠杆** [inference]。
3. **TGRS 终稿与 arXiv v3 不同** [verified: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TGRS.2026.3720373]。终稿摘要写的是 "evaluated 25 state-of-the-art algorithms"，并新增了 "experiments based on real imagery from the Chang'e missions"。终稿正文仍无法获取，IEEE Xplore 返回 HTTP 418/202。
4. **官方 GitHub README 的 "Baseline Results" 表不可信**：FCN 76.3、SegNet 78.9、PSPNet 82.1、DeepLabV3+ 84.6。这些数字与论文 Table V 冲突（论文中 DeepLabv3+ 为 87.39），论文也从未评测 FCN、SegNet、PSPNet [verified: https://github.com/zqyu9/LuSNAR-dataset ；论文 v3]。多个聚合站和搜索引擎在转载这组数字，**不要把它当目标或基线**。
5. **公开渠道没有按场景划分训练的 LuSNAR 2D 分割方案或排行榜**。唯一找到的第三方结果是一份 CS231N 报告，它用随机 80/20 帧划分得到 SegFormer mIoU 0.9686，存在严重的近邻帧泄漏 [verified]。
6. **DINOv3 官方 Mask2Former 配方已核实**（7B 冻结，Appendix D.10）[verified: https://arxiv.org/html/2508.10104v1]：
   - batch 16、AdamW、cosine、1.5k warmup。ADE20K 阶段 LR 3e-5、20k iter。
   - 之前先在 COCO-Stuff 上训 80k iter（LR 1.5e-5），再在 Hypersim 上训 10k iter（LR 2.5e-5），分辨率 896。
   - 推理用 896 滑窗（stride 596），TTA 比例 [0.9,0.95,1.0,1.05,1.1] × hflip。结果 SS 62.6 / TTA 63.0。
   - 冻结时去掉了 ViT-Adapter 的 injector，并对中间层特征加 LayerNorm + 可学习 BN。
7. **DINOv3 linear probe（ADE20K @512）** [verified: 同上 Table 14/3]：ViT-B 51.8 → ViT-L 54.9（+3.1），ViT-H+ 54.8，7B 55.9。**ViT-L 是性价比拐点**。
8. **冻结还是微调**：没有 DINOv3 的分割微调对照数据。DINOv2-L + M2F 在 GTAV→Cityscapes 上的结果为 Full FT 63.7、Freeze 61.1、Rein(PEFT) 66.4 [verified: https://arxiv.org/html/2312.04265]。说明在域偏移下"低 LR 的少量可训练参数"优于两个极端；全量微调也略好于冻结，但收益小。
9. **大上下文对跨域很关键** [verified: https://arxiv.org/html/2204.13132]。HRDA 显示 crop 缩小到 1/4 时，UDA 性能下降 39%，监督学习下降 14%。这支持保留 **1024 整图训练**，同时只加温和的尺度抖动。
10. **目前的优化步数可能偏少** [inference]。80 epoch × 6403 张（开发集 1/3/6/7/8）÷ 70 ≈ **7.3k 次参数更新**。作为对比：DINOv3 的 ADE20K 微调阶段 20k、Rein 40k、M2F Cityscapes 90k、ADE20K 160k。另外 M2F 官方 grad clip 为 full-model 0.01，项目为 0.1。
11. **合规**：在测试 RGB 上做 pseudo-label、自训练、TENT 或 AdaBN，等同于"用训练清单外数据参与训练"，风险高，不要做。纯推理期 TTA（flip / 多尺度，无参数更新）风险低。

---

## 1. LuSNAR 论文细节

### 1.1 版本与终稿

- **arXiv 版本**：v1 2024-07-09、v2 2024-09-05、v3 2024-09-26，**没有 v4** [verified: https://arxiv.org/abs/2407.06512]。
- **TGRS 终稿**：IEEE TGRS vol. 64，article 4706723，2026 年 [verified: https://api.openalex.org/works/https://doi.org/10.1109/TGRS.2026.3720373]。终稿标题为 "...Navigation, and Reconstruction Dataset Based on Multisensor..."。
- **终稿摘要的新增内容** [verified: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TGRS.2026.3720373]：
  - "we evaluated **25** state-of-the-art algorithms across semantic segmentation, autonomous navigation, and 3-D reconstruction"
  - "experiments based on **real imagery from the Chang'e missions** demonstrate the sim-to-real transfer potential of LuSNAR for semantic perception tasks"
  - v3 的摘要中没有这两句。
- **终稿参考文献**：OpenAlex 解析出 39 条。与 v3 相比新增了 FoundationStereo (2025)、Light4Mars (2024)、AI4Mars (2021)、Chang'e-4 overview、3-D Semantic Terrain Reconstruction of Martian close-up images (2024)。解析出的条目中**没有** Mask2Former、SAM、DINO 系列 [verified: OpenAlex referenced_works]。
  - [inference] 终稿的 2D 分割基线很可能仍是 CNN 或 SegFormer 一类，也许加入了 Light4Mars，并新增了嫦娥真实图像的 sim-to-real 定性或定量实验。OpenAlex 的引用列表常不完整，这一点不能作为定论。
- **出版社页面无法访问**：IEEE Xplore 正文、REST、PDF 返回 HTTP 418 或空的 202，Semantic Scholar 标注 openAccessPdf 为 CLOSED [verified: https://ieeexplore.ieee.org/document/11643283]。作者项目页的 Paper 链接仍指向 arXiv（repo 文档已记录）。
- **GitHub README 的状态**：最新 commit 是 2026-09-15 "添加 LuSNAR 发表在 TGRS 的信息"。仓库根目录只有 `README.md`、`index.html`、`static/`，**没有任何训练或评测代码** [verified: https://api.github.com/repos/zqyu9/LuSNAR-dataset/contents/]。
- **README 的 "Baseline Results" 表与论文矛盾** [verified: https://api.github.com/repos/zqyu9/LuSNAR-dataset/readme]：
  - 表中列出 FCN 76.3 / SegNet 78.9 / PSPNet 82.1 / DeepLabV3+ 84.6，以及 3D 重建 "Ours 0.082 m RMSE"、COLMAP、MVSNet。
  - 这些模型和数字都不在论文 v3 中（论文 Table V 的 DeepLabv3+ 为 87.39）。
  - [inference] 该表是 README 重写时加入的占位内容，不可作为参考。注意 HF 数据集 id：README 写的是 `JeremyLuo/LuSNAR`，HF 搜索实际只返回 `xuboluo2001/LuSNAR` [verified: https://huggingface.co/api/datasets?search=lusnar]。

### 1.2 场景设计网格

论文 §III-B 有一句："Scenes 1, 4, and 6 exhibit relatively flat terrain, while scenes 2, 5, and 7 feature minor undulations, and scenes 3, 6, and 9 display significant terrain variations"。场景 6 在这句里出现了两次，属于**笔误**。立体匹配实验段落的表述无歧义，可以据此确定网格 [verified: https://arxiv.org/html/2407.06512v3 立体匹配结果段]：

> "scenes 2, 5 and 8 with **moderate** topographic relief … scenes 1, 4 and 7 with **low** topographic relief … scenes 3, 6 and 9 with **high** topographic relief … scenes 4, 5 and 6 with **medium** ground feature richness … scenes 1, 2 and 3 with **low** … scenes 7, 8 and 9 with **high** ground feature richness"

比赛对测试场景的描述与这个行主序网格一致（2 = 中起伏 / 较稀疏，4 = 较平缓 / 中等，9 = 显著 / 较丰富）。

| 场景 | 起伏 | 丰富度 | 比赛划分 | Table III 列出的撞击坑（直径 m / 深度 m / d/D） | 原始左图数（repo 已统计） |
|---|---|---|---|---|---|
| 1 | 低 | 稀疏 | train | **无（Table III 不含场景 1）** | 1094 |
| 2 | 中 | 稀疏 | **test** | 1 个：69.50/5.37/0.07 | 1597 |
| 3 | 高 | 稀疏 | train | 1 个：73.31/6.07/0.08 | 1230 |
| 4 | 低 | 中等 | **test** | 3 个：49.10/6.78/0.13；25.61/3.69/0.14；62.00/6.85/0.11 | 1320 |
| 5 | 中 | 中等 | train（建议本地验证） | 3 个：63.63/11.48/0.18；35.05/3.73/0.10；65.25/7.13/0.10 | 1599 |
| 6 | 高 | 中等 | train | 3 个：27.47/5.30/0.19；24.22/2.41/0.09；67.31/11.56/0.17 | 1102 |
| 7 | 低 | 丰富 | train | 5 个：24.64…33.72（d/D 0.09–0.16） | 1626 |
| 8 | 中 | 丰富 | train | 4 个：52.36/8.04/0.15；27.91/2.82/0.10；**100.68**/6.56/0.06；59.35/8.00/0.13 | 1351 |
| 9 | 高 | 丰富 | **test（最高权重）** | 4 个：22.88/3.57/0.15；59.64/5.11/0.08；22.11/1.16/**0.05**；32.19/1.73/**0.05** | 2087 |

- 网格、Table III 和撞击坑数据 [verified: https://arxiv.org/html/2407.06512v3 §III-B、Table III]。
- 论文给出的其他事实 [verified: 同上]：
  - 总数 13,006 组，每个场景约 1000–2000 组，覆盖约 300 m × 300 m，行驶 200–600 m。
  - 论文**没有逐场景计数表**。repo 从 HF zip 统计的 9 个场景计数之和恰为 13,006，与论文吻合。
  - 加入了 0.1–0.3 m 的小石块以丰富场景。
  - 每个场景只有 1–2 座山。
  - "the terrain complexity gradually increases from scenes 1 to 9"。
  - Fig. 2 给出了岩石占比和撞击坑数量，但只是图像，无法提取数字。
- [inference] 对本项目的含义：
  - 测试场景 2 位于稀疏行的 1 与 3 之间，场景 4 位于低起伏列的 1 与 7 之间，都属于**内插**。
  - 场景 9 是**外推角**。最接近它的训练场景是 6（同为高起伏）和 8（同为丰富，且有 d/D 0.06 的大浅坑）。
  - 只用场景 5（中心点）做模型选择，可能偏向"平均场景"的最优，而不是场景 9 的最优。
  - 场景 1 的 GT 很可能没有 crater，所以它对 crater 的学习没有贡献，只贡献 regolith、rock、sky。

### 1.3 2D 分割基线表与误差分析

- **Table V**：训练场景 1/2/4/6/8/9，测试场景 3/5/7。各类 IoU 的顺序为 regolith / rock / crater / mountain / sky / mAcc / mIoU [verified: https://arxiv.org/html/2407.06512v3]：

  | 模型 | regolith | rock | crater | mountain | sky | mAcc | mIoU |
  |---|---|---|---|---|---|---|---|
  | DeepLabv3+ | 98.01 | 75.64 | 79.62 | 83.11 | 99.68 | 92.90 | 87.39 |
  | MobileNetV3 | 98.72 | 75.16 | 76.96 | 82.98 | 99.57 | 91.99 | 86.68 |
  | U-Net | 98.70 | 73.50 | 73.09 | 79.46 | 99.34 | 91.01 | 84.82 |
  | PointRend | 99.07 | 76.17 | **82.45** | 84.00 | 99.70 | 93.29 | 88.28 |
  | SegFormer | 99.05 | **81.69** | 80.37 | **89.16** | 99.54 | **94.22** | **89.96** |

- **训练设置**：SegFormer 型号、输入尺寸、epoch、增强、类别权重都没有报告。这一点 repo 文档已核实，此处不重复。
- **Table VII**：SegFormer 只用场景 1–3 训练，逐场景测试 [verified: 同上]：

  | 场景 | regolith | rock | crater | mountain | sky | mAcc | mIoU |
  |---|---|---|---|---|---|---|---|
  | 4 | 98.75 | 87.07 | 66.81 | 96.93 | 99.61 | 92.19 | 89.83 |
  | 5 | 98.22 | 70.68 | 60.48 | 74.95 | 99.53 | 89.19 | 80.77 |
  | 6 | 98.75 | 85.18 | 68.05 | 93.72 | 99.28 | 92.90 | 89.00 |
  | 7 | 98.43 | 86.73 | 66.36 | 93.75 | 99.25 | 92.59 | 88.90 |
  | 8 | 98.60 | 83.06 | 73.86 | 86.18 | 98.52 | 92.36 | 88.05 |
  | **9** | 96.20 | 76.20 | **19.19** | 76.23 | 96.43 | 78.47 | **72.85** |

- **论文的误差分析原话** [verified: 同上]：
  - "Both Deeplabv3+ and Unet have errors in the category of impact craters, which may be due to the presence of **shadow areas within the craters**."
  - 场景 5 精度最低，"may be due to the presence of **extensive shadow areas in scene 5**"。
  - 场景 4–6 的平均 mIoU 为 85.30，场景 7–9 降到 80.87："with the increase of lunar surface objects richness, the segmentation performance decreases"。
  - 在场景 4/5/6 和 7/8/9 内部，mIoU 都随起伏增大而下降。
  - 3D 的 Point Transformer 在 rock 上比最佳 2D 方法高 16.47，在 crater 上高 6.22，说明 2D 的短板集中在近景岩石和撞击坑。
- [inference] 本比赛的训练池包含场景 5/6/7/8 共 15 个撞击坑，而论文泛化实验的训练集只有 2 个。因此 19% 是悲观下界。即便如此，**crater 和 mountain 在场景 9 上仍然是最不稳定的类别**，阴影是第一嫌疑因素。

---

## 2. 公开代码、排行榜和使用 LuSNAR 做 2D 分割的工作

| 渠道 | 结果 | 证据 |
|---|---|---|
| GitHub 仓库搜索 "LuSNAR" | 只有 3 个仓库：官方数据集仓库（无代码），以及 `ArturPrentki/orb-slam3-lusnar`、`henrylanday/lunar-lidar-slam` 两个 SLAM 项目 | [verified: https://api.github.com/search/repositories?q=LuSNAR] |
| HF 模型搜索 | 只有 `LuMonDepth/DAv2-FT-LuSNAR`（深度模型），没有分割模型 | [verified: https://huggingface.co/api/models?search=lusnar] |
| Semantic Scholar 引用 | 12 篇引用，没有一篇给出场景划分下的 2D 分割配方或成绩 | [verified: https://api.semanticscholar.org/graph/v1/paper/arXiv:2407.06512/citations] |
| Papers with Code | 未找到 LuSNAR 排行榜（PwC 本身已停止服务） | [inference] |

Semantic Scholar 引用中相关的两项：

- **Stanford CS231N 2025 报告**（Brian Y. Wu）[verified: https://cs231n.stanford.edu/2025/papers/CS231N_Final_Project_Report%20-%20Brian%20Y%20Wu.pdf]
  - 划分方式：随机 80/20 帧划分，约 10k 训练 / 2k 验证，**不按场景**。
  - 增强：hflip、±10° 旋转、亮度/对比度。
  - 训练：5 epoch，Adam，SegFormer LR 5e-5，MaskFormer 1e-5，batch 4，单卡 3090。
  - 结果：SegFormer **0.9686**，MaskFormer 0.8955，UNet++ 0.7681，SAM-based unified 0.7766。
  - [inference] 这个数字比论文的跨场景结果高约 10–25 个点，再次说明**随机帧划分严重高估**，必须坚持按场景验证。
- **Frontiers in Robotics & AI 2026 视觉里程计论文** [verified: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1819714/full]
  - 只用 Moon_1/2/3 做 VO，没有 IoU。
  - 定性观察："the Segment Anything Model is robust to lunar lighting conditions; it successfully segments rocks even when they are partially obscured by deep shadows"。
  - 指出 Moon_2 "sparse landmark distribution most probably caused by severe terrain self-occlusions, deep shadows, or sparse rock distributions"。

**结论**：没有可复用的竞争性公开配方。论文的 Table V 和 VII 是唯一的同源参照，但划分不同，不能直接比较。

---

## 3. 行星表面分割文献中可迁移的经验

### 3.1 火星

- **AI4Mars**（MSL，类别为 soil/bedrock/sand/big rock）。MarsSeg 论文 Table III 给出的逐类 IoU [verified: https://arxiv.org/html/2404.04155]：
  - DeepLabv3+：92.88 / 93.51 / 85.26 / **46.28**，mIoU 79.48。
  - MarsSeg：93.08 / 93.81 / 86.17 / **50.50**，mIoU 80.89。
  - big rock 只占训练样本约 2%。
  - 在 MSL-Seg 上，rocks 类从 34.21 提到 52.81。
  - 训练配置：SGD LR 1e-3，batch 8。
  - **经验**：在所有行星数据集上，小而稀的岩石类都是瓶颈，架构带来的提升也集中在这一类。
- **Light4Mars / SynMars**（ISPRS J. 2024）被 TGRS 终稿新引用。它用合成火星图像训练，再在真实的 MarsScapes 上评测 [weak: https://www.sciencedirect.com/science/article/abs/pii/S0924271624002466，只看到了摘要级信息]。

### 3.2 月球

- **Synthetic Moon (SMo) / Kaggle Artificial Lunar Landscape**：9,766 张 480×720 渲染图，另有 36 张真实月面照片 (RMo)，类别为 sky / small rock / large rock [verified: https://arxiv.org/html/2504.15991 Supplementary]。
  - 该论文比较了在 U-Net 类 CNN 上的 adapter、全量微调和部分微调。
  - 真实月面上的最佳做法是"从合成数据模型出发做全量微调"；adapter 约低 3%，但 p=0.02 显著。
  - 增强网格搜索中，**random crop (p=0.5) 最有效** [verified: 同上]。
  - [inference] 这些结果是小 CNN 和真实小数据下得到的，迁移到 ViT-Adapter 的价值有限。
- **LuSeg**（RGB-D 月面障碍分割）：repo 已核实其增强代码实际没有被调用，此处不重复。

### 3.3 阴影、光照与小目标的通用证据

- **ShadowCast**（arXiv 2610.23440，遥感）[verified: https://arxiv.org/html/2610.23440]
  - 方法：从**分割标签**中提取凸起物体的掩码，根据周边亮度估计投影方向，再沿该方向投射合成阴影。
  - 参数：阴影长度 5–30 px，亮度衰减系数 0.5–0.8，p=0.5。
  - 在 10 种架构上都有提升，例如 DeepLabV3+ 在 LoveDA 上 43.29→44.89，OCRNet 在 Vaihingen 上 +2.49，InternImage 在 Vaihingen 上 +0.40。
  - 优于通用的 RandomShadow（DeepLabV3+ LoveDA 43.67 vs 44.89）。
  - [inference] 月面阴影主要由岩石和坑缘投射。"基于标签的局部变暗"只改动官方训练图像的像素强度，不改变几何和标签，属于颜色扰动的扩展。但它不在规则的明文列表里（详见第 6 节合规部分）。
- **DINOv3 遥感结果** [verified: https://arxiv.org/html/2508.10104v1 §6.x Table 19 及讨论]
  - 冻结的 DINOv3 在 LoveDA（1024×）、iSAID 等高分辨率遥感分割上达到 SOTA。
  - 论文原话："the DINOv3 **web** model achieves state-of-the-art results on semantic geospatial tasks … The DINOv3 satellite model excels in metric tasks like depth estimation"。
  - [inference] LuSNAR 是地面视角的渲染图，**应选 LVD-1689M 版本**。SAT-493M 是 Maxar 0.6 m 正射俯视影像，视角不匹配，不建议采用。
- **没有找到的证据**：
  - 行星场景中 "copy-paste rocks" 的定量证据，没有找到。
  - 关于 gamma 增强在月面数据上的定量证据，没有找到。
  - AI4Mars 原文 PDF（CVF）返回 403，未能直接读取。

---

## 4. DINOv3 下游语义分割实践

### 4.1 官方 Mask2Former 配方（ViT-7B，冻结）

来源 [verified: https://arxiv.org/html/2508.10104v1 §6.3.2、Appendix D.10、Table 11、Table 24]：

- **结构**：
  - ViT-Adapter **去掉 injector**，确保 backbone 激活不被改动。
  - Mask2Former 的 embedding 维度从 1024 增到 2048。
  - 取第 [10,20,30,40] 层特征，对各层施加 final LayerNorm，再加一个可学习 BN。BN 用于抑制 7B 模型在特征维度上的离群值。
  - 可训练 decoder 共 927M 参数。
- **训练**（全部为 batch 16、AdamW、cosine 调度）：

  | 阶段 | 迭代数 | LR | warmup | 分辨率 |
  |---|---|---|---|---|
  | COCO-Stuff | 80k | 1.5e-5 | 6k | 1280 |
  | Hypersim | 10k（约 2 epoch） | 2.5e-5 | 1.5k | — |
  | ADE20K | 20k | 3e-5 | 1.5k | 896 |

  - Cityscapes 阶段 LR 为 1.5e-5，VOC 为 1e-5。
- **推理**：先 resize 到训练分辨率，再用 896² 滑窗，stride 596。TTA 为 [0.9,0.95,1.0,1.05,1.1] × hflip，共 10 次前向后取平均。
- **结果**：
  - ADE20K SS 62.6 / TTA 63.0，与 ONE-PEACE 持平。对比 InternImage-H 62.5/62.9（需要微调），BEiT-3 62.0/62.8（需要微调）。
  - Cityscapes（1280）86.1 / 86.7。
- **官方仓库现状** [verified: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml]：
  - 只发布了 M2F **推理**配置（img_size 896、mode slide、crop 896、stride 596、tta_ratios 如上）和 linear 训练配置，**没有 M2F 训练配置**。
  - 官方 HEAD 就是项目 vendor 的 revision `6876159`（2026-07-15）[verified: https://api.github.com/repos/facebookresearch/dinov3/commits]。
  - 2026-06 新增了 FINO 分支，用 metadata 引导做无标签的 backbone 自适应 [verified: https://arxiv.org/abs/2606.05107 ，摘要级]。[inference] 与本赛无直接关系。
- [inference] **对本项目的启示**：
  - 官方在 batch 16 下用 LR 1.5e-5–3e-5、cosine、1.5k warmup，并用大规模分割数据预训练 decoder。
  - 本项目从随机初始化的 adapter/head 起步，用 LR 1e-4、batch 70、86 步 warmup，两者不可直接类比。
  - 但官方一个阶段 ≥20k 次更新，说明"更新次数"比"epoch 数"更关键。
  - 另外，用 COCO-Stuff 或 Hypersim 预训练 decoder 属于"额外数据训练任务头"。比赛规则写明"任务训练、微调…只能使用组织方指定训练集"，因此**不能**这样做。直接用 Meta 已发布的 M2F head 权重也属于同类风险，而且只有 7B 版本，详见第 7 节。

### 4.2 各尺寸性能

来源 [verified: https://arxiv.org/html/2508.10104v1 Table 3/14/15]：

- **ViT 系列，ADE20K linear @512**：

  | ViT-S | ViT-S+ | ViT-B | ViT-L | ViT-H+ | ViT-7B |
  |---|---|---|---|---|---|
  | 47.0 | 48.8 | 51.8 | **54.9** | 54.8 | 55.9 |

  - 同尺寸 DINOv2：B 48.4，L 48.8。
  - DINOv3 Cityscapes linear（7B）81.1。
- **ConvNeXt-DINOv3 蒸馏版，ADE20K linear @512**：

  | 模型 | T | S | B | L |
  |---|---|---|---|---|
  | ConvNeXt-DINOv3 | 42.7 | 44.8 | 46.3 | 47.8 |
  | ImageNet-22k 监督 ConvNeXt | 24.8 | 22.6 | 26.5 | 33.3 |

- **冻结与微调**：DINOv3 论文**只给冻结结果**，没有 ViT-S/B/L/H+ 接 M2F 的数字。
- [inference] ViT-L 相对 ViT-B 的 linear probe 提升 +3.1，H+ 没有进一步提升，所以**升级到 ViT-L 最划算**。ConvNeXt-DINOv3 更适合作为集成中的异构成员。

### 4.3 冻结还是解冻：域偏移下的证据

- **DINOv2 g + ViT-Adapter + M2F（冻结）**：ADE20K 60.2 [verified: https://arxiv.org/html/2304.07193v2]。
- **Rein（CVPR 2024），GTAV（合成）→ 真实域，DINOv2-L + M2F**
  - Table 2 中 DINOv2-L 的结果：**Full 63.7 / Freeze 61.1 / Rein-LoRA 66.4**。EVA02-L：Full 62.2 / Freeze 56.5 / Rein 65.3。
  - Table 1 中冻结 VFM 在 Cityscapes 上的结果：DINOv2-L 63.3，SAM-H 57.0，EVA02-L 56.5，CLIP-L 53.7，MAE-L 43.3。
  - 训练配置：AdamW，**backbone LR 1e-5 / head 1e-4**，40k iter，**batch 4**，crop 512，只用 M2F 的基础增强。
  - 论文观点："fine-tuning VFMs with their huge number of trainable parameters on these datasets result in limited generalizability"。
  - [verified: https://arxiv.org/html/2312.04265]
- **DINOv3 ConvNeXt 工业检测研究**：在 RGB 数据上，DINOv3 冻结时没有明显优势，**全量微调后初始化更强**；换到 X-ray 模态后 ImageNet 预训练反而更好 [verified: https://arxiv.org/abs/2605.23472 摘要]。
- **DINO Soars（DINOv3 遥感开放词汇分割）**：只解冻视觉编码器的最后 2 个 block，就比完全冻结"in the worst case"高 11.9 mIoU [verified: https://arxiv.org/html/2605.03175v1 §4.4.1]。这是开放词汇设置，迁移性有限。
- **常见 LLRD 配方**：ViT-Adapter BEiTv2-L + M2F 用 AdamW LR 2e-5、**layer_decay 0.90**、wd 0.05、**drop_path 0.3**、1500 步线性 warmup、poly(power 1.0)、80k iter、crop 896、从 COCO-Stuff 的 M2F 初始化，得到 61.2 / 61.5 [verified: https://github.com/czczup/ViT-Adapter/blob/main/segmentation/configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py ；https://github.com/czczup/ViT-Adapter/blob/main/segmentation/README.md]。
- [inference] **结论**：
  - 先把冻结方案调到收敛。然后用"低 LR 的部分解冻"做第二阶段：backbone LR 约为 head 的 0.1×，即 1e-5；LLRD 0.9；或只解冻最后 4–6 个 block。
  - 只有当它在**多个**留出场景上都提分时才保留。
  - 预期收益在 −1 到 +2 mIoU 之间，不确定性大。合成纹理过拟合是主要风险。
  - Rein 类 PEFT 目前没有 DINOv3 版的官方实现，自行实现的成本中等。

### 4.4 SAT-493M、获取渠道与 provenance

- **SAT-493M**：4.93 亿张 512×512 Maxar RGB 正射影像，分辨率 0.6 m [verified: https://arxiv.org/html/2508.10104v1 Geospatial 段]；只发布了 ViT-L 和 7B 两个版本 [verified: 本地 vendor `third_party/dinov3/README.md` 权重列表]。[inference] 与本赛无关。
- **LVD-1689M 的数据声明，可直接写进方法论文** [verified: 同上 §3.1]：
  - 来源是约 170 亿张 Instagram 公开帖子图片（已经过平台内容审核）。
  - 第一部分：用 DINOv2 embedding 做 5 级层次 k-means 平衡采样，得到 1,689M 张。
  - 第二部分：基于检索的种子数据集近邻。
  - 第三部分：ImageNet-1k、ImageNet-22k、Mapillary SLS 原始图像。
- **Hugging Face**：`facebook/dinov3-vitl16-pretrain-lvd1689m` 为 `gated: manual`，license 为 "other"（DINOv3 License）[verified: https://huggingface.co/api/models/facebook/dinov3-vitl16-pretrain-lvd1689m?blobs=true]。
  - model.safetensors 大小 1,212,559,808 bytes，LFS sha256 为 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`。
  - 这份 LFS 元数据**不需要 gated 授权就能读取**。
- **ModelScope 镜像**：存在 `facebook/dinov3-{vitb16,vitl16,vith16plus,vit7b16}-pretrain-lvd1689m`、`facebook/dinov3-convnext-large-pretrain-lvd1689m`、`facebook/dinov3-vitl16-pretrain-sat493m`，上传者均为 "Cherrytest"，license 为 "other" [verified: https://modelscope.cn/api/v1/models/facebook/dinov3-vitl16-pretrain-lvd1689m 等]。
  - [inference] 如果要证明 ModelScope 镜像与 Meta 发布的文件逐字节一致，最省事的办法是把镜像文件的 sha256 与上面的 HF LFS sha256 比对。repo 文档说"按项目要求没有做摘要计算"；是否要补这一步由用户决定，它对 provenance 声明有帮助。
- **DINOv3 License 中与比赛相关的条款** [verified: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md]：
  - 发表成果时 "must acknowledge the use of DINO Materials"。
  - 再分发（包括衍生权重）必须附带同一协议。
  - 必须遵守 Trade Controls 和 ITAR 限制。
  - [inference] 提交复现包时，要附上 DINOv3 License 和来源说明。

---

## 5. Mask2Former 语义分割配方

- **ADE20K 语义分割基础配置** [verified: https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml]：
  - IMS_PER_BATCH 16，AdamW，BASE_LR 1e-4，WEIGHT_DECAY 0.05，**BACKBONE_MULTIPLIER 0.1**。
  - WarmupPolyLR，**WARMUP_ITERS 0**，MAX_ITER 160k。
  - **CLIP_GRADIENTS full_model，CLIP_VALUE 0.01，L2**；开启 AMP。
  - crop 512²，短边在 {0.5…2.0}×512 中选择，长边 ≤2048。
  - COLOR_AUG_SSD，SINGLE_CATEGORY_MAX_AREA 1.0。
  - TTA 尺度 [256…896]（0.5–1.75×）+ flip，默认关闭。
- **Swin-L**：crop 640²，短边 320–1280，drop_path 0.3 [verified: https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml]。
- **Cityscapes 基础配置**：
  - 90k iter，batch 16，LR 1e-4，相同的 0.01 全模型裁剪，SyncBN。
  - **crop 512×1024**，短边 {0.5…2.0}×1024，长边 ≤4096。
  - TTA 尺度 [512…1792] + flip。
  - [verified: https://github.com/facebookresearch/Mask2Former/blob/main/configs/cityscapes/semantic-segmentation/Base-Cityscapes-SemanticSegmentation.yaml]
- **Model zoo 结果（SS / MS+flip）**，可据此估计 TTA 收益 [verified: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md]：

  | 数据集 | 模型 | SS | MS+flip | 增益 |
  |---|---|---|---|---|
  | ADE20K | R50 | 47.2 | 49.2 | +2.0 |
  | ADE20K | Swin-L | 56.1 | 57.3 | +1.2 |
  | Cityscapes | R50 | 79.4 | 82.2 | +2.8 |
  | Cityscapes | Swin-B | 83.3 | 84.5 | +1.2 |
  | Cityscapes | Swin-L | 83.3 | 84.3 | +1.0 |

- **高分辨率训练的考虑** [inference]：
  - Cityscapes 用 1/4 图的 crop 训练 90k iter，总样本约 144 万 crop，相当于 2975 张图的约 484 个"crop-epoch"。
  - ADE20K 160k×16 约 256 万样本，相当于约 128 epoch。
  - 本项目用整图、batch 70，80 epoch 约 51 万样本、**约 7.3k 次更新**。样本量不算少，但更新次数比官方少一个数量级。
  - 当前配置与 M2F 官方的差异：grad clip 为 0.1（官方 full-model 0.01），有 86 步 warmup（官方 0），batch 70（官方 16）。
  - 用户之前确定了有效 batch 必须 >64，这里不推翻。但如果保持 batch 70，可以考虑**提高 LR**（例如按 sqrt 缩放到约 2e-4），或**延长 epoch**，使更新次数达到 ≥15–20k，并以留出场景曲线为准。

---

## 6. 竞赛常用技巧（2022–2026）与证据

| 技巧 | 证据 | 对本赛的估计 |
|---|---|---|
| 多尺度 + flip TTA | M2F：ADE +1.2–2.0，Cityscapes +1.0–2.8 [verified: MODEL_ZOO]；DINOv3 7B 只用 0.9–1.1 窄比例就有 +0.4（62.6→63.0）[verified]；ViT-Adapter BEiTv2 用 [800/896…1.0]+flip 得 +0.3 [verified] | +0.5–2 [inference]，推理时间约 ×8 |
| 更高分辨率推理（上采样） | DINOv3 特征在宽分辨率范围内稳定；预训练用了 RoPE-box jitter，s∈[0.5,2] [verified: https://arxiv.org/html/2508.10104v1 "Updated Model Architecture" 段] | 对小岩石和浅坑可能有利，需在验证集上选尺度 [inference] |
| 稀有类 / 类均衡采样 | DAFormer 的 RCS：UDA 中 +5.8 mIoU；只用源域（DG）时 RCS+FD 合计 **+5.1** mIoU；温度 T=0.01 [verified: https://arxiv.org/html/2111.14887] | 按 crater/rock 像素频率做帧级过采样，+0.5–2 [inference] |
| 大上下文 / 多分辨率 | HRDA：crop 缩到 1/4 时，UDA 下降 39%，监督下降 14%；增大 context crop +5.3；GTA→CS +5.5 [verified: https://arxiv.org/html/2204.13132] | 保留整图，尺度抖动别把视野裁太小 [inference] |
| 标签驱动的阴影增强 | ShadowCast 在多架构上 +0.2–2.5 [verified] | 有可能，需消融；合规风险见下 |
| EMA / checkpoint 平均 | HuBMAP-2023 第 1 名 write-up 把 EMA 列为最关键的技巧 [weak: https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/tascj-1st-place-solution，Kaggle 页面被拦截，只见搜索摘要] | +0.2–1，成本极低 [inference] |
| logits 集成（种子或架构） | Contrails 第 1 名用 unet1024 + vit4 的集成 [weak: 搜索摘要] | +0.5–1.5 [inference] |
| Copy-paste | 只有通用证据；行星场景没有找到定量证据 | 不确定，且有合规疑问 [inference] |
| Boundary loss / OHEM / label smoothing | M2F 的 mask loss 已经在不确定点上做 importance sampling（oversample 3、importance 0.75，repo 已实现），等价于一种难例挖掘 | 额外收益小，优先级低 [inference] |
| 更长训练 | 见第 5、8 节 | 视验证曲线而定 |
| 在测试 RGB 上做 pseudo-label / 自训练 / TENT / AdaBN | 规则原文见下 | **禁止**（高风险） |

**合规判断**（规则原文取自本地 `challenge.md`）：

- 规则原文：
  - "参赛队伍不得使用官方发布训练集以外的任何数据参与训练"
  - "任务训练、微调及数据驱动的模型开发只能使用组织方指定训练集"
  - "隐藏测试集必须由算法完全自动处理"
  - "允许对官方训练 RGB 和对应 mask 进行裁剪、缩放、翻转、颜色扰动等增强，前提是增强来源限于官方训练材料，图像与标签的空间关系正确，类别含义不变。数据增强许可不扩展为使用外部图像、背景或标注的许可"
- [inference] 各项做法的风险：
  - 测试 RGB 上的自训练、伪标签、TENT 式参数更新，以及用测试统计量重算 BN（AdaBN），都等同于用测试数据训练或调参：**高风险，不做**。
  - 固定权重的 flip / 多尺度 TTA：**低风险**。
  - 训练集内部的 rock copy-paste：素材来自官方训练图，空间关系保持正确，但不在明文列表中，属**中风险**。应在官方论坛公开询问后再用。
  - 标签驱动的阴影或 gamma：属于颜色扰动的延伸，**低到中风险**，需要在方法论文中声明。
  - 公开的 LuSNAR 场景 2/4/9 数据（含标签）**不能下载使用**；repo 文档已强调这一点。

---

## 7. 候选替代或集成模型

| 模型 | ADE20K mIoU（head，crop，SS/MS） | 预训练数据（需声明） | 参数量 | 1024² 训练可行性（2×48GB）[inference] | 权重来源 | 协议 |
|---|---|---|---|---|---|---|
| **DINOv3 ViT-L/16** | linear 54.9 @512；没有官方 M2F 数字 [verified: DINOv3 Table 14] | LVD-1689M [verified] | ~0.3B | 冻结时可行，约 3–5 张/卡；解冻需要 grad checkpoint，约 1–2 张/卡 | HF（gated）；ModelScope 镜像 [verified] | DINOv3 License |
| DINOv3 ViT-H+/16 | linear 54.8 [verified] | LVD-1689M | 0.84B | 冻结勉强可行，收益不明显 | ModelScope [verified] | 同上 |
| DINOv3 7B + M2F | 62.6/63.0 @896（冻结，decoder 经 COCO-Stuff/Hypersim/ADE 训练）[verified] | LVD-1689M；head 用了额外分割数据 | 6.7B + 0.9B | 不可行；head 权重也有合规问题 | 官方需申请；ModelScope 有 backbone [verified] | 同上 |
| DINOv3 ConvNeXt-L | linear 47.8 @512 [verified] | LVD-1689M（蒸馏） | ~0.2B | 容易，可作为异构集成成员 | ModelScope [verified] | 同上 |
| DINOv2 ViT-g + Adapter + M2F | 60.2（冻结）[verified] | LVD-142M [inference，本次未重读] | 1.1B | 吃力 | ModelScope `facebook/dinov2-giant` [verified] | Apache-2.0 [verified] |
| InternImage-H | UperNet 59.9/60.3 @896；M2F 62.6/62.9（COCO-Stuff→ADE）[verified] | "Joint 427M → IN-22K" [verified]；427M 的组成（LAION-400M/YFCC-15M/CC12M）[inference，待核] | 1.08B | 吃力，且需要 DCNv3 CUDA 算子 | HF OpenGVLab；ModelScope [verified] | MIT [verified] |
| InternImage-XL | UperNet 55.0/55.3 @640 [verified] | IN-22K | 0.37B | 可行 | 同上 [verified] | MIT |
| EVA-02-L | UperNet 59.8/60.4 @512；60.1 @640 [verified] | MIM 在 Merged-38M 上 [verified]；组成（IN-22K/CC12M/CC3M/COCO/ADE20K/O365/OpenImages）和 EVA-CLIP 教师 [inference，待核] | 0.3B | 可行；provenance 链较复杂 | HF Yuxin-CV/EVA-02；ModelScope timm 版 [verified] | MIT [verified] |
| BEiT-3 + Adapter + M2F | 62.0/62.8 [verified: DINOv3 Table 11] | 多模态大规模数据 [inference] | 1.0B + 0.55B | 不可行 | — | — |
| ViT-Adapter-L (BEiTv2) + M2F | 61.2/61.5 @896（经 COCO-Stuff）[verified] | IN-1k MIM → IN-21k | 0.57B | 可行但慢 | GitHub release | — |
| OneFormer | Swin-L 57.0/57.7 @640；DiNAT-L 58.3/58.4；ConvNeXt-L 56.6/57.4 [verified: https://github.com/SHI-Labs/OneFormer] | IN-22K | ~0.22B | 可行 | shi-labs.com | MIT [inference] |
| Mask DINO | R50 语义 48.7（checkpoint 名）[verified: https://github.com/IDEA-Research/MaskDINO] | IN-1k（Swin-L 版本常带 O365 检测预训练）[inference] | — | 可行 | GitHub | Apache-2.0 [inference] |
| ConvNeXt-XL UPerNet | 53.6/54.0 @640 [verified: https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/README.md] | IN-22K | 0.39B | 可行 | dl.fbaipublicfiles；ModelScope [verified] | MIT / Apache（HF 版）[verified ModelScope] |
| SegFormer-B5 | Cityscapes 84.0（论文摘要）[verified: https://arxiv.org/abs/2105.15203]；B4 ADE20K 50.3 [verified]；B5 ADE20K 约 51 [inference] | IN-1k | 85M | 很容易，适合作为 CNN-like 异构成员 | ModelScope 无 `nvidia/mit-b5` [verified]；需用 HF 镜像 | NVIDIA 源码协议（非商用）[inference] |
| SAM-H / SAM2 Hiera | 冻结 SAM-H + M2F 在 GTAV→Citys 上 57.0，低于 DINOv2-L 的 63.3 [verified: Rein Table 1] | SA-1B（SAM2 另有 SA-V） | 0.6B | 可行 | ModelScope `facebook/sam2.1-hiera-large` [verified] | Apache-2.0 [verified] |
| Depth Anything V2-L 编码器 | 没有 ADE 分割数字 | 从 DINOv2-L 初始化，再用合成数据和约 62M 伪标签真实图训练 [inference] | 0.3B | 可行 | ModelScope [verified] | **CC-BY-NC-4.0** [verified]；provenance 链复杂 |
| AM-RADIOv2.5 g | linear 53.0 [verified: DINOv3 Table 3] | 从 DINOv2/SigLIP/SAM 等蒸馏 | 1.1B | 吃力 | NVIDIA | 来源链复杂 [inference] |

[inference] 小结：

- 在"同一套 provenance 声明（LVD-1689M）"的前提下，最稳妥的提分路径是 **DINOv3 ViT-L**。
- 集成时的第二成员可选 ConvNeXt-DINOv3-L 或 SegFormer-B5 + UPerNet 类结构，以增加误差多样性。
- InternImage-H 和 EVA-02 属于高成本、高 provenance 复杂度的选择。

---

## 8. 实践问题：收敛、过拟合、整图与随机裁剪

- **更新次数**（以 8.33 img/s 的实测吞吐估算）[inference]：

  | 配置 | 每 epoch 更新数 | 80 epoch 更新数 | 墙钟时间（2×48GB） |
  |---|---|---|---|
  | 开发集 1/3/6/7/8（6403 张），batch 70 | 91 | ≈7.3k | ≈17 h |
  | 全部 6 个训练场景（8002 张），batch 70 | 114 | ≈9.1k | ≈21 h |

  - 文献中的参照 [verified]：DINOv3 的 ADE20K 微调阶段 20k、Rein 40k（batch 4）、ViT-Adapter 80k、M2F 90k–160k（batch 16）。
  - 冻结 backbone 时，可训练的只有 adapter 和 head（约 1 亿参数，从随机初始化起步）。Mask2Former 的匹配式训练一般需要 ≥1–2 万次更新，类别 logit 才会稳定。
  - 建议每 2–4 epoch 在留出场景上评一次，看 crater 和 mountain IoU 是否已经进入平台期 [inference]。
- **合成数据过拟合的信号** [inference]：
  - 训练损失持续下降，而留出场景的 crater、rock IoU 先升后降。
  - 不同留出场景的走势出现分歧（例如场景 5 继续升，场景 6 或 8 下降）。
  - 暗区（阴影）被系统性地预测成 crater。
  - DAFormer 指出更强的架构更容易"overfitting to the source domain" [verified: https://arxiv.org/html/2111.14887 方法部分]；Rein 也有同样观察 [verified]。
  - 10 Hz 连续帧高度冗余，"epoch" 中实际独立的信息量远小于帧数。
- **整图还是随机尺度 crop**：
  - HRDA 的证据支持大上下文：crop 越小越差，跨域时尤其明显 [verified]。
  - DINOv3 预训练用了 RoPE 尺度抖动，对输入尺度变化鲁棒 [verified]。
  - [inference] LuSNAR 的相机内参和安装高度固定，尺度变化主要来自距离；山体和天空的判别依赖全局布局。
  - [inference] 推荐：以整图为主，再以 p≈0.5 做温和的尺度抖动——整图 resize 到 s×1024，s∈[0.75,1.25]（或到 1.5）。放大时随机 crop 回 1024，保留 ≥67% 的视野；缩小时 pad 并 ignore。必要时配合"多尺度推理选尺度"。
  - 不推荐 ADE 式的 [0.5,2.0] + 512 crop，因为它会丢掉 3/4 以上的视野。

---

## 9. 优先建议（共 12 条，按"期望收益 / 成本"排序）

预期 mIoU 影响都是 [inference] 的粗略估计，按场景加权口径给出。

| # | 建议 | 预期影响 | 工程成本 | 合规风险 | 依据 |
|---|---|---|---|---|---|
| 1 | **验证协议**：除场景 5 外，再用场景 6 或 8 做 leave-one-scene-out。两者是场景 9 外推角的最近邻。按场景累积混淆矩阵，按 GT 构造 C_s。用"5 与 6/8 的平均，或给 6/8 更高权重"来选 epoch、TTA 尺度和增强。可以先用短训做 fold，最终用 6 个场景重训 | 间接；避免选错模型，约 ±1–3 | 低到中（额外 1–2 次训练，可缩短） | 无（规则允许在训练池内验证） | 论文网格和 Table VII 的场景 9 塌陷 [verified] |
| 2 | **增加优化更新次数，对齐 M2F 官方细节**：有效 batch 70 不变时，LR 提到约 1.5–2e-4，或把 epoch 增到能达到 ≥15–20k 次更新；试 full-model grad clip 0.01（官方值）对比当前 0.1。注意这会改变用户已确认的配方，需要用户同意 | +0.5–2 | 低 | 无 | M2F 配置、DINOv3 D.10、Rein [verified] |
| 3 | **多尺度 + flip TTA**：在 {0.75,1.0,1.25,1.5}×1024 × hflip 上平均 softmax 语义概率。先在留出场景上选尺度子集；大尺度可用 1024 滑窗、stride 约 2/3 | +0.5–2 | 低（推理约 ×8） | 低 | M2F zoo、DINOv3 TTA 配置 [verified] |
| 4 | **Backbone 升级到 DINOv3 ViT-L/16（LVD-1689M）**，保持冻结和同样的 adapter/M2F（interaction 层 [4,11,17,23]，repo 已支持）。provenance 声明不变 | +1–3 | 中（显存和时间约 ×2，batch 约 3–4/卡） | 低（同协议；ModelScope 镜像，可用 HF LFS sha256 自证） | Table 14（B 51.8 → L 54.9）[verified] |
| 5 | **稀有类和场景均衡采样**：用 RCS 式帧采样，按 crater、rock 像素频率加权，T 约 0.01–0.1，并对场景 6/8 的困难帧适度加权 | +0.5–2（集中在 crater/rock） | 低 | 无 | DAFormer：DG 设置 +5.1、UDA +5.8 [verified] |
| 6 | **光度鲁棒性**：做三项消融——(a) 亮度/对比度恢复到源码强度（±32 / 0.5–1.5）；(b) gamma 0.7–1.5；(c) 标签驱动的投影阴影（衰减 0.5–0.8，p 约 0.3–0.5）。在阴影多的场景 5 和场景 6/8 上看 crater/rock | 0 到 +1.5 | 低到中 | 低到中（属"颜色扰动等"的延伸，需在论文中声明；阴影建议先在论坛确认） | ShadowCast [verified]；论文的阴影误差分析 [verified] |
| 7 | **保留上下文的温和尺度抖动**：p=0.5，整图 resize s∈[0.75,1.25]；放大时 crop 回 1024（视野 ≥67%），缩小时 pad 并 ignore。其余样本保持整图 | 0 到 +1 | 低 | 无（明文允许裁剪、缩放） | HRDA [verified]；DINOv3 RoPE jitter [verified] |
| 8 | **EMA 和末段 checkpoint 平均**：对可训练参数做 EMA（decay 0.999–0.9998，按更新次数设定），并与最后 K 个 checkpoint 的平均对比 | +0.2–1 | 低 | 无 | Kaggle 冠军经验 [weak]、通用实践 [inference] |
| 9 | **第二阶段低 LR 部分解冻**：收敛后解冻最后 4–6 个 block，或全量 + LLRD 0.9；backbone LR 1e-5，训 5–10k 次更新，开 grad checkpoint。只有多个留出场景都提升时才保留 | −1 到 +2（不确定） | 中 | 无 | Rein（Full 63.7 > Freeze 61.1，PEFT 66.4）、DINO Soars、DINOv3 工业研究 [verified] |
| 10 | **异构集成**：2–3 个模型做 logits 平均，例如 ViT-B 与 ViT-L，或 ViT-L 加 ConvNeXt-DINOv3-L / SegFormer-B5。权重可在留出场景上搜索 | +0.5–1.5 | 高（多次完整训练，受 4 周和 2 卡限制） | 低（每个权重单独声明来源） | 通用竞赛经验 [weak/inference] |
| 11 | **最终用全部 6 个训练场景重训**：沿用第 1 条选定的 epoch、TTA 和增强，固定种子，开 EMA。场景 5 的阴影帧和 3 个坑都会加入训练 | +0.5–2 | 低（1 次训练） | 无（规则明确允许） | 比赛规则 [verified: challenge.md] |
| 12 | **明确不做**：测试 RGB 上的伪标签、自训练、TENT、AdaBN；下载公开的场景 2/4/9 数据；把 README 的 FCN/SegNet/PSPNet 表当基线；未确认就用 rock copy-paste；用 COCO-Stuff/Hypersim 等额外分割数据预训练 head，或使用官方 7B M2F head | 避免取消资格 | 0 | — | 规则原文 [verified: challenge.md]；README 矛盾 [verified] |

**时间预算建议** [inference]：

- 训练数据 10-22 发放，测试输入 11-02 开放，截止 11-20。
- 第 1 周：在 ViT-B 上完成第 1、2、3、5、7、8 条的快速消融，每个 fold 短训约 0.5 天。
- 第 2 周：启动 ViT-L（第 4 条）全长训练，同时做第 6 条光度消融。
- 第 3 周：做第 9 条解冻实验和第 10 条第二个集成成员。
- 第 4 周初：第 11 条全量重训，冻结配方，生成提交并核对格式。每队只有一次正式提交。

---

## 附：本次读取但未展开的来源

- FINO（metadata-guided DINOv3 自适应）：https://arxiv.org/abs/2606.05107（摘要级）。
- nnMNet（火星分割基线，SynMars-TW 86.61 / MarsScapes 88.24 mIoU）：https://arxiv.org/abs/2608.29609（摘要级）。
- 本地 vendor 的 DINOv3 README 和 LICENSE，路径 `third_party/dinov3/`，对应 revision 6876159。
