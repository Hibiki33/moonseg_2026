# 参考文献与第三方工作档案

核对日期：2026-10-21。本文登记两类材料：A 是使用 LuSNAR 数据的第三方工作，逐项记录任务、划分、模型和报告数字，并判断能否与比赛的按场景划分对照；B 是本仓库文档引用的方法论文。表中数字均摘自原文或作者发布的结果文件，"未核实"表示没能读到正文或原文未写明；"推断"表示本文自己的判断，不是原作者的说法。

## 存档方式

本地文件放在 `references/`，这是指向 `/home/disk02/wjj/moonseg/references` 的符号链接，已由 `.gitignore` 的 `/references` 排除，不进 git。

`references/index.json`（同样不入库）每项记录 title、authors、venue/year、url、local_file、sha256、bytes、fetched_at（UTC）、任务/划分/结果摘要和相关性标签；没有下载文件的条目 local_file、sha256、bytes 为 null。核对文件完整性用 `sha256sum -c` 或对照 index.json 的 sha256。

arXiv 从主机下载只有约 20–35 KB/s；LuSNAR v3（17,869,228 字节）、DINOv3（15,823,906 字节）和 LuMon（23,023,708 字节）三个 PDF 首次抓取时未完成，已于 2026-10-22 01:37（UTC+8）用 `curl -C -` 续传补齐并校验 `%PDF`/`%%EOF`，`index.json` 中的 sha256 与字节数已更新；当前 `references/` 内没有 `.part` 文件。

只存合法公开的文件：arXiv、开放获取出版物（CC BY）、课程报告页、学位论文开放库和 Zenodo 复现包。没有下载任何付费墙 PDF 或数据集文件；Zenodo 复现包按其 README 不含 LuSNAR 图像，只有代码、权重、帧清单和结果表。

## A. 使用 LuSNAR 的第三方工作

检索范围：Semantic Scholar 引用接口（arXiv 与 TGRS 两条记录返回同一份 12 篇引用）、OpenAlex 引用过滤（3 篇）与全文检索"LuSNAR"（16 条）、Crossref、arXiv 站内检索（元数据，3 条）、GitHub 仓库检索（名称命中 3 个，README 命中 19 个）、hf-mirror（1 个模型、1 个数据集、0 个 Space）、Stanford CS231N 2024–2026 项目/报告页，以及多轮网页检索（含中文）。

受阻情况：Semantic Scholar 的关键词检索与全文 snippet 检索持续返回 429，网站页面返回 403；OpenReview API、Google Scholar 和 export.arxiv.org 从主机超时；ScienceDirect 返回 403（LESFormer、ISPRS 两篇无法核实）；IOP 直连只返回验证页、publications.polymtl.ca 直连超时，这两份 PDF 改由网页抓取工具取得；PMC 页面为 reCAPTCHA，改读 Frontiers 原站。

结论：公开渠道只有一项（Padilla Romero 的 Zenodo 复现包）给出**按场景留出**的 2D 五类分割结果；其余 2D 分割数字或者划分未写明，或者测试帧与训练帧来自同一批场景，不能与比赛划分对照。场景 9 有两份独立的第三方证据：Zenodo 复现包中只用场景 5/6/7 训练的模型在场景 9 上撞击坑 IoU 0.2432，BEA-RandLA-Net 的 3D 点云撞击坑 IoU 在场景 9 为 67.84/76.56、场景 5 为 86.08/91.62。两者都与论文表 VII 场景 9 撞击坑 19.19 的结论一致：场景 9 的撞击坑是最难的类别。

### A1. 2D 语义分割（与比赛同任务）

| 工作 | 年份 | 发布形式 | 任务 | 划分 | 关键数字 | 本地文件 | 结论 |
|---|---|---|---|---|---|---|---|
| Padilla Romero, *Lightweight difficulty-adaptive early-exit inference for lunar surface segmentation on flight-representative embedded hardware*，复现包 | 2026 | Zenodo 10.5281/zenodo.22089457（v1.0.0，CC BY 4.0；对应稿件未找到公开版本） | 2D 五类分割，嵌入式早退推理 | **按场景**：训练/验证取场景 5/6/7，测试留出场景 8；场景 1、9 只做分析 | 场景 8 基线 mIoU 0.7887；早退模型全深度 0.7846；场景 9 mIoU 0.6832、撞击坑 0.2432 | `references/2026_zenodo_22089457_padilla_earlyexit_lusnar_seg.zip` | 唯一可对照的第三方场景留出结果，但模型很小、输入 256，只作难度参照，不作目标 |
| Brian Wu, *Towards A Unified Deep Learning Architecture for Extraterrestrial Surface Perception* | 2025 | Stanford CS231N 课程报告 | 2D 分割 + 单目深度 | "suggested 80-20 split"，约 10k/2k；未写明随机还是按场景 | SegFormer 0.9686、MaskFormer 0.8955、UNet++ 0.7681、SAM 统一模型 0.7766（mIoU） | `references/2025_cs231n_wu_lusnar_segformer.pdf` | 不可比；比论文跨场景结果高约 7–25 个点，反映非场景验证的乐观偏差 |
| Théo Gachet, *Quantification de l'incertitude et fusion sémantique multimodale pour la navigation critique des rovers spatiaux en environnement extraterrestre* | 2026 | Polytechnique Montréal 硕士论文（导师 Giovanni Beltrame） | 2D 分割（SegFormer-B0）+ 3D 点云 + 不确定性校准 | 未说明 2D 验证集如何抽取 | 验证集 mIoU 0.9516；ECE 9.47%→1.09%（2D） | `references/2026_polymtl_msc_gachet_uncertainty_lusnar_segformer.pdf` | 划分不明，不可比；只对不确定性校准有参考价值 |
| Germano-Pansini/NSSS（GitHub，`master_thesis` 分支 `5e345da`） | 2025 | 代码仓库 | 2D 分割（SegFormer-B0）+ 神经符号推理 | 论文主划分：训练 Moon_1/2/4/6/8/9（其中随机 85/15 分训练/验证），测试 Moon_3/5/7 | 仓库未公开任何 mIoU | 无（仅登记 URL 与提交） | 有按场景测试的代码，但没有结果可对照 |
| halilaydin-tru/tua-hackathon（"Zenit"，GitHub `318a323`） | 2026 | 黑客松代码仓库 | RGB-D 分割（ESANet_Ultra） | 未说明 | mIoU 93.27%，18 epoch，输入 256 | 无 | 不可比：输入含深度，划分与评测代码均未给出 |

**Padilla Romero 复现包**（作者单位：北航国际学院，杭州）。协议写在包内 README 和 `code/moon.ipynb` 中：场景 5/6/7 隔帧取样，每条轨迹按时间顺序前 85% 训练、后 15% 验证，中间隔 5 帧，得 1,824/310 帧（训练 675/463/686、验证 115/78/117）；场景 8 隔帧取样，整体留出，"evaluated once, after the checkpoint and the confidence threshold have been fixed"；场景 1 因没有撞击坑像素被排除出训练。输入双线性缩放到 256×256，标签最近邻缩放；网络为约 0.49M 参数的紧凑 U-Net（5.34 GFLOPs）；Adam lr 1e-3、余弦退火、30 epoch、batch 8，逆频率加权交叉熵，没有数据增强，按验证 mIoU 选权重，训练不设种子。

该复现包的场景 8 基线每类 IoU：月壤 0.9753、撞击坑 0.6916、岩石 0.5098、山体 0.7799、天空 0.9870（`results/table2_baseline_test.csv`，mIoU 0.7887 见 `baseline_results.json`）。同表 `ref_UNet31M` 列实为 LuSNAR 论文表 V 的 U-Net（98.70/73.09/73.50/79.46/99.34，论文划分不同），不是作者自己的结果。场景 9、场景 1 的数字来自早退模型全深度推理（`table8_scene_analysis.csv`）：场景 9 mIoU 0.6832、撞击坑 0.2432、岩石 0.5261；场景 1 mIoU 0.6818，其中撞击坑 IoU 0 被计入。

该复现包的 mIoU 口径是全测试集像素累计 IoU，对 union>0 的类别求均值：参考标签没有某类但模型误报时记 0 并计入平均（场景 1 撞击坑即如此），这与比赛"只平均参考标签中存在的类别"不同。比赛训练池包括场景 5–8，所以该包的场景 8 结果对应本项目的训练场景；场景 9 是比赛测试场景，数字只能说明难度，不能当作分数下界或目标。

**Wu 的 CS231N 报告**原文为 "We follow a suggested 80-20 split for training and validation data respectively which translates to just over 10,000 sequences for the training set and just over 2,000 sequences for the validation set"。报告把 13,006 帧误称为 13,006 个各含 1,000–2,000 帧的"sequences"，全文没有出现 random 或按场景划分的说明。[paper_alignment.md](paper_alignment.md#公开代码与第三方结果) 中"随机 80/20 帧划分"的"随机"是推断：随机 80/20 约为 10,405/2,601，整场景留出场景 9 则为 10,919/2,087，两者都符合原文措辞。

Wu 报告的其他设置：增强为水平翻转、±10° 旋转、亮度/对比度；所有基线 5 epoch、Adam（Depth Anything v2 与 MaskFormer 改用 AdamW，wd 0.01），lr 分别为 SegFormer 5e-5、MaskFormer 1e-5、UNet++ 1e-4；batch 4，单卡 RTX 3090。SegFormer 型号与分割基线的输入分辨率均未写明，比较的是 MaskFormer 而不是 Mask2Former，没有每类 IoU 或 mAcc。深度部分：统一模型 ARE 0.074、RMSE 0.024、δ1 0.947，微调后的 Depth Anything v2 为 0.688/0.080/0.901。

**Gachet 硕士论文**（法语，105 页）：2D 部分为 SegFormer-B0（MiT-B0），输入 512×512，五类；表 3.1 在验证集上的每类 IoU 为月壤 0.9539、撞击坑 0.9374、岩石 0.9029、山体 0.9728、天空 0.9909，mIoU 0.9516，按五类平均。论文只说测试集"strictement disjoint"于训练和校准集，没有列出各集合所用的场景或帧，所以划分方式未核实。校准后 ECE 2D 从 9.47% 降到 1.09%，3D 从 11.50% 降到 2.13%；3D 部分为 PointNet++，每个样本 2048 点。

**NSSS 仓库**：`config.py` 写明 "SegFormer-B0 (ADE20K), train/test at 512×512"，推理用 1024；`SCENES_TRAIN = Moon_1/2/4/6/8/9`、`SCENES_TEST = Moon_3/5/7`，`TRAIN_VAL_RATIO = 0.85` 用 `train_test_split(shuffle=True)` 在训练场景内随机分帧，所以验证集与训练集同场景，测试集才按场景划分；`EPOCHS = 200`，`random_flip`/`random_crop` 为 False。仓库树里只有本体、USD 输出和无关论文 PDF，没有指标文件。数据下载链接指向巴里理工（Politecnico di Bari）的 SharePoint，项目可能出自该校硕士论文（推断）。

**tua-hackathon**：README 的类别表为天空、Taş（石）、撞击坑、Kaya（岩）、Toprak（土）五类，没有"山体"，与 LuSNAR 五类的映射未说明；mIoU 93.27% 没有给出划分、分辨率以外的评测细节，深度按 `dep / 100.0` 归一化后作为第二输入。

### A2. 其他任务（深度、点云、SLAM、定位、新视角合成）

| 工作 | 年份 | 发布形式 | 任务 | 划分 | 关键数字 | 本地文件 | 结论 |
|---|---|---|---|---|---|---|---|
| J. Liu 等, *BEA-RandLA-Net: a boundary-enhanced and globally-aware network for lunar surface obstacle detection from point clouds* | 2026 | J. Phys.: Conf. Ser. 3240, 012038（AIPC 2026，CC BY 4.0） | 3D 点云三类分割（月壤/撞击坑/岩石） | **按场景**：训练 1/2/3/6/7，验证 4/8，测试 5/9 | 场景 5 mIoU 93.22→95.26；场景 9 87.84→89.45；场景 9 撞击坑 67.84→76.56 | `references/2026_jpcs_3240_012038_liu_bea_randla_net_lusnar_pointcloud.pdf` | 任务不同，不可比；佐证场景 9 撞击坑最难 |
| Sekmen 等, *LuMon: A Comprehensive Benchmark and Development Suite with Novel Datasets for Lunar Monocular Depth Estimation* | 2026 | arXiv 2604.09352v1；CVPR 2026 AI4Space Workshop | 单目深度 | **按场景**：训练 Moon_2/3/4/5/7/9，验证 Moon_6，测试 Moon_1/8 | DAv2-FT 在 LuSNAR 测试集 δ1 0.93、AbsRel 0.07、RMSE 1.07；零样本 DAv2 0.31/0.45/2.89 | 未完成（`references/2026_cvprw_arxiv_2604.09352v1_lumon_lusnar_depth.pdf`，内容依据 arXiv HTML） | 深度任务，不可比；权重 `LuMonDepth/DAv2-FT-LuSNAR` |
| Pérez-Ramos 等, *Zero-shot semantic landmark-based visual odometry using foundation models for unstructured planetary exploration* | 2026 | Front. Robot. AI, 10.3389/frobt.2026.1819714（CC BY） | 视觉里程计（SAM + DINOv2） | 只用 Moon_1/2/3，各 100 帧 | ATE RMSE：Moon_3 1.93 m、Moon_2 4.88 m、Moon_1 5.70 m；无 IoU | `references/2026_frobt_1819714_perezramos_sam_dinov2_vo_lusnar.pdf` | 非分割；只有阴影下 SAM 表现的定性观察 |
| *MoonGS: High-quality Representation of the Lunar Surface via Gaussian Splatting Using Robust Depth Features from Image Pairs* | 2026 | arXiv 2610.07110（代码 InRobots/MoonBlender） | 稀疏视角 3DGS 新视角合成 | "the last 4 of the 9 scenes" 作训练和验证，256×256 | 摘要称比前馈 NeRF/3DGS 基线 PSNR +4.9 dB | 无 | 无关 |
| Dai、Casadesus Vila、Gao, *Visual SLAM with DEM Anchoring for Lunar Surface Navigation* | 2026 | arXiv 2603.17229；IEEE Aerospace Conf. 2026 | 视觉 SLAM | 九个场景逐一评测，用 LiDAR 累积构造局部 DEM | 表 I 逐场景 RPE/ATE | 无 | 无关 |
| *Globally Localizing Lunar Rover in Pixels via Graph Alignment*（WARG） | 2026 | arXiv 2606.10602（代码 maochen-casia/warg） | 跨视角全局定位 | 10,020/1,625/1,350 张，划分依据未核实 | 平均定位误差 0.32 m（OrienterNet 20.89 m） | 无 | 无关 |
| Cai Meng 等, *Render3D: A self-supervised learning method for lunar surface stereo matching based on 3D rendering* | 2026 | 航空学报 47(10) | 自监督立体匹配 | 只在 LuSNAR 上评测（表 3） | 未核实（仅读到摘要级信息） | 无 | 无关 |
| GitHub：ArturPrentki/orb-slam3-lusnar、henrylanday/lunar-lidar-slam（Colby College CS366）、eugene0429/WALJU、toto-jjw/Stereo_autonomous、MetuLuMon/LuMon-depth、umutcanustuntas/lunar-depth、shuyujinqu/PSF-PGO、phyllis-TANG/lunar_PNT | 2025–2026 | 代码仓库 | SLAM、立体/深度、导航 | 多为单场景或若干场景回放 | lunar-lidar-slam：ATE RMSE 0.655 m | 无 | 无关 |

**BEA-RandLA-Net**：网格下采样 0.03 m，分块 4096 点，batch 10，100 epoch，Adam 带 warm-up，轻度随机噪声增强，每组实验跑 4 次取平均。表 1 括号内依次为月壤/撞击坑/岩石 IoU：RandLA-Net 场景 5 为 93.22（99.21|86.08|93.81），场景 9 为 87.84（97.97|67.84|93.36）；BEA-RandLA-Net 场景 5 为 95.26（99.35|91.62|94.68），场景 9 为 89.45（98.20|76.56|93.60）。

原文给出的帧数为训练 6749、测试（5/9）2651、验证（4/8）3686。按本仓库左相机 RGB 计数，场景 1/2/3/6/7 为 6649，5/9 为 3686，4/8 为 2671，所以原文可能把测试与验证的帧数写反，或点云帧数与图像不同；未核实。

**Frontiers 视觉里程计论文**的定性观察："The Segment Anything Model is robust to lunar lighting conditions; it successfully segments rocks even when they are partially obscured by deep shadows"。Moon_2 地标稀疏归因于 "severe terrain self-occlusions, deep shadows, or sparse rock distributions"。

### A3. 只引用 LuSNAR、或使用情况未能核实

- Vila、Dai、Gao, *Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting*（arXiv 2603.18218，IEEE Aerospace 2026）：分割模型（U-Net、MANet 等）在 LuPNT 仿真的 spirals 数据集上训练和评测，**不是 LuSNAR**；LuSNAR 只在相关工作中被引用，并为 Fig. 6 的特征匹配示意图提供图像。
- 只在相关工作或数据集对比中引用：LLaVA-LE（arXiv 2603.24696）、Lunar-G2R（arXiv 2601.10449）、MoonAnything（arXiv 2604.00682）、StereoLunar（arXiv 2510.18172）、*Mining beyond Earth with Space Robots*（arXiv 2608.21358，综述）、Lunar-Bench（ACL Findings 2026，LLM 任务基准）、SSRN 综述 10.2139/ssrn.7553541、MDPI *Aerospace* 13(10):929 视觉惯性导航综述（未读正文）。Lunar-G2R 与 MoonAnything 都指出 LuSNAR 由 Unreal Engine 渲染，没有物理的月面反射（BRDF）模型；这与评估仿真到真实的迁移有关，与比赛内部划分无关。
- 使用情况未能核实：LESFormer（*Advances in Space Research* 2026，10.1016/j.asr.2026.07.124，地外地形分割）被 OpenAlex 全文检索命中"LuSNAR"，但没有公开摘要且 ScienceDirect 返回 403，任务细节、划分和数字均未知，取得正文后应优先补读；ISPRS J. 2026 位姿图优化（10.1016/j.isprsjprs.2026.07.024）同样被检索命中但无法访问；FASTA 2025 颠簸环境系统、ICCRE 2026 多智能体探索两篇为 IEEE 会议，没有摘要。
- 明确不使用：AlanSEncinas/LunarSite 把 "LuSNAR supplementary training data integration" 列为跳过项；kshitijjdixit/lunar-rover-mde 只引用；mennanasa12-cloud/paradox_perception 把 LuSNAR 列为主数据集，但尚无结果。

## B. 方法论文

| 论文 | arXiv | 本地文件 | 本仓库引用原因 |
|---|---|---|---|
| LuSNAR: A Lunar Segmentation, Navigation and Reconstruction Dataset based on Multi-sensor for Autonomous Exploration | 2407.06512v3 | `references/2024_arxiv_2407.06512v3_lusnar.pdf` | 数据集原论文：类别、场景网格、表 V/VII 基线和误差分析均以此版为据（[paper_alignment.md](paper_alignment.md)、[data.md](data.md)）；TGRS 终稿（10.1109/TGRS.2026.3720373）未取得全文 |
| DINOv3 | 2508.10104v1 | `references/2025_arxiv_2508.10104v1_dinov3.pdf` | 冻结骨干 ViT-B/16、ViT-L/16（LVD-1689M）及其分割适配做法（[model.md](model.md)、[weights.md](weights.md)、[training.md](training.md)） |
| Masked-attention Mask Transformer for Universal Image Segmentation（Mask2Former） | 2112.01527v3 | `references/2022_cvpr_arxiv_2112.01527v3_mask2former.pdf` | 分割头、匹配损失 CE/BCE/Dice = 2/5/5、12,544 个采样点、lr 与训练长度参照 |
| Vision Transformer Adapter for Dense Predictions（ViT-Adapter） | 2205.08534v4 | `references/2023_iclr_arxiv_2205.08534v4_vit_adapter.pdf` | 连接冻结 DINOv3 与 Mask2Former 的适配器；80k 训练长度参照 |
| Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation（Rein） | 2312.04265v5 | `references/2024_cvpr_arxiv_2312.04265v5_rein.pdf` | 域偏移下冻结、全量微调与参数高效微调的对比，对应 [roadmap.md](roadmap.md) 候选 #8 部分解冻 |
| DAFormer | 2111.14887v2 | `references/2022_cvpr_arxiv_2111.14887v2_daformer.pdf` | 稀有类采样 RCS（T = 0.01）的出处，本项目在训练侧实现为采样器，默认关闭 |
| HRDA | 2204.13132v2 | `references/2022_eccv_arxiv_2204.13132v2_hrda.pdf` | 上下文 crop 尺度对分割的影响，支撑保持 1024 整图、只做温和尺度抖动 |
| ShadowCast: On-the-Fly Context-Aware Shadow Augmentation for Remote Sensing Image Segmentation | 2610.23440v2 | `references/2026_arxiv_2610.23440v2_shadowcast.pdf` | 阴影增强的文献证据（[augmentation_research.md](augmentation_research.md#文献证据)）；本项目的阴影不是其实现 |
| LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar | 2503.11409v1 | `references/2025_iros_arxiv_2503.11409v1_luseg.pdf` | 月面 RGB-D 障碍分割（LunarSeg 数据集，不是 LuSNAR），其增强代码核查见 [external_datasets/lunarseg_luseg.md](external_datasets/lunarseg_luseg.md) |
| SegFormer | 2105.15203v3 | `references/2021_neurips_arxiv_2105.15203v3_segformer.pdf` | LuSNAR 表 V 最佳基线（mIoU 89.96）；[training.md](training.md) 中的轻量异构候选 |
| PointRend: Image Segmentation as Rendering | 1912.08193v2 | `references/2020_cvpr_arxiv_1912.08193v2_pointrend.pdf` | LuSNAR 表 V 基线，撞击坑 IoU 最高（82.45） |

ShadowCast 的 arXiv 号已核对：2610.23440 即 Yoon 等的 *ShadowCast: On-the-Fly Context-Aware Shadow Augmentation for Remote Sensing Image Segmentation*（当前 v2），与 [augmentation_research.md](augmentation_research.md#文献证据) 引用的是同一篇。已对照 v2 正文核实 DeepLabV3+ 在 LoveDA 上 43.29→44.89、RandomShadow 43.67，以及表 S1 默认阴影长度 (5, 30)、衰减 (0.5, 0.8)；该文档中的其余数字（10 种架构、OCRNet、InternImage 增益）没有逐一核对。
