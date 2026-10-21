# LuMon 整理的 Chang’e-3

调研日期：2026-10-21。Grok 独立调研；主代理复核 Hugging Face 元数据、代码仓库元数据，以及 `eval_change.py` 和 `core/masks.py`。用途为未来独立研究储备，本次比赛不使用这些额外数据。

## 身份与内容

LuMon 是 METU 团队的月面**单目深度估计**基准，论文为 *LuMon: A Comprehensive Benchmark and Development Suite with Novel Datasets for Lunar Monocular Depth Estimation*，CVPR 2026 AI4Space Workshop，arXiv:2604.09352。`LuMonDepth` 是其 Hugging Face 组织名；`LuMonDepth/ChangE-3` 是其中的真实月面子集。它与合成数据 LuSNAR、同组织的 LunarSim 分别发布。[项目页](https://metulumon.github.io/)、[论文](https://arxiv.org/abs/2604.09352)、[官方代码](https://github.com/MetuLuMon/LuMon-depth)。

数据来自嫦娥三号玉兔巡视器的月面成像。作者通过立体校正、ORB 特征和 FoundationStereo 构造深度伪真值；不能把它称为载荷直接测量的激光深度。发布分辨率和 27 cm 基线与 PCAM 彩色立体相机一致。它代表真实月海场景，不能因 LuMon 还包含极暗环境子集，就将这批嫦娥三号图像称为月球南极数据。[数据卡](https://huggingface.co/datasets/LuMonDepth/ChangE-3)、[校正流水线](https://github.com/MetuLuMon/LuMon-depth/tree/main/rectification)。

## 规模、格式与划分

| 字段 | 已取得的信息 |
|---|---|
| 发布量 | 168 组；主代理完整读取 API，确认 images、depth、masks 各 168 个文件 |
| 分辨率 | 数据卡标为宽 2352 × 高 1728 |
| 图像 | `images/001.png` 至 `images/168.png`，数据卡称 RGB |
| 深度 | `depth/001.npy` 至 `depth/168.npy`，数据卡称米制深度；未读取数组头或像素 |
| 掩码 | `masks/001.npy` 至 `masks/168.npy`，不是语义类别图；具体用途存在下述文档/代码冲突 |
| 页面体积 | 约 4.61 GB；本次未下载数据本体，未实测完整目录字节数 |
| 版本 | `f776d8c355253a1d58da45dc56f561a06d2872fa`，API 的最后修改日期为 2026-04-06 |
| 获取权限 | API 明确 `private=false`、`gated=false` |
| 划分 | HF Viewer 的 `train` 是存储 split 名；LuMon 论文将其用于评测，不应据此声称有官方监督训练划分 |

来源：[HF 原站 API](https://huggingface.co/api/datasets/LuMonDepth/ChangE-3)，本机通过 [HF Mirror API](https://hf-mirror.com/api/datasets/LuMonDepth/ChangE-3) 完整复核。没有取得原始 PCAM 产品号到 001–168 的对照、航点表、左右目配对表或正式分割 train/val/test。

文件名和完整传感器尺寸不足以证明这 168 张彼此独立，也不足以证明完全未经裁剪、重采样或增强。需要未来检查原图和原始产品对应关系。本报告不将整理包的 168 张当成嫦娥三号任务全部原始照片数。

## 标签与代码中需要注意的冲突

数据卡将 `masks/` 定义为用于过滤无效深度的二值掩码。但官方 [`evaluation/eval_change.py`](https://github.com/MetuLuMon/LuMon-depth/blob/main/evaluation/eval_change.py) 将 `--mask-dir` 描述为暗区掩码缓存，调用 [`load_or_create_dark_mask`](https://github.com/MetuLuMon/LuMon-depth/blob/main/evaluation/core/masks.py)；已有同名 `.npy` 时直接加载，缺失时才按图像亮度生成。主评测有效区另外由深度有限、深度大于零和最大距离 25 m 构造。

因此，**不能直接把发布的 `masks/` 当作这个参数传入并假设含义一致**。本次未读取数组，掩码的实际内容和 0/1 极性仍待确认。原始 Grok 报告的“masks 数量未完整取得”已被本次完整 API 清点补齐，语义冲突仍然保留。

五类映射均未建立：月壤、岩石、撞击坑、山体、天空都没有现成类别 ID。深度、暗区和语义是不同标签；黑色像素也不能直接判为天空或撞击坑。未来若用于分割定量评估，需要人工标注。代码中的距离分箱、深度有效区和阴影诊断可以参考，不能直接替代分割损失或分割真值。

## 下载入口与准备方式

本次完成入口和元数据整理，尚未下载图像、深度或掩码。未来下载使用独立目录：

```bash
mkdir -p /home/disk02/wjj/moonseg/external_data/lumon_change3
HF_ENDPOINT=https://hf-mirror.com hf download LuMonDepth/ChangE-3 \
  --repo-type dataset \
  --revision f776d8c355253a1d58da45dc56f561a06d2872fa \
  --max-workers 4 \
  --local-dir /home/disk02/wjj/moonseg/external_data/lumon_change3
```

这是供已安装 `hf` CLI 的环境手工执行的入口，本次未执行，也未向训练环境安装额外依赖。单文件入口可从 [固定版本文件树](https://huggingface.co/datasets/LuMonDepth/ChangE-3/tree/f776d8c355253a1d58da45dc56f561a06d2872fa) 获取；连接数保持适中。

如需回到原始仪器产品，使用 [NSSDC 嫦娥三号任务页](https://www.nssdc.ac.cn/nssdc_en/html/task/change3.html) 所列数据平台。Grok 未成功取得 `moon.bao.ac.cn` / `clpds.bao.ac.cn` 的当前完整下载流程，注册、产品级别及原始影像再分发条件仍待平台核实。公开 HF 包的可访问性不等于原始 L2 产品的授权说明已经核对。

## 许可、划分与后续价值

HF 数据卡声明 `license: mit`；本次元数据清单未见单独 LICENSE。原始任务影像的授权范围未取得，不把卡片字段自动扩大为对全部上游影像的独立授权。代码仓库 API 的 `license` 为 null；没有取得明确代码许可证，不能将数据卡的 MIT 套到代码。引用至少保留 LuMon 论文，并按原始任务平台要求记录数据来源。

**建议用途：真实月面外部定性检查，以及另行标注后的独立分割评测。** 这是研究价值判断，尚未做模型实验。整理时保留 `dataset_revision`、原文件名、原始产品号（未知则空）、左右目/航点/时间（未知则空）、深度来源、掩码含义、数据许可声明和实际下载状态。相邻视角、同一全景、同一原始曝光的不同处理版应归在同一组；不能随机拆开后声称场景泛化。

原始 Grok 报告与主代理复核文件保存在 `/home/disk02/wjj/moonseg/research/external_datasets/2026-10-21/`，不纳入训练清单。
