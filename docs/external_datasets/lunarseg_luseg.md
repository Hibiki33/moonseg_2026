# LunarSeg / LuSeg

调研日期：2026-10-21。Grok 独立调研；主代理核对论文、作者仓库及数据加载路径。以下区分论文统计、代码约定和下载包的实际核验范围。

## 身份与数据来源

**LunarSeg 是数据集，LuSeg 是分割方法，LESS 是生成数据的仿真系统。** 作者为国防科技大学 NuBot 团队；论文 *LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar*，IROS 2025。LESS 基于 Unreal Engine 4 / AirSim，在仿真巡视器视角采集 RGB-D；这是一套合成月面障碍物数据，与本仓库使用的 LuSNAR 分别发布。[作者仓库](https://github.com/nubot-nudt/LuSeg)、[公开论文 v1](https://arxiv.org/html/2503.11409v1)。

论文描述四种场景：低照/高照与平坦/崎岖地形组合。RGB-D 相机为 640×480、10 Hz，包含 RGB 和论文所称的 16 位深度图。人工标注的任务是正障碍、负障碍和背景，未提供月壤、岩石、撞击坑、山体、天空这套完整五类标签。

## 规模与实际可见文件

| 项目 | 结果与证据边界 |
|---|---|
| 论文总数 | 5,143 组 RGB-D |
| 论文划分 | train 3,085 / validation 1,028 / test 1,030；随机划分，合计 5,143 |
| 论文场景分项 | LR 1,650、LF 1,270、HR 1,144、HF 812；合计 **4,876**，少 267，原文未解释 |
| 包名 | `LunarSeg.rar` |
| 压缩体积 | Grok 通过 Google Drive Range 响应记录为 **1,404,494,257 字节**；主代理未另行下载复测 |
| 本次下载范围 | Grok 查看目录及少量 RAR 头部；未下载整包、未完整解压清点 |
| 原始采集总量 | 论文未说明发布的 5,143 组是否从更大原始序列抽取 |

以上数量与划分见论文 IV、V-A；不要用分项自行“修正”作者总数，也不要把论文张数写成已经完成的包内清点。[论文数据章节](https://arxiv.org/html/2503.11409v1#S4)、[Google Drive 数据目录](https://drive.google.com/drive/folders/1u5eHiEPSuPZm1nmjdD4iixaATTorBbI8)。

作者 [`LunarSeg_dataset`](https://github.com/nubot-nudt/LuSeg/blob/main/util/RGB_Depth_dataset.py) 预期：

```text
LunarSeg/
  train.txt
  validation.txt
  test.txt
  image/<name>.png
  depth/<name>.png
  label/<name>.png
  label_color/<name>.png     # Grok 的归档头抽查发现；加载器不使用
```

文件干名形如 `00_0000`；Grok 看到 00–03 前缀，但前缀和场景的完整对应关系尚未逐文件验证。加载器使用根目录的 split 文本及同名三模态文件。README 的评估路径示例与实际加载器存在差别，后续接入应以源码和完整解包结果为准。

## 类别、颜色与五类映射

| LuSeg 代码 ID | 含义 | 预测可视化 RGB | 对本项目的意义 |
|---|---|---|---|
| 0 | background / unlabeled | 0,0,0 | 无法分出月壤、山体、天空；不能全部映射到月壤 0 |
| 1 | negative obstacle | 0,163,226 | 可作为凹陷障碍的参考；需核对边界后才考虑与撞击坑 2 的关系 |
| 2 | positive obstacle | 173,69,31 | 论文举例为月岩；可参考岩石 1，但不能未经检查假定正障碍只包含岩石 |

来源：[类别与可视化实现](https://github.com/nubot-nudt/LuSeg/blob/main/util/util.py)、[训练脚本](https://github.com/nubot-nudt/LuSeg/blob/main/train_RGB.py)。这些 RGB 是**代码输出的调色板**，本次没有逐像素核对 `label_color`，不能直接据此断言原始彩色标签完全一致。加载器把 `label/` 作为整型类别索引使用，不是读取可视化彩色图。

负障碍包括能使车轮陷入的凹陷，定义与地质“撞击坑”存在差异；背景还合并了多种语义。以后即便做外部研究，也应先做障碍二/三类评估或补充标注，不能简单重着色后计算五类 mIoU。无实例 ID 或目标框的公开规范。

论文的“16 位深度”描述与训练时的实际尺度还应分开：加载器按每张深度图的最大值归一化，绝对尺度被改变。Grok 查到作者对深度归一化的答复，但本次未解码深度 PNG，实际位深、米制换算和完整标定仍待确认。[作者 issue #6](https://github.com/nubot-nudt/LuSeg/issues/6)。

## 下载与许可

作者 README 的 [公开 Google Drive 目录](https://drive.google.com/drive/folders/1u5eHiEPSuPZm1nmjdD4iixaATTorBbI8) 指向数据包；Grok 记录 file ID 为 `1CBmJ66_igKMIFvtp5Qb2czxUEFq3hak2`，并通过确认页后的 Range 请求读取到 RAR 文件头。未来手工下载入口：

```bash
mkdir -p /home/disk02/wjj/moonseg/external_data/lunarseg
curl --fail --location --retry 3 --connect-timeout 20 \
  --output /home/disk02/wjj/moonseg/external_data/lunarseg/LunarSeg.rar \
  'https://drive.usercontent.google.com/download?id=1CBmJ66_igKMIFvtp5Qb2czxUEFq3hak2&export=download&confirm=t'
```

本次未执行整包下载。Google Drive 可能改变确认页或触发配额限制；下载后应先检查文件类型和字节数，避免把 HTML 提示页当作 RAR。失效时从作者文件夹重新进入，不另猜文件 ID。

仓库代码为 **MIT**；作者给出的数据包没有在本次可见资料中明确单独许可，**数据再分发条款未确认**。论文公开访问或论文自身的 CC BY 不能代替数据许可。数据、LESS 工程包和 LuSeg 权重也是三个不同资产，不应混计为数据体积。引用保留 LuSeg 论文与作者仓库。[代码许可证](https://github.com/nubot-nudt/LuSeg/blob/main/LICENSE)。

## NPO 与可参考代码

`NPO_dataset` 是作者用于泛化实验的另一套**真实道路**正负障碍数据，来自 InconSeg；不是 LunarSeg 的真实月面部分。LuSeg 论文记为 5,000 张、划分 2,500/1,250/1,250，按 512×288 处理。其 `left/`、`labels/`、`depth/` 目录与 LunarSeg 的路径不同。只保留 [InconSeg 官方入口](https://github.com/lab-sun/InconSeg) 作为解释，不纳入这五组月面储备数据。

可参考 RGB-D 配对、最近邻标签缩放、两阶段特征对齐和障碍类评估。**原始 Grok 报告提到的 RandomCrop 不能直接认定生效**：已有 [增强调研](../augmentation_research.md) 核对出训练脚本虽然构造 transform 列表，相关 dataset 的 `__getitem__` 没有调用它。本次整理采用实际调用路径的结论。

## 未来独立研究的建议

适合做额外合成域的正负障碍分割对照，或在补充标签后研究五类任务迁移；这是待实验的用途判断。划分最好按场景、轨迹或连续片段隔离：10 Hz 相邻帧和共用场景资产高度相关，随机拆帧不能回答跨场景泛化问题。原有论文划分可保留用于复现，但应与新建场景划分分开报告。

后续需要补齐：完整文件计数、场景前缀对应、标签实际值域/位深、深度单位、图像相邻关系及数据许可。所有后续原始资产放 `/home/disk02/wjj/moonseg/external_data/lunarseg/`，本次仅准备资料和入口。
