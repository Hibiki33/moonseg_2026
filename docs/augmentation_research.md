# 数据增强：当前实现、源码依据与文献

核对日期：2026-10-21。本文先写当前默认和已实现的可选算子的精确语义，再列所读官方源码的实际做法和相关文献。增强的取值与开关由配置决定（[training.md](training.md)）；候选消融记在 [roadmap.md](roadmap.md)。所列算子和强度是本项目的选择，不是 LuSNAR 作者或任何所列仓库的配方，也没有已验证的 LuSNAR 增益。

## 当前默认

两份基线配置都是 `train_mode: full_image`、`image_size: [1024,1024]`，训练、验证与单尺度推理都使用完整图像：

| 项目 | 配置值与操作 | 来源与取舍 |
|---|---|---|
| 几何 | 只做水平翻转，`hflip_probability: 0.5`，RGB 与标签用同一决定 | DINOv3、SegFormer、Mask2Former 均有直接实现；不做垂直翻转或旋转，保留天空/地面的上下关系 |
| 亮度 | `brightness_probability: 0.5`，`brightness_delta: 32`：`β ~ U[−32,32]`，`clip(float32(I)+β, 0, 255).astype(uint8)` | DINOv3/MMSeg PhotoMetricDistortion 的标准强度与逐步 clip、uint8 量化 |
| 对比度 | `contrast_probability: 0.5`，`contrast_range: [0.5,1.5]`：`α ~ U[0.5,1.5]`，`clip(float32(I)·α, 0, 255).astype(uint8)` | 同一源码的标准强度 |
| 像素下限 | `pixel_floor: 22`：全部光度操作后 `np.maximum(I, 22)` | LuSNAR 渲染像素最小值恒为 22（[data.md](data.md#像素值与下限-22)） |
| 关闭项 | `gamma_probability`、`shadow_probability`、`scale_jitter_probability` 为 0；没有外层光度概率门控；色相、饱和度、噪声、模糊、擦除、MixUp/CutMix、随机裁剪均未实现或关闭 | 独立 0.5 已定义两项概率，外层再乘 0.5 会把单项实际概率降到 0.25；九个场景近乎灰度，色相/饱和度扰动没有依据 |

配置缺少这些键时，`moonseg/data/dataset.py` 的回退值为亮度 16、对比度 [0.9,1.1]、两项概率取 `photometric_probability`（默认 0.5）、`pixel_floor` 0；两份基线配置都显式写了上表的值。

实现语义：亮度是加性像素偏移，对比度是所有 RGB 通道围绕 0 乘同一因子，`PIL.ImageEnhance.Brightness` 的乘性亮度和 `ImageEnhance.Contrast` 围绕图像均值的操作都不与之等价。每一步都先 clip 再截断为 uint8，会改变暗部和高光，所以不能交换顺序、融合为一次仿射或改成 gamma 后仍称同一实现。只保留两项后，原 SSD 流程在色相/饱和度前后随机放置对比度已无必要，因此固定为先亮度后对比度。

完整顺序（RGB 以 uint8 numpy 处理，中间值 float32）：

`[full_image 确定性 resize] → 尺度抖动 → 水平翻转 → 亮度 → 对比度 → gamma → 阴影 → 像素下限 → /255 与 ImageNet mean/std`

标签只经过几何操作，光度操作不改变标签和类别 ID。原图已经是 1024×1024 时不重采样；若输入尺寸不同，只做同步的确定性整图 resize（RGB 双线性、标签最近邻），这改变采样分辨率而不截去视野。

整图与视野：固定 1024 输入不等于固定视野。先放大 2 倍再裁 1024，实际只看到原图 512×512（面积约四分之一）；先缩小 0.5 再补齐 1024 则保留内容，但真实图像只占画布四分之一，其余是 ignore 填充。`full_image` 模式完全跳过随机 resize/crop 和类别比例选框，所以不能把旧的 crop 配置改成 1024 后仍称整图训练。

## 可选算子（已实现，默认关闭）

只有概率大于 0 时才抽取对应随机数，因此关闭时的像素、标签和随机数序列与不含这些算子的实现逐位一致（场景 5 的 12 帧 × 4 个种子共 48 例同种子对照完全相同）。

| 配置键（`data.*`） | 代码默认 | 语义 |
|---|---|---|
| `scale_jitter_probability` | 0 | 仅 `full_image` 模式生效 |
| `scale_jitter_range` | [0.75, 1.25] | `s ~ U[a,b]`，尺寸 `round(W·s)×round(H·s)`；RGB 用 PIL 双线性（缩小时带抗混叠），标签最近邻。s>1 时均匀随机裁回 `image_size`；s<1 时在下/右补齐，RGB 填 `round(mean×255) = (124,116,104)`，标签填 255 |
| `gamma_probability` | 0 | 独立门控 |
| `gamma_range` | [0.7, 1.5] | `γ ~ U[a,b]`，`I' = 255·(I/255)^γ`，按 float32 预计算 256 项查表再 clip 和截断，与逐像素计算等价；γ>1 变暗 |
| `shadow_probability` | 0 | 独立门控 |
| `shadow_count_range` | [1, 3] | 形状个数，闭区间整数均匀 |
| `shadow_size_range` | [0.05, 0.35] | 形状尺度 r，单位为 `min(H,W)` 的比例（1024 图上 51–358 px） |
| `shadow_strength_range` | [0.3, 0.8] | 每个形状 `strength ~ U[a,b]`，形状内乘以 `1−strength` |
| `shadow_blur_range` | [8, 48] | 掩码高斯模糊 σ（px，PIL `GaussianBlur` 半径），用于软边 |
| `pixel_floor` | 0 | 全部光度操作后 `np.maximum(I, floor)`；基线配置设为 22 |

- 尺度抖动的视野：s=1.25 时裁回画布，宽、高各保留 80% 视野（面积 64%）；s=0.75 时保留完整视野，但 43.75% 的画布是 ignore 填充。下限取 0.75，是因为输入缩到 0.5 倍会让 27.7% 的岩石连通域消失（[data.md](data.md#目标尺寸与边界)）。
- 阴影：每个形状以各 50% 的概率取 3–7 顶点的星形多边形（顶点角度随机排序，半径 `r·U[0.4,1]`，不自交）或旋转椭圆（半长轴 r，半短轴 `r·U[0.15,1]`，方向 `U[0,π)`），中心在整幅画布上均匀放置。掩码模糊后与强度相乘，多个形状重叠处取最暗因子 `factor = min_i(1 − strength_i·mask_i)`，不累乘；随后天空（标签 4）和填充（255）处的因子置 1，最后 `I' = clip(float32(I)·factor)`。标签不修改，所以这是官方训练图像自身的空间变化亮度扰动；只在形状包围盒及 3σ 边带内光栅化与模糊。
- 与数据审计建议的已知差异：审计建议在线性空间做曝光和阴影，再按 ≥2/255 截断并重新编码回 sRGB 的 181 级。本实现直接乘在存储的 sRGB 值上，sRGB 因子 0.2–0.7 约对应线性亮度因子 0.03–0.46（按 γ≈2.2 估算），输出也不回到 181 级格点，`pixel_floor` 只保证最小值不低于 22。真实月面无大气散射，阴影边缘通常很硬，8–48 px 的模糊偏软；较小模糊（例如 2–12 px）和线性空间版本都是待消融项。
- `train_mode: random_resize_crop` 是另一种显式模式（`resize_range` 默认 [0.5,2.0]、`crop_size`、`cat_max_ratio` 0.75、`crop_attempts` 10，移植自 ADE20K 式流程），不执行尺度抖动；与 `scale_jitter_probability>0` 同用会报错。当前不使用。
- 范围校验：倒序区间、γ≤0、强度>1、形状个数<1 或非整数、下限超出 0–255 或非整数、负模糊、概率>1，以及 `random_resize_crop` 与尺度抖动同用，都在 `MoonDataset.__init__` 中报错。

实现核对（2026-10-21，CPU，场景 5 的 12 帧，开发池清单）：

- 几何对齐：把着色标签按同一种子走 RGB（双线性）路径，与标签（最近邻）路径比较，s∈[0.761,1.231] 共 72 例，含随机水平翻转。每一例中"着色结果解码类别与标签一致率"最高的偏移都是 (0,0)，±2 px 内其他偏移都更低，(0,0) 处一致率不低于 99.90%。去掉 3 px 画布边、标签 5×5 邻域一致的内部像素中，色差>2 的有 86/66,768,960（1.3×10⁻⁶），全部出现在 s<1 的例子：最近邻缩小跳过 1 px 岩石而双线性抗混叠把它混入，不是错位。s=1 的 72 例（其中 37 例翻转）逐像素完全相同；填充区 6,652,226 个像素全部为均值色，对应标签全为 255。
- 天空保护：阴影概率 1、其余光度操作开启的 60 例中，天空像素改变数为 0；非天空像素被改变的比例平均 19.4%（0.003%–63.6%），标签与几何输出完全相同。
- 像素下限：强制最强设置（strength 0.8、γ=1.5、亮度/对比度必开）时，`pixel_floor: 22` 的输出最小值为 22，不设下限时为 0。
- 单进程耗时（每样本，含 PNG 与标签解码）：默认 52.5 ms；四项可选算子概率 0.5 时 72.7 ms；全部强制开启（3 个阴影）90.2 ms。只计增强：默认 5.4 ms，全开 41.7 ms，其中 3 个阴影 14.1 ms。

## 数据依据

数字与出处都在 [data.md](data.md)，这里只列它们对增强的约束：像素下限恒为 22（→ `pixel_floor`）；标签与绝对行号强相关（→ 禁用垂直翻转、旋转和小 crop）；岩石是小目标（→ 尺度抖动下限 0.75）；九个场景近乎灰度、B−R 恒为约 +3.5（→ 不做色相/饱和度）；测试场景 9 的阴影超出全部训练场景的分布（→ gamma 与阴影作为候选）。

## 源码证据

| 项目 | 固定 revision | 核对的源码 |
|---|---|---|
| Meta DINOv3（与本项目 vendor 一致） | `6876159a11b4df116f30f667f8c9888617df0751` | [ADE20K linear 训练配置](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml)、[分割 transforms](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/transforms.py)、[RGB decoder](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/data/datasets/decoders.py)、[config 定义](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/config.py) |
| ViT-Adapter 作者仓库 | `fd64a5190f9f4530e8e5961cc87b2a3ad591190b` | [ViT-B UPerNet 完整配置](https://github.com/czczup/ViT-Adapter/blob/fd64a5190f9f4530e8e5961cc87b2a3ad591190b/segmentation/configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py)、[ADE20K base](https://github.com/czczup/ViT-Adapter/blob/fd64a5190f9f4530e8e5961cc87b2a3ad591190b/segmentation/configs/_base_/datasets/ade20k.py) |
| Meta Mask2Former | `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a` | [ADE20K semantic base 配置](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml)、[实际 semantic mapper](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py) |
| NVIDIA SegFormer 作者仓库 | `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` | [ADE20K 训练 pipeline](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/local_configs/_base_/datasets/ade20k.py)、[随仓库发布的 transform 实现](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/mmseg/datasets/pipelines/transforms.py) |
| OpenMMLab MMSegmentation | `b040e147adfa027bbc071b624bedf0ae84dfc922` | [ADE20K pipeline](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/configs/_base_/datasets/ade20k.py)、[transform 实现](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/datasets/transforms/transforms.py) |
| Detectron2（Mask2Former 调用接口的补充） | `b0fe2f0a5bb6496f65246ef35687a386a9f19202` | [ColorAugSSDTransform](https://github.com/facebookresearch/detectron2/blob/b0fe2f0a5bb6496f65246ef35687a386a9f19202/projects/PointRend/point_rend/color_augmentation.py)、[RandomFlip 等实现](https://github.com/facebookresearch/detectron2/blob/b0fe2f0a5bb6496f65246ef35687a386a9f19202/detectron2/data/transforms/augmentation_impl.py) |
| 月面 LuSeg 作者仓库 | `0f64efed77a08aa9f873a21e1c3098903d005baf` | [train_RGB.py](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/train_RGB.py)、[train_TS.py](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/train_TS.py)、[dataset 实际调用](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/util/RGB_Depth_dataset.py)、[augmentation 类](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/util/augmentation.py) |

DINOv3 的 revision 取自 `third_party/dinov3/SOURCE_REVISION`（源码归档，没有独立 `.git`，不能用归档目录内的 `git rev-parse HEAD`，它会读到 MoonSeg 父仓库的提交），其余 revision 由各自 GitHub commits API 读取后用固定 raw URL 抓取。Detectron2 是单独锁定、用于解释被调用 API 的源码，不表示 Mask2Former 作者当年的依赖恰好在此 revision。未把自监督预训练的 multi-crop/solarize 配方套到有监督分割。

- DINOv3 分割：ADE20K linear 配置为短边基准 512、随机倍率 [0.5,2.0]、随机 512×512 裁剪（`RandomCropWithLabel` 默认 `cat_max_ratio=0.75`，最多重选 10 轮，裁小后像素补 0、标签补 255）、水平翻转 0.5，随后无外层概率地调用 `PhotoMetricDistortion`。其中亮度、对比度、饱和度、色相各自 p=0.5，亮度 ±32，对比度和饱和度 [0.5,1.5]，色相 [−0.5,0.5]，对比度随机放在色相/饱和度之前或之后；`convert()` 用 float32、clip 到 0–255 后转 uint8。实际 RGB decoder 显式 `convert("RGB")`，饱和度/色相走 torchvision RGB 接口，类注释写的 BGR→HSV 不能照搬，其色相范围也不是 MMSeg 的 ±18。normalize 默认 mean/std 为 ImageNet 值乘 255（输入仍在 0–255），本项目先除 255 再用 ImageNet 值，等价。ADE20K 的 `reduce_zero_label=True` 用于去掉该数据集的 0 类，月壤 ID 0 有效，不能照搬。
- ViT-Adapter / SegFormer / MMSeg：所核 ADE20K 配置都是保持比例 resize 到 (2048,512) 尺度再乘 [0.5,2.0]、随机 512×512 裁剪、`cat_max_ratio=0.75`、水平翻转 0.5 和 `PhotoMetricDistortion`（2048 是长边上界、512 是短边基准）。SegFormer 仓库的光度实现为亮度 ±32、对比度/饱和度 [0.5,1.5]、色相整数 ±18 并对 180 取模，各项独立 0.5，在 BGR 上做，随后 `Normalize(..., to_rgb=True)` 才转 RGB；docstring 提到通道交换，但 `__call__` 没有该步骤。ViT-Adapter 所选 ViT-B 配置把 normalize 改为 RGB mean/std 全 127.5，不能为了模仿它改掉 DINOv3 需要的 ImageNet 归一化。
- Mask2Former semantic mapper：ADE20K 配置随机选短边 `int(x·0.1·512), x=5..20`、长边 ≤2048，绝对 512×512 裁剪，`SINGLE_CATEGORY_MAX_AREA=1.0`，启用 SSD 颜色增强；mapper 依次调用 resize、裁剪、ColorAugSSDTransform、RandomFlip，并把几何同步给语义图。Detectron2 的 `RandomFlip()` 默认水平 0.5、垂直关闭；SSD 颜色增强在 BGR 上做亮度/对比度/HSV 后换回 RGB，默认 ±32、[0.5,1.5]、色相 ±18，各 0.5，与 DINOv3 的 RGB torchvision 色相接口不逐像素等价。
- 月面 LuSeg（RGB-D 障碍分割，任务与类别不同）：两份 train 脚本构造了水平翻转 0.5、随机去边裁剪 `crop_rate=0.1,p=1.0` 的列表，亮度、噪声和 cropout 在注释中；但 `LunarSeg_dataset` 与 `NPO_dataset` 只把列表存为 `self.transform`，`__getitem__` 从不调用它，只读取、确定性 resize、转 float 并拼 RGB-D。因此不能说这些随机增强在该 revision 实际生效，更不能据此证明月面亮度增强有效。
- LuSNAR 作者仓库没有分割训练入口（[paper_alignment.md](paper_alignment.md#公开代码与第三方结果)），论文描述不能用来填补缺失的增强实现。

## 文献证据

- ShadowCast（[arXiv 2610.23440](https://arxiv.org/html/2610.23440)，遥感）：从分割标签提取凸起物体，按周边亮度估计的光照方向投射合成阴影，阴影长度 5–30 px、亮度衰减 0.5–0.8、p=0.5；在 10 种架构上提升 0.2–2.5（例如 DeepLabV3+ LoveDA 43.29→44.89，OCRNet Vaihingen +2.49，InternImage Vaihingen +0.40），优于通用 RandomShadow（LoveDA 43.67 对 44.89）。本项目的阴影**不是 ShadowCast 实现**：形状随机生成，不从物体标签和光照方向推导，标签只用于保护天空；强度范围与其衰减系数量级相近。
- HRDA（[arXiv 2204.13132](https://arxiv.org/html/2204.13132)）：crop 缩小到 1/4 时 UDA 性能下降 39%、监督学习下降 14%，增大上下文 crop +5.3。这支持保持 1024 整图，只做温和尺度抖动；不采用 ADE20K 式 [0.5,2.0] + 512 crop。
- DAFormer 稀有类采样 RCS（[arXiv 2111.14887](https://arxiv.org/html/2111.14887)，T=0.01）：UDA 中 +5.8 mIoU，只用源域的域泛化中 RCS 与特征距离合计 +5.1。本项目在训练侧实现为采样器（[model.md](model.md#rare-class-sampling)），不在 `MoonDataset` 中，默认关闭。
- gamma：调研没有找到 gamma 在月面数据上的定量证据，也没有找到行星场景 copy-paste 的定量证据；gamma 只作为"整体中间调变暗/变亮"的候选（场景 9 非天空均值比训练场景 3/5–8 约低 8 个灰度级）。

## 合规

比赛允许对官方训练 RGB 和对应 mask 做裁剪、缩放、翻转、颜色扰动等增强，前提是增强来源限于官方训练材料、空间关系正确、类别含义不变，且不扩展为使用外部图像、背景或标注（[challenge.md](../challenge.md)）。本项目的增强只作用于官方训练图像及其标签：几何操作同步作用于 RGB 和标签，光度操作不改变标签；阴影形状是随机几何，不来自任何其他图像。阴影与 gamma 属于颜色扰动的延伸（调研评为低到中风险），启用它们的模型须在方法论文的数据使用部分逐项声明。其余红线见 [roadmap.md](roadmap.md#红线)。
