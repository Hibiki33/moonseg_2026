# 1024×1024 月面语义分割的数据增强调研

核对日期：2026-10-21。本文直接读取官方仓库的配置、变换实现及数据集调用路径；固定 revision 见下表。这里区分“源码实际做什么”和“本项目如何取舍”，不把论文实验描述、未调用的类或自定强度写成作者训练配方。未读取测试场景标签，未引入外部训练数据。

## 建议交给实现方的默认契约

保留用户要求的 **1024×1024 全幅训练**：`train_mode=full_image`，训练、验证与单尺度推理均使用完整图像。原图已经是1024×1024时不重采样；若输入尺寸另有变化，仅同步确定性整图 resize 到目标尺寸，RGB用双线性、标签用最近邻，并明确这会改变采样分辨率而非截去视野。不要把随机缩放后裁剪、类别比例选框或滑窗推理作为此 baseline 的隐含行为。

最小默认采用：

| 项目 | 默认参数和操作 | 来源与取舍 |
|---|---|---|
| 几何 | 仅水平翻转，`p=0.5`，RGB/标签用同一决定 | DINOv3、SegFormer、Mask2Former均有直接实现；不默认垂直翻转或旋转，以保留天空/地面的上下关系 |
| 亮度 | 独立`p=0.5`；`beta ~ U[-16,16]`；`clip(float32(I)+beta,0,255).astype(uint8)` | 加性亮度、逐步clip及uint8量化来自DINOv3/MMSeg；16是本项目相对源码32收窄后的保守超参数 |
| 对比度 | 独立`p=0.5`；`alpha ~ U[0.9,1.1]`；`clip(float32(I)*alpha,0,255).astype(uint8)` | 乘性对比度来自同一源码；0.9–1.1是本项目相对源码0.5–1.5收窄的范围 |
| 顺序/空间 | RGB uint8上：水平翻转 → 亮度 → 对比度 → 转float/255 → ImageNet mean/std；标签只执行相同几何 | 只有亮度与对比度子集，无HSV步骤；不对着色标签或类别ID施加光度操作 |
| 禁用项 | 无外层photometric概率门控；gamma、色相、饱和度、噪声、模糊、擦除、MixUp/CutMix、垂直翻转、旋转和随机裁剪均关闭 | 独立0.5已定义两项概率，外层再乘0.5会将单项实际概率降到0.25；其他操作未在本任务证明必要 |

上述轻量默认**不是任何作者原封不动的配方，也没有已验证的LuSNAR增益**。算子和原始强度有源码依据，收窄强度是为本项目保守起步作出的选择，应在配置和报告中如实写明。若需要完全不自行选择光度强度，先跑“全幅+水平翻转”的对照，再将此子集作为首个消融。

注意实现语义：这里亮度是加性像素偏移，对比度是所有RGB通道围绕0乘同一因子；`PIL.ImageEnhance.Brightness` 的乘性亮度及 `ImageEnhance.Contrast` 围绕图像均值的操作都不与之等价。每步clip会改变暗部/高光，不能把顺序交换、融合为一次仿射或改成gamma后仍称同一实现。只保留两项后，原SSD流程在色相/饱和度前后随机放置对比度已无必要，因此固定亮度后对比度。

## 已直接读取的固定源码

| 项目 | 固定revision | 核对的源码 |
|---|---|---|
| Meta DINOv3（与本项目vendor一致） | `6876159a11b4df116f30f667f8c9888617df0751` | [ADE20K linear训练配置](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml)、[分割transforms](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/transforms.py)、[RGB decoder](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/data/datasets/decoders.py)、[config定义](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/config.py) |
| ViT-Adapter作者仓库 | `fd64a5190f9f4530e8e5961cc87b2a3ad591190b` | [ViT-B UPerNet完整配置](https://github.com/czczup/ViT-Adapter/blob/fd64a5190f9f4530e8e5961cc87b2a3ad591190b/segmentation/configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py)、[ADE20K base](https://github.com/czczup/ViT-Adapter/blob/fd64a5190f9f4530e8e5961cc87b2a3ad591190b/segmentation/configs/_base_/datasets/ade20k.py) |
| Meta Mask2Former | `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a` | [ADE20K semantic base配置](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml)、[实际semantic mapper](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py) |
| NVIDIA SegFormer作者仓库 | `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` | [ADE20K训练pipeline](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/local_configs/_base_/datasets/ade20k.py)、[随仓库发布的transform实现](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/mmseg/datasets/pipelines/transforms.py) |
| OpenMMLab MMSegmentation | `b040e147adfa027bbc071b624bedf0ae84dfc922` | [ADE20K pipeline](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/configs/_base_/datasets/ade20k.py)、[当前transform实现](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/datasets/transforms/transforms.py)、[IoU/accuracy metric](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/evaluation/metrics/iou_metric.py) |
| Detectron2（Mask2Former调用接口的补充） | `b0fe2f0a5bb6496f65246ef35687a386a9f19202` | [ColorAugSSDTransform](https://github.com/facebookresearch/detectron2/blob/b0fe2f0a5bb6496f65246ef35687a386a9f19202/projects/PointRend/point_rend/color_augmentation.py)、[RandomFlip等实现](https://github.com/facebookresearch/detectron2/blob/b0fe2f0a5bb6496f65246ef35687a386a9f19202/detectron2/data/transforms/augmentation_impl.py) |
| 月面LuSeg作者仓库 | `0f64efed77a08aa9f873a21e1c3098903d005baf` | [train_RGB.py](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/train_RGB.py)、[train_TS.py](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/train_TS.py)、[dataset实际调用](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/util/RGB_Depth_dataset.py)、[augmentation类](https://github.com/nubot-nudt/LuSeg/blob/0f64efed77a08aa9f873a21e1c3098903d005baf/util/augmentation.py) |

其余仓库revision由本次各自GitHub commits API读取后使用固定raw URL抓取。DINOv3为不含独立`.git`目录的源码归档，固定revision取自`third_party/dinov3/SOURCE_REVISION`，与本次官方GitHub commits API返回一致；直接读取vendor对应文件。不能用归档目录内的`git rev-parse HEAD`识别上游版本，因为它会向上读取MoonSeg父仓库的提交。未把自监督预训练的multi-crop/solarize配方套到有监督分割。Detectron2是单独锁定用于解释被调用API的当前源码，**不表示Mask2Former作者当年的依赖环境恰好锁在此revision**。

## 各实现实际采用的增强

### DINOv3 segmentation

ADE20K linear配置：短边基准512、随机倍率`[0.5,2.0]`、随机`512×512`裁剪、水平翻转`p=0.5`。`RandomCropWithLabel`默认`cat_max_ratio=0.75`，最多重选10轮，仍可能接受最后的框；裁小后用像素0/标签255补齐。图像双线性、标签最近邻。`make_segmentation_train_transforms`随后无外层概率地调用`PhotoMetricDistortion`。

光度类中，亮度、对比度、饱和度、色相各自`p=0.5`；亮度`±32`，对比度和饱和度`[0.5,1.5]`，色相`[-0.5,0.5]`。对比度随机放在色相/饱和度之前或之后。`convert()`用float32、clip到0–255后转uint8。实际RGB decoder显式`convert("RGB")`，饱和度/色相使用torchvision RGB接口；类注释仍写BGR→HSV，不能照注释给RGB数据额外换通道。其色相范围也不是MMSeg的OpenCV `±18`。

normalize的默认mean/std分别为`[0.485,0.456,0.406]`和`[0.229,0.224,0.225]`乘255，因其输入张量仍在0–255；本项目先除255后用未乘255的mean/std等价。ADE20K `reduce_zero_label=True`用于去掉该数据集0类，**月壤ID0有效，本项目不能照搬**。

### ViT-Adapter / SegFormer / MMSeg

三者核对的ADE20K配置均为保持比例resize到`(2048,512)`尺度再乘`[0.5,2.0]`、随机`512×512`裁剪、`cat_max_ratio=0.75`、水平翻转0.5和`PhotoMetricDistortion`。2048是长边上界、512是短边基准，不是强制拉伸成2048×512。当前MMSeg用`RandomResize`，旧SegFormer用`Resize`的ratio_range接口；语义相近，不代表版本可直接混用。

SegFormer仓库内的光度实现给出明确数值：亮度`±32`，对比度/饱和度`[0.5,1.5]`，色相整数`±18`并对180取模；各项独立0.5，图像先BGR，饱和度和色相使用BGR↔HSV。旧pipeline随后`Normalize(...,to_rgb=True)`才转RGB。源码docstring提到通道交换，但实际`__call__`没有该步骤，不能声称启用通道交换。当前MMSeg相同光度类的关键参数与BGR语义一致。

ViT-Adapter所选完整ViT-B配置会覆盖base的normalize，实际是RGB下mean/std全为127.5（映射到约−1到1）；SegFormer base是RGB ImageNet mean/std。**不能为了模仿Adapter而改掉DINOv3预训练骨干需要的ImageNet归一化。** Adapter配置引用外部MMSeg API，本次其transform语义由SegFormer随仓库代码及当前MMSeg交叉核对，不声称已重建Adapter当年的全部依赖环境。

### Mask2Former semantic mapper

ADE20K配置随机选择短边`int(x*0.1*512), x=5..20`，长边最多2048；绝对`512×512`裁剪，`SINGLE_CATEGORY_MAX_AREA=1.0`，启用SSD颜色增强，输入格式RGB。实际mapper依次调用resize、裁剪、ColorAugSSDTransform、RandomFlip，并把几何同步给语义图；因此不能把它概括成一套没有裁剪的1024整图增强。

所读Detectron2源码中，`RandomFlip()`默认水平0.5、垂直关闭。SSD颜色增强按RGB参数先换到BGR，进行亮度/对比度/HSV，再换回RGB；默认`±32`、`[0.5,1.5]`、色相`±18`，每项概率0.5，标签光度保持不变。它与DINOv3实际RGB torchvision色相接口并不逐像素等价。

### 月面LuSeg：必须检查调用是否生效

LuSeg是月面RGB-D障碍分割的作者实现，任务、类别及模型均不同于本项目。两份train脚本都构造水平翻转0.5、随机去边裁剪`crop_rate=0.1,p=1.0`的列表；亮度`bright_range=0.15,p=0.9`、噪声及cropout位于注释中。

但实际`LunarSeg_dataset`和`NPO_dataset`仅把列表存为`self.transform`，两个`__getitem__`都没有调用该列表，只读取、确定性resize、转float并拼RGB-D。因此**不能说这些随机增强在该固定revision实际训练生效，更不能据此证明月面亮度增强有效**。数据由PIL读取，RGB不换BGR，标签resize明确最近邻；LuSeg的RGB resize未显式指定插值参数，不给它补造某种插值。此结果也说明“作者仓库有RandomBrightness类”不能作为本项目默认gamma或噪声的证据。

LuSNAR作者仓库此前核对可见README/静态项目页，没有取得可核实的分割训练入口；论文描述及上游数据场景不能用来填补缺失的增强实现。未把其他个人AI4Mars课程项目当作官方实现。

## 对1024全幅视野的具体影响

固定1024输入不等于固定视野。假设原图1024×1024，先放大2倍再裁1024：实际只看到原图512×512，宽高视野各减半，面积约四分之一。先缩小0.5再补齐1024则保留场景内容，但真实图像只占输出四分之一面积，边缘出现大量ignore padding。这两者都不同于原图完整占据1024画布的baseline。

所以不能仅把旧的`crop_size`从`[512,1024]`改为`[1024,1024]`，仍保留随机倍率、`cat_max_ratio`选框后称“1024整图训练”。推荐显式模式分开：默认`full_image`彻底跳过随机resize/crop和类别选框；只有可选`random_resize_crop`实验才调用源自上述ADE20K流程的几何策略，并单独记录视野变化。水平镜像保留全部像素，但反转左右关系；它是所查分割实现普遍采用的训练操作，仍需本地验证。

## 最少消融与指标

只用官方训练池：开发训练1/3/6/7/8、本地验证5；冻结增强选择后可用1/3/5/6/7/8重训最终模型。任何比较都不能读取2/4/9测试标签或据测试反馈挑策略。

| 实验 | 改动 | 回答的问题 |
|---|---|---|
| A：全幅对照 | 1024全幅，无随机增强 | 与既有预训练特征适配后的起点 |
| B：最小几何 | A + 水平翻转0.5 | 左右镜像是否有益 |
| C：建议默认 | B + 上述独立亮度/对比度 | 轻量光度子集是否优于只翻转 |
| D：可选强度消融 | C的光度改为源码强度32、0.5–1.5，仍关闭HSV | 收窄强度是否必要；不混入FOV变化 |
| E：可选几何消融 | 单独启用随机resize/crop，明确模式、倍率和crop；不作为默认 | 局部视野/尺度扰动是否值得牺牲整图上下文 |

不预设gamma、垂直翻转、旋转、擦除或外部图像合成实验；本次读取没有形成足以优先加入本任务的依据。对验证集同时报告mIoU、每类IoU、mAcc、每类accuracy/recall、类别支持像素和总像素accuracy；保留混淆矩阵并按场景汇总，固定种子/训练预算做比较。

mAcc采用`Acc_c=TP_c/(TP_c+FN_c)`，在GT支持大于0的类别中平均；缺类记录为空值，0–4都是有效类。该定义与所读MMSeg `IoUMetric`中的`total_area_intersect / total_area_label`及其均值汇总一致。它是附加诊断指标，不替代比赛场景加权mIoU，也不能把像素总准确率标成mAcc。正式场景权重尚无具体数值，验证场景5分数不伪装成官方总分。

## 增补（2026-10-21，第二版）：可选尺度抖动、gamma、阴影与像素下限

本节为第二版增补，不修改上文第一版契约。新增四项操作全部**默认关闭**：概率默认为0，`pixel_floor` 默认为0。只有概率大于0时才抽取对应随机数，因此默认配置与第一版逐位一致，包括像素、标签和随机数序列。已用场景5的12帧×4个种子共48例，与 HEAD 实现做同种子对照，结果完全相同。下列算子和数值是**本项目超参数**，有待场景5/6本地消融；它们不是 LuSNAR 作者或任何所列仓库的配方，也没有已验证的 LuSNAR 增益。

### 数据依据（docs/audit_2026-10-21/data_audit.md）

- 像素是8位线性亮度经 sRGB 编码的结果，全部场景只出现181个离散值，**最小值恒为22**（= sRGB(2/255)）。深阴影和天空都截断在22/28。合成变暗若低于22，会产生渲染器从不输出的数值。
- 测试场景9的阴影超出全部训练场景的分布，且只按 RGB 统计。非天空像素 Y<25 为11.2%，训练场景3/5/6/7/8为5.8–6.7%。物体带 rows 256–384 的 Y<25 为19.6%，训练场景为3.1–9.5%。72.4%的帧在 rows 256–512 有超过10%的暗像素，训练场景最高为场景5的22.3%。场景9到各训练场景的非天空亮度 W1 为8.3–11.9，训练场景（除场景1）两两之间为0.3–3.1。
- 标签与绝对行位置强相关：天空最低行不超过365，rows 512–1024 有99.1–100%为月壤。因此继续禁用垂直翻转、旋转及小 crop。审计建议的尺度抖动为0.75–1.25倍。
- 岩石是小目标：连通域面积中位数63 px。输入缩到0.5倍时，27.7%的岩石连通域会消失，所以尺度下限取0.75。
- 九个场景近乎灰度，B−R 恒为约+3.5，色相/饱和度扰动没有依据，本版仍不加入。

### 文献依据（docs/audit_2026-10-21/method_research.md）

- **ShadowCast**（arXiv 2610.23440，遥感）：从分割标签提取凸起物体，按估计的光照方向投射合成阴影，阴影长度5–30 px，亮度衰减0.5–0.8，p=0.5。在10种架构上提升0.2–2.5，优于通用 RandomShadow（LoveDA DeepLabV3+ 43.67 vs 44.89）。**本项目的阴影不是 ShadowCast 实现**：形状随机生成，不从物体标签和光照方向推导；标签只用于排除天空。它属于“受其启发、带天空保护的随机投影阴影”，强度范围与其衰减系数量级相近。该调研把标签引导的阴影或 gamma 评为“颜色扰动的延伸、低到中风险、须在方法论文中声明”。
- **DAFormer 稀有类采样 RCS**（arXiv 2111.14887，T=0.01；只用源域的 DG 中 RCS+FD 合计+5.1 mIoU）：**在训练侧实现**，不在 `MoonDataset` 中。它读取 `tools/label_stats.py` 输出的逐样本类别像素数，采样权重必须由所训练清单本身的统计得到（见 [开发池](development_pool.md)）。
- **HRDA 上下文证据**（arXiv 2204.13132）：crop 缩小到1/4时，UDA 下降39%，监督学习下降14%。这支持保持1024整图，只做温和尺度抖动。s=1.25时裁回画布，宽、高各保留80%视野（面积64%）；s=0.75时保留完整视野，但43.75%的画布是 ignore 填充。
- gamma：该调研**未找到** gamma 在月面数据上的定量证据。加入它只为提供“整体中间调变暗/变亮”的候选消融（γ>1 变暗，场景9非天空均值比训练场景3/5–8约低8灰度级），不作为默认。

### 精确语义与顺序

RGB 以 uint8 numpy 处理，中间值为 float32，每步先 clip 到0–255，再用 `astype(uint8)` 截断，与第一版亮度/对比度一致。顺序为：

`[full_image 确定性 resize] → 尺度抖动 → 水平翻转 → 亮度 → 对比度 → gamma → 阴影 → 像素下限 → /255 与 ImageNet mean/std`

标签只经过前两步几何操作。`random_resize_crop` 模式保持原样，不执行尺度抖动；在该模式下设置 `scale_jitter_probability>0` 会报错。

| 配置键（data.*） | 默认 | 语义 |
|---|---|---|
| `scale_jitter_probability` | 0 | 仅 `full_image` 模式生效 |
| `scale_jitter_range` | [0.75, 1.25] | `s~U[a,b]`，尺寸 `round(W·s)×round(H·s)`。RGB 用 PIL 双线性（缩小时带 PIL 抗混叠），标签用最近邻。s>1 时均匀随机裁回 `image_size`；s<1 时在下/右补齐，RGB 填 `round(mean×255)=(124,116,104)`，标签填255 |
| `gamma_probability` | 0 | 独立门控 |
| `gamma_range` | [0.7, 1.5] | `γ~U[a,b]`，`I'=255·(I/255)^γ`。按 float32 预计算256项查表，再 clip 和截断，与逐像素计算等价 |
| `shadow_probability` | 0 | 独立门控 |
| `shadow_count_range` | [1, 3] | 形状个数，闭区间整数均匀 |
| `shadow_size_range` | [0.05, 0.35] | 形状尺度 r，单位为 `min(H,W)` 的比例（1024 图上为51–358 px）。本键为本项目新增 |
| `shadow_strength_range` | [0.3, 0.8] | 每个形状 `strength~U[a,b]`，形状内乘以 `1−strength` |
| `shadow_blur_range` | [8, 48] | 掩码高斯模糊 σ（px，PIL `GaussianBlur` 半径），用于软边 |
| `pixel_floor` | 0 | 全部光度操作后执行 `np.maximum(I, floor)`。依据审计，配置建议设为22 |

阴影的每个形状以各50%概率取两类之一。一类是3–7顶点多边形：顶点角度随机排序，半径为 `r·U[0.4,1]`，构成不自交的星形多边形。另一类是旋转椭圆：半长轴 r，半短轴 `r·U[0.15,1]`，方向 `U[0,π)`。中心在整幅画布上均匀放置。形状掩码模糊后与强度相乘，多形状重叠处取最暗因子，即 `factor = min_i(1 − strength_i·mask_i)`，不累乘。随后天空（标签4）及填充（255）处的因子置1，最后 `I' = clip(float32(I)·factor)`。**标签不修改**，所以这仍是官方训练图像自身的空间变化亮度扰动。实现只在形状包围盒及3σ边带内光栅化与模糊。

与审计建议的已知差异：审计建议在**线性空间**做曝光和阴影，然后按 ≥2/255 截断并重新编码为 sRGB 的181级。本实现直接乘在存储的 sRGB 值上，因此 sRGB 因子0.2–0.7大约对应线性亮度因子0.03–0.46（按 γ≈2.2 估算），输出也不会回到181级量化格点。`pixel_floor` 只保证最小值不低于22。另外，真实月面无大气散射，阴影边缘通常很硬，可视化中8–48 px 的模糊显得偏软。较小模糊（例如2–12 px）和线性空间版本都应作为消融项，不能视为已定。

### 实现验证（2026-10-21，CPU，场景5的12帧，开发池清单）

- 几何对齐：将着色标签按同一种子经过 RGB（双线性）路径，与标签（最近邻）路径比较，s∈[0.761,1.231] 共72例，含随机水平翻转。在每一例中，“着色结果解码类别与标签一致率”最高的偏移都是 (0,0)，±2 px 内其他偏移都更低，(0,0) 处一致率不低于99.90%。标签5×5邻域一致的内部像素（去掉3 px 画布边）中，色差>2的有86/66,768,960（1.3×10⁻⁶），全部出现在 s<1 例：最近邻缩小会跳过1 px 岩石，双线性抗混叠会把它混入，这不是错位。s=1 时72例（其中37例翻转）逐像素完全相同；填充区6,652,226个像素全部为均值色，对应标签全为255。
- 天空保护：阴影概率1，其余光度操作开启，60例中天空像素改变数为**0**。非天空像素被改变的比例平均为19.4%（0.003%–63.6%），标签与几何输出完全相同。
- 像素下限：强制最强设置（strength 0.8、γ=1.5、亮度/对比度必开）时，`pixel_floor=22` 的输出最小值为22，不设下限时为0。
- 范围校验：倒序区间、γ≤0、强度>1、形状个数<1或非整数、floor 超出0–255或非整数、负模糊、概率>1，以及 `random_resize_crop` 与尺度抖动同用，均在 `__init__` 中报错。
- 单进程耗时（每样本，含 PNG 解码与标签解码）：默认配置52.5 ms；四项新操作概率为0.5时72.7 ms；全部操作强制开启（3个阴影）时90.2 ms。只计增强时，默认5.4 ms，全开41.7 ms，其中3个阴影14.1 ms。同时将 P 模式标签解码改为调色板查表，8002张标签结果与原实现逐位一致，单张耗时由99.9 ms降到7.0 ms。

### 合规

只使用官方训练 RGB 及其对应标签。开发期使用公开训练场景开发池，最终训练只用组织方清单（见 [开发池](development_pool.md)）。不引入外部图像、背景、纹理或标注；阴影形状是随机几何，不来自任何其他图像。几何操作同步作用于 RGB 和标签，光度操作不改变标签和类别含义。从不读取测试场景2、4、9的标签，也不依据测试反馈选择增强。启用这些操作的模型须在方法论文的数据使用部分逐项声明。

### 建议消融（只用训练池，固定种子与训练预算）

以第一版C（全幅+水平翻转+轻量亮度/对比度）为对照，分别在 `dev_train`/`dev_val`（场景5）和 `dev_val6_train`/`dev_val6`（场景6）上比较：F = C+尺度抖动0.5；G = C+阴影0.5+`pixel_floor:22`；H = C+gamma 0.5+`pixel_floor:22`；I = 全部。除常规指标外，单独报告审计定义的暗帧子集（物体带 Y<25 超过10%，场景5共357帧），并按块长≥50帧的 block bootstrap 估计差异的置信区间。没有稳定增益的选项保持关闭。
