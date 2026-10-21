# 第一版实验约定

采用 DINOv3 ViT-B/16（LVD-1689M 通用预训练）+ 官方 ViT-Adapter 组件 + Mask2Former。先冻结原始 ViT，训练空间先验、可变形特征提取和分割头。显存/吞吐实测后可以把配置中的 `backbone` 与权重改为 ViT-L/16；本次未运行 ViT-L。全部超参数先在训练池内场景5验证。

原始官方 LuSNAR 仓库可核实的是类别颜色和数据布局说明，没有找到可直接复用的训练脚本或完整增强配方。论文里的训练/测试划分与比赛不同。本项目以下增强来自常见分割训练做法和任务特点，不声称是 LuSNAR 官方实现。

2026-10-21再次检查[项目页的Code链接](https://luoxubo.github.io/LuSNAR-dataset/)及[作者仓库完整文件树](https://github.com/zqyu9/LuSNAR-dataset)：当前仅有 `README.md`、`index.html`、网页静态资源，只有 `main` 分支，没有tags或releases，未找到分割训练入口、模型实现、训练配置或增强代码。README虽然写有code released，但公开文件不足以复现其分割实验；[询问SegFormer具体型号的issue #18](https://github.com/zqyu9/LuSNAR-dataset/issues/18)也尚无回复。

可参考的实现来自模型作者：我们直接复用DINOv3官方Adapter和Mask2Former组件；[Mask2Former ADE20K配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml)包含AdamW、poly学习率、随机缩放/裁剪及颜色增强；[SegFormer官方数据pipeline](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py)包含0.5–2.0随机缩放、`cat_max_ratio=0.75`裁剪、水平翻转、光度扰动和255 padding。当前增强参考这些公开做法，但它们不是LuSNAR专用配方。LuSNAR类别0为有效月壤，因此不能照搬ADE20K的 `reduce_zero_label=True`。

尺寸核对：LuSNAR公开论文v3的表IV与数据说明明确原始RGB/语义标签为 **1024×1024**；IV-A没有报告分割训练裁剪尺寸，全文也没有512这一尺寸。因此 **512×1024是本项目选定的初始训练crop，不是论文配置**。其用意是在较少像素计算量下保留横向上下文，但尚未通过验证集消融证明优于512×512或1024×1024。

| 策略 | 第一版设置 | 目的与后续检查 |
|---|---|---|
| 空间增强 | 等比例随机缩放 0.5–2.0，512×1024 crop | 兼顾岩石尺寸变化与横向地形上下文；不拉伸原始图像 |
| 类别约束裁剪 | 单类最多约75%，最多尝试10次 | 减少大片月壤/天空主导；条件无法满足时使用末次裁剪 |
| 翻转 | 仅水平翻转，概率0.5 | 保留上下地形关系；不默认垂直翻转或大角度旋转 |
| 光照 | brightness/contrast/gamma 各0.9–1.1，概率0.5 | 轻度变化阴影与曝光；不注入外部背景 |
| 标签处理 | RGB精确映射，nearest resize，padding=255 | 避免RGB插值污染类别；天空4始终是有效类 |
| 优化 | AdamW，LR 1e-4，warmup+poly，梯度裁剪0.1 | 先训练Adapter和head；骨干解冻时仅其LR乘0.1 |
| 批量 | 每卡12，双卡全局24，无梯度累积 | 512×1024/BF16实测每卡allocated35.62GiB、reserved39.65GiB |
| Warmup | 250个优化器更新 | 与原配置保持约6000张图像曝光量；保持基础LR1e-4，未盲目按batch比例放大学习率 |
| 损失 | Hungarian匹配，类别CE/BCE/Dice权重2/5/5，no-object 0.1 | 按类构造目标mask，辅助层监督，忽略padding |
| 推理 | 重叠滑窗512×1024，stride341×682 | 保留原分辨率的细小目标，重叠区平均语义分数 |

建议按顺序做小规模消融：固定第一版；调整crop与分辨率；在训练集统计岩石/撞击坑比例后决定是否需要稀有类采样；对比冻结骨干与低学习率解冻；最后再测 `[0.75,1.0,1.25]` 多尺度和水平翻转 TTA。当前没有实现 Copy-Paste、伪标签或额外数据训练；如后续加入，增强源仍必须限于官方训练清单。

每次实验保存配置、正式样本清单快照、各类IoU与混淆矩阵、训练状态和随机状态。比较模型时同时看岩石/撞击坑IoU和场景5 mIoU。场景9虽然权重更高，但不能读取其公开标签来设计增强或选模型。

参考：

- [DINOv3 官方语义分割实现](https://github.com/facebookresearch/dinov3/tree/main/dinov3/eval/segmentation)。
- [Mask2Former 官方配置](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/config.py)及[criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)。
- [LuSNAR 官方项目](https://github.com/zqyu9/LuSNAR-dataset)及[论文](https://arxiv.org/abs/2407.06512)。
