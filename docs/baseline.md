# 1024×1024 第一版训练方案

当前方案按用户确认采用1024×1024整图训练，DINOv3 ViT-B/16 + 官方ViT-Adapter组件 + Mask2Former。冻结原始DINOv3，训练Adapter和分割头。论文、比赛与工程选择的区别见 [paper_alignment.md](paper_alignment.md)；具体增强调研和固定源码链接见 [augmentation_research.md](augmentation_research.md)。

LuSNAR作者公开仓库尚未提供可复现的2D分割训练脚本。TGRS终稿优先，本次未取得正文，按用户允许暂参考arXiv v3。论文原图为1024×1024，但没有报告训练crop、增强与优化器配方；以下设置不能声称是LuSNAR原作者实现。

| 项目 | 当前默认 | 依据与边界 |
|---|---|---|
| 训练视野 | `full_image`，1024×1024 | 用户确认；原图同尺寸时不重采样，完整保留画面 |
| 随机几何 | 仅同步水平翻转，p=0.5 | 所查DINOv3/Mask2Former/SegFormer实现均采用；保持上下结构 |
| 亮度 | 独立p=0.5，加性像素偏移U[-16,16] | DINOv3/MMSeg算子；强度由源码±32收窄，尚无本任务增益结论 |
| 对比度 | 独立p=0.5，RGB乘U[0.9,1.1] | 同上；强度由源码0.5–1.5收窄 |
| 光度顺序 | uint8 RGB亮度→clip/量化→对比度→clip/量化，再normalize | 标签不做光度变换；没有外层概率门控或gamma |
| 可选裁剪 | 仅显式`random_resize_crop`模式启用 | 默认不随机缩放、不选crop，不把裁剪伪装成整图 |
| 标签 | 比赛ID0–4，mask最近邻；ignore255仅用于无效/补齐区域 | 天空与月壤均为有效类别，不能照搬ADE20K reduce_zero_label |
| 归一化 | ImageNet mean/std | 与DINOv3预训练骨干输入约定一致 |
| 批量 | 每卡4 × 2卡 × 累积8 = 全局有效64 | 真实完整模型双卡验证；启动检查`min_effective_batch:64` |
| epoch尾部 | 只做完整64张更新，按每轮随机顺序跳过不足组 | 不允许最后一次优化器更新偷偷变成小batch；日志记录使用/跳过数量 |
| 优化 | AdamW、LR1e-4、weight decay0.05、poly0.9、clip0.1 | 当前初始工程设置，待验证集选择 |
| Warmup | 94次优化器更新，即6016张图像 | 约保持既有6000张预热图像预算；不是论文设置 |
| 损失 | Mask2Former Hungarian + CE/BCE/Dice权重2/5/5，辅助层监督 | 与所选模型结构相符；不是LuSNAR的SegFormer配方 |
| 训练预算 | 80 epoch，BF16 | 初始预算，尚无正式数据验证结果 |
| 验证/推理 | 整图、单尺度、无flip TTA | `sliding_window:false`；输出恢复原输入尺寸 |
| 报告 | mIoU、各类IoU、mAcc、各类accuracy/recall/support、pixel accuracy | [metrics.md](metrics.md)定义；主比赛指标仍是场景加权mIoU |

## 批量选择的实测依据

物理GPU0/7均为RTX 6000 Ada 48GB，固定PCI顺序。每卡4的1024²容量测试约8.40张/秒（双卡总计），每卡6追加预热后的结果约8.29张/秒；后者没有更高吞吐，却将预留显存提高到41.01GiB，仅余约5.7GiB设备可用空间。

选择每卡4、累积8后，实际执行5次AdamW更新（2次预热、3次计时），双卡约8.51张/秒，每卡峰值allocated24.09GiB、reserved27.72GiB，实际设备尚余约19GiB。报告为 `/home/disk02/wjj/moonseg/verification/1024_batch_selection.json`。这是合成输入的短程容量/流程验证，不是精度证据，吞吐不包含真实读图与增强。

累积64只说明每次优化器更新累计的图像数：Adapter的普通BatchNorm仍按每卡microbatch=4计算统计；损失中的mask数量也按当前分布式microbatch归一化。因此不宣称与一次前向64张严格数学或逐位等价。

## 实验顺序

正式样本清单到位后，只在训练池内开发：场景1/3/6/7/8训练、场景5验证；模型选择结束再用1/3/5/6/7/8重训。首先检查无随机增强、仅水平翻转、翻转加轻量光度三个版本，保持模型、训练预算与有效batch一致，观察每类IoU及mAcc。后续才单独讨论标准强度光度、随机resize/crop或TTA。

所有增强源仅限正式训练RGB及其对应标签，不能读取2/4/9测试标签来选择增强或参数。公开场景池索引仍不是组织方的正式样本清单。更改图像模式、有效batch或增强配置后，使用新实验目录，不从旧512配置做完整状态续训。
