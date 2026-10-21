# DINOv3 预训练权重

核对日期：2026-10-21。

## ckpts 目录各文件的含义

这里只下载了一种预训练模型：**DINOv3 ViT-B/16（LVD-1689M）**。`vitb16` 表示骨干架构是 ViT-Base、patch size=16，并不是另一个普通 ViT 模型；DINOv3 描述其模型与预训练方式。ViT-Adapter 是后续可训练的分割适配模块，也不是另一份预训练 ViT。

| 路径 | 内容 | 训练是否直接使用 |
|---|---|---|
| `ckpts/modelscope/vitb16/model.safetensors` | ModelScope 下载的原始 DINOv3 权重，Transformers 键名 | 保留来源；经转换后使用 |
| `ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 同一份 DINOv3 权重转换为 Meta 原生 state_dict | 当前配置实际加载此文件 |
| 同名 `.source.json` | 下载来源、版本及转换说明 | 不是模型 |
| `ckpts/pretrained_model_probe.json` | 临时前向/反向验证报告 | 不是模型 |
| `ckpts/modelscope/vitb16/*.json`、README、LICENSE | 模型配置、预处理配置、来源清单与许可 | 不是额外权重 |

目前没有 LuSNAR 任务训练后的完整分割 checkpoint。Adapter 与 Mask2Former 在构建模型时初始化，正式训练后才与 backbone 一起保存到 `runs/<实验名>/last.pt`、`best.pt`。

已从 [ModelScope ViT-B/16 仓库](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m)下载完整快照，固定 revision 为 `23d0280ae6ee4ced592a3459674ad027d3c18906`。原始文件位于 `ckpts/modelscope/vitb16/`（实际磁盘目录 `/home/disk02/wjj/moonseg/checkpoints/modelscope/vitb16/`），包含 `.gitattributes`、`config.json`、`configuration.json`、`preprocessor_config.json`、`README.md`、`LICENSE.md` 和 `model.safetensors`。权重文件为 **342,662,192 bytes**，真实 safetensors header 包含 211 个 FP32 张量，并非 Git LFS pointer。

`SOURCE_REVISION` 和 `modelscope_file_metadata.json` 保留了固定仓库 revision、文件清单、大小、逐文件 revision 和服务器原始响应。文件大小与远端清单一致，aria2 下载全部完成；按项目要求没有进行文件摘要计算。远端记录的最新提交者为 `Cherrytest`，不能仅凭 `facebook/` 命名空间断言其为 Meta 官方认证发布者。

下载的模型卡声明模型由 Meta AI 开发、采用 LVD-1689M 预训练，并链接 [Meta 官方源码](https://github.com/facebookresearch/dinov3)及 DINOv3 许可；这些声明也出现在 [Meta 的 Hugging Face 模型页](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。已保留原始模型卡和许可，下载的 `LICENSE.md` 与固定 Meta 源码中的许可文件内容完全相同。这里记录真实 ModelScope 下载来源；没有将该镜像权重与 Meta 原始发行权重逐字节核对，也不声称转换后的文件是原始 Meta `.pth`。

## 原生格式与 Transformers 格式

[Meta 官方 backbone 实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py)以 `strict=True` 加载原生 PyTorch `state_dict`。原生键名包括 `blocks.0.attn.qkv.weight`、`storage_tokens`、`rope_embed.periods`。本次下载的 safetensors 使用 `embeddings.*`、`layer.*` 键名，需转换后传给 `model.backbone_weights`。

[Transformers 官方转换代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py)将 QKV 拆开，并省略 masked K bias、bias mask 和确定性 RoPE buffer。本项目反向合并 QKV，恢复官方确定性 RoPE buffer，将已经丢弃的 masked K bias 填为零并恢复 K bias mask。被遮蔽的 K bias 不影响有效注意力运算，但输出不是原始发行文件的逐字节恢复。

```bash
uv run --no-sync python tools/convert_dinov3_weights.py \
  --input ckpts/modelscope/vitb16 \
  --arch vitb16 \
  --output ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

输出路径沿用官方发行文件名以匹配现有配置；来源说明必须与文件一起保留。转换工具只支持 LVD-1689M ViT-B/16 与 ViT-L/16 的非 gated MLP 结构，拒绝未知键、缺键、重复键、错误形状及不兼容配置，不会自动退回随机初始化。转换不会验证训练数据声明或模型质量。ViT-L 本次没有下载；转换时使用 `--arch vitl16`。

## 验证范围

早期模拟检查已为 B/L 两种结构生成完整 Transformers 键集合，检查反向映射后的全部原生键名与形状，并以原生模型 `strict=True` 加载。随机官方 ViT-B 模拟往返转换后，64×64 输入的 CLS、storage、patch 特征逐元素一致；错误 QKV 形状会被拒绝。这些检查不代表真实预训练精度验证。

真实 ModelScope safetensors 已在独立 uv 环境（managed Python 3.12、PyTorch 2.7.1+cu126）读取并转换：211 个 Transformers 张量映射为 188 个原生张量，官方原生模型 `strict=True` 键名/形状检查通过。输出 `ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` 为 342,852,817 bytes；同名 `.source.json` 记录实际下载 revision、转换及来源限制。

真实预训练 backbone 再经全模型构建器独立 strict 加载，在物理 GPU 0（RTX 6000 Ada 48 GB）完成 128×128 和 512×1024、batch=1 的默认完整模型 BF16 前向与反向。全部已有梯度有限，deformable sampling-offset 非零梯度检查通过，冻结 backbone 无梯度；没有 optimizer step。512×1024 峰值 allocated 为 3.38 GiB、reserved 为 4.11 GiB，未计 optimizer 状态。详细记录见 `ckpts/pretrained_model_probe.json` 和 [模型说明](model.md)。没有数据集训练、验证集指标、与官方预训练参考输出的独立数值对齐或精度结论。

真实转换还验证了 PyTorch 2.7.1 的初始化顺序：官方模型模块须在 meta-device context 之前导入，避免其首次 lazy Dynamo 导入被 default-device interception 递归触发。工具已修正该顺序，未修改官方源文件。
