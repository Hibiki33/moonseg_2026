# DINOv3 预训练权重

核对日期：2026-10-21。

## ckpts 目录各文件的含义

这里有两种预训练骨干：**DINOv3 ViT-B/16 与 ViT-L/16**，均为 LVD-1689M 预训练，分别由 `dnv3_va_m2f_vitb16.yaml` 与 `dnv3_va_m2f_vitl16.yaml` 加载。`vitb16` 表示骨干架构是 ViT-Base、patch size 16，不是另一个普通 ViT 模型；ViT-Adapter 是可训练的分割适配模块，也不是另一份预训练 ViT。

| 路径 | 内容 | 训练是否直接使用 |
|---|---|---|
| `ckpts/modelscope/vitb16/model.safetensors` | ModelScope 下载的原始 DINOv3 ViT-B/16 权重，Transformers 键名 | 保留来源；经转换后使用 |
| `ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 同一份权重转换为 Meta 原生 state_dict | `dnv3_va_m2f_vitb16.yaml` 加载 |
| `ckpts/modelscope/vitl16/model.safetensors` 与 `ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | ViT-L/16 的 ModelScope 原始文件及其原生转换（368 个张量，strict 校验通过） | `dnv3_va_m2f_vitl16.yaml` 加载后者 |
| 同名 `.source.json` | 下载来源、版本及转换说明 | 不是模型 |
| `ckpts/pretrained_model_probe.json` | 真实权重的前向/反向检查记录 | 不是模型 |
| `ckpts/modelscope/vitb16/*.json`、README、LICENSE | 模型配置、预处理配置、来源清单与许可 | 不是额外权重 |

`ckpts` 中的 safetensors 和 pth 是同一个 DINOv3 模型的两种格式。LuSNAR 分割 checkpoint 不在这里，而在 `runs/<实验名>/last.pt`、`best.pt`；它们不包含冻结的 DINOv3 张量，加载时从本目录的预训练文件重建骨干，因此复现包必须附带对应的 `.pth`（见 [model.md](model.md#checkpoints)）。

## 来源与预训练数据声明

已从 [ModelScope ViT-B/16 仓库](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m)下载完整快照，固定 revision 为 `23d0280ae6ee4ced592a3459674ad027d3c18906`。原始文件位于 `ckpts/modelscope/vitb16/`（实际磁盘目录 `/home/disk02/wjj/moonseg/checkpoints/modelscope/vitb16/`），包含 `.gitattributes`、`config.json`、`configuration.json`、`preprocessor_config.json`、`README.md`、`LICENSE.md` 和 `model.safetensors`。权重文件为 **342,662,192 bytes**，真实 safetensors header 包含 211 个 FP32 张量，并非 Git LFS pointer。`SOURCE_REVISION` 和 `modelscope_file_metadata.json` 保留了固定仓库 revision、文件清单、大小、逐文件 revision 和服务器原始响应；文件大小与远端清单一致，按项目要求没有做文件摘要计算。

ViT-L/16 以同样流程下载并转换（`--arch vitl16`），ModelScope 提交 `99b6205b…`。其 safetensors 的 sha256 与 Meta 在 Hugging Face 发布的 `facebook/dinov3-vitl16-pretrain-lvd1689m` 一致：该仓库为 `gated: manual`、license "other"（DINOv3 License），`model.safetensors` 为 1,212,559,808 bytes，LFS sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`，这份 LFS 元数据不需要 gated 授权即可读取（[API](https://huggingface.co/api/models/facebook/dinov3-vitl16-pretrain-lvd1689m?blobs=true)）。ViT-B 的哈希已记录，但没有取得可公开比对的官方值。两份 `.source.json` 保留这些事实；哈希只作记录，不作为下载流程的强制校验。

ModelScope 上的 `facebook/dinov3-*` 镜像（vitb16、vitl16、vith16plus、vit7b16、convnext-large、vitl16-sat493m）上传者均为 "Cherrytest"，license 为 "other"（[API 示例](https://modelscope.cn/api/v1/models/facebook/dinov3-vitl16-pretrain-lvd1689m)），不能仅凭 `facebook/` 命名空间断言其为 Meta 官方认证发布者。下载的模型卡声明模型由 Meta AI 开发、采用 LVD-1689M 预训练，并链接 [Meta 官方源码](https://github.com/facebookresearch/dinov3)及 DINOv3 许可；这些声明也出现在 [Meta 的 Hugging Face 模型页](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。下载的 `LICENSE.md` 与固定 Meta 源码中的许可文件内容完全相同。ViT-B 没有与 Meta 原始发行权重逐字节核对，也不声称转换后的文件是原始 Meta `.pth`。

LVD-1689M 的数据声明（[DINOv3 论文](https://arxiv.org/html/2508.10104v1) §3.1，可直接写进方法论文）：来源为约 170 亿张 Instagram 公开帖子图片（已经过平台内容审核）；用 DINOv2 embedding 做 5 级层次 k-means 平衡采样得到 1,689M 张，另加基于检索的种子数据集近邻，以及 ImageNet-1k、ImageNet-22k、Mapillary SLS 原始图像。卫星版 SAT-493M（4.93 亿张 512×512 Maxar 0.6 m 正射影像，只发布 ViT-L 与 7B）视角与地面渲染图不匹配，不使用；论文也写明 web 版在语义类地理任务上最好，卫星版擅长深度等度量任务。

[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md) 中与比赛相关的条款：发表成果时须 "acknowledge the use of DINO Materials"；再分发（包括衍生权重）必须附带同一协议；须遵守 Trade Controls 与 ITAR 限制。提交复现包时附上 DINOv3 License 和本文的来源说明。

## 原生格式与 Transformers 格式

[Meta 官方 backbone 实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py)以 `strict=True` 加载原生 PyTorch `state_dict`，原生键名包括 `blocks.0.attn.qkv.weight`、`storage_tokens`、`rope_embed.periods`。下载的 safetensors 使用 `embeddings.*`、`layer.*` 键名，需转换后传给 `model.backbone_weights`；不要把 Transformers `model.safetensors` 直接改名为 `.pth`。

[Transformers 官方转换代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py)将 QKV 拆开，并省略 masked K bias、bias mask 和确定性 RoPE buffer。本项目反向合并 QKV，恢复官方确定性 RoPE buffer，将已经丢弃的 masked K bias 填为零并恢复 K bias mask。被遮蔽的 K bias 不影响有效注意力运算，但输出不是原始发行文件的逐字节恢复。

```bash
uv run --no-sync python tools/convert_dinov3_weights.py \
  --input ckpts/modelscope/vitb16 \
  --arch vitb16 \
  --output ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

输出路径沿用官方发行文件名以匹配配置，来源说明必须与文件一起保留。转换工具只支持 LVD-1689M ViT-B/16 与 ViT-L/16 的非 gated MLP 结构，拒绝未知键、缺键、重复键、错误形状及不兼容配置，不会自动退回随机初始化；它不验证训练数据声明或模型质量。

## 验证范围

- 模拟检查为 B/L 两种结构生成完整 Transformers 键集合，检查反向映射后的全部原生键名与形状，并以原生模型 `strict=True` 加载；随机官方 ViT-B 模拟往返转换后，64×64 输入的 CLS、storage、patch 特征逐元素一致，错误 QKV 形状会被拒绝。
- 真实 ModelScope ViT-B safetensors 在 uv 环境（Python 3.12、PyTorch 2.7.1+cu126）中转换：211 个 Transformers 张量映射为 188 个原生张量，官方原生模型 `strict=True` 键名/形状检查通过，输出 `.pth` 为 342,852,817 bytes。全模型构建器再独立 strict 加载该文件，在物理 GPU 0 上完成 BF16 前向与反向：全部梯度有限，deformable sampling-offset 梯度非零，冻结骨干无梯度（记录见 `ckpts/pretrained_model_probe.json`）。ViT-L 的 368 个原生张量同样通过 strict 校验。
- 真实转换还暴露了 PyTorch 2.7.1 的初始化顺序问题：官方模型模块须在 meta-device context 之前导入，避免其首次 lazy Dynamo 导入被 default-device interception 递归触发；工具已修正该顺序，未修改官方源文件。
- 这些检查不是精度验证；没有与官方预训练参考输出做独立数值对齐。

如果已有 Meta 授权的原生 `.pth` URL，可在本机设置 `DINOV3_WEIGHTS_URL` 后用 `--weights-provider meta` 下载；签名 URL 不写入项目记录。
