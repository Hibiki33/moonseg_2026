# DINOv3 预训练权重

核对日期：2026-10-21。

## ckpts 目录各文件的含义

这里有三种预训练骨干：**DINOv3 ViT-B/16、ViT-L/16 与 ViT-H+/16**，均为 LVD-1689M 预训练。B/L 分别由 `dnv3_va_m2f_vitb16.yaml` 与 `dnv3_va_m2f_vitl16.yaml` 加载；H+ 已下载并转换，模型构建器接受 `backbone: vith16plus`，但还没有对应的训练配置。`vitb16` 表示骨干架构是 ViT-Base、patch size 16，不是另一个普通 ViT 模型；ViT-Adapter 是可训练的分割适配模块，也不是另一份预训练 ViT。

| 路径 | 内容 | 训练是否直接使用 |
|---|---|---|
| `ckpts/modelscope/vitb16/model.safetensors` | ModelScope 下载的原始 DINOv3 ViT-B/16 权重，Transformers 键名 | 保留来源；经转换后使用 |
| `ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 同一份权重转换为 Meta 原生 state_dict | `dnv3_va_m2f_vitb16.yaml` 加载 |
| `ckpts/modelscope/vitl16/model.safetensors` 与 `ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | ViT-L/16 的 ModelScope 原始文件及其原生转换（368 个张量，strict 校验通过） | `dnv3_va_m2f_vitl16.yaml` 加载后者 |
| `ckpts/modelscope/vith16plus/model.safetensors` 与 `ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` | ViT-H+/16（840.6M 参数，SwiGLU MLP）的 ModelScope 原始文件及其原生转换（615 个 Transformers 张量 → 552 个原生张量，strict 校验通过） | 尚无配置；设 `backbone: vith16plus` 并以后者为 `backbone_weights` |
| 同名 `.source.json` | 下载来源、版本及转换说明 | 不是模型 |
| `ckpts/pretrained_model_probe.json` | 真实权重的前向/反向检查记录 | 不是模型 |
| `ckpts/modelscope/vitb16/*.json`、README、LICENSE | 模型配置、预处理配置、来源清单与许可 | 不是额外权重 |

`ckpts` 中的 safetensors 和 pth 是同一个 DINOv3 模型的两种格式。LuSNAR 分割 checkpoint 不在这里，而在 `runs/<实验名>/last.pt`、`best.pt`；它们不包含冻结的 DINOv3 张量，加载时从本目录的预训练文件重建骨干，因此复现包必须附带对应的 `.pth`（见 [model.md](model.md#checkpoints)）。

## 来源与预训练数据声明

已从 [ModelScope ViT-B/16 仓库](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m)下载完整快照，固定 revision 为 `23d0280ae6ee4ced592a3459674ad027d3c18906`。原始文件位于 `ckpts/modelscope/vitb16/`（实际磁盘目录 `/home/disk02/wjj/moonseg/checkpoints/modelscope/vitb16/`），包含 `.gitattributes`、`config.json`、`configuration.json`、`preprocessor_config.json`、`README.md`、`LICENSE.md` 和 `model.safetensors`。权重文件为 **342,662,192 bytes**，真实 safetensors header 包含 211 个 FP32 张量，并非 Git LFS pointer。`SOURCE_REVISION` 和 `modelscope_file_metadata.json` 保留了固定仓库 revision、文件清单、大小、逐文件 revision 和服务器原始响应；文件大小与远端清单一致，按项目要求没有做文件摘要计算。

ViT-L/16 以同样流程下载并转换（`--arch vitl16`），ModelScope 提交 `99b6205b…`。其 safetensors 的 sha256 与 Meta 在 Hugging Face 发布的 `facebook/dinov3-vitl16-pretrain-lvd1689m` 一致：该仓库为 `gated: manual`、license "other"（DINOv3 License），`model.safetensors` 为 1,212,559,808 bytes，LFS sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`，这份 LFS 元数据不需要 gated 授权即可读取（[API](https://huggingface.co/api/models/facebook/dinov3-vitl16-pretrain-lvd1689m?blobs=true)）。ViT-B 的哈希已记录，但没有取得可公开比对的官方值。两份 `.source.json` 保留这些事实；哈希只作记录，不作为下载流程的强制校验。

ViT-H+/16（2026-10-22 核对）来自 [ModelScope 仓库](https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m)，固定提交 `98b7096b2938406ade58801a0bf75eca3198b5bd`（master；`model.safetensors` 最后修改于 `a7e110a46be5aab90d0d369a986728826fec06ee`），下载 `config.json`、`model.safetensors`、`README.md`、`LICENSE.md` 到 `ckpts/modelscope/vith16plus/`。`model.safetensors` 为 **3,362,432,800 bytes**，615 个 FP32 张量，sha256 `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d`，与 [ModelScope API](https://modelscope.cn/api/v1/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/repo/files?Revision=master&Root=) 在该提交报告的 sha256 和大小一致，另外三个文件同样一致。这只证明与镜像一致：没有记录过 Hugging Face `facebook/dinov3-vith16plus-pretrain-lvd1689m` 的 LFS sha256，huggingface.co 在训练主机上不可达（TLS 证书被拦截），因此尚未与 Meta 发布的文件比对。下载的模型卡声明 H+ 有 840M 参数、由 ViT-7B 蒸馏、LVD-1689M 预训练；`LICENSE.md` 与固定 Meta 源码中的许可文件内容完全相同。`.source.json` 记录了 `source_sha256`、`source_bytes`、`modelscope_commit` 与 `cross_check`。

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

ViT-H+ 用下载工具一步完成下载与转换（`--modelscope-revision` 省略时按 `--backbone` 取上述固定提交）：

```bash
uv run --no-sync python tools/download_assets.py --only weights \
  --weights-provider modelscope --backbone vith16plus \
  --modelscope-revision 98b7096b2938406ade58801a0bf75eca3198b5bd
```

该命令等价于先下载四个文件，再运行 `tools/convert_dinov3_weights.py --input ckpts/modelscope/vith16plus --arch vith16plus --output ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`，并写出基础 `.source.json`；sha256、提交与比对字段是之后补入的。

输出路径沿用官方发行文件名以匹配配置，来源说明必须与文件一起保留。转换工具只支持 LVD-1689M ViT-B/16、ViT-L/16（非 gated MLP）与 ViT-H+/16（SwiGLU MLP），按架构检查 `use_gated_mlp`、`hidden_act` 与 `intermediate_size`，拒绝未知键、缺键、重复键、错误形状及不兼容配置，不会自动退回随机初始化；它不验证训练数据声明或模型质量。

ViT-H+ 使用 SwiGLU（gated）MLP：Transformers 配置为 `use_gated_mlp: true`、`hidden_act: silu`、`intermediate_size: 5120`，每层有 `mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`（均带 bias）。Transformers 的 `DINOv3ViTGatedMLP` 计算 `down_proj(silu(gate_proj(x)) * up_proj(x))`，官方 `SwiGLUFFN` 计算 `w3(silu(w1(x)) * w2(x))`，隐藏宽度为 `int(1280×6×2/3)=5120`；因此映射为 `gate_proj→w1`、`up_proj→w2`、`down_proj→w3`。`gate_proj` 与 `up_proj` 形状相同，形状校验无法区分二者，映射依据是 Transformers 4.57.3 的 `modeling_dinov3_vit.py` 与官方 `dinov3/layers/ffn_layers.py` 源码。

## 验证范围

- 模拟检查为 B/L 两种结构生成完整 Transformers 键集合，检查反向映射后的全部原生键名与形状，并以原生模型 `strict=True` 加载；随机官方 ViT-B 模拟往返转换后，64×64 输入的 CLS、storage、patch 特征逐元素一致，错误 QKV 形状会被拒绝。
- 真实 ModelScope ViT-B safetensors 在 uv 环境（Python 3.12、PyTorch 2.7.1+cu126）中转换：211 个 Transformers 张量映射为 188 个原生张量，官方原生模型 `strict=True` 键名/形状检查通过，输出 `.pth` 为 342,852,817 bytes。全模型构建器再独立 strict 加载该文件，在物理 GPU 0 上完成 BF16 前向与反向：全部梯度有限，deformable sampling-offset 梯度非零，冻结骨干无梯度（记录见 `ckpts/pretrained_model_probe.json`）。ViT-L 的 368 个原生张量同样通过 strict 校验。
- 真实转换还暴露了 PyTorch 2.7.1 的初始化顺序问题：官方模型模块须在 meta-device context 之前导入，避免其首次 lazy Dynamo 导入被 default-device interception 递归触发；工具已修正该顺序，未修改官方源文件。
- ViT-H+：真实 safetensors 的 615 个张量映射为 552 个原生张量，`dinov3_vith16plus(pretrained=False)` 以 `strict=True` 加载通过，参数 840,633,600，比 safetensors 的 840,592,640 个元素多 40,960 = 32 层 × 1280，即补回的零 K bias；输出 `.pth` 为 3,363,214,029 bytes。CPU FP32 224×224 前向的 CLS、storage、patch 输出全部有限。venv 中没有 Transformers，因此没有与 HF 实现做数值对齐。作为启发式检查（6 张 LuSNAR 图），按当前映射 patch 特征的平均两两余弦为 0.50，16 px 平移的最近邻对应率为 87.9%；故意交换 gate/up 后特征塌缩（余弦 0.92，对应率 84.3%）。这支持当前映射，但不是参考输出比对。
- ViT-H+ 全模型（ViT-Adapter + Mask2Former）在 CPU 上 strict 加载上述 `.pth`，256×256 前向输出 `pred_logits [1,100,6]`、`pred_masks [1,100,64,64]`；总参数 885,060,390，骨干冻结时可训练 44,426,790。构建器的 H+ 默认值：`interaction_indexes` 为 [7, 15, 23, 31]（上游 FOUR_EVEN_INTERVALS 规则），adapter 头数 `embed_dim // 64` = 20（与 B 的 12、L 的 16 一样每头 64 通道）。
- 加入 H+ 后，对 ViT-L 快照重新转换，以相同文件名写出的 `.pth` 与现有 `ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` 逐字节相同；B/L 的构建器在相同随机种子下与修改前状态字典完全一致。
- 这些检查不是精度验证；没有与官方预训练参考输出做独立数值对齐。

如果已有 Meta 授权的原生 `.pth` URL，可在本机设置 `DINOV3_WEIGHTS_URL` 后用 `--weights-provider meta` 下载；签名 URL 不写入项目记录。

## DA3MONO-LARGE（11-08追加骨干实验）

用户要求先下载权重，已完成后再启动训练。资产在`ckpts/da3mono-large/`（真实目录disk02），包含`model.safetensors`、官方模型配置/模型卡及`source.json`。下载来源为[ModelScope depth-anything/DA3MONO-LARGE](https://modelscope.cn/models/depth-anything/DA3MONO-LARGE)，修订`ad5c2cd9fd6245ce3591227b4c53fea4b20bb240`；官方HF模型元数据修订`f465978e618db8cc79c83b8bbf24964857db1875`。模型卡许可为Apache-2.0。

文件1,336,734,448字节，406个F32张量、334,171,394参数；模型配置、键/形状、参数总量及文件长度与官方HF元数据一致，未另下载HF整文件做逐字节对照。分割模型只加载其中342个encoder张量/304,367,616参数，排除64个深度head张量。源码固定为官方仓库修订`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`，最小未改动源码与许可证保存在`third_party/depth_anything_3/`；完整稀疏checkout在disk02的references。详细CPU加载与回载证据见[evidence/da3mono_verification_20261108.json](evidence/da3mono_verification_20261108.json)。

## DA3-LARGE-1.1（独立DPT融合分割路线）

资产位于 `ckpts/da3-large-1.1/`，实际目录为 `/home/disk02/wjj/moonseg/checkpoints/da3-large-1.1/`，包括原始 `model.safetensors`、配置、模型卡及 `source.json`。权重来自 [ModelScope depth-anything/DA3-LARGE-1.1](https://modelscope.cn/models/depth-anything/DA3-LARGE-1.1)，固定修订 `36524905cae6200ad00b4b085a9fdde51db9842b`；对应[官方HF仓库](https://huggingface.co/depth-anything/DA3-LARGE-1.1)，模型卡声明 Apache-2.0。ModelScope是下载来源，未另下载HF整文件做逐字节对比。

文件为 **1,643,843,860字节**，包含637个FP32张量、410,941,767个元素。使用现有 `tools.download_ranges.download_ranges` 完成HTTP分段下载，核对每段 `Content-Range`、文件总长度、safetensors元数据和原生严格加载；没有计算文件摘要。仅提取 `model.backbone.pretrained.*` 的 **407个张量/304,373,760参数**，`strict=True` 无缺键、无多键；保留camera token、QK norm、RoPE和local/global拼接。其余230个预训练张量不加载，DPT融合和语义decoder随机初始化，避免将额外几何head预训练混入首版对照。

该权重属于any-view的DA3-LARGE-1.1，不能与DA3MONO/DA3METRIC文件互换。配置使用 `architecture: da3_dpt_mask2former`，完整结构与CPU验证见 [model.md](model.md#independent-da3-large-11-segmentation)。官方 `dpt.py` 和 `model/utils/head_utils.py` 从与encoder相同的源码修订原样纳入 `third_party/depth_anything_3`，来源清单保留文件说明；其 `addict==2.4.0` 依赖已纳入uv锁文件，现有torch/CUDA版本未升级。
