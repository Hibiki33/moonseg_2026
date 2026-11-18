# MoonSeg

MoonSeg 面向月球车 RGB 图像，逐像素识别月壤、岩石、撞击坑、山体和天空。模型以单目深度估计预训练的 DA3MONO-LARGE 为骨干，通过 ViT-Adapter 融合多尺度特征，再由 Mask2Former 预测分割掩码。训练和推理分别从仓库根目录的 `train.py` 和 `predict.py` 启动，默认配置为 [da3mono_vitl14_finetuned_e25.yaml](configs/da3mono_vitl14_finetuned_e25.yaml)。

## 安装

使用 Linux x86_64、Python 3.12 和 CUDA Toolkit 12.6。以下命令通过 uv 安装锁定的依赖（包括 PyTorch 2.7.1），并编译可变形注意力算子。项目也提供 [Dockerfile](Dockerfile)。

```bash
uv sync --locked
BUILD_GPU=0 bash scripts/build_ops.sh
```

[configs/](configs/) 还包含 DeepLabV3+、SegFormer-B5、SAM 3、DINOv3 和其他 DA3 配置。运行基线所需的可选依赖可用 `uv sync --locked --extra baselines` 安装。SAM 3 使用预训练权重进行零样本推理。

## 数据与权重

数据来自 MoonSeg AI Challenge 2026 的 LuSNAR 月表场景。将图像、样本清单和权重放在以下位置。训练读取 RGB 图像和标签，预测只读取测试集 RGB 图像。

```text
data/LuSNAR_official/
├── train/Moon_<场景>/color/<样本编号>.png
├── train/Moon_<场景>/label/<样本编号>.png
├── test/Moon_<场景>/color/<样本编号>.png
├── official_train_allowlist.json
├── official_test_allowlist.json
└── manifests/
    ├── final_train.json
    └── test.json

ckpts/da3mono_vitl14_finetuned_e25.pt       # 用于推理的完整分割模型
pretrained/da3mono-large/model.safetensors  # 用于训练的骨干初始化权重
```

训练使用 Depth Anything 团队发布的 DA3MONO-LARGE 单目权重。模型卡将预训练数据范围列为公开学术数据集。原实验使用 ModelScope 修订 `ad5c2cd9fd6245ce3591227b4c53fea4b20bb240`。复现时可从发布方的 Hugging Face 固定版本获取权重，也可通过 `train.py --backbone-weights` 指定本地文件。[官方预训练权重](https://huggingface.co/depth-anything/DA3MONO-LARGE/blob/f465978e618db8cc79c83b8bbf24964857db1875/model.safetensors)

## 预测

在仓库根目录运行以下命令，默认加载 `ckpts/da3mono_vitl14_finetuned_e25.pt`，并按官方测试清单处理场景 2、4 和 9 的 5,004 幅图像。

```bash
uv run --no-sync python predict.py
```

预测写入 `predictions/test/Moon_<场景>/submission/<样本编号>.png`。模型在 0.9、1.0 和 1.1 倍尺度下分别对原图及其水平翻转图像推理。每个视图先将掩码 logits 插值到填充后的输入尺寸并计算类别得分，再裁去填充、撤销翻转，用双线性插值恢复到原图尺寸。对六个视图的类别得分取均值后，逐像素选择得分最高的类别。

输出为与原图同尺寸的 8 位单通道 PNG。像素值 0、1、2、3、4 依次表示月壤、岩石、撞击坑、山体和天空。

## 训练

最终模型在两张 NVIDIA RTX A6000 上训练 25 轮，微调整个骨干网络。每张 GPU 每步处理 4 幅图像，梯度累积 8 步，等效批量大小为 64。双卡训练命令如下，GPU 编号按本机配置调整。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
  uv run --no-sync torchrun --standalone --nproc-per-node=2 train.py
```

开发阶段使用场景 5 做验证。最终训练将场景 5 并入场景 1、3、6、7、8，共使用 8,002 对图像和标签。模型取第 25 轮、即第 3,125 次参数更新后的权重。开发阶段的方案比较使用过公开 LuSNAR 测试场景标注进行离线评估。

新训练的检查点和日志写入配置中的 `train.output_dir`，默认为 `runs/da3mono_vitl14_finetuned_e25/`。最终模型对应的训练记录见 `logs/history.jsonl`、`logs/train.log`、`logs/batch_plan.json` 和 `logs/resolved_config.json`。

## 许可证

MoonSeg 的原创代码采用 [MIT 许可证](LICENSE)。`third_party/` 中的代码沿用各项目的许可证。数据和预训练权重的使用与分发遵循各自发布方的条款。
