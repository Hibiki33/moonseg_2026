# MoonSeg：DINOv3 + ViT-Adapter + Mask2Former

LuSNAR 月面五类语义分割第一版。类别编号固定为月壤 0、岩石 1、撞击坑 2、山体 3、天空 4；输入仅左相机 RGB。比赛信息见 [challenge.md](challenge.md)，模型细节见 [docs/model.md](docs/model.md)。

当前先核对论文实现细节，逐项证据与差距见 [论文对照表](docs/paper_alignment.md)。已有YAML是工程配置，能通过显存/流程检查不代表已经对齐作者的训练配方；本轮只整理依据，未继续改参数或启动训练。

## 当前状态（2026-10-21）

- 已实现数据准备、开发/最终训练配置、双卡训练、验证、双卡滑窗推理和预测检查打包；默认 DINOv3 ViT-B/16，骨干冻结，Adapter 和 Mask2Former 从头训练。
- 已收录固定版本的完整 DINOv3 官方源码，并下载、转换和严格加载ModelScope ViT-B/16预训练权重；默认512×1024/BF16单卡前向与反向已通过，峰值allocated约3.38GiB。
- **uv 独立管理 Python 3.12.15** 的 `.venv` 已安装，`uv.lock` 已生成，`uv pip check` 通过；不借用外部 Conda 解释器。
- 九个LuSNAR原始包已完整下载，共52,301,555,752字节；已整理8002对训练RGB/标签与5004张测试RGB。资产位于 `/home/disk02/wjj/moonseg`，仓库通过 `data`、`ckpts` 软链接访问。
- 已实际读取官方评价与提交要求，详见 `challenge.md`。具体场景权重尚未公布；官方训练包计划10月22日起发放，正式样本清单尚未取得。

## 环境：uv

在能联网、能访问 GPU 的宿主环境执行：

```bash
bash scripts/setup_env.sh
```

该脚本让 uv 下载并管理 Python 3.12，安装至项目 `.cache/uv-python`，创建 `.venv`。PyTorch 2.7.1 / torchvision 0.22.1 使用官方 CUDA 12.6 wheel 索引，默认面向 Linux x86_64。当前主机驱动为580.126.09，物理GPU0、7均为48GB RTX 6000 Ada。

`uv.lock` 随代码保留。迁移时复制代码及锁文件，重新运行安装脚本（内部使用 `uv sync --locked`）；不复制 `.venv`，也不依赖其他人的解释器路径。数据和模型存储位置通过下载脚本的 `--storage` 设置。运行入口使用 `uv run --no-sync`，避免训练时隐式修改环境。

## 下载与存储

下载默认使用 `aria2c`，也提供 `--dataset-engine curl` 分块下载方式。每文件8连接、同时3文件，保留已完成部分供续传；数据请求使用 `https://hf-mirror.com`。只检查远端字节数、HTTP范围和数据格式，不另做哈希校验。

```bash
# 无网络、无写入地查看配置
python3 tools/download_assets.py --plan

# 获取固定版本官方源码；已完成时跳过
bash scripts/bootstrap.sh --only source

# 下载9个原始包（约52.3 GB压缩），整理左相机RGB/训练标签并建立链接
bash scripts/bootstrap.sh --only data --extract

# 固定版本的ModelScope来源；下载后严格转换为官方原生state_dict
bash scripts/bootstrap.sh --only weights --weights-provider modelscope --backbone vitb16
```

执行成功后的目录：

```text
/home/disk02/wjj/moonseg/
  data/
    archives/Moon_1.zip ... Moon_9.zip
    LuSNAR/Moon_1/color/       # 保留原sample_id
    LuSNAR/Moon_1/label/       # 仅场景1、3、5、6、7、8
    LuSNAR/Moon_2/color/       # 场景2、4、9只提取RGB
    official/                 # 组织方正式样本清单
    manifests/                # 派生索引
  checkpoints/
    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

本仓库/data  -> /home/disk02/wjj/moonseg/data
本仓库/ckpts -> /home/disk02/wjj/moonseg/checkpoints
```

实际ZIP的带标注第一相机使用 `image0/color` 与 `image0/label`，两者PNG按原文件名逐一配对，作为比赛左RGB整理；`image1/color` 是另一相机流。原始ZIP保持完整，但整理过程不打开或解压测试场景标签，不提取另一相机、深度或LiDAR；`Thumbs.db` 等非PNG文件不计入样本。解压后的实际数量见 `data/archives/zip_directory_discovery.json` 与各场景 `extraction.json`。

本机系统DNS曾将CAS下载域名解析到错误证书端点。curl方式支持 `--dns-resolver https://dns.alidns.com/resolve` 或 `--resolve-host HOST:443:IP`，仅影响当前下载进程，并保留TLS证书校验。不要通过关闭证书校验掩盖这类故障。分块下载复用HF Mirror的重定向地址，避免每个并发分块重复访问镜像而触发限流。

ModelScope 来源与权重转换边界见 [docs/weights.md](docs/weights.md)。如果已有 Meta 授权原生 `.pth` URL，可在本机设置 `DINOV3_WEIGHTS_URL` 后使用 `--weights-provider meta`；签名 URL 不写入项目记录。来源清单保留模型卡、许可和预训练数据说明。不要将 Transformers `model.safetensors` 直接改名为 `.pth`。

## 准备正式清单

只扫描公开原始包时：

```bash
uv run --no-sync python tools/prepare_data.py --root data/LuSNAR
```

只会生成 `exploration_*.json`，不会授权用于比赛训练。组织方固定清单到位后，将其转写为 JSON、JSONL、CSV 或每行一个 sample_id 的文本；JSON 示例为 `[{"scene":1,"sample_id":"原时间戳"}]`，不能用扫描公开数据得到的列表冒充正式清单。

```bash
uv run --no-sync python tools/prepare_data.py \
  --root data/LuSNAR \
  --official-allowlist data/official/train_allowlist.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --validate-labels
```

输出 `train.json`（1、3、6、7、8）、`val.json`（5）、`final_train.json`（1、3、5、6、7、8）和不含标签的 `test.json`。严格校验五类 RGB 颜色、样本存在性与清单成员资格；不会读取测试标签。未知标签颜色报错，不自动纠正。

## 训练、验证与推理

启动脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，默认 `CUDA_VISIBLE_DEVICES=0,7`，两张物理卡分别映射为进程内 `cuda:0,1`，训练和预测都由两个进程执行。该主机默认CUDA顺序与 `nvidia-smi` 不同，不能省略 `CUDA_DEVICE_ORDER`。

```bash
# 开发：场景5验证，只按开发验证结果选择参数
bash scripts/train.sh configs/moonseg_vitb16.yaml

# 从完整epoch边界恢复，保持配置及数据清单一致
bash scripts/train.sh configs/moonseg_vitb16.yaml --resume runs/moonseg_vitb16/last.pt

# 场景5本地诊断，mIoU不是排行榜分数
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,7 uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m moonseg.evaluate \
  --config configs/moonseg_vitb16.yaml --checkpoint runs/moonseg_vitb16/best.pt

# 方案冻结后，用全部官方训练场景重新训练
bash scripts/train.sh configs/moonseg_vitb16_final.yaml

# 双卡自动预测，输出原尺寸8位单通道PNG
bash scripts/predict.sh configs/moonseg_vitb16_final.yaml \
  runs/moonseg_vitb16_final/last.pt data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json

# 校验完整覆盖、尺寸、PNG模式、值域，并按官方目录打包；不执行上传
uv run --no-sync python tools/package_submission.py \
  --root data/LuSNAR --manifest data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --predictions runs/predictions --archive submissions/predictions.zip
```

默认 512×1024 crop、每 GPU batch=12、无梯度累积（`accumulation_steps: 1`）、bf16、AdamW 和 warmup+poly 学习率，全局batch为24。学习率维持1e-4；warmup调整为250个优化器更新，对应约6000张训练图像，与此前配置的样本曝光量一致。80 epoch 是初始实验参数，尚无真实数据验证依据。增强包括保持宽高比的随机缩放、类别比例约束裁剪、水平翻转及轻度亮度/对比度/gamma。多尺度/翻转 TTA 可在配置中开启，先用场景5衡量收益与开销。更多选择见 [docs/baseline.md](docs/baseline.md)。

2026-10-21在物理GPU0、7上补测完整默认模型、512×1024、BF16、冻结骨干、无梯度累积；每档运行6个AdamW更新，吞吐按后4步统计，包含优化器状态：

| 每卡batch | 全局batch | 每卡峰值allocated | 每卡峰值reserved | 双卡合计吞吐 |
|---:|---:|---:|---:|---:|
| 8 | 16 | 24.01 GiB | 27.59 GiB | 约15.77张/秒 |
| **12（默认）** | **24** | **35.62 GiB** | **39.65 GiB** | **约15.70张/秒** |

两档吞吐基本持平；选择每卡12，将batch增大并保留约8GiB显存余量。结果见 `/home/disk02/wjj/moonseg/verification/ddp_batch_{8,12}_no_accumulation.json`。这是合成张量短程训练实测，吞吐不含真实图像读取与增强；峰值不含完整验证、存盘阶段及长程分配变化，reserved也不包含全部驱动/NCCL直接申请的内存。更改batch后不能使用旧配置checkpoint做完整状态续训，须开启新实验。

LuSNAR原始RGB和标签为1024×1024；公开论文v3没有给出分割训练crop。当前512×1024由本项目选择，不能称为LuSNAR官方训练尺寸。`ckpts`中的safetensors和pth是同一个DINOv3 ViT-B/16的不同格式，文件对应关系见 [docs/weights.md](docs/weights.md)。

预测ZIP只包含 `test/Moon_2/submission/*.png`、`test/Moon_4/submission/*.png`、`test/Moon_9/submission/*.png`。mIoU按场景累积混淆矩阵，并只平均参考标签中出现的类别；参考标签缺类不能按预测结果扩张平均类别集合。完整评分与一次提交限制见 `challenge.md`。仓库同时提供顶层 `train.py`、`predict.py` 作为复现入口。

## Git

使用标准 `.git` 管理。初始化脚本设置本仓库身份为 `jiajun <jiajun9264@gmail.com>`：

```bash
bash scripts/init_git.sh
git status
```

仓库已通过标准 `git init` 初始化，身份已设置。直接使用普通 `git` 命令，数据、权重、环境与训练输出均由 `.gitignore` 排除。

## 验证边界

uv环境及依赖检查已通过；ModelScope真实safetensors的211个张量已转换为188个官方原生state_dict张量并严格加载。实际预训练DINOv3+Adapter+Mask2Former在RTX 6000 Ada上完成128×128和512×1024/B1的BF16前向与反向，全部已有梯度有限，原始冻结骨干无梯度。完整尺寸单批峰值allocated为3.38GiB、reserved为4.11GiB；这不是带优化器状态和梯度累积的训练显存上限。

双卡流程已用真实预训练骨干及缩小的分割头通过：2个epoch训练、完整epoch状态恢复、分布式评估、双卡预测、官方ZIP目录检查。该流程使用64×64训练裁剪及65×81验证/推理图像，验证记录在 `/home/disk02/wjj/moonseg/verification/pipeline.json`。恢复后两rank的随机状态、调度器和优化器步数一致；CUDA计算未保证逐位确定性，恢复运行与连续运行的数值存在小幅差异。

上述运行仅使用临时合成数据，未进行正式比赛训练或报告比赛精度；未添加unittest或仓库测试文件。正式训练前仍需取得组织方样本清单，按官方范围准备训练与验证索引。
