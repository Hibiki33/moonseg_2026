# MoonSeg：DINOv3 + ViT-Adapter + Mask2Former

LuSNAR 月面五类语义分割第一版。类别编号固定为月壤 0、岩石 1、撞击坑 2、山体 3、天空 4；输入仅左相机 RGB。比赛信息见 [challenge.md](challenge.md)，模型细节见 [docs/model.md](docs/model.md)。

当前方案为用户确认的1024×1024整图训练、双卡梯度累积、全局有效batch至少64。增强来源见 [增强调研](docs/augmentation_research.md)，指标定义见 [评测说明](docs/metrics.md)；[论文对照表](docs/paper_alignment.md)保留论文事实、比赛规则与工程选择的边界。

## 当前状态（2026-10-21）

- 已实现数据准备、开发/最终训练配置、双卡训练、验证、双卡整图推理和预测检查打包；默认 DINOv3 ViT-B/16，骨干冻结，Adapter 和 Mask2Former 从头训练。
- 固定版本DINOv3源码及ModelScope预训练权重已就绪；完整1024×1024模型在物理GPU0/7上通过每卡4、累积8步的真实AdamW更新验证。
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
bash scripts/train.sh configs/moonseg_vitb16.yaml --resume runs/moonseg_vitb16_1024/last.pt

# 场景5本地诊断，mIoU不是排行榜分数
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,7 uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m moonseg.evaluate \
  --config configs/moonseg_vitb16.yaml --checkpoint runs/moonseg_vitb16_1024/best.pt

# 方案冻结后，用全部官方训练场景重新训练
bash scripts/train.sh configs/moonseg_vitb16_final.yaml

# 双卡自动预测，输出原尺寸8位单通道PNG
bash scripts/predict.sh configs/moonseg_vitb16_final.yaml \
  runs/moonseg_vitb16_1024_final/last.pt data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json

# 校验完整覆盖、尺寸、PNG模式、值域，并按官方目录打包；不执行上传
uv run --no-sync python tools/package_submission.py \
  --root data/LuSNAR --manifest data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --predictions runs/predictions --archive submissions/predictions.zip
```

默认 `train_mode: full_image`、`image_size: [1024,1024]`，保留完整视野；原图已是1024×1024时不重采样。推理使用整图、单尺度、无翻转TTA。随机缩放/裁剪只能通过显式 `random_resize_crop` 模式启用，默认不做。增强为同步水平翻转0.5、独立亮度扰动0.5（RGB像素加性±16）和独立对比度扰动0.5（乘性0.9–1.1），标签不做光度变换；gamma、HSV扰动和外部图像合成均未启用。算子来源与本项目收窄的强度范围详见 [增强调研](docs/augmentation_research.md)。

每卡batch=4，2卡，累积8步：`4×2×8=64`张图像参与每次优化器更新。使用BF16、AdamW、LR1e-4、warmup+poly；warmup94步对应6016张图像，约保持此前6000张图像的工程预热预算。80 epoch、学习率和预热量仍是初始超参数，不声称来自LuSNAR论文或已经验证精度最佳。

`min_effective_batch: 64`在启动时校验有效batch，单卡沿用此配置会因仅有32而报错。训练只进行完整的累积更新：分布式采样、microbatch及epoch尾部不完整组会丢弃，避免最后一次更新小于64。每轮重新打乱，实际使用/跳过的样本数写入 `batch_plan.json`；例如6403张训练图像每轮用6400张、跳过3张，8002张则用8000张、跳过2张。具体数量以官方清单为准。梯度只在第8次反向传播同步，裁剪、优化器和调度器各更新一次。

2026-10-21，完整1024×1024模型、BF16、冻结骨干、真实AdamW状态的双卡实测：

| 每卡batch | 实测累积步数 | 实测有效batch | 每卡峰值allocated | 每卡峰值reserved | 双卡合计吞吐 |
|---:|---:|---:|---:|---:|---:|
| 4（容量对比） | 1 | 8 | 23.97 GiB | 27.68 GiB | 约8.40张/秒 |
| 6（容量对比） | 1 | 12 | 35.55 GiB | 41.01 GiB | 约8.29张/秒 |
| **4（默认）** | **8** | **64** | **24.09 GiB** | **27.72 GiB** | **约8.51张/秒** |

每卡6没有吞吐收益，设备可用显存仅剩约5.7GiB；默认组合仍余约19GiB。默认组合验证了5个实际优化器更新，前2个预热、后3个统计。完整报告在 `/home/disk02/wjj/moonseg/verification/1024_batch_selection.json`。这些是合成输入的短程结果，吞吐不含真实读图与增强，不能当作精度或长跑显存保证。累积64描述优化器有效batch；Adapter的普通BatchNorm仍按每卡microbatch=4统计，不等同一次前向64张。

此配置使用新输出目录。旧512×1024配置的checkpoint不能用来做完整状态续训。LuSNAR原始图像/标签为1024×1024，但公开v3未给出训练crop；当前整图设定来自用户确认。`ckpts`中的safetensors和pth是同一个DINOv3模型的不同格式，见 [权重说明](docs/weights.md)。

预测ZIP只包含 `test/Moon_2/submission/*.png`、`test/Moon_4/submission/*.png`、`test/Moon_9/submission/*.png`。mIoU按场景累积混淆矩阵，并只平均参考标签中出现的类别；参考标签缺类不能按预测结果扩张平均类别集合。现在同时输出mAcc、每类accuracy/recall、每类GT支持数及pixel accuracy，定义见 [指标说明](docs/metrics.md)；这些补充指标不替代比赛主指标。完整评分与一次提交限制见 `challenge.md`。仓库同时提供顶层 `train.py`、`predict.py` 作为复现入口。

## Git

使用标准 `.git` 管理。初始化脚本设置本仓库身份为 `jiajun <jiajun9264@gmail.com>`：

```bash
bash scripts/init_git.sh
git status
```

仓库已通过标准 `git init` 初始化，身份已设置。直接使用普通 `git` 命令，数据、权重、环境与训练输出均由 `.gitignore` 排除。

## 验证边界

uv环境及依赖检查已通过；ModelScope真实权重已严格加载。默认完整1024×1024模型已完成双卡容量比较和有效batch64的5次AdamW更新，所有损失与裁剪前梯度范数有限，冻结骨干无梯度。完整视野与增强几何同步、mAcc/缺类/全ignore等CPU检查通过，报告在 `/home/disk02/wjj/moonseg/verification/full_image_data_metrics.json`。

完整1024×1024配置的CLI集成检查包括两轮训练、epoch尾部不足64的处理、断点恢复、含mAcc的分布式验证、整图预测与官方ZIP打包。结果记录在 `/home/disk02/wjj/moonseg/verification/full1024_integration.json`。恢复完整epoch状态不保证CUDA逐位确定性。

上述运行仅使用临时合成数据，未进行正式比赛训练或报告比赛精度；未添加unittest或仓库测试文件。正式训练前仍需取得组织方样本清单，按官方范围准备训练与验证索引。
