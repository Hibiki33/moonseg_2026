# MoonSeg：DINOv3 + ViT-Adapter + Mask2Former

LuSNAR 月面五类语义分割第一版。类别编号固定为月壤 0、岩石 1、撞击坑 2、山体 3、天空 4；输入仅左相机 RGB。比赛信息见 [challenge.md](challenge.md)，模型细节见 [docs/model.md](docs/model.md)。

当前方案为用户确认的1024×1024整图训练、双卡梯度累积、全局有效batch严格大于64。增强来源见 [增强调研](docs/augmentation_research.md)，指标定义见 [评测说明](docs/metrics.md)；[论文对照表](docs/paper_alignment.md)保留论文事实、比赛规则与工程选择的边界。

未来独立研究的外部数据资料见 [五组月面数据调研](docs/external_datasets/README.md)：LunarSeg/LuSeg、LuMon Chang’e-3、Apollo、DFKI Rocks and Craters、POLAR/POLAR-Sim，包含来源、标签、许可、下载入口与待核实项。

## 当前状态（2026-10-21 晚）

- 主线：DINOv3 ViT-B/16 **冻结** + 官方 ViT-Adapter + Mask2Former，1024×1024 整图。配置命名为 `dnv3_va_m2f_<骨干>`：`configs/dnv3_va_m2f_vitb16.yaml`（主线，GPU 0/7）和 `configs/dnv3_va_m2f_vitl16.yaml`（同配方换 ViT-L/16，GPU 5/6）。今天之前的旧草稿 `configs/moonseg_vitb16*.yaml` 已删除。
- 四项审计已完成并归档在 `docs/audit_2026-10-21/`：代码审查、数据实测（`data_audit.md`）、公开方法调研（`method_research.md`）、GPU 容量实测（`gpu_capacity.md`）。
- 官方 MSDeformAttn CUDA 扩展已编译（`scripts/build_ops.sh` → `build/ms_deform_attn/`），`model.deformable_attention: auto` 自动选用；与纯 PyTorch 路径损失一致到 1e-7 量级，batch 7 显存从 41.5 GiB 降到 30.9 GiB，双卡吞吐 8.3 → 10.8 img/s。
- 有效 batch 改为每卡 8 × 2 卡 × 累积 4 = **64**（要求：不低于 64）。
- checkpoint 只保留 `last.pt`（完整恢复状态，约 460 MB）和 `best.pt`（仅权重，约 230 MB），不再存冻结的 DINOv3 张量；推理与恢复时从 `model.backbone_weights` 重新加载骨干。`runs` 软链到本地盘 `/home/disk02/wjj/moonseg/runs`，不写 NAS。
- 官方清单到位前，开发训练使用公开 LuSNAR 训练场景池（`data.development_pool: true`，清单 `data/manifests/dev_*.json`），边界见 [docs/development_pool.md](docs/development_pool.md)；最终提交模型必须在官方清单上从骨干重训。
- 2026-10-21 晚已启动两条基线（ViT-B 在 PCI 5/6，ViT-L 在 PCI 0/7）；已定事项、候选路线与实验记录见 [docs/roadmap.md](docs/roadmap.md)，GPU 配对约定见 [docs/gpu_topology.md](docs/gpu_topology.md)。
- 固定版本DINOv3源码及ModelScope预训练权重（ViT-B/16、ViT-L/16）已就绪，来源见 [docs/weights.md](docs/weights.md)。
- **uv 独立管理 Python 3.12.15** 的 `.venv` 已安装，`uv.lock` 已生成，`uv pip check` 通过；不借用外部 Conda 解释器。
- 九个LuSNAR原始包已完整下载，共52,301,555,752字节；已整理8002对训练RGB/标签与5004张测试RGB。资产位于 `/home/disk02/wjj/moonseg`，仓库通过 `data`、`ckpts` 软链接访问。
- 已实际读取官方评价与提交要求，详见 `challenge.md`。具体场景权重尚未公布；官方训练包计划10月22日起发放，正式样本清单尚未取得。

## 环境：uv

在能联网、能访问 GPU 的宿主环境执行：

```bash
bash scripts/setup_env.sh
```

该脚本让 uv 下载并管理 Python 3.12。Python和安装缓存实际位于 `/home/disk02/wjj/moonseg/.cache`，虚拟环境位于 `/home/disk02/wjj/moonseg/.venv`；代码目录的 `.cache` 与 `.venv` 都是指向它们的软链接。可用 `MOONSEG_STORAGE_ROOT=/其他目录 bash scripts/setup_env.sh` 指定新的存储根目录。PyTorch 2.7.1 / torchvision 0.22.1 使用官方 CUDA 12.6 wheel 索引，默认面向 Linux x86_64。当前主机驱动为580.126.09，物理GPU0、7均为48GB RTX 6000 Ada。

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
  .cache/                     # uv缓存与独立Python
  .venv/                      # 运行环境
  data/
    archives/Moon_1.zip ... Moon_9.zip
    LuSNAR/Moon_1/color/       # 保留原sample_id
    LuSNAR/Moon_1/label/       # 仅场景1、3、5、6、7、8
    LuSNAR/Moon_2/color/       # 场景2、4、9只提取RGB
    official/                 # 组织方正式样本清单
    manifests/                # 派生索引
  checkpoints/
    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  runs/                       # 训练输出（last.pt/best.pt/日志），不放 NAS

本仓库/.cache -> /home/disk02/wjj/moonseg/.cache
本仓库/.venv  -> /home/disk02/wjj/moonseg/.venv
本仓库/data  -> /home/disk02/wjj/moonseg/data
本仓库/ckpts -> /home/disk02/wjj/moonseg/checkpoints
本仓库/runs  -> /home/disk02/wjj/moonseg/runs
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

官方清单到位前的开发池模式（只用公开训练场景 1/3/5/6/7/8，边界见 [docs/development_pool.md](docs/development_pool.md)）：

```bash
uv run --no-sync python tools/prepare_data.py --root data/LuSNAR --development-pool --validate-labels
uv run --no-sync python tools/label_stats.py --manifest data/manifests/dev_train.json \
  --output data/manifests/label_stats_dev_train.json
```

生成 `dev_train.json`（1/3/6/7/8，6403 张）、`dev_val.json`（5，1599 张）、`dev_val6_train.json`/`dev_val6.json`（场景 6 作确认性验证）和 `dev_final_train.json`（8002 张）。

## 训练、验证与推理

启动脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，默认 `CUDA_VISIBLE_DEVICES=0,7`，两张物理卡分别映射为进程内 `cuda:0,1`，训练和预测都由两个进程执行。该主机默认CUDA顺序与 `nvidia-smi` 不同，不能省略 `CUDA_DEVICE_ORDER`。

```bash
# 一次性：编译官方可变形注意力 CUDA 扩展（需要一张可见 GPU，约 40 秒）
bash scripts/build_ops.sh

# 主线基线：ViT-B/16 冻结，GPU 0/7，场景 5 验证
PYTHON=.venv/bin/python bash scripts/train.sh configs/dnv3_va_m2f_vitb16.yaml

# 第二配置：ViT-L/16 冻结，GPU 5/6。该卡对的 PCIe P2P 有故障，必须关闭 NCCL P2P
CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 PYTHON=.venv/bin/python \
  bash scripts/train.sh configs/dnv3_va_m2f_vitl16.yaml

# 从完整epoch边界恢复（只能用 last.pt；best.pt 仅含权重）
PYTHON=.venv/bin/python bash scripts/train.sh configs/dnv3_va_m2f_vitb16.yaml \
  --resume runs/dnv3_va_m2f_vitb16/last.pt

# 场景5本地诊断，mIoU不是排行榜分数；--weights auto 按 checkpoint 的 best_source 选 model/ema_model
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,7 uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m moonseg.evaluate \
  --config configs/dnv3_va_m2f_vitb16.yaml --checkpoint runs/dnv3_va_m2f_vitb16/best.pt

# 方案冻结且官方清单到位后，用全部官方训练场景重新训练（配置待建）
# bash scripts/train.sh configs/dnv3_va_m2f_vitb16_final.yaml

# 双卡自动预测，输出原尺寸8位单通道PNG；--weights-key auto|model|ema_model
bash scripts/predict.sh configs/dnv3_va_m2f_vitb16.yaml \
  runs/dnv3_va_m2f_vitb16/best.pt data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json

# 校验完整覆盖、尺寸、PNG模式、值域，并按官方目录打包；不执行上传
uv run --no-sync python tools/package_submission.py \
  --root data/LuSNAR --manifest data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --predictions runs/predictions --archive submissions/predictions.zip
```

默认 `train_mode: full_image`、`image_size: [1024,1024]`，保留完整视野；原图已是1024×1024时不重采样。推理使用整图、单尺度、无翻转TTA。基线增强：同步水平翻转0.5、独立亮度扰动0.5（加性±32）、独立对比度扰动0.5（乘性0.5–1.5），即 DINOv3/MMSeg PhotoMetricDistortion 的标准强度；光度操作后像素钳到渲染数据的下限 22（`pixel_floor`）。gamma、按标签合成阴影、尺度抖动、稀有类采样、部分解冻骨干都已实现但默认关闭，作为基线之后的候选，见 [增强调研](docs/augmentation_research.md) 与 [模型说明](docs/model.md)。

每卡batch=8，2卡，累积4步：`8×2×4=64`张图像参与每次优化器更新（要求不低于64，`min_effective_batch: 64`）。使用BF16、TF32、AdamW、LR 2e-4、warmup 250步 + poly 0.9、梯度裁剪0.1（记录裁剪前范数）、EMA 0.999。80 epoch及优化参数是本项目选择，不声称来自LuSNAR论文或已验证精度最优；LR 2e-4 对应 Mask2Former 原配方 batch 16/1e-4 的平方根缩放。

训练只进行完整64张的累积更新，epoch尾部不完整组跳过；每轮重新打乱，并在 `batch_plan.json` 记录使用/跳过数量：6403张每轮使用6400张、跳过3张，共100次更新；8002张使用8000张、跳过2张。梯度在第4次反向传播同步，再统一裁剪和更新。

GPU 实测（合成输入，CUDA 扩展，激活检查点开）：每卡 batch 7 峰值 allocated 30.85 GiB、reserved 35.37 GiB，双卡 10.8 img/s；batch 10 为 43.67/46.65 GiB。batch 8 的长跑显存尚未单独实测。A6000（49140 MiB 型号）可放 batch 9。详见 [docs/audit_2026-10-21/gpu_capacity.md](docs/audit_2026-10-21/gpu_capacity.md)。

GPU 配对：PCI 5 与 6 之间的 PCIe P2P 拷贝会静默丢数据，默认 NCCL 会在初始化时挂死；设置 `NCCL_P2P_DISABLE=1` 后 all-reduce 正确，6.9 GB/s。0/7、5/7、6/7 默认正常。

不使用梯度累积时的独立容量测试见 [48GB/24GB容量报告](docs/memory_capacity.md)：48GB实卡batch7通过、8 OOM；24GiB预算模拟batch3通过、4 OOM。模拟不等同于真实24GB显卡测试。所有吞吐/容量数据来自短程合成输入，不含真实读图、增强及长跑变化，也不代表精度。

checkpoint：每轮覆盖 `last.pt`（模型可训练张量、EMA、优化器、调度器、RNG），验证提升时覆盖 `best.pt`（仅 `model`/`ema_model` 及元数据，不能用于 `--resume`），不写周期快照（`save_every: 0`）。冻结的 DINOv3 张量不保存，加载时从 `model.backbone_weights` 重建，因此复现包必须同时提供该预训练文件。LuSNAR原始图像/标签为1024×1024，但公开v3未给出训练crop；当前整图设定来自用户确认。`ckpts`中的safetensors和pth是同一个DINOv3模型的不同格式，见 [权重说明](docs/weights.md)。

预测ZIP只包含 `test/Moon_2/submission/*.png`、`test/Moon_4/submission/*.png`、`test/Moon_9/submission/*.png`。mIoU按场景累积混淆矩阵，并只平均参考标签中出现的类别；参考标签缺类不能按预测结果扩张平均类别集合。现在同时输出mAcc、每类accuracy/recall、每类GT支持数及pixel accuracy，定义见 [指标说明](docs/metrics.md)；这些补充指标不替代比赛主指标。完整评分与一次提交限制见 `challenge.md`。仓库同时提供顶层 `train.py`、`predict.py` 作为复现入口。

## Git

使用标准 `.git` 管理。初始化脚本设置本仓库身份为 `jiajun <jiajun9264@gmail.com>`：

```bash
bash scripts/init_git.sh
git status
```

仓库已通过标准 `git init` 初始化，身份已设置。直接使用普通 `git` 命令，数据、权重、环境与训练输出均由 `.gitignore` 排除。

## 验证边界

uv环境及依赖检查已通过；ModelScope真实权重已严格加载。当前完整1024×1024模型已完成单卡容量边界检查和双卡有效batch70的5次AdamW更新，所有损失与裁剪前梯度范数有限，冻结骨干无梯度。完整视野与增强几何同步、mAcc/缺类/全ignore等CPU检查通过，报告在 `/home/disk02/wjj/moonseg/verification/full_image_data_metrics.json`。

此前有效batch64的完整1024 CLI集成检查覆盖两轮训练、尾部处理、恢复、mAcc、整图预测及ZIP，历史记录在 `/home/disk02/wjj/moonseg/verification/full1024_integration.json`。本次另验证了当前70的真实累积更新、尾部样本数及64配置被拒绝；不将旧报告冒称为70的全流程重跑。恢复完整epoch状态不保证CUDA逐位确定性。

上述运行仅使用临时合成数据或少量训练场景帧，未进行正式比赛训练或报告比赛精度；未添加unittest或仓库测试文件。2026-10-21 曾误启动一次 ViT-B 基线约 3 分钟后终止，无 checkpoint，日志在 `runs/aborted_2026-10-21_baseline_vitb16_frozen_3min/`。正式训练前仍需取得组织方样本清单，按官方范围准备训练与验证索引。
