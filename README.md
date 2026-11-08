# MoonSeg：DINOv3 + ViT-Adapter + Mask2Former

LuSNAR 月面五类语义分割。类别编号固定为月壤 0、岩石 1、撞击坑 2、山体 3、天空 4；输入仅左相机 RGB。比赛规则见 [challenge.md](challenge.md)。

## 当前状态（2026-11-08）

代码与Git的本地工作副本现位于 `/home/disk02/wjj/moonseg/code`，用于避开NAS读写阻塞完成开发、验收和推送。原A/B训练进程仍以 `/home/nas_1/wjj/moon` 为工作目录；DA3冻结运行从本地副本启动，但已启动训练保存的旧解释器路径引用仍保留。后续任务统一使用修正后的本地Python、源码、权重、数据、输出和可变形注意力扩展；现有训练暂不停止或重启。

- 默认配方：冻结 DINOv3 ViT-L + ViT-Adapter + Mask2Former，1024×1024 RRC，decoder6层/hidden256，梯度裁剪 **0.01**，每卡8×两卡×累积4=有效batch64。后续从 [默认模板](configs/dnv3_va_m2f_vitl16_default.yaml) 派生独立实验，历史配置保持原值；详见 [训练说明](docs/training.md)。
- 权重选择：当前场景5独立验证，可按验证mIoU或验证loss选best；后续场景5并入训练、无独立验证集时用last。`best_loss.pt`指最低验证loss，不按训练loss选模型；10-29的末轮对照结果按原口径保留，新三项训练的验证best尚未做公开测试。 验证best仅为待测候选，收益判断和后续微调起点须有对应公开测试结果；clip001已测last与尚待测best不能混称“最佳模型”。
- clip0.01、decoder9、width384及各自末轮评估均已完成。width384的场景9 TTA87.892%低于同clip0.01的width256基线88.021%，三个公开场景mIoU均下降，默认保持6层/width256/clip0.01。以上为10-29已完成实验；11-08已启动下述A/B，12/18轮搜索继续暂缓，数据划分保持原状。状态与结果见 [路线记录](docs/roadmap.md)。
- 当前训练：B在PCI0/7从原DINOv3预训练联合训练全部ViT-L，25轮/2500更新、Adapter/head与骨干lr为2e-4/2e-5；A在PCI4/5从clip001第15轮验证best初始化后全解冻，5轮/500更新、新优化器、lr为2e-5/2e-6。两者每卡4×2卡×累积8=64；父权重的best_miou与A/B的best_miou、last公开单尺度/TTA使用原评测队列，待同两组卡释放。计划/状态在`runs/plans/2026-11-08/fullft/`，勿重复启动。
- DA3首组已纠正为冻结骨干：`da3mono_va_m2f_vitl14_rrc_frozen`从disk02在PCI2/3启动，严格对齐冻结clip001的每卡8×2卡×累积4、25轮/2500更新与全部训练设置，仅换官方DA3MONO encoder及patch14→stride16特征接口。独立计划在`runs/plans/2026-11-08/da3mono_frozen/`，完成后在原独立队列评best_miou与last的5004张single/narrowflip：best对比clip001验证best，25轮last对比已有clip001末轮结果。误启动的DA3 fullft因路线纠正停止，全部产物保留且旧队列禁用；详见[训练说明](docs/training.md#gpu-放置与启动)。
- 数据：公开版九个ZIP与官方训练/测试RAR均已下载。官方包位于 `data/LuSNAR_official`，8002张训练RGB、8002张训练标签、5004张测试RGB共21,008个PNG与公开版逐字节一致；无新增、缺失或内容变化，详见 [数据核对与正式清单](docs/data.md#官方数据一致性核对)。
- 正式样本清单已从官方包内时间戳文本转录到 `data/LuSNAR_official/`，四份训练/验证/全训练池/测试清单已生成，8002张训练标签全部通过校验。用户已确认沿用现有数据路径与训练/验证划分，已有训练和测试结论继续有效，无需因数据来源差异重训或重复评估；官方场景权重数值尚未取得。
- 环境与资产：uv 独立管理 Python 3.12.15 与 `.venv`（`uv.lock` 已生成，`uv pip check` 通过）；固定版本 DINOv3 源码与 ModelScope ViT-B/16、ViT-L/16 权重已就绪；官方 MSDeformAttn CUDA 扩展已编译（`build/ms_deform_attn/`）。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/training.md](docs/training.md) | 当前配方与每项理由、batch 计划、验证协议、GPU 放置、墙钟时间、参考配方与文献证据 |
| [docs/data.md](docs/data.md) | 数据实测（格式与调色板、像素下限、类别分布、光照、标注惯例、目标尺寸、时间冗余、验证场景）、开发池边界、清单工具 |
| [docs/handover.md](docs/handover.md) | 交接文档：现状、实验结果、保守消融阶梯、大胆创新设计、决策规则、操作手册与流程规则 |
| [docs/model.md](docs/model.md) | 模型结构、损失、兼容性改动、CUDA 扩展、采样/解冻/验证抽样选项、验证损失、checkpoint 格式 |
| [docs/augmentation_research.md](docs/augmentation_research.md) | 增强默认值与可选算子的精确语义、源码依据、文献 |
| [docs/gpu_topology.md](docs/gpu_topology.md) | 主机 GPU 拓扑、P2P 故障与启动约定、显存与吞吐容量 |
| [docs/metrics.md](docs/metrics.md) | 评测指标定义 |
| [docs/paper_alignment.md](docs/paper_alignment.md) | LuSNAR 论文事实、公开结果与本项目选择的边界 |
| [docs/weights.md](docs/weights.md) | 预训练权重来源、转换与许可 |
| [docs/rules_sources.md](docs/rules_sources.md) | 比赛规则的取证来源 |
| [docs/roadmap.md](docs/roadmap.md) | 正在跑的实验、候选路线、最终流程、红线、实验记录 |
| [docs/evidence/](docs/evidence/) | 数据统计与 GPU 基准的原始 JSON |
| [docs/external_datasets/README.md](docs/external_datasets/README.md) | 五组月面外部数据的调研（未来独立研究，不用于本比赛） |

## 环境：uv

在能联网、能访问 GPU 的宿主环境执行：

```bash
bash scripts/setup_env.sh
```

该脚本让 uv 下载并管理 Python 3.12。Python 和安装缓存实际位于 `/home/disk02/wjj/moonseg/.cache`，虚拟环境位于 `/home/disk02/wjj/moonseg/.venv`；代码目录的 `.cache` 与 `.venv` 都是指向它们的软链接。可用 `MOONSEG_STORAGE_ROOT=/其他目录 bash scripts/setup_env.sh` 指定新的存储根目录。PyTorch 2.7.1 / torchvision 0.22.1 使用官方 CUDA 12.6 wheel 索引，默认面向 Linux x86_64；本机 GPU 见 [docs/gpu_topology.md](docs/gpu_topology.md)。

`uv.lock` 随代码保留。迁移时复制代码及锁文件，重新运行安装脚本（内部使用 `uv sync --locked`）；不复制 `.venv`，也不依赖其他人的解释器路径。数据和模型存储位置通过下载脚本的 `--storage` 设置。训练、预测脚本直接使用本地 `.venv/bin/python`，也可用 `PYTHON` 指定同一环境的解释器；环境安装完成后不在运行入口隐式同步依赖。

## 下载与存储

下载默认使用 `aria2c`，也提供 `--dataset-engine curl` 分块下载方式。每文件 8 连接、同时 3 文件，保留已完成部分供续传；数据请求使用 `https://hf-mirror.com`。只检查远端字节数、HTTP 范围和数据格式，不另做哈希校验。

```bash
# 无网络、无写入地查看配置
python3 tools/download_assets.py --plan

# 获取固定版本官方源码；已完成时跳过
bash scripts/bootstrap.sh --only source

# 下载 9 个原始包（约 52.3 GB 压缩），整理左相机 RGB/训练标签并建立链接
bash scripts/bootstrap.sh --only data --extract

# 固定版本的 ModelScope 来源；下载后严格转换为官方原生 state_dict（ViT-L 用 --backbone vitl16）
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
  runs/                       # 训练输出（last.pt/best_miou.pt/best_loss.pt/日志），不放 NAS

本仓库/.cache -> /home/disk02/wjj/moonseg/.cache
本仓库/.venv  -> /home/disk02/wjj/moonseg/.venv
本仓库/data  -> /home/disk02/wjj/moonseg/data
本仓库/ckpts -> /home/disk02/wjj/moonseg/checkpoints
本仓库/runs  -> /home/disk02/wjj/moonseg/runs
```

原始 ZIP 的目录结构、整理规则和实际数量见 [docs/data.md](docs/data.md#文件格式与标签解码)。

本机系统 DNS 曾将 CAS 下载域名解析到错误证书端点。curl 方式支持 `--dns-resolver https://dns.alidns.com/resolve` 或 `--resolve-host HOST:443:IP`，仅影响当前下载进程，并保留 TLS 证书校验；不要通过关闭证书校验掩盖这类故障。分块下载复用 HF Mirror 的重定向地址，避免每个并发分块重复访问镜像而触发限流。权重来源、Meta 原生 URL 选项与转换边界见 [docs/weights.md](docs/weights.md)。

## 数据清单

```bash
# 只扫描公开原始包：生成 exploration_*.json，不授权用于训练
uv run --no-sync python tools/prepare_data.py --root data/LuSNAR

# 组织方清单到位后：生成 train.json（1/3/6/7/8）、val.json（5）、final_train.json（1/3/5/6/7/8）和不含标签的 test.json
uv run --no-sync python tools/prepare_data.py \
  --root data/LuSNAR \
  --official-allowlist data/official/train_allowlist.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --validate-labels

# 官方清单到位前的开发池：生成 dev_train/dev_val/dev_val6_train/dev_val6/dev_final_train.json，再统计类别像素
uv run --no-sync python tools/prepare_data.py --root data/LuSNAR --development-pool --validate-labels
uv run --no-sync python tools/label_stats.py --manifest data/manifests/dev_train.json --development-pool
```

清单格式、校验规则、开发池边界和官方数据一致性核对的步骤见 [docs/data.md](docs/data.md)。

## 训练、验证与推理

后续训练、验证、公开测试队列和预测均从本地完整代码副本启动。本机先 `cd /home/disk02/wjj/moonseg/code`；其他机器使用自己的本地代码目录，并通过 `MOONSEG_STORAGE_ROOT` 安装环境及建立本地资产链接。仅将 `data/runs/.venv` 链到本地盘不足以避开NAS：代码、`third_party`、CUDA扩展、Python解释器本体及标准库也必须位于本地。旧NAS副本的入口未更新，不会自动转发，不能继续从那里启动。已在运行的训练保留原进程和目录，路径修复不改变其已保存的运行环境。

`train.sh`、`predict.sh`和四个计划/队列入口会先切到自身代码目录的物理路径；子进程使用物理虚拟环境目录下的 `bin/python`，保留venv识别。数据加载和训练/预测输出在入口解析软链一次，配置中的原路径及清单记录保持原值，恢复配置比较、数据来源和采样不变。直接执行 `python -m moonseg.evaluate`、`moonseg.evaluate_public_test` 或顶层 `train.py/predict.py` 时，也须遵守上述本地目录和解释器约定；自定义 `PYTHON`、源码或扩展路径同样须指向本地。

启动脚本固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，默认 `CUDA_VISIBLE_DEVICES=0,7`，两张物理卡映射为进程内 `cuda:0,1`，训练和预测都由两个进程执行。该主机默认 CUDA 顺序与 `nvidia-smi` 不同，不能省略 `CUDA_DEVICE_ORDER`。

```bash
# 一次性：编译官方可变形注意力 CUDA 扩展（需要一张可见 GPU，约 40 秒）
bash scripts/build_ops.sh

# ViT-L/16 默认模板：PCI 0/7，场景5验证；新实验须复制配置并设置唯一output_dir
PYTHON=.venv/bin/python bash scripts/train.sh configs/dnv3_va_m2f_vitl16_default.yaml

# ViT-B/16 基线：PCI 5/6。该卡对的 PCIe P2P 有故障，必须关闭 NCCL P2P
CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 PYTHON=.venv/bin/python \
  bash scripts/train.sh configs/dnv3_va_m2f_vitb16.yaml

# 从完整 epoch 边界恢复（只能用 last.pt，且须用原运行的卡对与环境变量）
CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 PYTHON=.venv/bin/python \
  bash scripts/train.sh configs/dnv3_va_m2f_vitb16.yaml --resume runs/dnv3_va_m2f_vitb16/last.pt

# 场景 5 全量本地评估（不是排行榜分数），同时输出验证损失；--weights auto 即 model（旧 checkpoint 的 ema_model 需显式指定）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m moonseg.evaluate \
  --config configs/dnv3_va_m2f_vitl16.yaml --checkpoint runs/dnv3_va_m2f_vitl16/best_miou.pt

# 双卡预测，输出原尺寸 8 位单通道 PNG；--weights-key auto|model|ema_model
bash scripts/predict.sh configs/dnv3_va_m2f_vitb16.yaml \
  runs/dnv3_va_m2f_vitl16/best_miou.pt data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json

# 校验完整覆盖、尺寸、PNG 模式、值域，并按官方目录打包；不执行上传
uv run --no-sync python tools/package_submission.py \
  --root data/LuSNAR --manifest data/manifests/test.json \
  --official-test-allowlist data/official/test_allowlist.json \
  --predictions runs/predictions --archive submissions/predictions.zip
```

推理默认整图、单尺度、无翻转 TTA，输出恢复原输入尺寸。预测 ZIP 只包含 `test/Moon_2/submission/*.png`、`test/Moon_4/submission/*.png`、`test/Moon_9/submission/*.png`；完整评分口径与一次提交限制见 [challenge.md](challenge.md)，本地指标定义见 [docs/metrics.md](docs/metrics.md)。新训练保存 `last.pt`（完整恢复状态）与 `best_miou.pt`、`best_loss.pt`（仅权重，不能用于 `--resume`）；旧 B/L 的最佳文件名为 `best.pt`。它们不含冻结的 DINOv3 张量，评估、预测和复现包都需要 `model.backbone_weights` 指向的预训练文件（[docs/model.md](docs/model.md#checkpoints)）。最终在全部训练场景上重训的流程见 [docs/roadmap.md](docs/roadmap.md#最终流程方案冻结后)。仓库同时提供顶层 `train.py`、`predict.py` 作为复现入口。

按用户 2026-10-23 的明确要求，公开 LuSNAR 场景 2/4/9 的 5004 张图另用 `moonseg.evaluate_public_test` 做离线研究评估。标签位于 `data/research_test/`，不进入训练加载器；权重分别按场景5验证选择best_miou及固定预算取last，固定评估单尺度和窄尺度TTA，按相同选择口径比较。这不是比赛正式成绩。11-08当前后台队列为 `scripts/run_public_test_queue.py`，计划在 `runs/plans/2026-11-08/fullft/public_test.json`，可用 `--dry-run` 检查；已启动实例不要重复运行。父权重best及A/B的best+last评估等待PCI0/7或4/5释放，DINO队列共10项；DA3原独立队列共4项，每队列最多2个单卡任务。clip001的25轮last直接复用[既有单尺度证据](docs/evidence/public_development_last_20261029/rrc_clip001_single.json)与[TTA证据](docs/evidence/public_development_last_20261029/rrc_clip001_narrowflip.json)，不重复评估。A的last为从第15轮best继续微调5轮的阶段末轮，仅作不同预算的阶段诊断。DINO结果在 `runs/public_development_fullft_2026-11-08/aggregate.json`。10-23历史计划 `runs/plans/2026-10-23/vitl_priority/public_test.json` 及结果 `runs/public_test_research_2026-10-23/aggregate.json` 保留追溯；指标与数据口径见 [docs/metrics.md](docs/metrics.md) 和 [docs/data.md](docs/data.md#开发清单)。当前训练与队列状态见 [docs/handover.md](docs/handover.md#4-正在跑与排队2026-11-08)。

## Git

使用标准 `.git` 管理，本仓库身份为 `jiajun <jiajun9264@gmail.com>`（`bash scripts/init_git.sh` 设置）。直接使用普通 `git` 命令；数据、权重、环境、编译产物与训练输出均由 `.gitignore` 排除。
