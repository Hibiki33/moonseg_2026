# 公开训练场景开发池（development pool）

日期：2026-10-21。本文说明 `tools/prepare_data.py --development-pool` 生成的开发清单能做什么、不能做什么。它是**显式开启**的工程开发通道，不是组织方样本清单，也不改变 [challenge.md](../challenge.md) 的任何规则。

## 为什么需要

组织方正式训练包和固定样本清单计划自2026-10-22起滚动发放，当前尚未取得。此前加载器只接受 `official_train_manifest`，因此在清单到位前无法进行任何开发训练、消融或增强/采样超参数的初步比较。开发池允许在**公开 LuSNAR 包中比赛训练池的场景**上先行开发，并保留与正式清单的硬边界。

## 边界

| 项目 | 开发池 | 正式清单 |
|---|---|---|
| 清单 `kind` | `public_development_manifest` | `official_train_manifest` |
| `official_allowlist` | `null`（不存在组织方清单） | 组织方清单路径，逐样本校验成员资格 |
| 样本来源 | 公开包中场景1、3、5、6、7、8的左相机 `image0/color` 与 `image0/label`，按公开场景纳入 | 组织方固定样本列表 |
| 加载条件 | 配置 `data.development_pool: true`；否则加载器报错 | 必须设置 `data.official_allowlist` |
| 可用于 | 方法开发：训练/本地验证、消融、增强与采样超参数初选、标签统计 | 方法开发及最终提交模型 |
| 最终提交模型 | **不可用** | 唯一来源 |

明确允许：

- 在 `dev_*.json` 上训练和本地验证，比较模型、增强（见[增强调研第二版增补](augmentation_research.md)）和采样策略；
- 用 `tools/label_stats.py --development-pool` 统计开发清单的类别像素，用于开发期的稀有类采样；
- 报告开发期结果时注明 `provenance = public_development_pool`。

明确禁止：

- 读取、解压或统计场景2、4、9的标签。整理阶段只为测试场景提取 RGB；`prepare_data.py` 生成开发清单时断言不含测试场景，`load_manifest` 对开发清单再次拒绝任何非1/3/5/6/7/8场景，`MoonDataset` 也只接受训练场景；
- 用测试场景 RGB 做训练、伪标签、统计调参或 BN 重估。推理时读取测试 RGB 生成预测不受影响；
- 把开发清单当作或改写为正式清单。规则写明“每个分区实际选取的样本列表由官方清单固定”“不得引入官方训练清单以外的数据。禁止使用未分配的 LuSNAR 图像、标签或其衍生材料”；
- 用开发池训练出的 checkpoint 提交，或用它初始化、继续训练最终模型。最终模型必须在组织方清单上从预训练骨干**重新训练**。

规则风险须如实记录：本项目此前写明“开发训练、调参、数据统计与增强均限于官方训练清单”。开发池是在清单发放前为工程开发作出的显式例外，范围限于比赛已公布的训练场景，从不涉及测试场景。如果正式清单只是这些场景中的子集，开发池中不在清单内的样本即属于“未分配图像”。这些样本可能已经影响开发期的超参数选择，所以清单到位后必须完成下文的“切换到正式清单”，并在方法论文中声明开发期使用过公开训练场景池。

## 生成与校验（实际运行结果）

```bash
uv run --no-sync python tools/prepare_data.py --root data/LuSNAR --development-pool --validate-labels
```

`--validate-labels` 与 `--development-pool` 同用时，校验全部已发现的训练场景标签（不涉及测试场景）：五类 RGB 颜色严格匹配，标签尺寸与 RGB 一致。校验由64个进程执行（`--workers`）。不带 `--development-pool` 时，仍只能与 `--official-allowlist` 同用，且只校验正式清单内的样本，官方路径行为不变。

2026-10-21 实际输出：发现训练场景8002对、测试RGB 5004张；**8002/8002个标签校验通过**，用时约19秒，打开的测试标签数为0。

| 清单 | 场景 | 样本数 | 用途 |
|---|---|---:|---|
| `dev_train.json` | 1、3、6、7、8 | 6403 | 开发训练（对应场景5验证） |
| `dev_val.json` | 5 | 1599 | 本地验证（官方建议场景） |
| `dev_val6_train.json` | 1、3、5、7、8 | 6900 | 开发训练（对应场景6验证） |
| `dev_val6.json` | 6 | 1102 | 第二本地验证 |
| `dev_final_train.json` | 1、3、5、6、7、8 | 8002 | 开发期全训练池，用于长程/容量试验；不是最终提交模型 |

场景6作为第二验证场景：LuSNAR论文场景网格中它属于“高起伏、中等丰富”，与测试场景9（高起伏、丰富）相邻（见 `docs/audit_2026-10-21/method_research.md` §0）。数据审计建议的主验证场景仍是5。两个开发验证分数都不是排行榜成绩，也不能换算成场景加权总分。

每个开发清单都写入 `kind: "public_development_manifest"`、`official_allowlist: null`、`scenes` 和说明文字：这些样本来自公开 LuSNAR 训练场景池，只用于方法开发，最终提交模型必须在组织方清单上重新训练。

## 标签统计

```bash
uv run --no-sync python tools/label_stats.py --manifest data/manifests/dev_train.json --development-pool
```

输出 `data/manifests/label_stats_<清单名>.json`，结构为 `{"kind":"label_stats","num_classes":5,"manifest":…,"samples":{sample_id:{"class_pixels":[5],"total":N}},"class_pixel_fraction":[5],"class_presence_fraction":[5]}`，另附 `provenance`、`scenes`、`class_pixels_total`。统计使用64个进程，并通过 `load_manifest` 读取，所以开发清单同样需要 `--development-pool`，正式清单同样需要 `--official-allowlist`。实际结果（类别顺序：月壤、岩石、撞击坑、山体、天空）：

| 清单 | 像素占比 % | 出现于图像的比例 % |
|---|---|---|
| `dev_train`（6403） | 68.556 / 2.193 / 1.377 / 3.709 / 24.165 | 100 / 100 / 67.0 / 97.7 / 100 |
| `dev_val6_train`（6900） | 68.804 / 2.328 / 1.305 / 3.416 / 24.145 | 100 / 100 / 61.8 / 85.9 / 100 |
| `dev_final_train`（8002） | 68.871 / 2.189 / 1.372 / 3.496 / 24.072 | 100 / 100 / 65.6 / 87.8 / 100 |

`dev_final_train` 与 [数据审计](audit_2026-10-21/data_audit.md) §1–2 的全训练池数字一致。这些统计属于开发池的衍生数据，最终训练使用的采样权重必须由正式清单重新计算（见下）。

## 配置接口

开发配置写法示例（不需要 `official_allowlist`）：

```yaml
data:
  root: data/LuSNAR
  development_pool: true            # 显式开启；默认 false
  train_manifest: data/manifests/dev_train.json
  val_manifest: data/manifests/dev_val.json
  train_scenes: [1, 3, 6, 7, 8]
  val_scenes: [5]                   # 使用 dev_val6.json 时改为 [6]，训练场景改为 [1,3,5,7,8]
```

`MoonDataset` 把 `allow_development_pool=bool(config.get("development_pool", False))` 传给 `load_manifest`，并提供 `dataset.provenance`，取值为 `"official_allowlist"` 或 `"public_development_pool"`。即使开启开发池，正式清单仍要求 `official_allowlist`。若同时给出 `official_allowlist`，开发清单内每个样本也必须属于该清单，否则报错。训练入口只在配置含 `data.official_allowlist` 时读取它，在开发池模式下打印警告，并应把 `provenance` 写入训练记录。

## 切换到正式清单

1. 将组织方清单转写到 `data/official/`，运行 `prepare_data.py --official-allowlist … --validate-labels`，生成 `train.json`、`val.json` 和 `final_train.json`。
2. 配置改回正式清单，删除 `development_pool` 或设为 `false`，设置 `data.official_allowlist`。
3. 用 `tools/label_stats.py --manifest data/manifests/final_train.json --official-allowlist …` 重新生成统计，最终训练的稀有类采样只使用这一份。
4. 若正式清单与开发池不同，用正式 `train.json`/`val.json` 复核开发期选定的超参数。
5. 最终模型在 `final_train.json` 上从预训练骨干重新训练，开发池 checkpoint 不参与。
