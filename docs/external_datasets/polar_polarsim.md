# POLAR / POLAR-Sim

调研日期：2026-10-21。Grok 独立调研；主代理复核 NASA 文档、POLAR-Sim 仓库完整树、类别转换代码和 Dryad 文件清单。

## 两者的关系

**NASA POLAR Stereo 提供实验室实拍影像和几何信息；POLAR-Sim 为这些影像补充标注、地形网格及渲染示例。** 不能把两者的图像数相加当成两个独立数据集。

POLAR 全称 Polar Optical Lunar Analog Reconstruction，NASA Ames 在地球实验室沙盘中模拟月球极区的低角度、强阴影光照；不是实际月面照片，也不是月球轨道遥感图。POLAR-Sim 由 University of Wisconsin–Madison 的 Simulation-Based Engineering Lab 发布，论文 *POLAR-Sim: Augmenting NASA's POLAR Dataset for Data-Driven Lunar Perception and Rover Simulation*，arXiv:2309.12397；数据 DOI 为 `10.5061/dryad.ksn02v7hf`。[NASA 项目页](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/)、[POLAR-Sim 仓库](https://github.com/uwsbel/POLAR-Sim)、[论文](https://arxiv.org/abs/2309.12397)、[Dryad](https://doi.org/10.5061/dryad.ksn02v7hf)。

旧名 POLAR3D 出现在仓库引用信息中，指同一项扩展工作。POLAR Traverse 是另外的视觉里程计数据，见末节，不混入本专题数量。

## 原始影像与规模

| 字段 | 已核实内容 |
|---|---|
| 场景 | 12 个主要沙盘试验，另有 Terrain 13 高地模拟物附加场景；后者校准较弱 |
| 数量口径 | NASA 写超过 2,500 对 HDR 立体像对；POLAR-Sim 论文约 2,600 对。未解包精确清点，不能写成精确的独立场景数 |
| 原始图像 | 宽 1936 × 高 1216，单色 PNG，12 位有效值存放在 16 位容器中 |
| 曝光 | 静态场景的多档曝光，左右相机、太阳方位、车灯状态和 A/B/C 机位；不能把“HDR”理解为每张都是已融合 RGB |
| 几何 | 立体标定、LiDAR 点云、可选整理点云/距离图；不是语义类别标签 |
| 全包 | `dataset_public_release.zip`；Grok HEAD 记录 **11,738,549,533 字节**，未整包下载 |
| 划分 | 未取得官方固定 train/val/test 文件；论文实验切分不等于本次已取得对应清单 |

图像和传感器细节来自 [NASA User Documentation V1.4](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_Readme.pdf)。场景在数米尺度沙盘构造，室内背景、光源距离、模拟月壤和实际月面有差异，不具有真实远山/天空。文档将高地/月海反照率分别列为约 30%/12%；原始 Grok 报告有一处写反，本报告采用 NASA 原文。

## POLAR-Sim 标注及编码

| 资产 | 路径 | 编码 |
|---|---|---|
| 语义多边形 | `SegmentationLabels/terrainXX/labels/*.txt` | YOLO polygon：类别号后接归一化顶点；0 background、1 ground、2 rock、3 shadow |
| 检测框 | `Labels/TerrainXX/*.txt` | YOLO bbox；0 rock、1 shadow，类别顺序与分割不同 |
| 网格 | `TerrainMeshes/` 及外部网格包 | 独立岩石与地面 OBJ；不是逐图语义 mask |
| 转换脚本 | `yolo2bitmap.py` | 输出灰度值：背景 0、地面 86、岩石 128、阴影 43 |

来源：[Dryad 标注规范](https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7hf)、[转换脚本](https://github.com/uwsbel/POLAR-Sim/blob/main/yolo2bitmap.py)。**脚本输出的 86/128/43 是显示灰度值，不能直接当类别 ID。** 后绘制多边形会覆盖先绘制区域，后续转换应检查重叠顺序及未覆盖像素。仓库 README 的分割段落误用了 bounding box 字样，但转换代码确实按多边形顶点读取。

虽然 Terrain 11/12 包含新鲜/侵蚀的坑形，**当前分割 schema 没有 crater 类**。不能只依据论文摘要里的 “craters” 判断已经有坑的像素标签。地面和阴影覆盖了坑区域，仍需单独标注才能评测本项目的撞击坑类。

## 仓库实际计数与异常

主代理完整读取 GitHub 树；本次树 revision 为 `593ed65c901bb53a05dc0a7363f5192d1171b85d`，`truncated=false`：

- 分割：**3,977 个 `.txt`**，Terrain 01–13 均有目录；这不是 3,977 个独立场景，也未证明每份都能正确配到原图。
- 检测：共有 3,985 个 `.txt`，其中 **12 个是 `classes.txt`**，所以候选逐图框文件为 **3,973**，尚未扣除 `copy` / `good copy` 等副本。原始 Grok 报告的 3,985 不能直接当标注照片数。
- `SegmentationLabels/terrain11/labels/09_A_off_no_R_0032.txt` 的文件名前缀与目录地形号不符；本次只核对名字，未判断内容是否误放。
- 网格目录包含 Terrain 01–12；部分大地面网格不在 GitHub，不能把该目录当成完整网格集。

来源：[完整树 API](https://api.github.com/repos/uwsbel/POLAR-Sim/git/trees/main?recursive=1)，原始 API 响应已保存在外置研究记录目录。上述 revision 是 Git 树标识，不是对数据文件做的内容校验。

## 五类任务如何使用

| 本项目类别 | 可用信息 | 主要限制 |
|---|---|---|
| 月壤 0 | ground 可作地面参考 | ground 同时包含坑内/坑沿等几何；被阴影覆盖的像素另标为 shadow |
| 岩石 1 | 分割 rock / 检测 rock | 可研究岩石识别，需检查小岩石与遮挡的标注范围 |
| 撞击坑 2 | Terrain 11/12 的场景几何 | 没有现成 crater 像素类 |
| 山体 3 | 无单独类别 | 小沙盘起伏不能替代远处山体 |
| 天空 4 | 无单独类别 | 实验室 background 不能映射为天空 |

shadow 是照明现象，不能一律改成月壤或撞击坑。灰度图复制成三通道只是满足网络输入形状，不会补回 RGB 信息。适合未来独立研究中的岩石/阴影分离、曝光鲁棒性和可控光照实验；这是用途判断，尚未做模型效果验证。

## 下载入口与体积选择

原图可从 NASA [完整包](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/dataset_public_release.zip) 或按地形下载。Grok 对全包及 Terrain 01/11/13 等链接取得 HTTP 200。未来先取一个地形更容易核对命名和标注配对：

```bash
mkdir -p /home/disk02/wjj/moonseg/external_data/polar
wget --continue --directory-prefix=/home/disk02/wjj/moonseg/external_data/polar \
  'https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/Terrain01_LargeOnly.zip'
```

Terrain 01 的 HEAD 字节数由 Grok 记录为 748,321,473。本次未执行上述整包下载。NASA 项目页还提供 [地形 11](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/Terrain11_FreshCrater.zip)、[地形 13](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/Terrain13_LHT.zip) 和 [辐射定标](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/RadiometricCalibrationData.zip) 等入口。

标注与网格优先使用作者 [GitHub](https://github.com/uwsbel/POLAR-Sim) / [Dryad](https://doi.org/10.5061/dryad.ksn02v7hf)。Dryad 文件总量约 **6.87 GB**，其中 **Videos.zip 约 6.12 GB**；做图像分割不必同时取渲染演示视频和全部网格。Dryad README 已说明 Videos.zip 是多视角、光照、BRDF 和曝光设置下的仿真视频，不能据此声称有与 NASA 实拍并列的、完整逐帧带标签的合成图像训练集。

Grok 对 Dryad `file_stream` 请求遇到 403，数据集页面和清单可读取，**未验证直链成功下载**；[linkset.json](https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7hf/linkset.json) 可用于取得文件清单。README 另列 [Box 地面网格入口](https://uwmadison.box.com/s/gyhe2ggwb0v40ivbt7jx4pn6temadkbe)，Grok 未取得完整文件列表。不能将页面可见写成文件已到本地。

## 许可及切分

仓库 [LICENSE](https://github.com/uwsbel/POLAR-Sim/blob/main/LICENSE) 是 **CC0-1.0**，Dryad 元数据也声明 CC0；但仓库 README 的 Copyright 段仍写 BSD-3。记录为**声明冲突**，保留两份来源，不把 README 的陈述悄悄删除。NASA 原始照片的数据专项许可在本次读取资料中未明确取得，不能自动套用扩展标注的 CC0。论文与数据 DOI 按作者要求引用。

未来划分至少按 terrain 分组。同一场景的左右目、多档曝光、太阳方位和车灯状态相互关联；若复用网格渲染，新图也共享几何。Terrain 1/2、3/4、9/10 等共享或重排岩石，11/12 是相关坑形，应额外记录场景族。先按原图标识去除副本并修正目录/文件名冲突，再谈实验划分。

建议 manifest 保留 `terrain_id, camera_position, eye, sun_azimuth, rover_light, exposure, source_frame, source_geometry, annotation_kind, class_schema, license_note, downloaded`。阴影、曝光、灰度和完整几何来源都应成为可追溯字段。

## POLAR Traverse 的区别

[POLAR Traverse](https://ti.arc.nasa.gov/dataset/PolarTrav/) 是另一个面向视觉里程计的地面模拟数据集，Grok 查到 3,960 对、2048×2048 灰度图及位姿/几何信息。**它不是 POLAR-Sim 标注对应的原始图像集合。** 本次只保留识别入口，不把它的图像数、相机参数、下载体积或划分混入以上统计。
