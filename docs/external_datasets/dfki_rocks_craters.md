# DFKI Lunar Rocks and Craters

调研日期：2026-10-21。Grok 读取 Zenodo 元数据、两个 ZIP 的中央目录及少量包内说明/标注文本；主代理复查这些原始响应，并独立重算文件数量与源编号交集。未下载整包影像。

## 身份和采集域

准确名称为 *Lunar Rocks and Craters Dataset for Instance Segmentation (YOLO and COCO Format)*，Siddhant Shete 于 2025-01-16 发布，机构为 DFKI Robotics Innovation Center，贡献者为 Hilmi Dogu Kücüker。固定记录 DOI：**10.5281/zenodo.14673237**。[官方存档](https://zenodo.org/records/14673237)、[记录 API](https://zenodo.org/api/records/14673237)。

记录称数据在 DFKI 机器人创新中心场地采集，应作为地面机构场地数据理解，不能称为真实月面遥测。没有取得具体相机、原始分辨率、帧率、场景/轨迹表及场地内部实拍和合成内容的逐项来源。Grok 查到后续论文曾将此 DOI 与更大规模的实拍/合成训练池关联，但该训练池数量不等于这里公开的压缩包，不能直接移用。[后续部署论文](https://arxiv.org/abs/2609.02219)

未核到作者官方 Kaggle 或 Hugging Face 版本；应从 Zenodo 固定记录获取。包内标明 Roboflow 项目 `dsad-omghq/combined-mgeca`，但 Grok 访问 Roboflow 页面遇到 403，未把它认定为已验证的第二下载源。

## 两个包不只是格式转换

| 包 | 压缩字节数 | train | valid | test | JPEG 总数 |
|---|---:|---:|---:|---:|---:|
| `Dataset_yolo.zip` | 186,819,579 | 7,734 | 313 | 307 | **8,354** |
| `Dataset_coco.zip` | 72,191,694 | 2,578 | 313 | 307 | **3,198** |

两个包**都含图像和标注**。Zenodo 页面写的 7,734/313/307 对应 YOLO 包，不应套到 COCO 包。主代理从 Grok 保存的中央目录重算，数量一致。两包 valid/test 文件名相同；COCO 训练图包含在 YOLO 训练图中。YOLO 训练图数量恰为 COCO 的三倍，但未取得完整增强配方，不能据数量关系断言具体用了哪些增强。

```text
Dataset_yolo/
  data.yaml
  train/images/*.jpg        train/labels/*.txt
  valid/images/*.jpg        valid/labels/*.txt
  test/images/*.jpg         test/labels/*.txt
  # 还包含 labels.cache，不是新的标注样本

Dataset_coco/
  README.dataset.txt
  README.roboflow.txt
  train/*.jpg               train/_annotations.coco.json
  valid/*.jpg               valid/_annotations.coco.json
  test/*.jpg                test/_annotations.coco.json
```

包内 README 记录自动校正图像方向、**拉伸至 640×640**。COCO test JSON 的 307 张全部声明 640×640；本次未解码所有 JPEG，因此不将元数据尺寸说成逐图实测。没有深度、LiDAR 或语义 PNG 标签。COCO 包的导出说明称该次未应用增强，与大量同源编号副本的形成过程仍未完全解释，保留为待查项。

## 多边形、类别 ID 与五类映射

这是**实例分割**数据。Grok 抽查的 YOLO 行为类别号加多边形顶点，不是只有五个数的检测框；COCO test 的 1,022 个实例均有 `segmentation`、`bbox`、`area` 和 `iscrowd`。主代理读取了这次 Range 请求的输出记录，确认以下编码：

| 格式 | 类别对应 | 注意 |
|---|---|---|
| YOLO `data.yaml` | 0=crater，1=rock | `nc=2`；多边形坐标归一化 |
| COCO test | 1=crater，2=rock | `categories` 还出现一个 id=0、name=rock 的条目，该 split 实例没有使用它 |

COCO test 中 143 个 crater、879 个 rock；train/valid 的类别使用尚未完整解析。不得盲目把 COCO 类别号减一，也不能只依据 `supercategory` 把 crater 并入 rock。

对本项目可考虑把核实后的 rock 轮廓映射到岩石 1、crater 轮廓映射到撞击坑 2；栅格化之前仍需看图判断坑的标注边界，以及重叠实例的覆盖顺序。未标注背景没有月壤、山体、天空的语义定义，**不能整片填成月壤 0**。五类定量评测需要补标或显式忽略无法对应的区域。

## 文件名重复与划分问题

去掉 Roboflow 导出后缀后，两包均出现 `rgb_0` 到 `rgb_735`，共 **736 个源编号候选**。主代理重算这些编号在 split 中的交集：

| 交集 | 重复编号数 |
|---|---:|
| train ∩ valid | 259 |
| train ∩ test | 249 |
| valid ∩ test | 75 |
| train ∩ valid ∩ test | 63 |

这些数字说明公开划分存在明显的同源/同名检查需求。**736 不等于已经证明只有 736 张独立原图**：合并来源也可能发生编号碰撞，本次没有通过像素和采集记录排除它。原始 Grok 报告把它直接称为严重泄漏，本报告采用更精确的结论：在原始出处和图像内容核清前，不将这些 split 当作独立样本评估。

未来至少按源编号候选分组，并进一步用图像相似性、原始路径和场景记录核验，防止同一照片的裁切、增强或重导出跨 split。仅按编号重新划分仍不能自动保证场景独立。

## 下载和许可

Zenodo API 明确 `access_right=open`、`license.id=cc-by-4.0`；包内 README、YOLO YAML 和 COCO licenses 也声明 **CC BY 4.0**。网页渲染的 License 区为空不能据此说许可未知；这里依据机器元数据和包内文本。使用/再分发应保留作者、数据标题、DOI、许可链接及必要的修改说明。[许可说明](https://creativecommons.org/licenses/by/4.0/)

已核实匿名获取入口和 HTTP Range。未来手工下载可先选择较小的 COCO 包进行配对及标注边界检查：

```bash
mkdir -p /home/disk02/wjj/moonseg/external_data/dfki_rocks_craters
curl --fail --location --retry 3 --connect-timeout 20 \
  --output /home/disk02/wjj/moonseg/external_data/dfki_rocks_craters/Dataset_coco.zip \
  'https://zenodo.org/api/records/14673237/files/Dataset_coco.zip/content'

# 需要核对 YOLO 版本时另行获取，不把它当作不同的独立数据来源
curl --fail --location --retry 3 --connect-timeout 20 \
  --output /home/disk02/wjj/moonseg/external_data/dfki_rocks_craters/Dataset_yolo.zip \
  'https://zenodo.org/api/records/14673237/files/Dataset_yolo.zip/content'
```

本次未执行整包下载。完整文件清单入口为 [Zenodo files API](https://zenodo.org/api/records/14673237/files)。没有取得该数据集随附的官方训练代码或其软件许可，不能将 CC BY 4.0 当成未找到的代码许可证。

## 未来独立研究建议

获取体积小，rock/crater 多边形比较接近本项目关心的两类，适合优先做下载与格式核对。接入成本主要是原图去重、来源划分、类别重映射和未标注区域处理，不能因有 polygon 就直接合并成五类训练集。实验室域、640 拉伸及未报告相机条件也需要单独记录。

建议 manifest 保留 `zenodo_record, archive_name, export_version, original_filename, source_stem_candidate, split, annotation_format, class_schema, preprocessing, license, download_bytes, downloaded`。未来资产根目录为 `/home/disk02/wjj/moonseg/external_data/dfki_rocks_craters/`；本次仅形成调研和下载入口。
