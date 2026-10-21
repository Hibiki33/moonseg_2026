# NASA Apollo 月面地面照片

调研日期：2026-10-21。Grok 调研 NASA、LPI、ASU 和相关档案入口；主代理复核 NASA 图库说明、LPI 扫描/全景处理说明及 ASU 使用条款。本专题准备照片来源与筛选方案，尚未建立图像集合。

## 范围与身份

这是历史影像档案，不是具有统一版本、样本数和标签的机器学习数据集。主要目标为 Apollo 11、12、14、15、16、17 六次着陆任务中，从月面或已着陆登月舱拍摄的场景照片，尤其是有地平线的近地视角。70 mm Hasselblad 档案也包含轨道、地球和舱内影像，必须逐帧筛选。Apollo 13 没有着陆。

| 入口 | 能提供什么 | 使用方式 |
|---|---|---|
| [LPI Apollo Image Atlas / 70 mm 目录](https://www.lpi.usra.edu/resources/apollo/catalog/70mm/) | 任务、胶卷、帧号、surface/orbital/other 分类 | 建立候选帧目录，不将目录缩略图作为研究图像 |
| [NASA Apollo Lunar Surface Journal 图库](https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/images11.html) | 帧级说明、任务活动、来源和部分较高分辨率图片 | 确认 EVA、站点、视角及处理版本 |
| [ASU March to the Moon](https://tothemoon.ser.asu.edu/) | 手持相机扫描档案，包含 Mercury/Gemini/Apollo | 从 Apollo 分区继续筛选；Grok 记录旧域重定向到 `tothemoon.im-ldi.com` |
| [NASA Image and Video Library](https://images.nasa.gov/) | 可检索照片与公开 JSON API | 适合用已知帧号定位资产；不是完整胶卷目录 |
| [LPI Apollo Surface Panoramas](https://www.lpi.usra.edu/resources/apollopanoramas/) | 多帧拼接全景、部分地名注释和源帧 | 记录派生关系，不能和源帧分别当作独立样本 |
| [Project Apollo Archive](https://www.flickr.com/photos/projectapolloarchive/albums/) | Kipp Teague 的第三方整理和再发布 | 逐项保留来源与许可，不当成 NASA 官方统一数据包 |

不应混入：Apollo Metric / ITEK Panoramic 轨道测绘图、返回地球后的样品实验室照片、地球训练场、近距土壤显微式立体照片。USGS 的全景和实验室样品档案也要分别判断；名称含 Apollo 并不意味着是月面地平线视角。

## 数量、格式和分辨率

[NASA 对 Apollo Image Atlas 的说明](https://data.nasa.gov/dataset/apollo-image-atlas) 写约 25,000 张，但包含轨道、月面、地球、宇航员和设备。**这不是可直接使用的地面照片数。** Grok 根据 LPI 各任务目录的 surface 分项得到约 6,514 条，但同时发现目录计数不一致；这只是未逐帧清理的候选口径，不纳入本报告的已确认规模字段。

同一曝光存在多种数字化版本，没有统一分辨率或全库体积：

- LPI 老目录浏览图约 450×450；其处理说明明确要求研究者另取高分辨率产品。它适合认图和查帧号。[LPI 扫描流程](https://www.lpi.usra.edu/resources/apollo/processing/)
- NASA ALSJ 的 Apollo 11 图库说明列出 4096×4096 扫描、约 2350×2350 的 OF300 处理版，以及更小的浏览版。[NASA 图库说明](https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/images11.html)
- ASU 页面给出 Hasselblad 彩色名义扫描 6400×6400、黑白 12800×12800；这是扫描规格，不能据此认定每一帧的该版本都已成功取得。原始扫描、处理后的 TIFF/PNG/JPEG要分别记录。[ASU 扫描规格](https://apollo.im-ldi.com/ABOUT_SCANS/index.html)

彩色和黑白胶片并存；没有统一 RGB-D、逐帧深度或分割 train/val/test。最终样本数、文件字节数及图像尺寸应由未来实际取得的文件决定，本次未批量下载。

## 标签与适配成本

本次核对的 NASA/LPI/ASU 官方档案没有现成的五类像素标签。地名标注、拍摄点坐标、照片说明和全景里的坑名都不是语义 mask。这个结论限定于上述档案，不等于宣称全网不存在第三方 Apollo 标注集。

真实画面可出现月壤、岩石、坑、远山和黑色天空，但分布随着陆点及取景变化。还常见宇航员、登月舱、月球车、工具、旗帜、电缆，以及十字丝、眩光、胶片划痕。未来标注应定义人工物体及不可判区域的处理，不能强塞进现有五类。

LPI 全景的处理说明明确记录了去除眩光、把天空像素置零和手工接缝。这类派生图可帮助理解场景，不能直接用于研究原始天空边界或光度分布。源帧与拼接全景也不能跨训练/测试划分。[全景制作说明](https://www.lpi.usra.edu/resources/apollopanoramas/about/)

## 已核实的获取入口

先用任务/胶卷/帧号定位，例如 `AS11-40-5874`。NASA API 无需专用访问密钥；Grok 已取得该帧的资产清单，并对其 JPEG 做了 HEAD 请求。以下是未来手工获取示例，本次未执行影像下载：

```bash
# 查帧号，先核对返回描述是否真是所需月面视角
curl --fail --get --connect-timeout 20 --max-time 60 \
  'https://images-api.nasa.gov/search' \
  --data-urlencode 'q=AS11-40-5874' \
  --data-urlencode 'media_type=image'

# 查看可取的版本和元数据
curl --fail --location --connect-timeout 20 --max-time 60 \
  'https://images-assets.nasa.gov/image/as11-40-5874/collection.json'

# 一张已确认入口的照片；先查看版权/处理信息，再决定是否执行
mkdir -p /home/disk02/wjj/moonseg/external_data/apollo
curl --fail --location --retry 3 --connect-timeout 20 \
  --output /home/disk02/wjj/moonseg/external_data/apollo/as11-40-5874.jpg \
  'https://images-assets.nasa.gov/image/as11-40-5874/as11-40-5874~orig.jpg'
```

`~orig.jpg` 是 NASA 图库内该条目的原始上传资产命名，**不能据文件名推断它是未经处理的胶片原始扫描**。该样例 Grok 记录为 2,236,993 字节、3922×3882，属于具体版本的元数据；主代理未下载该 JPEG 复测。NASA API 文档见 [官方接口 PDF](https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf)。

LPI 单帧可从 [AS11-40-5874 页面](https://www.lpi.usra.edu/resources/apollo/frame/?AS11-40-5874) 进入；ASU 整库最高分辨率文件清单、USGS 全景直接文件入口和所有 Flickr 原图地址未在本次形成可复现清单，不编造批量 URL。

## 许可和来源记录

NASA 的一般媒体说明允许教育、信息及事实性使用，并要求适当标明来源；同时保留第三方受版权保护内容、机构标识等例外。逐项记录图片来源及 credit，不能用“NASA 网站可访问”推断所有处理版的条件相同。[NASA 媒体使用说明](https://www.nasa.gov/nasa-brand-center/images-and-media/)

ASU 的页面区分原始未处理扫描与其派生处理产品：原始扫描按 NASA 政策处理；派生产品附有特定使用和再分发限制。使用时保留 `raw_scan` / `processed_scan` 字段，并针对具体资产核对条款。LPI 网站的通用 CC BY 标记也不代替上游照片和第三方处理内容的来源记录。[ASU Acceptable Use](https://apollo.im-ldi.com/ABOUT_SCANS/index.html)

这里没有官方分割训练代码或其代码许可证；不要将网页、说明文字或第三方脚本的许可套给照片。

## 建议的准备顺序

1. 先用 LPI 和 ALSJ 建候选表，筛出月面、地平线/近地视角，并记录胶卷、帧号、EVA、站点和镜头信息。
2. 为每个原始帧选一个可用的较高质量扫描版本；同一帧的不同裁剪、色彩处理和再发布记录为派生版本。
3. 小规模人工标注后再评估五类模型；灰度与彩色、近景与地平线、不同着陆点分别统计。
4. 以任务/站点/全景序列划分。同一原始帧及其所有派生图必须在同一组，避免重复扫描和重叠全景造成泄漏。

推荐保留 `mission, magazine, frame, source_url, product_type, source_frames, eva, station, viewpoint, width, height, color_mode, credit, license_note, downloaded`。这是未来独立研究的准备方案，不是本次比赛新增训练数据。
