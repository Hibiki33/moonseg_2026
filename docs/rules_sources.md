# 比赛规则与数据来源核对

核实日期：**2026-10-21（UTC+8）**。本文记录比赛规则实际读取的来源、可支持的事实和仍未取得的内容；比赛规则正文见 [challenge.md](../challenge.md)。

| 来源 | 核对结果 | 用途与限制 |
|---|---|---|
| [比赛评价与结果提交](https://insight.onescience.cn/competition/moonseg?event_id=c2f557a58ff2e9f4265d56f112a7e01&tab=info&section=validation) | 标准 HTTPS GET 成功；返回 SPA HTML，引用当前入口 JS | 官方规则的页面入口，HTML本身不含完整正文，须读取实际加载模块 |
| [SPA 入口 index-Cxly3Qme.js](https://insight.onescience.cn/assets/index-Cxly3Qme.js) | 成功读取，比赛路由懒加载 CompetitionPage-uKTt9A4g.js | 将页面入口与具体比赛模块关联；资源名可能随部署更新 |
| [比赛页面模块 CompetitionPage-uKTt9A4g.js](https://insight.onescience.cn/assets/CompetitionPage-uKTt9A4g.js) | 成功读取，含 MoonSeg 文案 `Hr`、评价渲染函数 `di`、概述赛程 `Fr`、数据说明 `Lr`、提交界面 `qr` | 本次评分公式、缺类集合、同分、截止、目录、材料和次数的直接证据，确为本赛事页面实际使用的内容 |
| [比赛详情 API](https://insight.onescience.cn/api/competition/event/detail?event_id=c2f557a58ff2e9f4265d56f112a7e01) 与 [规则 API](https://insight.onescience.cn/api/competition/rules?event_id=c2f557a58ff2e9f4265d56f112a7e01) | 未登录请求均返回 `Authentication required!` / `unauthorized` | 未取得账户授权状态、完整登录后规则或技术协议；不得声称已核实这些内容 |
| [LuSNAR 官方 GitHub](https://github.com/zqyu9/LuSNAR-dataset) | 已读取 README 与文件列表；根目录只有 README.md、index.html、static | 支持五类颜色表、左相机示意目录、1024×1024 原始分辨率；没有分割训练或评测代码，README 的基线表与论文矛盾（见 [paper_alignment.md](paper_alignment.md#公开代码与第三方结果)） |
| [LuSNAR Hugging Face 文件列表](https://huggingface.co/datasets/xuboluo2001/LuSNAR/tree/main) | 已读取，列出 Moon_1.zip 至 Moon_9.zip | 支持原始场景包名称；不是比赛官方样本清单。包含评测场景标签的上游包不构成训练授权 |
| LuSNAR 论文：[arXiv v3](https://arxiv.org/html/2407.06512v3)、[TGRS 终稿](https://ieeexplore.ieee.org/document/11643283/)、[issue #18](https://github.com/zqyu9/LuSNAR-dataset/issues/18) | 版本、可读范围与表 V/VII 见 [paper_alignment.md](paper_alignment.md) | 论文划分与本比赛不同，论文成绩不是本比赛排行榜基线；终稿出版社版本已由用户提供（见 [references.md](references.md)），[paper_alignment.md](paper_alignment.md) 以其为准 |

## 当前官网的直接证据

以下定位为当前模块中的对象/函数标识，供再次取证时核对；它们是构建产物内的名称，不是平台 API 承诺。

| 位置 | 已核实规则 |
|---|---|
| `di()`：IoU 公式与定义 | `TP[s,c]`、`FP[s,c]`、`FN[s,c]` 是场景级像素数；`IoU = TP/(TP+FP+FN)`，按场景计数计算，不能先算逐图 mIoU 再平均 |
| `di()`：`C_s` 与 mIoU | 计分类别只由场景参考标签中确有有效像素的类别确定，评分前固定，各类别等权。参考标签缺失的类别不因预测出现而进入平均，不能用 `union > 0` 决定集合 |
| `di()`：最终成绩 | `S = (w2*mIoU2+w4*mIoU4+w9*mIoU9)/(w2+w4+w9)`，只公布 `w9>w2=w4>0`；`Fr.laterSections` 说明数值正式评测前公布冻结 |
| `Hr.extraNotes` | 各场景/各类 IoU 作为分析报告；同分暂同名次，平台若有其他政策须开赛前公布 |
| `Hr.submissionTime`、`Fr.schedule` | 结果截止 `2026-11-20 23:59(UTC+8)`；赛程同时列预测结果包、复现包与技术报告截止 |
| `Hr.correspondenceRows` | 唯一对应：`test/Moon_i/color/<sample_id>.png` → `test/Moon_i/submission/<sample_id>.png` |
| `Hr.maskRows`、`packagingBullets` | PNG、单通道、8位、原尺寸、值域0–4；一个 ZIP，直接压缩要求的顶层结构，无额外包装文件夹，不接受tar/rar；无固定ZIP基名 |
| `Hr.completenessBullets` | 缺文件、名不符或尺寸错误，相关图片得0并跳过，无自动容错；额外掩码警告；完整有效测试集是入榜条件 |
| `Hr.ranking` | 提交预测、train.py、predict.py、权重、配置、环境、README、训练日志、模型/数据来源；README含训练/推理命令、目录、验证划分、模型选择及恢复尺寸；最终排名需≤6页方法文档，并按要求提供Docker/演示/其他技术材料 |
| `qr.results` | 平台“提交结果→测试集”上传ZIP，需组织方通知授权及发放提交编号；每队唯一正式机会，上传失败/校验失败不占次数，成功接收不可替换 |
| `qr.method`、`qr.materials` | 方法论文、补充材料分别提交，组织方通知后开放；方法论文不占正式成绩次数；具体截止、格式以通知为准 |
| `Ur.notes` | 联系邮箱不可默认用于数据或正式成绩提交，除非主办方另行书面通知 |
| `Fr.schedule`、`Lr` | 训练包 `LuSNAR_train.rar`，10月22日起滚动发放；测试输入11月2日；11月25日公布结果、11月29日展示。训练池1/3/5/6/7/8、测试2/4/9，实际样本由官方清单固定，场景5可本地验证后用于最终训练 |

## 仍待官方公布或取得的内容

- 具体场景权重数值，以及得分显示/比较精度；不能默认使用1:1:2。
- 正式样本清单、技术协议、校验工具及其实际运行结果。
- ZIP大小限制、完整技术模板、脚本参数、容器打包要求、软硬件与推理时限。`Hr.rankingPending` 明确表示文件大小限制、技术模板等另行公布。
- 参考场景完全无有效类别的处理，以及额外ignore值/非法像素的细项规则。
- 各队实际提交权限、论文与补充材料的通知、截止及格式；未登录接口不能用于证明已获授权。

## 文档与实现的边界

challenge.md 原有比赛正文保留，其后的核对补充据当前官网写成。官方已定义的场景像素计数、参考标签类别集合、归一化加权公式、提交路径及脚本材料清单不再标为项目猜测；项目自定的CLI参数、默认权重、模型及增强强度仍属实现选择。

来源记录不转载受限数据链接或提取密码。报名与数据访问仍须按官方流程审核；上游公开许可证或公开可下载状态不能覆盖比赛对训练样本、测试真值和非公开材料的限制。上游仓库、HF 和论文条目沿用此前的核对，联网核实集中于比赛官网。
