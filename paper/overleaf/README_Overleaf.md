# AAS Overleaf 版本 20261009 Public release

目标期刊：Advances in Atmospheric Sciences（AAS）。

论文标题保持原定标题：

**Extreme-Aware Fusion of Multi-Model AI Weather Forecasts: Joint Optimization of Overall Skill and Event Detection**

## 如何编译

1. 在 Overleaf 中选择 New Project → Upload Project，上传 `PS-PaE-Fuse_AAS_Overleaf_public_release_20261009.zip`。
2. 主文件选择 `main.tex`，编译器选择 **pdfLaTeX**。
3. 补充材料单独编译：将主文件临时切换为 `supplement.tex`。
4. 图件全部通过 `figures/` 的相对路径引用；表格使用 `tables/` 的相对路径。本包不依赖 Windows 本机绝对路径。

## 本次已经补充

- 作者顺序：Ruxue XING、Jianjun ZHU、Lang ZHENG、Yaojun WANG。Ruxue XING 与 Jianjun ZHU 使用相同的 † 标记并注明共同第一作者；四人机构均为 College of Information and Electrical Engineering, China Agricultural University, Beijing 100083, China。通讯作者 Yaojun WANG，邮箱 wangyaojun@cau.edu.cn。主文和补充材料同步更新。

- 已按作者确认替换 Fig. 6：Brier/CRPS 移到独立指标行；共享横轴标签和底部图例分开；主文按通栏宽度显示，局部调整该图图注的行距，以保持图和图注完整。数据、其他图、正文科学表述及标题不变。

- 六变量（T2M、U10、V10、MSL、Z500、T850）× 72/120/168 h 总体技巧与风速概率可靠性的六面板组图，插入主文 Fig. 6。此图中的 PS-PaE-Fuse 数值明确对应保障前输出。
- 图件 PDF 和 SVG 为矢量格式，PNG 为 300 dpi；绘图代码、两份原始汇总数据及输入/输出校验值保存在 `figure_sources/fig_multivariate_probability_2021/`。
- GraphCast/FuXi checkpoint 版本、初始化来源、时次、rollout/cascade 设置和 SHA-256。
- PS-PaE-Fuse 全称：Phase-Selective Phase-Aware Ensemble Fusion。
- global-skill expert 的训练样本量、训练配置、loss、最优 epoch 16、validation loss 0.2984 和 checkpoint SHA-256。
- Data availability 和 Code availability，使用 https://github.com/NIKI1924/PS-PaE-Fuse。
- 补充材料图号说明统一为 Figs. S1–S3。

## GitHub 当前状态

代码、作者确认版论文和 Figure 6 修正版已推送。四个 checkpoint 已上传完成，GitHub 返回的 SHA-256 digest 与本地冻结清单逐项一致。

论文现采用真实状态：checkpoint **已公开发布**到
https://github.com/NIKI1924/PS-PaE-Fuse/releases/tag/v1.0-paper-assets。
Code availability 已改为已公开。Release 同时提供主文 PDF、补充材料 PDF 和完整 Overleaf ZIP；优先下载名称含 `public_release_20261009` 的最新版论文文件。Release 标签保留初版代码快照，最新论文源码在仓库 main 分支；不要误把自动生成的标签 Source code ZIP 当成最新版 Overleaf 包。

完整版本/来源说明、逐 epoch CSV 和四个 checkpoint 校验清单附在 Overleaf 包的 `reproducibility/` 目录。

## 还需要作者填写的黄色信息

基金/资源致谢、作者贡献、利益冲突确认，以及 LLM 使用声明。作者姓名、机构、共同第一作者及通讯邮箱已按作者提供的信息填写。

## 本地编译复核

主文和补充材料均已成功编译。日志没有未定义引用或未定义 citation；原稿的三张旧图仍会产生 float 高度警告，已逐页渲染检查，没有正文/图注裁切。AAS 模板自身还产生 headheight、perpage 和页脚 destination 的警告。

