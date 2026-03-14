# paper_vs_code_diff

## 公式与代码不一致

- epsilon 公式形式与 `src/CC_HIHH.cpp` 一致，但默认参数口径不一致：`src/main.cpp` 默认 `eps_k=0.01`，`src/CC_HIHH.cpp` 构造函数默认 `epsilon_k=2.0`。如果按论文参数表 `0.01`，则 epsilon 从 0.2 到 10000 代几乎不下降。
- 论文若把 makespan 写成整体完工时间，则与 `src/Problems.cpp` 不一致。代码中 `time_max` 只对 `CE_ET[i]` 取最大值。
- 论文若将制造加工时间写成 `p_{j,k,d}`，则与代码不一致。代码使用 `MTask_Time[CJ * M_OPTnum + CO]`，即 `p_{j,k}`。

## 实验表述与实际协议不一致

- PPO 图注称“fixed trained policy; fitness is constant across evaluations; Std≈0”，但 `outputs/results/PPO/statistics.txt` 显示跨 10 个 seed 的 std 显著非零，说明表格来自多次重新训练，不是一个固定策略的重复评估。
- 论文中“population size (per island): 40”若不同时说明 `nsubpop=8`，会掩盖 CCHIHH 总人口 320、每代约 121 次 evaluation 的真实预算。
- `f1_ref/f2_ref` 并非“fixed across runs”。`src/Multimethod.cpp` 会在每次 run 初始化后重新计算 reference values。

## 统计检验与表格不一致

- 论文写的是双侧 Wilcoxon signed-rank，但仓库里多份产表脚本使用 `mannwhitneyu`。
- 若干论文 p 值不在 `n=10` 双侧 exact Wilcoxon 的可取值集合内，因此不可能来自文中声明的方法。
- 当前仓库缺少 GA、DE、noHI 与论文同版本口径一致的 10-run 原始日志，导致这些行无法完全复核。

## 目标函数/约束与实现不一致

- CE 与制造的耦合确实存在，但主要通过 `Job_Constraints` 驱动的开始/结束同步，而不是论文若暗示的统一 overall makespan 目标。
- cloud/edge server 上多个 CE tasks 在代码中不是显式串行队列，也不是标准 capacity-limited parallel machine；更接近负载惩罚近似模型。

## 复杂度叙述与 profiling 不一致

- 当前仓库没有模块级 profiling 结果，无法严谨支撑 decoding/simulation/bandit/migration 的耗时占比。
- `outputs/mnt/figure_data_provenance.txt` 明确写了 runtime figure 中 DSAC-DE runtime 曾按“user instruction”放大（T100 x3，T200/T500 x6）。若论文采用该图，结论不可接受。

## 基线实现与原始方法不一致

- 论文对 DSAC-DE 当前实现的描述与代码不符：不是“每 10 代更新、batch 32、只更新输出层”，而是更接近“每 2 次触发更新、batch 256、全网络更新”。
- 当前 DSAC-DE 更严谨的称呼应是 simplified/lightweight re-implementation，而不是未经限定地称为 original DSAC-DE。

## 其他值得审稿人质疑的点

- device gene 按 execution-order rank 映射到 operation，导致 sequence block 与 device block 强耦合；若正文把三块说成弱耦合独立分解，会被追问。
- `makespan_energy_scatter.pdf` 与 `communication_time.pdf` 的 provenance 已写明是推断值而非直接日志，若正文把这些图当作直接实验观测，会被质疑。
