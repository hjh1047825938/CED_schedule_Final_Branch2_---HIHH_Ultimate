# CC-HIHH 算子性能消融分析报告（示例版）

## 说明
本报告基于 `analyze_all.py --demo` 生成的模拟数据与完整分析流程产出，用于验证实验框架、日志字段与图表/表格链路是否完整可执行。正式论文结论应以真实 6 个变体（Full/Random/RoundRobin/ReducedOps/TopOps/FixedBest）在 T100/T200/T500、10 次独立运行、10000 代实验数据重跑后替换。

## 发现1：动态学习效果（Contextual Bandit 的学习行为）
从频率演化图 `fig_operator_frequency_evolution.pdf` 可观察到典型的“探索到利用”过程：在前期（0-2k）各算子频率更接近均匀，随后在中后期逐步向优势算子集中。热力图中后段（6k-10k）颜色集中度明显高于前段，符合 bandit 在稳定反馈下提升 exploitation 的机制预期。权重范数图 `fig_weight_evolution.pdf` 显示多数算子在前 2k 左右完成主要权重幅度建立，随后进入缓慢微调区，这与 ε 衰减和学习率衰减共同作用一致。七维权重细分图 `fig_weight_dimensions.pdf` 体现出不同算子在状态维度上的偏好分化：常见模式是对 stagnation、success_rate、last_reward 维度更敏感，说明模型确实在利用历史反馈，而非仅按固定算子序。

## 发现2：问题规模依赖性
跨规模对比图 `fig_cross_scale_comparison.pdf` 与熵图 `fig_diversity_vs_scale.pdf` 支持“规模越大越需要自适应”的研究假设。在示例数据中，T100 的算子分布更容易出现主导算子，熵较低；T500 分布更均衡，熵更高。对应解释是：小规模搜索空间较窄，固定策略或低复杂策略可较快逼近局部优区；大规模问题存在更多结构性阶段变化，需要通过 bandit 在不同搜索阶段切换算子以维持全局搜索与局部收敛的平衡。`table_scale_dependent_entropy.tex` 提供了“主导算子 + 熵值”的可论文化摘要格式。

## 发现3：算子贡献度
贡献度定义 `Contribution = Frequency × AvgReward` 在 `table_operator_contribution.tex` 与 `fig_operator_contribution.pdf` 中实现。示例结果中，高贡献算子通常同时具备较高频率和正向平均奖励，体现“高频高质”特征；而低频算子可能在特定阶段有价值，但总体贡献不高。该指标可直接指导 ReducedOps/TopOps 构造：
1. ReducedOps（Top 50%）用于测试是否可在削减算子集合后保持接近性能并降低策略复杂度。
2. TopOps（Top 30%）用于测试激进裁剪是否导致探索能力不足。

正式实验中，建议将贡献度按 block 分开排名，避免跨块直接比较造成解释偏差，因为不同块算子作用对象与尺度不同。

## 发现4：选择策略对比（CB vs Random vs RR vs FixedBest）
策略对比表 `table_strategy_comparison.tex` 与收敛曲线 `fig_selection_strategy_comparison.pdf` 给出统一分析视图。示例数据呈现 CB 在均值性能与收敛稳定性上均占优，Random 与 RoundRobin 更易出现波动，FixedBest 在部分小规模场景可能接近 CB，但在复杂规模下通常落后。Wilcoxon 配对检验已在脚本中集成（见 `compare_selection_strategies`），可输出 p-value 作为显著性支撑。论文撰写建议将“均值改进幅度 + p-value + 效应量（可选）”并列报告，避免仅凭均值差异下结论。

## 发现5：Gating 机制作用
`table_gating_statistics.tex` 记录了 blocked/fallback 总量与阻挡率，`fig_gating_effect.pdf` 给出 RESAMPLE 频率轨迹。示例中 gate 机制有效限制了早期无效的大扰动操作，并通过 fallback 保持迭代连续性，避免因硬阻断导致搜索停摆。随着规模增大，理论上 stagnation 触发机会提高，gate 对“何时允许大步重采样”的调控价值会更明显。建议正式实验补充“with/without gate”的同配置对照，以明确 gate 对收敛速度与最终质量的边际贡献。

## 总结
从流程验证角度，当前补丁已实现从“算子选择-奖励反馈-权重更新-全局统计”到“可视化/显著性/表格导出”的完整闭环。若在真实数据上复现实验矩阵，预计可以回答 Section 5.2.2 的核心问题：
1. CB 是否学习到有效的动态算子分配；
2. 不同规模是否对应不同的算子多样性需求；
3. 哪些算子是性能核心、哪些算子可裁剪；
4. 简单策略（Random/RR/Fixed）与学习策略（CB）的实证差异是否显著；
5. Gating 是否提升了算子调度稳定性与后期收敛质量。

建议下一步直接用 `run_experiments.sh` 跑真实数据，再用 `analyze_all.py` 生成最终论文图表并替换本示例报告数值。
