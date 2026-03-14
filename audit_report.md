# CCHIHH 论文可复核性与内部一致性审计

本报告只依据当前仓库中的 `CCHIHH.tex`、源代码、现有日志/结果文件与本地 DSAC-DE PDF 生成。找不到的证据明确写为 `NOT FOUND` 或 `REQUIRES RERUN`。

## A. 统计检验审计

### 已核到的事实
- 论文在 `CCHIHH.tex` 中声明：所有 pairwise comparison 使用双侧 Wilcoxon signed-rank，`n=10`。
- 但仓库中存在多份旧产表脚本使用 `mannwhitneyu`，说明统计口径至少在仓库历史中并不统一。
- 重算结果已落盘到：
  - `stats_pairwise_raw.csv`
  - `stats_recomputed.csv`
  - `stats_pvalue_diff.csv`
  - `final_fitness_raw.csv`

### 关键代码/脚本证据
`scripts/build_table_and_plot_cchihh_vs_ppo.py` / `scripts/build_table_and_plot_cchihh_vs_dsac_de.py` / `scripts/build_table_and_plot_full_vs_nocc_blocks.py` 中存在 `mannwhitneyu`。

论文声称位置：
```tex
    \item \textbf{DSAC-DE}~\cite{laili2023dsac}: DSAC-DE is a Discretized Soft Actor-Critic configured Differential Evolution method specifically designed for CED task scheduling. It maintains five DE mutation operators and uses a UCB-based adaptive operator selection mechanism informed by operator success statistics (offspring improvement over parent, global best, and population average). In our implementation, for computational tractability, the SAC training component is simplified: Q-network updates are performed every 10 generations with a reduced mini-batch size of 32, and only the output-layer weights are updated per training step. The UCB-based operator selection mechanism, which dominates practical performance, is faithfully implemented.
\end{itemize}

\textbf{Statistical test}: 
All pairwise comparisons are conducted using the two-sided Wilcoxon signed-rank test ($\alpha=0.05$) over 10 paired runs (seeds 1-10). 

```

### 明确判断
- 当前论文“统一使用双侧 Wilcoxon signed-rank”的说法，不能被当前仓库中的全部产表脚本支撑。
- 当前能够重算的比较里，以下 p 值明显不可能来自 `n=10` 双侧 exact Wilcoxon：
- T100 / CCHIHH-full vs CCHIHH-noCC / paper p=0.000181651146
- T200 / CCHIHH-full vs CCHIHH-noCC / paper p=0.00100797624
- T500 / CCHIHH-full vs CCHIHH-noCC / paper p=0.000182671791
- T100 / CCHIHH-full vs PPO / paper p=1.0825e-05
- T200 / CCHIHH-full vs PPO / paper p=1.0825e-05
- T500 / CCHIHH-full vs PPO / paper p=1.0825e-05
- T100 / CCHIHH-full vs DSAC-DE / paper p=1.0825e-05
- T200 / CCHIHH-full vs DSAC-DE / paper p=1.0825e-05
- T500 / CCHIHH-full vs DSAC-DE / paper p=1.0825e-05
- 需要整体替换的表：所有包含 ablation/baseline pairwise p-value 且尚未与 `stats_pvalue_diff.csv` 对齐的表。

结论：
- 论文当前表述是否成立：不完全成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## B. 预算公平性与实验协议审计

### 已核到的事实
- 预算表已写入 `budget_fairness.csv`。
- 代码口径下：
  - `CCHIHH-full`: pop 40 per island, 8 islands, 约 121 evals/gen, 总约 1,210,000 evals。
  - `CCHIHH-noCC`: 约 40 evals/gen, 总约 400,000 evals。
  - `DSAC-DE`: 约 400,000 evals。
  - `PPO`: 约 10,000 evals（1 eval/episode）。
  - `CGA`: 默认人口 300，且可能有额外灾变评估，预算显著更高。

### 关键代码证据
`src/main.cpp` 和 `src/CC_HIHH.cpp` 的主循环/子块评估逻辑显示 CCHIHH 不是“40 人口、10000 代”的单一预算概念。

### 一句话回答
论文当前比较不是按相同 generations、不是按相同 total evaluations、也不是按相同 wall-clock time。

### 样例行
- CCHIHH-full T100: total_evals=1210000, note=每代 3 blocks x 40 offspring，再额外 1 次全局 context eval；论文“per island 40”容易让总预算被误读
- CCHIHH-noCC T100: total_evals=400000, note=去掉 block 分解后 total evaluations 显著更少，与 CCHIHH-full 不公平
- CGA T100: total_evals=>= 3000000, note=默认人口 300，灾变阶段还会追加评估，总 evaluations 明显高于 CCHIHH-full
- PPO T100: total_evals=10000, note=预算与进化算法完全不同：10000 次调度评估，不是 10000 代 x population
- DSAC-DE T100: total_evals=400000, note=

结论：
- 论文当前表述是否成立：不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## C. epsilon-greedy 公式与代码一致性审计

### 关键代码位置
`src/CC_HIHH.cpp`
```cpp
double CC_HIHH_Solver::ComputeEpsilon(int gen) const
{
    if (!stable_mode) {
        double eps = epsilon0 * std::pow(epsilon_decay, gen);
        if (eps < epsilon_min) eps = epsilon_min;
        return eps;
    }
    double eps = epsilon0 * std::exp(-epsilon_k * gen / max_generations);
    if (eps < epsilon_min) eps = epsilon_min;
```

### 已核到的事实
- 公式形式与论文一致：`epsilon = max(epsilon_min, epsilon0 * exp(-epsilon_k * gen / max_generations))`
- 但参数默认值存在双口径：
  - `src/main.cpp` 默认 `eps_k=0.01`
  - `src/CC_HIHH.cpp` 构造函数默认 `epsilon_k=2.0`
- `epsilon_schedule.csv` 已按主程序默认口径导出。
- 若按论文/主程序默认 `eps_k=0.01`，从 0 到 10000 代几乎不下降。

结论：
- 论文当前表述是否成立：公式形式成立，参数解释不充分
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：中

## D. 目标函数、makespan 与 CE/MFG 耦合审计

### 关键代码位置
`src/Problems.cpp`
```cpp
        int job_cons = task.Job_Constraints;
        if (job_cons == 1 || job_cons == 3)
            ce_st_val = fast_fmax(ce_st_val, ST[i][m_opt_minus_1]);
        
        double ce_et_val = ce_st_val + t_comm + t_comp;
        ce_et_val = fast_fmax(ce_et_val, max_End_EndTime);
        ce_et_val = fast_fmax(ce_et_val, max_Iter_EndTime);
        if (job_cons == 2 || job_cons == 3)
            ce_et_val = fast_fmax(ce_et_val, ET[i][m_opt_minus_1]);

        CE_ST[i] = ce_st_val;
        CE_ET[i] = ce_et_val;

        if (!task.Interact.empty())
        {
            for (int idx : task.Interact)
                CE_ET[idx] = ce_et_val;
        }
    }

    for (int i = 0; i < CE_Tnum; i ++)
    {
        double et_val = CE_ET[i];
        if (et_val > time_max)
            time_max = et_val;
    }

    const double energy_scale = 1.0 / 1000.0;
    for (int i = 0; i < Cnum; i ++)
    {
        const auto& load = CloudLoad[i];
        size_t cloud_size = load.size();
        if (cloud_size == 0)
            continue;
        int u_ratio = (int)((cloud_size / 20.0) * 10);
        if (u_ratio > 10) u_ratio = 10;
        int time_expand = 0;
        for (int idx : load)
        {
            int dur = (int)(CE_ET[idx] - CE_ST[idx]);
            if (dur > time_expand) time_expand = dur;
        }
        energy += EnergyList[u_ratio] * time_expand * energy_scale;
    }
    for (int i = 0; i < Enum; i ++)
    {
        const auto& load = EdgeLoad[i];
        size_t edge_size = load.size();
        if (edge_size == 0)
            continue;
        int u_ratio = (int)((edge_size / 6.0) * 10);
        if (u_ratio > 10) u_ratio = 10;
        int time_expand = 0;
        for (int idx : load)
        {
            int dur = (int)(CE_ET[idx] - CE_ST[idx]);
            if (dur > time_expand) time_expand = dur;
        }
        energy += EnergyList[u_ratio] * time_expand * energy_scale;
        energy += EnergyList[u_ratio] * time_expand / 1000.0;
    }
#ifdef PROFILE_EVAL
    ws.profile.tasks_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
#endif

    const double eps = 1e-6;
    const double f1_ref = (std::abs(ws.f1_ref) > eps) ? ws.f1_ref : 1.0;
    const double f2_ref = (std::abs(ws.f2_ref) > eps) ? ws.f2_ref : 1.0;
    const double alpha = std::clamp(ws.alpha, 0.0, 1.0);

    const double f1_normalized = time_max / f1_ref;
    const double f2_normalized = energy / f2_ref;
```

### 已核到的事实
- 最终 `time_max` 只对 `CE_ET[i]` 取最大值。
- 因此当前 fitness 中的 makespan 定义是 `max over CE tasks only`。
- CE 与 manufacturing 的耦合通过 `Job_Constraints` 实现：
  - `job_cons == 1 or 3`: CE 开始时间受制造侧时间约束
  - `job_cons == 2 or 3`: CE 结束时间受制造侧时间约束
- `makespan_audit.csv` 目前只能写 `REQUIRES RERUN`，因为日志没有保存 best solution 向量，无法从现有文件回放每个代表性 run 的 `max_ET_CE/max_ET_MFG`。

### 明确判断
- 论文当前公式若写成 `f1(x)=max_i ET_i^CE`，则与代码严格一致。
- 若论文把 `f1` 写成整体系统完工时间，则与代码不一致。
- 正文建议改写方向：明确说明优化目标中的 makespan 是 CE completion time 上界，制造侧通过同步约束耦合进入该量，而不是直接取 MFG/CE 的联合最大值。

结论：
- 论文当前表述是否成立：部分成立，取决于正文是否把 f1 写成 CE-only
- 是否需要修改正文：大概率需要
- 修改风险等级（高/中/低）：高

## E. PPO 基线协议审计

### 已核到的事实
- `ppo_protocol_audit.csv` 已导出每个 scale 的 10 个 seed 原始 final fitness。
- `outputs/results/PPO/statistics.txt` 的 std 非零，说明表格不是“一个固定策略重复评估”。
- 单个 seed 目录如 `results/PPO_runs/T500_seed1/statistics.txt` 的 std 为 0，说明同一个训练产出的固定策略在该 seed 内确实是常数。

### 明确判断
- 表格是“10 个训练 seed 的汇总”时才成立。
- 图注是“单个固定策略重复评估”时才成立。
- 两者不能同时成立。

结论：
- 论文当前表述是否成立：不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## F. DSAC-DE 基线实现差异审计

### 已核到的事实
- 差异表已落盘到 `dsac_de_diff.csv`。
- 本地 PDF 可支持的原方法信息包括：5 个 DE operators、state/obs 定义、reward、replay buffer=40000、batch size=512。
- 代码实际实现：
  - `buffer_size=20000`
  - `batch_size=256`
  - `train_step` 不是每 10 代，而是每 2 次触发一次
  - 更新范围覆盖 `w1/w2_mid/w3` 和偏置，不是仅输出层

### 明确判断
- 当前实现更严谨地应称为 simplified/lightweight re-implementation，而不是原始 DSAC-DE。

结论：
- 论文当前表述是否成立：不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## G. 制造加工时间建模审计

### 关键代码位置
`src/Problems.cpp`
```cpp
        geneO[CJ] ++;
        int CO = geneO[CJ];
        int CM = mvar[op];
        int Cprev = CO - 1;
        double* st_row = ST[CJ];
        double* et_row = ET[CJ];
        double mtask_time = MTask_Time[CJ * M_OPTnum + CO];
        
        if (Cprev < 0)
```

### 已核到的事实
- processing time 数据结构是 `p_{j,k}`，不是 `p_{j,k,d}`。
- 代表性实例样本已导出到 `processing_time_device_dependency.csv`。

结论：
- 论文当前表述是否成立：若写成 p_{j,k,d} 则不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：中

## H. 云/边服务器资源竞争模型审计

### 已核到的事实
- 说明文档已写入 `server_scheduling_model.md`。
- 当前代码更接近“依赖释放 + 通信/负载惩罚”的近似模型，而非显式串行排队或标准容量受限并行机。

结论：
- 论文当前表述是否成立：若写成标准队列/容量模型则不成立
- 是否需要修改正文：建议修改
- 修改风险等级（高/中/低）：中

## I. device gene 语义与分解合理性审计

### 已核到的事实
- 文档已写入 `device_gene_semantics.md`。
- 当前 `m_i` 对应 execution order rank，不是固定 operation ID。
- 这会导致 sequence block 与 device block 强耦合。

结论：
- 论文当前表述是否成立：若声称按 operation ID 独立编码则不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## J. objective normalization 审计

### 关键代码位置
`src/Multimethod.cpp`
```cpp
                                        CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                                        AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                                        ws.cloud_load.data(), ws.edge_load.data(), DeviceLoad, CETask_coDevice,
                                        ws.edge_device_comm.data(), ws.st_rows.data(), ws.et_rows.data(),
                                        ws.ce_st.data(), ws.ce_et.data());

            ws.set_alpha(0.0);
            double energy = EvaluFunc(pop[i], ws, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                                      CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                                      AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                                      ws.cloud_load.data(), ws.edge_load.data(), DeviceLoad, CETask_coDevice,
                                      ws.edge_device_comm.data(), ws.st_rows.data(), ws.et_rows.data(),
                                      ws.ce_st.data(), ws.ce_et.data());
            if (makespan > local_max_makespan) local_max_makespan = makespan;
            if (energy > local_max_energy) local_max_energy = energy;
        }

        #pragma omp critical
        {
            if (local_max_makespan > max_makespan) max_makespan = local_max_makespan;
            if (local_max_energy > max_energy) max_energy = local_max_energy;
        }
    }

    workspace.f1_ref = (max_makespan > 1e-6) ? max_makespan : 1.0;
    workspace.f2_ref = (max_energy > 1e-6) ? max_energy : 1.0;
    workspace.set_alpha(prev_alpha);
    for (int t = 0; t < ws_pool.size(); ++t) {
        Workspace& ws = ws_pool.get(t);
        ws.f1_ref = workspace.f1_ref;
        ws.f2_ref = workspace.f2_ref;
        ws.alpha = workspace.alpha;
    }

    cout << "[Normalization] f1_ref (makespan) = " << workspace.f1_ref << endl;
    cout << "[Normalization] f2_ref (energy) = " << workspace.f2_ref << endl;
```

### 已核到的事实
- `f1_ref/f2_ref` 来自当前 run 初始化种群上的扫描，不是每个 instance 固定一次后在所有 run 共享。
- `normalization_refs.csv` 已导出当前日志中可见的 ref 值。
- 至少 `f1_ref` 已观察到跨算法、跨 seed 变化。

### 明确判断
- 论文声称“shared across all methods and fixed across runs”不属实。

结论：
- 论文当前表述是否成立：不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## K. 复杂度与 profiling 审计

### 已核到的事实
- `runtime_profile.csv` 目前只能标为 `REQUIRES RERUN`，因为仓库没有模块级计时点。
- `avg_eval_time.csv` 给出了基于 wall-clock / 估算 eval budget 的粗略单次 evaluation 时间。
- `outputs/mnt/figure_data_provenance.txt` 说明 runtime figure 中 DSAC-DE runtime 曾被人工倍率调整。

### 明确判断
- 当前无法严谨证明 runtime 由 fitness evaluation 主导，只能给出弱推断。

结论：
- 论文当前表述是否成立：证据不足
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## L. gating mechanism 活跃度审计

### 已核到的事实
- `gating_activity.csv` 中按 block / resample / gate_open 的细粒度统计全部为 `NOT FOUND`。
- 当前仓库只保存了 `gate_blocked_total` 与 `gate_fallback_total` 的 overall 汇总，另存于 `gating_activity_overall.csv`。
- 例如 T100/T500 的总体 blocked/fallback 均值分别约为 3806.2 / 14370.6，说明大规模实例上的 gate 更活跃。

### 明确判断
- “主要对大规模问题有效”有一定总体证据支持。
- 但“小规模是否过度干预”无法按 block 粒度证明，因为缺少原始日志。

结论：
- 论文当前表述是否成立：只能部分支持
- 是否需要修改正文：建议补充限定
- 修改风险等级（高/中/低）：中

## M. operator dynamics 底层统计审计

### 已核到的事实
- `operator_dynamics_raw.csv` 已从 `CCHIHH_full_opstats_seed*.csv` 重建。
- probability 定义：
  - 在单个 run 内，对一个 50-generation window 汇总
  - 对所有 islands 汇总
  - 对 block 内 operator 调用次数按 `window_size * nSubpop` 归一化
  - 原始文件不是先对 runs 平均；平均应由后续作图脚本完成

### 明确判断
- 早期/后期窗口表值原则上可以由这些原始 opstats 复现。
- 若论文图表数值与该文件对不上，应优先检查后处理脚本的跨 seed 平均方式。

结论：
- 论文当前表述是否成立：基本可复核
- 是否需要修改正文：可选
- 修改风险等级（高/中/低）：低

## N. QAS 指标审计

### 已核到的事实
- `qas_audit.csv` 已导出 mean/std/CV/improvement/QAS。
- 当前脚本按 CGA 作为 improvement baseline 重算。
- IMOMA 为 N/A 的原因是其均值劣于 CGA，若强行代入会得到负 improvement，因此作者选择 N/A 而非负数或 0。

结论：
- 论文当前表述是否成立：需看正文是否明确 baseline 和 N/A 规则
- 是否需要修改正文：建议补充
- 修改风险等级（高/中/低）：中
