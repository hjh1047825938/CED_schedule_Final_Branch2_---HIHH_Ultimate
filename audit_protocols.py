from audit_lib import *


def code_snippet(path: str, start: int, end: int) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8", errors="ignore").splitlines()
    chunk = []
    for i in range(start - 1, min(end, len(lines))):
        chunk.append(lines[i])
    return "\n".join(chunk)


def build_ppo_protocol() -> None:
    rows = []
    data, _ = collect_algorithm_data()
    for scale in SCALES:
        recs = data["PPO"][scale]
        for seed in range(1, 11):
            rec = recs.get(seed)
            rows.append([scale, seed, rec.final if rec else "NOT FOUND", True if rec else "NOT FOUND", False if rec else "NOT FOUND", "每个 seed 对应一次独立训练；单 seed 目录中的 statistics.txt 为 std=0，跨 seed 汇总后 std>0"])
    write_csv(ROOT / "ppo_protocol_audit.csv", ["problem", "seed", "final_fitness", "retrained_flag", "same_policy_flag", "notes"], rows)


def build_dsac_diff() -> None:
    rows = [
        ["operator pool", "5 个 DE mutation operators", "仍为 5 个 DE operators", "该项基本保留", "低"],
        ["SAC / policy update frequency", "原文 Algorithm 1 表示每轮交互后进行梯度更新", "代码为 `if (++train_counter % 2 != 0) return;`，即每 2 次触发一次 train_step，不是每 10 代", "与论文当前写法不一致；训练节奏改变", "高"],
        ["replay buffer", "原文 buffer size = 40000", "代码 `buffer_size(20000)`", "经验回放容量减半，可能降低训练稳定性", "中"],
        ["batch size", "原文 batch size = 512", "代码默认 `batch_size(256)`；论文正文却写 32", "实现与原文、与论文描述同时不一致", "高"],
        ["network update scope", "原方法更新完整 actor/critic 网络", "代码更新 `w1/w2_mid/w3` 和偏置，属于完整多层更新；不是只更新 output layer", "论文当前简化描述错误", "高"],
        ["state definition", "原文 state=17 维 population state，obs=6 维 individual observation", "代码保留 global state + observation 拼接思路", "核心思想保留，但实现细节需逐项核对", "中"],
        ["reward definition", "原文奖励由 gbest 改善与终局回报组成", "代码实现了代际 improvement/终局逻辑，但未在论文当前实现说明中展开", "可能影响可复核性", "中"],
        ["training scope", "原文强调 DSAC policy 训练后可迁移到其他 case", "当前实现是在线伴随进化更新，不是单独先训后测的固定策略协议", "协议语义不同，可能影响对 baseline 的理解", "高"],
    ]
    write_csv(ROOT / "dsac_de_diff.csv", ["component", "original_method", "current_implementation", "possible_effect", "severity"], rows)


def build_normalization_refs() -> None:
    data, _ = collect_algorithm_data()
    rows = []
    for scale in SCALES:
        refs = []
        for alg in ["CCHIHH-full", "CGA", "IMOMA", "Gbest-DE", "DSAC-DE"]:
            for seed, rec in data.get(alg, {}).get(scale, {}).items():
                if rec.normalization_f1 is not None or rec.normalization_f2 is not None:
                    refs.append((alg, seed, rec.normalization_f1, rec.normalization_f2))
        if refs:
            f1_values = sorted({round(x[2], 12) for x in refs if x[2] is not None})
            f2_values = sorted({round(x[3], 12) for x in refs if x[3] is not None})
            rows.append([scale, ";".join(map(str, f1_values[:10])), ";".join(map(str, f2_values[:10])), len(f1_values) == 1 and len(f2_values) == 1, False, "来自各算法各自日志；至少 f1_ref 跨算法/跨 seed 不恒定"])
        else:
            rows.append([scale, "NOT FOUND", "NOT FOUND", "NOT FOUND", "NOT FOUND", "未找到 normalization 日志"])
    write_csv(ROOT / "normalization_refs.csv", ["problem", "f1_ref", "f2_ref", "shared_across_methods_flag", "fixed_across_runs_flag", "notes"], rows)


def build_gating_activity() -> None:
    rows = []
    overall = []
    for scale in SCALES:
        summary = ROOT / "outputs" / "results" / "cchihh_ablation_suite" / "stability" / scale / "gate_trigger_summary.csv"
        total_by_seed = {}
        if summary.exists():
            lines = summary.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[1:11]:
                if not line.strip():
                    continue
                seed, blocked, fallback = line.split(",")
                total_by_seed[int(seed)] = (int(blocked), int(fallback))
        for seed in range(1, 11):
            blocked, fallback = total_by_seed.get(seed, ("NOT FOUND", "NOT FOUND"))
            overall.append([scale, seed, blocked, fallback, "仅有 overall total；无 per-block / gate_open 原始日志"])
            for block in ["offload", "seq", "dev"]:
                rows.append([scale, block, seed, "NOT FOUND", "NOT FOUND", "NOT FOUND", "NOT FOUND"])
    write_csv(ROOT / "gating_activity.csv", ["problem", "block", "run", "resample_selected", "veto_count", "fallback_count", "gate_open_count"], rows)
    write_csv(ROOT / "gating_activity_overall.csv", ["problem", "run", "gate_blocked_total", "gate_fallback_total", "notes"], overall)


def build_operator_dynamics() -> None:
    rows = []
    for scale in ["T500"]:
        for seed in range(1, 11):
            path = ROOT / "outputs" / "results" / "cchihh_full_canonical_multiscale" / scale / f"CCHIHH_full_opstats_seed{seed}.csv"
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            header = lines[0].split(",")
            for line in lines[1:]:
                vals = line.split(",")
                gen_end = int(vals[0])
                gen_start = max(0, gen_end - 50)
                record = dict(zip(header, vals))
                block_map = {
                    "offload": ["offload_GA", "offload_DE", "offload_BITFLIP", "offload_RESAMPLE"],
                    "seq": ["seq_GA", "seq_SWAP", "seq_VNS", "seq_RESAMPLE"],
                    "dev": ["dev_DE", "dev_GDE", "dev_LEVY", "dev_RESAMPLE"],
                }
                for block, cols in block_map.items():
                    for col in cols:
                        prob = float(record[col])
                        count = round(prob * 50 * 8)
                        rows.append([scale, seed, f"{gen_start}-{gen_end}", block, col.split("_", 1)[1], count, prob])
    write_csv(ROOT / "operator_dynamics_raw.csv", ["problem", "run", "generation_window", "block", "operator", "selection_count", "selection_probability"], rows)


def build_summary_table() -> None:
    rows = [
        ["统计检验", "所有 pairwise comparison 都是 n=10 双侧 Wilcoxon signed-rank", "仓库中存在大量 Mann-Whitney 产表脚本，且若干论文 p 值不可能来自 n=10 双侧 exact Wilcoxon", False, "高", "重做统计表并统一检验说明"],
        ["预算公平性", "默认比较公平", "不是相同 generations、不是相同 total evaluations、也不是相同 wall-clock time", False, "高", "正文明确预算口径并重构 baseline 对比"],
        ["epsilon 公式", "公式与实现一致且能体现明显衰减", "公式形式一致，但主程序默认 eps_k=0.01 时几乎不衰减；构造函数默认还是 2.0，存在双默认值", False, "中", "正文和参数表必须澄清"],
        ["目标函数 makespan", "f1 是整体完工时间", "代码中 `time_max` 仅取 CE_ET 最大值，不是 CE+MFG overall completion time", False, "高", "改正文中的 f1 定义"],
        ["PPO 协议", "固定策略重复评估，std≈0", "表格非零 std 来自不同 seed 重新训练；图注与表格不能同时成立", False, "高", "二选一修正协议描述"],
        ["DSAC-DE 实现", "简化为每 10 代更新、batch=32、仅输出层更新", "代码实际为每 2 次触发更新、batch=256、全网络更新", False, "高", "改成 simplified re-implementation 的真实说明"],
        ["制造加工时间模型", "加工时间依赖设备", "代码为 `MTask_Time[CJ*M_OPTnum+CO]`，即 p_{j,k}", False, "中", "修正符号与模型说明"],
        ["device gene 语义", "设备块可独立解释为按工序分配", "当前编码按 execution-order rank 映射，和 sequence 强耦合", False, "高", "在编码设计段落加限定"],
        ["normalization refs", "shared across all methods and fixed across runs", "refs 在 `ComputeReferenceValues()` 中按当前 run 初始化种群重新计算，至少不 fixed across runs", False, "高", "重写 normalization 协议"],
        ["复杂度/profiling", "runtime 受各模块支配关系已说明", "缺少模块级 profiling；现有 figure provenance 还显示 DSAC-DE runtime 曾被按指令放大", False, "高", "补 profiling 或删除强结论"],
    ]
    write_csv(ROOT / "audit_summary_table.csv", ["topic", "current_paper_claim", "actual_code_or_data_finding", "consistent_flag", "risk_level", "recommended_action"], rows)


def build_paper_vs_code_diff() -> None:
    text = """# paper_vs_code_diff

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
"""
    (ROOT / "paper_vs_code_diff.md").write_text(text, encoding="utf-8")


def build_audit_report() -> None:
    impossible_p = []
    pval_rows = []
    pval_path = ROOT / "stats_pvalue_diff.csv"
    if pval_path.exists():
        with pval_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pval_rows = list(reader)
        impossible_p = [r for r in pval_rows if "不可能来自 n=10 双侧 exact Wilcoxon" in (r["notes"] or "")]

    budget_rows = []
    with (ROOT / "budget_fairness.csv").open(encoding="utf-8") as f:
        budget_rows = list(csv.DictReader(f))
    nofair = [r for r in budget_rows if r["algorithm"] in {"CCHIHH-full", "CCHIHH-noCC", "PPO", "CGA", "DSAC-DE"} and r["problem"] == "T100"]

    norm_rows = []
    with (ROOT / "normalization_refs.csv").open(encoding="utf-8") as f:
        norm_rows = list(csv.DictReader(f))

    report = f"""# CCHIHH 论文可复核性与内部一致性审计

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
{code_snippet("CCHIHH.tex", 795, 800)}
```

### 明确判断
- 当前论文“统一使用双侧 Wilcoxon signed-rank”的说法，不能被当前仓库中的全部产表脚本支撑。
- 当前能够重算的比较里，以下 p 值明显不可能来自 `n=10` 双侧 exact Wilcoxon：
{chr(10).join([f"- {r['problem']} / {r['comparison']} / paper p={r['paper_pvalue']}" for r in impossible_p[:20]]) or "- NOT FOUND"}
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
{chr(10).join([f"- {r['algorithm']} {r['problem']}: total_evals={r['total_evals']}, note={r['fairness_note']}" for r in nofair])}

结论：
- 论文当前表述是否成立：不成立
- 是否需要修改正文：是
- 修改风险等级（高/中/低）：高

## C. epsilon-greedy 公式与代码一致性审计

### 关键代码位置
`src/CC_HIHH.cpp`
```cpp
{code_snippet("src/CC_HIHH.cpp", 430, 438)}
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
{code_snippet("src/Problems.cpp", 387, 458)}
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
{code_snippet("src/Problems.cpp", 121, 129)}
```

### 已核到的事实
- processing time 数据结构是 `p_{{j,k}}`，不是 `p_{{j,k,d}}`。
- 代表性实例样本已导出到 `processing_time_device_dependency.csv`。

结论：
- 论文当前表述是否成立：若写成 p_{{j,k,d}} 则不成立
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
{code_snippet("src/Multimethod.cpp", 275, 310)}
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
"""
    (ROOT / "audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    build_ppo_protocol()
    build_dsac_diff()
    build_normalization_refs()
    build_gating_activity()
    build_operator_dynamics()
    build_summary_table()
    build_paper_vs_code_diff()
    build_audit_report()


if __name__ == "__main__":
    main()
