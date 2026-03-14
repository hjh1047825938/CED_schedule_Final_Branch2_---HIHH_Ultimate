from audit_lib import *


def runtime_stats(records: Dict[int, LogRecord]) -> Tuple[object, object]:
    vals = [r.runtime_sec for r in records.values() if r.runtime_sec is not None]
    if not vals:
        return "NOT FOUND", "NOT FOUND"
    return mean(vals), stdev(vals) if len(vals) > 1 else 0.0


def main() -> None:
    data, missing = collect_algorithm_data()
    rows = []
    threads_default = "OMP max threads (未记录，随环境变化)"
    for scale in SCALES:
        for alg in REQUESTED_ALGORITHMS:
            recs = data.get(alg, {}).get(scale, {})
            rt_mean, rt_std = runtime_stats(recs)
            pop_size, islands, evals_per_gen, total_gens, total_evals = "NOT FOUND", "NOT FOUND", "NOT FOUND", 10000, "NOT FOUND"
            parallel, granularity, note, threads = "No / NOT FOUND", "NOT FOUND", "", threads_default
            if alg == "CCHIHH-full":
                pop_size, islands, evals_per_gen, total_evals = 40, 8, 121, 1210000
                parallel, granularity = "Yes", "按 evaluation(OpenMP for)，不是按 island"
                note = "每代 3 blocks x 40 offspring，再额外 1 次全局 context eval；论文“per island 40”容易让总预算被误读"
            elif alg == "CCHIHH-noCC":
                pop_size, islands, evals_per_gen, total_evals = 40, 8, 40, 400000
                parallel, granularity = "Yes", "按 evaluation(OpenMP for)"
                note = "去掉 block 分解后 total evaluations 显著更少，与 CCHIHH-full 不公平"
            elif alg == "CCHIHH-noHI":
                pop_size, islands, evals_per_gen, total_evals = 40, 1, 121, 1210000
                parallel, granularity = "Yes", "按 evaluation(OpenMP for)"
                note = "这是 nsubpop=1 的预期预算；当前原始日志缺失，REQUIRES RERUN"
            elif alg in {"CCHIHH-noCB", "CCHIHH-noMig", "CCHIHH-noGate"}:
                pop_size, islands, evals_per_gen, total_evals = 40, 8, 121, 1210000
                parallel, granularity = "Yes", "按 evaluation(OpenMP for)"
                note = "与 full 的 evaluations 一致；但 noGate 现有日志对应 stable_false + gate_off，不是纯 only-noGate"
            elif alg in {"GA", "DE", "Gbest-DE", "DSAC-DE"}:
                pop_size, islands, evals_per_gen, total_evals = 40, 1, 40, 400000
                parallel, granularity = "Yes", "按 evaluation(OpenMP for)"
                if alg in {"GA", "DE"}:
                    note = "当前与论文量级一致的 10-run 日志缺失，REQUIRES RERUN"
            elif alg == "CGA":
                pop_size, islands, evals_per_gen, total_evals = 300, 1, "300 + catastrophe extra", ">= 3000000"
                parallel, granularity, threads = "No evidence of OpenMP in CGA loop", "串行主循环", 1
                note = "默认人口 300，灾变阶段还会追加评估，总 evaluations 明显高于 CCHIHH-full"
            elif alg == "IMOMA":
                pop_size, islands, evals_per_gen, total_evals = 40, 1, "variable", "NOT FOUND"
                parallel, granularity = "No clear OpenMP at algorithm level", "串行生成候选，fitness 内部可并行"
                note = "每代额外候选数依赖 archive/pulse/intensify/restart，需要插桩或重跑才能精确统计"
            elif alg == "PPO":
                pop_size, islands, evals_per_gen, total_gens, total_evals = "policy dim = Nvar", 1, "1 eval / episode", "10000 episodes", 10000
                parallel, granularity = "No clear OpenMP at training loop", "每个 episode 一次黑盒 fitness"
                note = "预算与进化算法完全不同：10000 次调度评估，不是 10000 代 x population"
            rows.append([alg, scale, pop_size, islands, evals_per_gen, total_gens, total_evals, rt_mean, rt_std, threads, parallel, note])
    write_csv(ROOT / "budget_fairness.csv", ["algorithm", "problem", "pop_size", "islands", "evals_per_gen", "total_gens", "total_evals", "runtime_mean_sec", "runtime_std_sec", "threads", "parallelism", "fairness_note"], rows)
    write_json(ROOT / "budget_missing_data.json", missing)


if __name__ == "__main__":
    main()
