from audit_lib import *


def main() -> None:
    data, missing = collect_algorithm_data()
    paper_tables = parse_paper_tables()
    exact_candidates = enumerate_exact_wilcoxon_pvalues(10)
    raw_rows, stats_rows, diff_rows, final_rows, qas_rows, method_trace = [], [], [], [], [], []

    for scale in SCALES:
        for alg in REQUESTED_ALGORITHMS:
            recs = data.get(alg, {}).get(scale, {})
            for seed in range(1, 11):
                rec = recs.get(seed)
                if rec:
                    value = rec.final
                else:
                    value = "REQUIRES RERUN" if alg in {"GA", "DE", "CCHIHH-noHI"} else "NOT FOUND"
                final_rows.append([scale, alg, seed, value])

    available_algs = [alg for alg in REQUESTED_ALGORITHMS if any(data.get(alg, {}).get(scale) for scale in SCALES)]
    for scale in SCALES:
        for i, alg_a in enumerate(available_algs):
            for alg_b in available_algs[i + 1:]:
                recs_a, recs_b = data[alg_a][scale], data[alg_b][scale]
                seeds = sorted(set(recs_a) & set(recs_b))
                if len(seeds) != 10:
                    continue
                vals_a = [recs_a[s].final for s in seeds]
                vals_b = [recs_b[s].final for s in seeds]
                for s in seeds:
                    raw_rows.append([scale, alg_a, alg_b, s, recs_a[s].final, recs_b[s].final])
                stat, p, mode = exact_wilcoxon(vals_a, vals_b)
                stats_rows.append([
                    scale, alg_a, alg_b, 10, "Wilcoxon signed-rank", mode, "two-sided",
                    stat if stat is not None else "NOT FOUND",
                    p if p is not None else "NOT FOUND",
                    (p is not None and p < 0.05),
                ])

    paper_pairs = {
        "CCHIHH-noCC": ("CCHIHH-full", "CCHIHH-noCC"),
        "CCHIHH-noHI": ("CCHIHH-full", "CCHIHH-noHI"),
        "CCHIHH-noCB": ("CCHIHH-full", "CCHIHH-noCB"),
        "CCHIHH-noGate": ("CCHIHH-full", "CCHIHH-noGate"),
        "CCHIHH-noMig": ("CCHIHH-full", "CCHIHH-noMig"),
        "CGA": ("CCHIHH-full", "CGA"),
        "IMOMA": ("CCHIHH-full", "IMOMA"),
        "PPO": ("CCHIHH-full", "PPO"),
        "DSAC-DE": ("CCHIHH-full", "DSAC-DE"),
    }
    for other_alg, label in PAPER_TABLES.items():
        alg_a, alg_b = paper_pairs[other_alg]
        for scale in SCALES:
            paper_p = paper_tables.get(label, {}).get(scale, {}).get("p_value")
            recs_a, recs_b = data.get(alg_a, {}).get(scale, {}), data.get(alg_b, {}).get(scale, {})
            seeds = sorted(set(recs_a) & set(recs_b))
            notes, recomputed, match_flag = [], "REQUIRES RERUN", False
            if len(seeds) == 10:
                vals_a = [recs_a[s].final for s in seeds]
                vals_b = [recs_b[s].final for s in seeds]
                w_stat, w_p, _ = exact_wilcoxon(vals_a, vals_b)
                mw_stat, mw_p = exact_mannwhitney(vals_a, vals_b)
                recomputed = w_p
                if paper_p is not None and w_p is not None and abs(paper_p - w_p) <= 1e-12:
                    match_flag = True
                    notes.append("与 exact Wilcoxon 完全一致")
                elif paper_p is not None and w_p is not None and abs(paper_p - w_p) <= 5e-7:
                    match_flag = True
                    notes.append("与 exact Wilcoxon 仅有表格舍入差异")
                elif paper_p is not None and mw_p is not None and abs(paper_p - mw_p) <= 1e-12:
                    notes.append("更像 Mann-Whitney U exact，而不是 paired Wilcoxon")
                elif paper_p is not None and not closest_match(paper_p, exact_candidates, tol=5e-7):
                    notes.append("该 p 值不可能来自 n=10 双侧 exact Wilcoxon")
                else:
                    notes.append("与重算 p 值不一致")
                method_trace.append({
                    "problem": scale,
                    "comparison": f"{alg_a} vs {alg_b}",
                    "paper_pvalue": paper_p,
                    "wilcoxon_exact_p": w_p,
                    "mannwhitney_exact_p": mw_p,
                    "wilcoxon_statistic": w_stat,
                    "mannwhitney_statistic": mw_stat,
                })
            else:
                notes.append("当前仓库缺少完整 10-run 原始配对数据")
            diff_rows.append([scale, f"{alg_a} vs {alg_b}", paper_p, recomputed, match_flag, "; ".join(notes)])

    for scale in SCALES:
        cga_vals = [r.final for _, r in sorted(data["CGA"][scale].items())]
        cga_mean = mean(cga_vals) if cga_vals else None
        for alg in ["CCHIHH-full", "CGA", "IMOMA", "Gbest-DE", "PPO", "DSAC-DE"]:
            vals = [r.final for _, r in sorted(data.get(alg, {}).get(scale, {}).items())]
            if not vals:
                qas_rows.append([scale, alg, "NOT FOUND", "NOT FOUND", "NOT FOUND", "NOT FOUND", "NOT FOUND", "原始日志缺失"])
                continue
            m, sd = mean(vals), pstdev(vals)
            cv = sd / m if m else None
            improvement = qas = None
            note = ""
            if alg == "CGA":
                note = "QAS 参考基线"
            elif cga_mean is not None:
                improvement = (cga_mean - m) / cga_mean * 100.0
                if improvement > 0 and cv not in (None, 0):
                    qas = improvement / cv
                elif alg == "IMOMA":
                    note = "论文将其写为 N/A，因为均值劣于 CGA"
            qas_rows.append([scale, alg, m, sd, cv, improvement if improvement is not None else "N/A", qas if qas is not None else "N/A", note])

    write_csv(ROOT / "stats_pairwise_raw.csv", ["problem", "algorithm_a", "algorithm_b", "seed", "fitness_a", "fitness_b"], raw_rows)
    write_csv(ROOT / "stats_recomputed.csv", ["problem", "algorithm_a", "algorithm_b", "n", "test_name", "exact_or_asymptotic", "sidedness", "statistic", "p_value", "significant"], stats_rows)
    write_csv(ROOT / "stats_pvalue_diff.csv", ["problem", "comparison", "paper_pvalue", "recomputed_pvalue", "match_flag", "notes"], diff_rows)
    write_csv(ROOT / "final_fitness_raw.csv", ["problem", "algorithm", "seed", "final_fitness"], final_rows)
    write_csv(ROOT / "qas_audit.csv", ["problem", "algorithm", "mean", "std", "cv", "improvement_pct", "qas", "notes"], qas_rows)
    write_json(ROOT / "stats_method_trace.json", method_trace)
    write_json(ROOT / "audit_missing_data.json", missing)


if __name__ == "__main__":
    main()
