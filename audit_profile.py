from audit_lib import *


def main() -> None:
    data, _ = collect_algorithm_data()
    rows = []
    approx_rows = []
    for scale in SCALES:
        recs = data.get("CCHIHH-full", {}).get(scale, {})
        runtimes = [r.runtime_sec for r in recs.values() if r.runtime_sec is not None]
        if len(runtimes) == 10:
            runtime_mean = mean(runtimes)
            total_evals = 1210000
            avg_eval = runtime_mean / total_evals
            approx_rows.append([scale, runtime_mean, total_evals, avg_eval, "基于 wall-clock / 估算 eval budget 的粗略值，不是插桩 profiling"])
        else:
            approx_rows.append([scale, "NOT FOUND", "NOT FOUND", "NOT FOUND", "缺少完整 runtime 日志"])
        for module in ["decoding", "sequencing sort", "simulation / fitness evaluation", "contextual bandit state extraction", "operator scoring", "migration", "logging / others"]:
            rows.append([scale, module, "REQUIRES RERUN", "REQUIRES RERUN"])

    write_csv(ROOT / "runtime_profile.csv", ["problem", "module", "total_time_sec", "percent_runtime"], rows)
    write_csv(ROOT / "avg_eval_time.csv", ["problem", "runtime_mean_sec", "estimated_total_evals", "estimated_avg_eval_sec", "notes"], approx_rows)


if __name__ == "__main__":
    main()
