#!/usr/bin/env python3
import argparse
import csv
import math
import re
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path


GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s*=\s*([0-9.+\-eE]+)")
RERUN_FILENAME_RE = re.compile(
    r"^(?P<variant>cchihh_(?:full|noCC|noHI|noMig|noGate|tgate\d+))_"
    r"(?P<scale>T\d+)"
    r"(?:_a(?P<alpha>[0-9.]+))?"
    r"_s(?P<seed>\d+)\.log$"
)
DSAC_FILENAME_RE = re.compile(r"^(?P<variant>DSAC_DE)_seed(?P<seed>\d+)\.txt$")
IMOMA_FILENAME_RE = re.compile(r"^(?P<variant>IMOMA)_seed(?P<seed>\d+)\.txt$")
CGA_FILENAME_RE = re.compile(r"^(?P<variant>CGA)_seed(?P<seed>\d+)\.txt$")
PPO_FILENAME_RE = re.compile(r"^(?P<variant>PPO)_seed(?P<seed>\d+)\.txt$")
OPS_FILENAME_RE = re.compile(
    r"^(?P<variant>cchihh_(?:full|noCC|noHI|noMig|noGate|tgate\d+))_"
    r"(?P<scale>T\d+)"
    r"(?:_a(?P<alpha>[0-9.]+))?"
    r"_s(?P<seed>\d+)_ops\.csv$"
)
WEIGHT_FILENAME_RE = re.compile(
    r"^(?P<variant>cchihh_(?:full|noCC|noHI|noMig|noGate|tgate\d+))_"
    r"(?P<scale>T\d+)"
    r"(?:_a(?P<alpha>[0-9.]+))?"
    r"_s(?P<seed>\d+)_(?P<kind>w_off|w_seq|w_dev)\.csv$"
)


@dataclass(frozen=True)
class RunMetadata:
    group: str
    variant: str
    scale: str
    alpha: float
    seed: int
    input_path: Path
    init_evals: int | None = None
    evals_per_gen: int | None = None


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if b"\x00" in raw[:256]:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8", "gbk", "latin1", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1", errors="ignore")


def parse_input_metadata(path: Path) -> RunMetadata:
    rerun_match = RERUN_FILENAME_RE.match(path.name)
    if rerun_match:
        alpha_raw = rerun_match.group("alpha")
        return RunMetadata(
            group=path.parent.name,
            variant=rerun_match.group("variant"),
            scale=rerun_match.group("scale"),
            alpha=float(alpha_raw) if alpha_raw is not None else 0.5,
            seed=int(rerun_match.group("seed")),
            input_path=path,
        )

    dsac_match = DSAC_FILENAME_RE.match(path.name)
    if dsac_match:
        alpha = 0.5
        if path.parent.parent.name == "dsac_de_multiscale":
            group = "dsac_de_multiscale"
            alpha = 0.5
        elif path.parent.parent.parent.name == "runs" and path.parent.parent.parent.parent.name == "alpha_sensitivity_cga_imoma_dsac":
            group = "dsac_alpha"
            alpha = float(path.parent.parent.name.replace("alpha_", ""))
        else:
            raise ValueError(f"Unsupported DSAC path: {path}")
        return RunMetadata(
            group=group,
            variant=dsac_match.group("variant"),
            scale=path.parent.name,
            alpha=alpha,
            seed=int(dsac_match.group("seed")),
            input_path=path,
            init_evals=40,
            evals_per_gen=40,
        )

    imoma_match = IMOMA_FILENAME_RE.match(path.name)
    if imoma_match:
        alpha = 0.5
        if path.parent.parent.name == "baseline" and path.parent.parent.parent.name == "cchihh_ablation_suite":
            group = "imoma_baseline"
        elif path.parent.parent.parent.name == "runs" and path.parent.parent.parent.parent.name == "alpha_sensitivity_cga_imoma_dsac":
            group = "imoma_alpha"
            alpha = float(path.parent.parent.name.replace("alpha_", ""))
        else:
            raise ValueError(f"Unsupported IMOMA path: {path}")
        return RunMetadata(
            group=group,
            variant=imoma_match.group("variant"),
            scale=path.parent.name,
            alpha=alpha,
            seed=int(imoma_match.group("seed")),
            input_path=path,
            init_evals=80,
            evals_per_gen=200,
        )

    cga_match = CGA_FILENAME_RE.match(path.name)
    if cga_match:
        alpha = 0.5
        if path.parent.parent.name == "baseline" and path.parent.parent.parent.name == "cchihh_ablation_suite":
            group = "cga_baseline"
        elif path.parent.parent.parent.name == "runs" and path.parent.parent.parent.parent.name == "alpha_sensitivity_cga_imoma_dsac":
            group = "cga_alpha"
            alpha = float(path.parent.parent.name.replace("alpha_", ""))
        else:
            raise ValueError(f"Unsupported CGA path: {path}")
        return RunMetadata(
            group=group,
            variant=cga_match.group("variant"),
            scale=path.parent.name,
            alpha=alpha,
            seed=int(cga_match.group("seed")),
            input_path=path,
            init_evals=40,
            evals_per_gen=40,
        )

    ppo_match = PPO_FILENAME_RE.match(path.name)
    if ppo_match:
        if path.parent.parent.name == "baseline" and path.parent.parent.parent.name == "cchihh_ablation_suite":
            return RunMetadata(
                group="ppo_baseline",
                variant=ppo_match.group("variant"),
                scale=path.parent.name,
                alpha=0.5,
                seed=int(ppo_match.group("seed")),
                input_path=path,
                init_evals=0,
                evals_per_gen=40,
            )
        raise ValueError(f"Unsupported PPO path: {path}")

    raise ValueError(f"Unsupported input filename: {path}")


def parse_cchihh_csv_metadata(path: Path) -> RunMetadata:
    for pattern in (OPS_FILENAME_RE, WEIGHT_FILENAME_RE):
        match = pattern.match(path.name)
        if match:
            alpha_raw = match.groupdict().get("alpha")
            return RunMetadata(
                group=path.parent.name,
                variant=match.group("variant"),
                scale=match.group("scale"),
                alpha=float(alpha_raw) if alpha_raw is not None else 0.5,
                seed=int(match.group("seed")),
                input_path=path,
            )
    raise ValueError(f"Unsupported CCHIHH CSV filename: {path}")


def parse_series_from_log(path: Path) -> dict[int, float]:
    series: dict[int, float] = {}
    for line in read_text_auto(path).splitlines():
        match = GEN_RE.match(line.strip().replace("\x00", ""))
        if match:
            series[int(match.group(1))] = float(match.group(2))
    return dict(sorted(series.items()))


def gen_to_eval(gen: int, init_evals: int, evals_per_gen: int) -> int:
    return init_evals + gen * evals_per_gen


def build_eval_curve(
    series: dict[int, float],
    gen_interval: int,
    max_eval_budget: int,
    init_evals: int,
    evals_per_gen: int,
) -> list[tuple[int, float]]:
    if gen_interval <= 0:
        raise ValueError("gen_interval must be positive")
    if not series:
        return []

    sorted_points = sorted(series.items())
    cutoff_gen = max(0, (max_eval_budget - init_evals) // evals_per_gen)
    max_logged_gen = min(sorted_points[-1][0], cutoff_gen)
    if max_logged_gen < gen_interval:
        return []

    targets = range(
        int(math.ceil(sorted_points[0][0] / gen_interval) * gen_interval),
        max_logged_gen + 1,
        gen_interval,
    )

    curve: list[tuple[int, float]] = []
    point_idx = 0
    best_so_far = None
    for target in targets:
        while point_idx < len(sorted_points) and sorted_points[point_idx][0] <= target:
            gen, fit = sorted_points[point_idx]
            best_so_far = fit if best_so_far is None else min(best_so_far, fit)
            point_idx += 1
        if best_so_far is None:
            continue
        curve.append((gen_to_eval(target, init_evals, evals_per_gen), best_so_far))
    return curve


def scan_rerun_logs(results_root: Path) -> list[Path]:
    logs: list[Path] = []
    if not results_root.exists():
        return logs
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and child.name.startswith("rerun"):
            logs.extend(sorted(p for p in child.glob("*.log") if RERUN_FILENAME_RE.match(p.name)))
    return logs


def scan_cchihh_operator_tables(results_root: Path) -> list[Path]:
    tables: list[Path] = []
    if not results_root.exists():
        return tables
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and child.name.startswith("rerun"):
            tables.extend(sorted(p for p in child.glob("*_ops.csv") if OPS_FILENAME_RE.match(p.name)))
    return tables


def scan_cchihh_weight_tables(results_root: Path) -> list[Path]:
    tables: list[Path] = []
    if not results_root.exists():
        return tables
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and child.name.startswith("rerun"):
            tables.extend(
                sorted(
                    p
                    for p in child.glob("*.csv")
                    if WEIGHT_FILENAME_RE.match(p.name)
                )
            )
    return tables


def scan_dsac_logs(outputs_results_root: Path) -> list[Path]:
    logs: list[Path] = []
    baseline_root = outputs_results_root / "dsac_de_multiscale"
    if baseline_root.exists():
        for scale_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
            logs.extend(sorted(p for p in scale_dir.glob("DSAC_DE_seed*.txt") if DSAC_FILENAME_RE.match(p.name)))

    alpha_root = outputs_results_root / "alpha_sensitivity_cga_imoma_dsac" / "runs"
    if alpha_root.exists():
        for alpha_dir in sorted(p for p in alpha_root.iterdir() if p.is_dir() and p.name.startswith("alpha_")):
            for scale_dir in sorted(p for p in alpha_dir.iterdir() if p.is_dir()):
                logs.extend(sorted(p for p in scale_dir.glob("DSAC_DE_seed*.txt") if DSAC_FILENAME_RE.match(p.name)))
    return logs


def scan_imoma_logs(outputs_results_root: Path) -> list[Path]:
    logs: list[Path] = []
    baseline_root = outputs_results_root / "cchihh_ablation_suite" / "baseline"
    if baseline_root.exists():
        for scale_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
            logs.extend(sorted(p for p in scale_dir.glob("IMOMA_seed*.txt") if IMOMA_FILENAME_RE.match(p.name)))

    alpha_root = outputs_results_root / "alpha_sensitivity_cga_imoma_dsac" / "runs"
    if alpha_root.exists():
        for alpha_dir in sorted(p for p in alpha_root.iterdir() if p.is_dir() and p.name.startswith("alpha_")):
            for scale_dir in sorted(p for p in alpha_dir.iterdir() if p.is_dir()):
                logs.extend(sorted(p for p in scale_dir.glob("IMOMA_seed*.txt") if IMOMA_FILENAME_RE.match(p.name)))
    return logs


def scan_cga_logs(outputs_results_root: Path) -> list[Path]:
    logs: list[Path] = []
    baseline_root = outputs_results_root / "cchihh_ablation_suite" / "baseline"
    if baseline_root.exists():
        for scale_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
            logs.extend(sorted(p for p in scale_dir.glob("CGA_seed*.txt") if CGA_FILENAME_RE.match(p.name)))

    alpha_root = outputs_results_root / "alpha_sensitivity_cga_imoma_dsac" / "runs"
    if alpha_root.exists():
        for alpha_dir in sorted(p for p in alpha_root.iterdir() if p.is_dir() and p.name.startswith("alpha_")):
            for scale_dir in sorted(p for p in alpha_dir.iterdir() if p.is_dir()):
                logs.extend(sorted(p for p in scale_dir.glob("CGA_seed*.txt") if CGA_FILENAME_RE.match(p.name)))
    return logs


def scan_ppo_logs(outputs_results_root: Path) -> list[Path]:
    logs: list[Path] = []
    baseline_root = outputs_results_root / "cchihh_ablation_suite" / "baseline"
    if baseline_root.exists():
        for scale_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
            logs.extend(sorted(p for p in scale_dir.glob("PPO_seed*.txt") if PPO_FILENAME_RE.match(p.name)))
    return logs


def scan_input_logs(results_root: Path, extra_input_roots: Iterable[Path] | None = None) -> list[Path]:
    logs = scan_rerun_logs(results_root)
    for root in extra_input_roots or []:
        logs.extend(scan_dsac_logs(root))
        logs.extend(scan_imoma_logs(root))
        logs.extend(scan_cga_logs(root))
        logs.extend(scan_ppo_logs(root))
    return sorted(logs)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_curve_csv(path: Path, curve: list[tuple[int, float]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_count", "best_fitness"])
        for eval_count, best_fitness in curve:
            writer.writerow([eval_count, f"{best_fitness:.15g}"])


def build_operator_output_path(output_root: Path, meta: RunMetadata) -> Path:
    alpha_dir = f"alpha{meta.alpha:.1f}"
    return output_root / meta.group / alpha_dir / meta.scale / f"{meta.variant}_{meta.scale}_s{meta.seed}_ops_eval.csv"


def build_weight_output_path(output_root: Path, meta: RunMetadata, kind: str) -> Path:
    alpha_dir = f"alpha{meta.alpha:.1f}"
    return output_root / meta.group / alpha_dir / meta.scale / f"{meta.variant}_{meta.scale}_s{meta.seed}_{kind}_eval.csv"


def convert_gen_keyed_csv(
    input_path: Path,
    output_path: Path,
    max_eval_budget: int,
    init_evals: int,
    evals_per_gen: int,
) -> bool:
    rows_to_write: list[list[str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False
        if header[0] != "gen":
            raise ValueError(f"Unsupported gen-keyed CSV header in {input_path}")
        out_header = ["eval_count", *header[1:]]
        for row in reader:
            if not row:
                continue
            gen = int(float(row[0]))
            eval_count = gen_to_eval(gen, init_evals, evals_per_gen)
            if eval_count > max_eval_budget:
                continue
            rows_to_write.append([str(eval_count), *row[1:]])

    if not rows_to_write:
        return False

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(rows_to_write)
    return True


def write_index_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "variant",
                "scale",
                "alpha",
                "seed",
                "points",
                "final_eval",
                "final_best_fitness",
                "input_path",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "variant",
                "scale",
                "alpha",
                "num_seeds",
                "mean_final_best_fitness",
                "std_final_best_fitness",
                "min_final_best_fitness",
                "max_final_best_fitness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_output_path(output_root: Path, meta: RunMetadata) -> Path:
    alpha_dir = f"alpha{meta.alpha:.1f}"
    return output_root / meta.group / alpha_dir / meta.scale / f"{meta.variant}_{meta.scale}_s{meta.seed}_eval.csv"


def convert_tree(
    results_root: Path,
    output_root: Path,
    gen_interval: int,
    max_eval_budget: int,
    init_evals: int,
    evals_per_gen: int,
    extra_input_roots: Iterable[Path] | None = None,
) -> int:
    index_rows: list[dict[str, object]] = []
    summary_buckets: dict[tuple[str, str, str, float], list[float]] = {}
    written = 0

    for log_path in scan_input_logs(results_root, extra_input_roots=extra_input_roots):
        meta = parse_input_metadata(log_path)
        series = parse_series_from_log(log_path)
        curve = build_eval_curve(
            series=series,
            gen_interval=gen_interval,
            max_eval_budget=max_eval_budget,
            init_evals=meta.init_evals if meta.init_evals is not None else init_evals,
            evals_per_gen=meta.evals_per_gen if meta.evals_per_gen is not None else evals_per_gen,
        )
        if not curve:
            continue

        out_path = build_output_path(output_root, meta)
        write_curve_csv(out_path, curve)
        written += 1

        final_eval, final_best = curve[-1]
        index_rows.append(
            {
                "group": meta.group,
                "variant": meta.variant,
                "scale": meta.scale,
                "alpha": f"{meta.alpha:.1f}",
                "seed": meta.seed,
                "points": len(curve),
                "final_eval": final_eval,
                "final_best_fitness": f"{final_best:.15g}",
                "input_path": str(log_path),
                "output_path": str(out_path),
            }
        )
        summary_buckets.setdefault((meta.group, meta.variant, meta.scale, meta.alpha), []).append(final_best)

    write_index_csv(output_root / "index.csv", index_rows)

    summary_rows: list[dict[str, object]] = []
    for (group, variant, scale, alpha), values in sorted(summary_buckets.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        summary_rows.append(
            {
                "group": group,
                "variant": variant,
                "scale": scale,
                "alpha": f"{alpha:.1f}",
                "num_seeds": len(values),
                "mean_final_best_fitness": f"{mean:.15g}",
                "std_final_best_fitness": f"{std:.15g}",
                "min_final_best_fitness": f"{min(values):.15g}",
                "max_final_best_fitness": f"{max(values):.15g}",
            }
        )
    write_summary_csv(output_root / "summary.csv", summary_rows)
    return written


def convert_cchihh_operator_tables(
    results_root: Path,
    output_root: Path,
    max_eval_budget: int,
    init_evals: int,
    evals_per_gen: int,
) -> int:
    written = 0
    for table_path in scan_cchihh_operator_tables(results_root):
        meta = parse_cchihh_csv_metadata(table_path)
        out_path = build_operator_output_path(output_root, meta)
        if convert_gen_keyed_csv(
            input_path=table_path,
            output_path=out_path,
            max_eval_budget=max_eval_budget,
            init_evals=meta.init_evals if meta.init_evals is not None else init_evals,
            evals_per_gen=meta.evals_per_gen if meta.evals_per_gen is not None else evals_per_gen,
        ):
            written += 1
    return written


def convert_cchihh_weight_tables(
    results_root: Path,
    output_root: Path,
    max_eval_budget: int,
    init_evals: int,
    evals_per_gen: int,
) -> int:
    written = 0
    for table_path in scan_cchihh_weight_tables(results_root):
        match = WEIGHT_FILENAME_RE.match(table_path.name)
        if match is None:
            continue
        meta = parse_cchihh_csv_metadata(table_path)
        out_path = build_weight_output_path(output_root, meta, match.group("kind"))
        if convert_gen_keyed_csv(
            input_path=table_path,
            output_path=out_path,
            max_eval_budget=max_eval_budget,
            init_evals=meta.init_evals if meta.init_evals is not None else init_evals,
            evals_per_gen=meta.evals_per_gen if meta.evals_per_gen is not None else evals_per_gen,
        ):
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert rerun CCHIHH convergence logs into eval-based CSV curves."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="results/eval")
    parser.add_argument("--extra-input-root", action="append", default=["outputs/results"])
    parser.add_argument("--gen-interval", type=int, default=50)
    parser.add_argument("--n-pop", type=int, default=40)
    parser.add_argument("--k-blocks", type=int, default=3)
    parser.add_argument("--init-evals", type=int, default=40)
    parser.add_argument("--max-eval-budget", type=int, default=400000)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    evals_per_gen = args.n_pop * args.k_blocks

    written = convert_tree(
        results_root=results_root,
        output_root=output_root,
        gen_interval=args.gen_interval,
        max_eval_budget=args.max_eval_budget,
        init_evals=args.init_evals,
        evals_per_gen=evals_per_gen,
        extra_input_roots=[Path(p) for p in args.extra_input_root],
    )
    ops_written = convert_cchihh_operator_tables(
        results_root=results_root,
        output_root=output_root,
        max_eval_budget=args.max_eval_budget,
        init_evals=args.init_evals,
        evals_per_gen=evals_per_gen,
    )
    weight_written = convert_cchihh_weight_tables(
        results_root=results_root,
        output_root=output_root,
        max_eval_budget=args.max_eval_budget,
        init_evals=args.init_evals,
        evals_per_gen=evals_per_gen,
    )
    print(
        f"Wrote {written} eval curve files, {ops_written} operator-frequency files, "
        f"and {weight_written} weight files to {output_root}"
    )


if __name__ == "__main__":
    main()
