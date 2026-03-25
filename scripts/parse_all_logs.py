from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CURVE_PATTERNS = [
    re.compile(r"Eval\s+(\d+):\s+best_fit\s*=\s*([\d.eE+-]+)"),
    re.compile(r"Gen\s+(\d+):\s+best_fit\s*=\s*([\d.eE+-]+)"),
]
FINAL_PATTERNS = [
    re.compile(r"The best (?:solution|scalar solution)\s*=\s*([\d.eE+-]+)"),
    re.compile(r"best_fit\s*=\s*([\d.eE+-]+)"),
]
TIME_PATTERN = re.compile(r"Time\s*=\s*([\d.]+)\s*s")
FILE_PATTERN = re.compile(
    r"(?P<solver>[A-Za-z0-9_-]+)_(?P<scale>T\d+)(?:_a(?P<alpha>\d+(?:\.\d+)?))?_s(?P<seed>\d+)",
    re.IGNORECASE,
)


def parse_log(path: Path) -> tuple[list[tuple[int, float]], float | None, float | None, str | None]:
    curve: list[tuple[int, float]] = []
    final_fit: float | None = None
    runtime_s: float | None = None
    x_kind: str | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        for pattern in CURVE_PATTERNS:
            match = pattern.match(line)
            if match:
                x_kind = "eval" if line.startswith("Eval") else "gen"
                curve.append((int(match.group(1)), float(match.group(2))))
                break
        else:
            for pattern in FINAL_PATTERNS:
                match = pattern.match(line)
                if match:
                    final_fit = float(match.group(1))
                    break
            match = TIME_PATTERN.match(line)
            if match:
                runtime_s = float(match.group(1))

    return curve, final_fit, runtime_s, x_kind


def infer_meta(path: Path) -> dict[str, str]:
    match = FILE_PATTERN.search(path.stem)
    meta = {"solver": "", "scale": "", "alpha": "", "variant": "", "seed": ""}
    if match:
        meta["solver"] = match.group("solver") or ""
        meta["scale"] = match.group("scale") or ""
        meta["alpha"] = match.group("alpha") or ""
        meta["seed"] = match.group("seed") or ""

    lowered = str(path).lower()
    if "nocc" in lowered:
        meta["variant"] = "noCC"
    elif "nohi" in lowered:
        meta["variant"] = "noHI"
    elif "nocb" in lowered:
        meta["variant"] = "noCB"
    elif "nomig" in lowered:
        meta["variant"] = "noMig"
    elif "nogate" in lowered:
        meta["variant"] = "noGate"
    elif "tgate" in lowered:
        meta["variant"] = "tgate"
    else:
        meta["variant"] = "full"
    return meta


def write_curve_csv(path: Path, curve: list[tuple[int, float]], x_kind: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_label = "eval" if x_kind == "eval" else "gen"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_label, "best_fit"])
        writer.writerows(curve)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse experiment log files into CSV summaries.")
    parser.add_argument("--input", default="results/eval", help="Root directory to scan for log files.")
    parser.add_argument("--output", default="results/eval/parsed_logs", help="Output directory.")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    conv_root = output_root / "convergence_data"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    logs = sorted(input_root.rglob("*.log"))
    for log_path in logs:
        curve, final_fit, runtime_s, x_kind = parse_log(log_path)
        meta = infer_meta(log_path)
        if curve:
            write_curve_csv(conv_root / f"{log_path.stem}.csv", curve, x_kind)
        rows.append(
            {
                "solver": meta["solver"],
                "scale": meta["scale"],
                "alpha": meta["alpha"],
                "variant": meta["variant"],
                "seed": meta["seed"],
                "final_fit": "" if final_fit is None else final_fit,
                "runtime_s": "" if runtime_s is None else runtime_s,
                "curve_points": len(curve),
                "curve_kind": x_kind or "",
                "log_path": str(log_path),
            }
        )

    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "solver",
                "scale",
                "alpha",
                "variant",
                "seed",
                "final_fit",
                "runtime_s",
                "curve_points",
                "curve_kind",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned logs: {len(logs)}")
    print(output_root / "summary.csv")
    print(conv_root)


if __name__ == "__main__":
    main()
