import csv
import math
import re
from pathlib import Path
import matplotlib.pyplot as plt

LINE_RE = re.compile(r"^The best solution\s*=\s*([0-9.+\-eE]+)")
BACKUP_RE = re.compile(r"^Best fitness\s*=\s*([0-9.+\-eE]+)")
GEN_RE = re.compile(r"^Gen\s+(\d+):\s+best_fit\s+=\s+([0-9.+\-eE]+)")

root = Path(r"C:\\\\Users\\\\Hu\\\\Desktop\\\\论文\\\\CED_schedule_Final\\\\results\\\\cchihh_t500_resample_gate_scan_gen10000_parallel15")
logs = sorted(root.glob("CCHIHH_stable_T500_gate*_seed*.txt"))
if len(logs) != 15:
    raise SystemExit(f"Expected 15 logs, got {len(logs)}")

raw_rows = []
by_gate = {}
for p in logs:
    name = p.stem
    m = re.match(r"CCHIHH_stable_T500_gate(\d+)_seed(\d+)", name)
    if not m:
        continue
    gate = int(m.group(1))
    seed = int(m.group(2))

    text = p.read_text(encoding='utf-8', errors='ignore').replace('\x00','')
    final = None
    last_gen = None
    for line in text.splitlines():
        s = line.strip()
        m1 = LINE_RE.match(s)
        if m1:
            final = float(m1.group(1))
        m2 = BACKUP_RE.match(s)
        if m2 and final is None:
            final = float(m2.group(1))
        mg = GEN_RE.match(s)
        if mg:
            last_gen = float(mg.group(2))
    if final is None:
        if last_gen is None:
            raise SystemExit(f"No fitness found in {p}")
        final = last_gen

    raw_rows.append((gate, seed, final))
    by_gate.setdefault(gate, []).append(final)

raw_rows.sort()
with (root / "t500_gate_fitness_raw.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gate", "seed", "fitness"])
    w.writerows(raw_rows)

summary = []
for gate in sorted(by_gate):
    vals = by_gate[gate]
    mean = sum(vals)/len(vals)
    var = sum((x-mean)**2 for x in vals)/len(vals)
    std = math.sqrt(var)
    summary.append((gate, mean, std, min(vals), max(vals)))

with (root / "t500_gate_fitness_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gate", "mean_fitness", "std_fitness", "min_fitness", "max_fitness"])
    w.writerows(summary)

gates = [r[0] for r in summary]
means = [r[1] for r in summary]
stds = [r[2] for r in summary]

plt.figure(figsize=(10,6))
plt.plot(gates, means, color="#d62728", linewidth=2, marker="o", markersize=6, label="CCHIHH stable (T500)")
low = [m-s for m,s in zip(means,stds)]
high = [m+s for m,s in zip(means,stds)]
plt.fill_between(gates, low, high, color="#d62728", alpha=0.18, label="mean ± std")
plt.xticks(gates)
plt.xlabel(r"$T_{gate}$")
plt.ylabel("Best fitness")
plt.title("T500: Fitness vs resample gate (gen=10000)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

png = root / "T500_fitness_vs_Tgate_mean_seed1_3.png"
pdf = root / "T500_fitness_vs_Tgate_mean_seed1_3.pdf"
plt.savefig(png, dpi=220)
plt.savefig(pdf)
print(png)
print(pdf)
print(root / "t500_gate_fitness_summary.csv")

