import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCALES = {
    100: dict(data_file='data_matrix_100.txt', cnum=100, enum=100, dnum=300, tnum=100, mopt=5),
    200: dict(data_file='data_matrix_T200_E100_D300.txt', cnum=100, enum=100, dnum=300, tnum=200, mopt=5),
    500: dict(data_file='data_matrix_T500_E200_D800.txt', cnum=200, enum=200, dnum=800, tnum=500, mopt=5),
}

VARIANTS = {
    'full': ['--op_mode', 'bandit'],
    'random': ['--op_mode', 'random'],
    'roundrobin': ['--op_mode', 'roundrobin'],
    'fixedbest': ['--cchihh_fixed_ops'],
}


def build_cmd(exe: Path, data_dir: Path, out_root: Path, gens: int, log_every: int, nsubpop: int,
              scale: int, run_id: int, variant: str):
    spec = SCALES[scale]
    seed = run_id + 1
    out_dir = out_root / f'T{scale}' / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        '--solver', 'CCHIHH',
        '--stable',
        '--resample_gate', '15',
        '--data_dir', str(data_dir),
        '--data_file', spec['data_file'],
        '--generations', str(gens),
        '--log_every', str(log_every),
        '--nsubpop', str(nsubpop),
        '--seed', str(seed),
        '--cnum', str(spec['cnum']), '--enum', str(spec['enum']), '--dnum', str(spec['dnum']), '--tnum', str(spec['tnum']), '--mopt', str(spec['mopt']),
        '--cchihh_op_stats', str(out_dir / f'op_freq_{variant}_T{scale}_run{run_id}.csv'),
        '--cchihh_op_stats_every', str(log_every),
        '--cchihh_global_stats', str(out_dir / f'global_stats_{variant}_T{scale}_run{run_id}.csv'),
        '--cchihh_global_stats_every', str(log_every),
    ]

    if variant == 'full':
        cmd += [
            '--cchihh_weight_log_offload', str(out_dir / f'op_weights_offload_T{scale}_run{run_id}.csv'),
            '--cchihh_weight_log_seq', str(out_dir / f'op_weights_seq_T{scale}_run{run_id}.csv'),
            '--cchihh_weight_log_dev', str(out_dir / f'op_weights_dev_T{scale}_run{run_id}.csv'),
            '--cchihh_weight_log_every', str(log_every),
            '--cchihh_reward_log', str(out_dir / f'op_rewards_T{scale}_run{run_id}.csv'),
        ]

    cmd += VARIANTS[variant]
    log_file = out_dir / f'final_{variant}_T{scale}_run{run_id}.txt'
    return cmd, log_file


def run_one(job):
    cmd, log_file = job
    with log_file.open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, str(log_file)


def main():
    ap = argparse.ArgumentParser(description='Run CC-HIHH operator ablation in parallel')
    ap.add_argument('--exe', default='build_codex/Release/CED_Schedule.exe')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--out_root', default='results/operator_ablation')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--gens', type=int, default=10000)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--nsubpop', type=int, default=8)
    ap.add_argument('--workers', type=int, default=15)
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f'Executable not found: {exe}')

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for scale in [100, 200, 500]:
        for variant in VARIANTS.keys():
            for run_id in range(args.runs):
                jobs.append(build_cmd(exe, data_dir, out_root, args.gens, args.log_every, args.nsubpop, scale, run_id, variant))

    total = len(jobs)
    print(f'Total jobs: {total}, workers={args.workers}, seeds=1..{args.runs}')
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            rc, path = fut.result()
            done += 1
            if rc != 0:
                failed.append(path)
            if done % 10 == 0 or done == total:
                print(f'Progress: {done}/{total}, failed={len(failed)}')

    if failed:
        print('Failed logs:')
        for p in failed[:20]:
            print('  ', p)
        raise SystemExit(1)
    print('All jobs completed successfully.')


if __name__ == '__main__':
    main()
