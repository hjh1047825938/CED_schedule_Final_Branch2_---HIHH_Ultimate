## Retry Script

This repo includes a retry helper to scan logs, detect incomplete runs, and re-run failed jobs.

### Usage

```powershell
python scan_and_retry.py --log_dir .\results\gen10000_seed0_20 --retry_dir .\retries --max_retry 5
```

### Outputs

- `retry_plan.json` — initial failed runs and parsed parameters
- `unresolved.json` — logs that could not be parsed (no solver/seed)
- `final_report.json` — summary after retries finish

### Notes

- Success / error markers are defined at the top of `scan_and_retry.py`.
- Retries go into `--retry_dir` with `*_retryN.txt` filenames.
- The script is idempotent: it will not re-run successful logs or successful retries.
