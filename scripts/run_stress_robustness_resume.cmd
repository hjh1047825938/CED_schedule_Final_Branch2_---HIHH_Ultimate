@echo off
setlocal
cd /d "%~dp0.."
python -u scripts\run_stress_robustness_eval.py --instances T200,T500 --max_parallel 1
