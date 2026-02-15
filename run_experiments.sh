#!/usr/bin/env bash
set -euo pipefail

# CC-HIHH operator ablation batch runner
# Usage:
#   bash run_experiments.sh [EXE_PATH]
#
# Optional envs:
#   DATA_DIR=./data
#   OUT_ROOT=./results/operator_ablation
#   RUNS=10
#   GENS=10000
#   LOG_EVERY=50
#   NSUBPOP=8
#   REDUCED_EXTRA_ARGS="--op_mode bandit ..."  # custom reduced-ops args (if implemented)
#   TOP_EXTRA_ARGS="--op_mode bandit ..."      # custom top-ops args (if implemented)

EXE_PATH="${1:-./build_codex/Release/CED_Schedule.exe}"
DATA_DIR="${DATA_DIR:-./data}"
OUT_ROOT="${OUT_ROOT:-./results/operator_ablation}"
RUNS="${RUNS:-10}"
GENS="${GENS:-10000}"
LOG_EVERY="${LOG_EVERY:-50}"
NSUBPOP="${NSUBPOP:-8}"

if [[ ! -f "$EXE_PATH" ]]; then
  echo "[ERROR] executable not found: $EXE_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

SCALES=(100 200 500)

scale_data_file() {
  case "$1" in
    100) echo "data_matrix_100.txt" ;;
    200) echo "data_matrix_T200_E100_D300.txt" ;;
    500) echo "data_matrix_T500_E200_D800.txt" ;;
    *) echo "unsupported scale: $1" >&2; exit 1 ;;
  esac
}

scale_dims() {
  case "$1" in
    100) echo "--cnum 100 --enum 100 --dnum 300 --tnum 100 --mopt 5" ;;
    200) echo "--cnum 100 --enum 100 --dnum 300 --tnum 200 --mopt 5" ;;
    500) echo "--cnum 200 --enum 200 --dnum 800 --tnum 500 --mopt 5" ;;
    *) echo "unsupported scale: $1" >&2; exit 1 ;;
  esac
}

run_variant() {
  local variant="$1"
  local scale="$2"
  local run_id="$3"
  local seed="$4"
  local extra_args="$5"

  local out_dir="$OUT_ROOT/T${scale}/${variant}"
  mkdir -p "$out_dir"

  local op_freq="$out_dir/op_freq_${variant}_T${scale}_run${run_id}.csv"
  local log_txt="$out_dir/final_${variant}_T${scale}_run${run_id}.txt"

  local weight_off="$out_dir/op_weights_offload_T${scale}_run${run_id}.csv"
  local weight_seq="$out_dir/op_weights_seq_T${scale}_run${run_id}.csv"
  local weight_dev="$out_dir/op_weights_dev_T${scale}_run${run_id}.csv"
  local reward_csv="$out_dir/op_rewards_T${scale}_run${run_id}.csv"
  local global_csv="$out_dir/global_stats_${variant}_T${scale}_run${run_id}.csv"

  local data_file
  data_file="$(scale_data_file "$scale")"
  local dims
  dims="$(scale_dims "$scale")"

  local cmd=(
    "$EXE_PATH"
    --solver CCHIHH
    --stable
    --resample_gate 15
    --data_dir "$DATA_DIR"
    --data_file "$data_file"
    --generations "$GENS"
    --log_every "$LOG_EVERY"
    --nsubpop "$NSUBPOP"
    --seed "$seed"
    --cchihh_op_stats "$op_freq"
    --cchihh_op_stats_every "$LOG_EVERY"
    --cchihh_global_stats "$global_csv"
    --cchihh_global_stats_every "$LOG_EVERY"
  )

  if [[ "$variant" == "full" ]]; then
    cmd+=(
      --op_mode bandit
      --cchihh_weight_log_offload "$weight_off"
      --cchihh_weight_log_seq "$weight_seq"
      --cchihh_weight_log_dev "$weight_dev"
      --cchihh_weight_log_every "$LOG_EVERY"
      --cchihh_reward_log "$reward_csv"
    )
  fi

  # shellcheck disable=SC2206
  cmd+=( $dims )
  # shellcheck disable=SC2206
  cmd+=( $extra_args )

  echo "[RUN] variant=${variant} scale=T${scale} run=${run_id} seed=${seed}"
  "${cmd[@]}" > "$log_txt" 2>&1
}

for scale in "${SCALES[@]}"; do
  for run_id in $(seq 0 $((RUNS - 1))); do
    seed=$((run_id + 1))

    run_variant "full" "$scale" "$run_id" "$seed" "--op_mode bandit"
    run_variant "random" "$scale" "$run_id" "$seed" "--op_mode random"
    run_variant "roundrobin" "$scale" "$run_id" "$seed" "--op_mode roundrobin"
    run_variant "fixedbest" "$scale" "$run_id" "$seed" "--cchihh_fixed_ops"

    # ReducedOps / TopOps are workflow hooks; pass custom args when corresponding support is enabled.
    if [[ -n "${REDUCED_EXTRA_ARGS:-}" ]]; then
      run_variant "reduced" "$scale" "$run_id" "$seed" "$REDUCED_EXTRA_ARGS"
    fi
    if [[ -n "${TOP_EXTRA_ARGS:-}" ]]; then
      run_variant "top" "$scale" "$run_id" "$seed" "$TOP_EXTRA_ARGS"
    fi
  done
done

echo "All runs completed. Outputs under: $OUT_ROOT"
