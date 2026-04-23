#!/usr/bin/env bash
# Neil's supervised-depth ablations. Runs in parallel with run_neil_ltd.sh.
#
#   14b, 10k: {base, titan} × {1, 20 ep}
#   8b,  50k: {base, titan} × {1, 20 ep}
#   14b, 50k: {base, titan} × {1, 20 ep}
# = 12 training runs + 12 benchmarks.
#
# Resume-safe. Billing: MODAL_PROFILE=neil.

set -euo pipefail
export MODAL_PROFILE=neil

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results"
LOG="$REPO_ROOT/neil_supervised_ablations.log"
SUP_SCRIPT="supervised_depth_modal/modal_supervised_depth_qwen3.py"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Neil Supervised ablations started at $(date -Iseconds)"
echo "# MODAL_PROFILE=$MODAL_PROFILE"
echo "############################################################"

modal profile list || true

train_done_sup() {
  local preset="$1"
  local train_subdir="$2"
  modal volume ls ltd-qwen3-results "$preset/supervised_depth/$train_subdir" 2>/dev/null \
    | grep -q "supervised_depth_model.pt"
}

benchmark_done() {
  local remote_path="$1"
  modal volume ls ltd-qwen3-results "$remote_path" 2>/dev/null \
    | grep -q "summary.csv"
}

run_supervised_ablation() {
  local preset="$1"
  local label="$2"
  local variant="$3"      # base | titan
  local timesteps="$4"
  local tk_label="$5"
  local epochs="$6"
  local ep_label="$7"
  local model_label="$8"

  local suffix=""
  local policy_flags=()
  local method_dir="SupervisedLearning"
  if [[ "$variant" == "titan" ]]; then
    suffix="_titan"
    policy_flags=(--policy-variant titan)
    method_dir="SupervisedLearningTitan"
  fi
  local train_subdir="train_t${timesteps}${suffix}"
  local benchmark_subdir="$preset/supervised_depth/benchmark_outputs_${variant}"
  local local_dir="$RESULTS_ROOT/${label}/${method_dir}/${tk_label}_${epochs}"
  local staging="$REPO_ROOT/.sup_staging_neil_${label}_${variant}_${tk_label}_${ep_label}"
  local sentinel="$RESULTS_ROOT/${label}/${method_dir}/.${tk_label}_${epochs}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] Supervised $label $variant $tk_label $ep_label"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel exists -> skipping."
    return 0
  fi

  if train_done_sup "$preset" "$train_subdir"; then
    echo "--- train already complete for $train_subdir ---"
  else
    echo "--- [$(date -Iseconds)] training preset=$preset variant=$variant timesteps=$timesteps epochs=$epochs ---"
    modal run "$SUP_SCRIPT" \
      --action train-depth --model-preset "$preset" \
      --total-timesteps "$timesteps" --epochs "$epochs" --checkpoint-epochs "$epochs" \
      --batch-size 256 --lr 0.001 \
      "${policy_flags[@]}"
  fi

  if benchmark_done "$benchmark_subdir"; then
    echo "--- benchmark already done ---"
  else
    echo "--- [$(date -Iseconds)] benchmarking $variant ---"
    modal run "$SUP_SCRIPT" \
      --action benchmark-suite --model-preset "$preset" \
      --depth-model-path "$preset/supervised_depth/$train_subdir/supervised_depth_model.pt" \
      --output-subdir "$benchmark_subdir" \
      --model-label "$model_label $variant $ep_label"
  fi

  echo "--- [$(date -Iseconds)] downloading to $local_dir ---"
  rm -rf "$staging" "$local_dir"
  mkdir -p "$staging" "$local_dir"
  mkdir -p "$(dirname "$sentinel")"
  modal volume get ltd-qwen3-results "$preset/supervised_depth/$train_subdir" "$staging/"
  modal volume get ltd-qwen3-results "$benchmark_subdir"                      "$staging/"
  cp -r "$staging/$train_subdir"                   "$local_dir/"
  cp -r "$staging/$(basename "$benchmark_subdir")" "$local_dir/"

  find "$local_dir" -type f -name "supervised_depth_model_step_*.pt" -delete
  find "$local_dir" -type f -name "supervised_depth_model_best.pt"   -delete
  rm -rf "$staging"

  mkdir -p "$(dirname "$sentinel")"
  date -Iseconds > "$sentinel"

  echo "--- wiping train + benchmark outputs on modal (keeping dataset) ---"
  modal volume rm -r ltd-qwen3-results "$preset/supervised_depth/$train_subdir" || true
  modal volume rm -r ltd-qwen3-results "$benchmark_subdir"                      || true

  echo "=== [$(date -Iseconds)] DONE Supervised $label $variant $tk_label $ep_label ==="
}

wipe_dataset() {
  local preset="$1"
  local variant="$2"
  local timesteps="$3"
  local suffix=""
  [[ "$variant" == "titan" ]] && suffix="_titan"
  echo "--- wiping dataset: $preset/supervised_depth/dataset_t${timesteps}${suffix} ---"
  modal volume rm -r ltd-qwen3-results "$preset/supervised_depth/dataset_t${timesteps}${suffix}" || true
}

# 14b, 10k
run_supervised_ablation qwen3_14b 14b base  10000 10k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  10000 10k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b base 10000
run_supervised_ablation qwen3_14b 14b titan 10000 10k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b titan 10000 10k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b titan 10000

# 8b, 50k
run_supervised_ablation qwen3_8b  8b  base  50000 50k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b  8b  base  50000 50k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b base 50000
run_supervised_ablation qwen3_8b  8b  titan 50000 50k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b  8b  titan 50000 50k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b titan 50000

# 14b, 50k
run_supervised_ablation qwen3_14b 14b base  50000 50k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  50000 50k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b base 50000
run_supervised_ablation qwen3_14b 14b titan 50000 50k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b titan 50000 50k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b titan 50000

echo ""
echo "############################################################"
echo "# Neil SUPERVISED ABLATIONS COMPLETE at $(date -Iseconds)"
echo "############################################################"
