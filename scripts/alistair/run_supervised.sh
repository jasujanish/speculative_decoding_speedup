#!/usr/bin/env bash
# Resumable fire-and-forget script for Alistair's supervised-depth ablations.
#
# Assignments (all 100k timesteps):
#   8b  × {base, titan} × {1 epoch, 20 epochs}
#   14b × {base, titan} × {1 epoch, 20 epochs}
# = 8 training runs + 8 benchmarks.
#
# Per (model, variant): the dataset is collected once by the first train call
# and reused between the 1-epoch and 20-epoch runs. Only the train_t<N>[_titan]
# and benchmark_outputs_<variant> paths get wiped between epoch settings; the
# dataset stays. After both epoch settings for a (model, variant) pair are
# done, the dataset is wiped too.
#
# Resume behavior:
#   - Per-ablation sentinel at results/.<label>_supervised_<variant>_<tk>_<ep>.done
#   - Checks Modal volume for canonical promoted checkpoint / summary.csv to
#     skip already-finished train/benchmark steps.
#
# Run in screen/tmux:
#   screen -S sup
#   bash run_supervised_ablations.sh

set -euo pipefail

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results"
LOG="$REPO_ROOT/scripts/alistair/supervised.log"
SUP_SCRIPT="supervised_depth_modal/modal_supervised_depth_qwen3.py"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Supervised ablations started at $(date -Iseconds)"
echo "############################################################"

train_done() {
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
  local preset="$1"       # qwen3_8b | qwen3_14b
  local label="$2"        # 8b | 14b
  local variant="$3"      # base | titan
  local timesteps="$4"    # e.g. 100000
  local tk_label="$5"     # e.g. 100k
  local epochs="$6"       # 1 | 20
  local ep_label="$7"     # 1epoch | 20epochs
  local model_label="$8"  # "Qwen3 8B" | "Qwen3 14B"

  local suffix=""
  local policy_flags=()
  local method_dir="SupervisedLearning"
  if [[ "$variant" == "titan" ]]; then
    suffix="_titan"
    policy_flags=(--policy-variant titan)
    method_dir="SupervisedLearningTitan"
  fi
  local train_subdir="train_t${timesteps}${suffix}"
  local dataset_subdir="dataset_t${timesteps}${suffix}"
  local benchmark_subdir="$preset/supervised_depth/benchmark_outputs_${variant}"
  local local_dir="$RESULTS_ROOT/${label}/${method_dir}/${tk_label}_${epochs}"
  local staging="$REPO_ROOT/.sup_staging_${label}_${variant}_${tk_label}_${ep_label}"
  local sentinel="$RESULTS_ROOT/${label}/${method_dir}/.${tk_label}_${epochs}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] Supervised $label $variant $tk_label $ep_label"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel $sentinel exists -> already complete, skipping."
    return 0
  fi

  if train_done "$preset" "$train_subdir"; then
    echo "--- [$(date -Iseconds)] train already complete for $train_subdir, skipping ---"
  else
    echo "--- [$(date -Iseconds)] training: preset=$preset variant=$variant timesteps=$timesteps epochs=$epochs ---"
    modal run "$SUP_SCRIPT" \
      --action train-depth \
      --model-preset "$preset" \
      --total-timesteps "$timesteps" \
      --epochs "$epochs" \
      --checkpoint-epochs "$epochs" \
      --batch-size 256 \
      --lr 0.001 \
      "${policy_flags[@]}"
  fi

  if benchmark_done "$benchmark_subdir"; then
    echo "--- [$(date -Iseconds)] benchmark already complete for $benchmark_subdir, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmarking $variant ---"
    modal run "$SUP_SCRIPT" \
      --action benchmark-suite \
      --model-preset "$preset" \
      --depth-model-path "$preset/supervised_depth/$train_subdir/supervised_depth_model.pt" \
      --output-subdir "$benchmark_subdir" \
      --model-label "$model_label $variant $ep_label"
  fi

  echo "--- [$(date -Iseconds)] downloading results to $local_dir ---"
  rm -rf "$staging" "$local_dir"
  mkdir -p "$staging" "$local_dir"
  mkdir -p "$(dirname "$sentinel")"
  modal volume get ltd-qwen3-results "$preset/supervised_depth/$train_subdir" "$staging/"
  modal volume get ltd-qwen3-results "$benchmark_subdir"                      "$staging/"
  modal volume get ltd-qwen3-results \
    "$preset/supervised_depth/$dataset_subdir/dataset_manifest.json" \
    "$staging/dataset_manifest.json"

  cp -r "$staging/$train_subdir"                          "$local_dir/"
  cp -r "$staging/$(basename "$benchmark_subdir")"        "$local_dir/"
  mkdir -p "$local_dir/$dataset_subdir"
  cp "$staging/dataset_manifest.json" "$local_dir/$dataset_subdir/dataset_manifest.json"

  # Keep only the canonical promoted checkpoint; drop per-step + best.
  echo "--- [$(date -Iseconds)] pruning intermediate .pt checkpoints ---"
  find "$local_dir" -type f -name "supervised_depth_model_step_*.pt" -delete
  find "$local_dir" -type f -name "supervised_depth_model_best.pt"   -delete

  rm -rf "$staging"

  # Write sentinel BEFORE wiping modal so a crash in the wipe window doesn't
  # cause the ablation to re-train from scratch on the next invocation.
  mkdir -p "$(dirname "$sentinel")"
  date -Iseconds > "$sentinel"

  echo "--- [$(date -Iseconds)] wiping train + benchmark outputs on modal (keeping dataset) ---"
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
  echo "--- [$(date -Iseconds)] wiping dataset on modal: $preset/supervised_depth/dataset_t${timesteps}${suffix} ---"
  modal volume rm -r ltd-qwen3-results "$preset/supervised_depth/dataset_t${timesteps}${suffix}" || true
}

# --- Alistair's supervised ablations (all 100k timesteps) ---
# 1-epoch first so the 20-epoch run reuses the already-collected dataset.

run_supervised_ablation qwen3_8b  8b  base  100000 100k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b  8b  base  100000 100k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b base 100000

run_supervised_ablation qwen3_8b  8b  titan 100000 100k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b  8b  titan 100000 100k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b titan 100000

run_supervised_ablation qwen3_14b 14b base  100000 100k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  100000 100k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b base 100000

run_supervised_ablation qwen3_14b 14b titan 100000 100k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b titan 100000 100k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b titan 100000

echo ""
echo "############################################################"
echo "# ALL SUPERVISED ABLATIONS COMPLETE at $(date -Iseconds)"
echo "############################################################"
