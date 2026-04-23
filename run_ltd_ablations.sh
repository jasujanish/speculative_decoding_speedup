#!/usr/bin/env bash
# Resumable fire-and-forget script for LTD ablations.
#
# Each ablation does:
#   1. Six PPO stages: iter0_size, iter0_depth, iter1_size, iter2_depth,
#      iter3_size, iter4_depth.
#   2. Two benchmark-suite runs: co-op (iter3/iter4) and not co-op (iter0/iter0).
#   3. Download results into results/<label>_LTD_<tk>timesteps_{coop,notcoop}/.
#   4. Prune *.zip PPO policy files locally to stay lightweight.
#   5. Wipe the Modal volume and drop a sentinel file so re-runs skip this ablation.
#
# Resume behavior:
#   - A sentinel at results/.<label>_LTD_<tk>.done marks a fully completed ablation
#     and causes the ablation to be skipped on re-run.
#   - If no sentinel, each step (per-stage train, per-benchmark) is re-checked
#     against the Modal volume via `modal volume ls` and skipped if already done.
#   - To force-redo an ablation: delete the sentinel AND `modal volume rm -r`
#     the preset's iterative + benchmark_outputs paths.
#
# Run in screen/tmux:
#   screen -S ltd
#   bash run_ltd_ablations.sh
#   # Ctrl-A D to detach

set -euo pipefail

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results"
LOG="$REPO_ROOT/ltd_ablations.log"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LTD ablations started at $(date -Iseconds)"
echo "############################################################"

# Returns 0 if the canonical promoted checkpoint for a stage already exists on
# the Modal volume, meaning that stage has completed successfully.
stage_done() {
  local preset="$1"
  local stage="$2"
  local canonical
  if [[ "$stage" == *_size ]]; then
    canonical="ppo_speculative_decoder_controller_rebuttal.zip"
  else
    canonical="ppo_speculative_decoder_controller_v1_single_action.zip"
  fi
  modal volume ls ltd-qwen3-results "$preset/iterative/$stage" 2>/dev/null \
    | grep -q "$canonical"
}

# Returns 0 if benchmark-suite has produced summary.csv at the given remote path.
benchmark_done() {
  local remote_path="$1"
  modal volume ls ltd-qwen3-results "$remote_path" 2>/dev/null \
    | grep -q "summary.csv"
}

run_ltd_ablation() {
  local preset="$1"       # qwen3_8b | qwen3_14b
  local label="$2"        # 8b | 14b
  local timesteps="$3"    # 10000 | 100000
  local tk_label="$4"     # 10k | 100k
  local model_label="$5"  # "Qwen3 8B" | "Qwen3 14B"

  local coop_dir="$RESULTS_ROOT/${label}/LTD/${tk_label}_coop"
  local notcoop_dir="$RESULTS_ROOT/${label}/LTD/${tk_label}_notcoop"
  local staging="$REPO_ROOT/.ltd_staging_${label}_${tk_label}"
  local sentinel="$RESULTS_ROOT/${label}/LTD/.${tk_label}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] LTD $label $tk_label"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel $sentinel exists -> already complete, skipping."
    return 0
  fi

  # Six PPO stages, in order. Skip any already complete on the volume.
  for stage in iter0_size iter0_depth iter1_size iter2_depth iter3_size iter4_depth; do
    if stage_done "$preset" "$stage"; then
      echo "--- [$(date -Iseconds)] stage=$stage already complete, skipping ---"
      continue
    fi
    echo "--- [$(date -Iseconds)] stage=$stage preset=$preset timesteps=$timesteps ---"
    modal run modal_qwen3.py \
      --action iterative-train \
      --model-preset "$preset" \
      --current-stage "$stage" \
      --size-total-timesteps "$timesteps" \
      --depth-total-timesteps "$timesteps" \
      --batch-size 256 \
      --n-steps 2048 \
      --lr 0.001 \
      --validation-fraction 0.2 \
      --split-seed 42
  done

  if benchmark_done "$preset/iterative/benchmark_outputs"; then
    echo "--- [$(date -Iseconds)] co-op benchmark already done, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmark co-optimized (iter3/iter4) ---"
    modal run modal_qwen3.py \
      --action benchmark-suite \
      --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --model-label "$model_label"
  fi

  if benchmark_done "$preset/iterative/benchmark_outputs_iter0"; then
    echo "--- [$(date -Iseconds)] not-coop benchmark already done, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmark not co-optimized (iter0/iter0) ---"
    modal run modal_qwen3.py \
      --action benchmark-suite \
      --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter0_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter0_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --output-subdir "$preset/iterative/benchmark_outputs_iter0" \
      --model-label "$model_label iter0"
  fi

  # Pull everything down to a staging directory, then reorganize into Nish's
  # folder layout (coop: iter3_size/iter4_depth/benchmark_outputs,
  # notcoop: iter0_size/iter0_depth/benchmark_outputs_iter0).
  echo "--- [$(date -Iseconds)] downloading results ---"
  rm -rf "$staging"
  mkdir -p "$staging"
  modal volume get ltd-qwen3-results "$preset/iterative" "$staging/"

  rm -rf "$coop_dir" "$notcoop_dir"
  mkdir -p "$coop_dir" "$notcoop_dir"
  mkdir -p "$(dirname "$sentinel")"

  cp -r "$staging/iterative/iter3_size"          "$coop_dir/"
  cp -r "$staging/iterative/iter4_depth"         "$coop_dir/"
  cp -r "$staging/iterative/benchmark_outputs"   "$coop_dir/"

  cp -r "$staging/iterative/iter0_size"              "$notcoop_dir/"
  cp -r "$staging/iterative/iter0_depth"             "$notcoop_dir/"
  cp -r "$staging/iterative/benchmark_outputs_iter0" "$notcoop_dir/"

  # Drop PPO policy zips to keep local copy lightweight. JSON/JSONL metrics stay.
  echo "--- [$(date -Iseconds)] pruning *.zip policy files from local results ---"
  find "$coop_dir" "$notcoop_dir" -type f -name "*.zip" -delete

  rm -rf "$staging"

  # Write sentinel BEFORE wiping modal so a crash in the wipe window doesn't
  # cause the ablation to re-train from scratch on the next invocation.
  mkdir -p "$(dirname "$sentinel")"
  date -Iseconds > "$sentinel"

  echo "--- [$(date -Iseconds)] wiping modal volume for $preset ---"
  modal volume rm -r ltd-qwen3-results "$preset/iterative" || true
  echo "=== [$(date -Iseconds)] DONE LTD $label $tk_label (sentinel: $sentinel) ==="
}

# Edit this list to change the ablations this script runs.
# Args: <preset> <label> <timesteps> <tk_label> "<model_label>"
run_ltd_ablation qwen3_14b 14b 10000  10k  "Qwen3 14B"
run_ltd_ablation qwen3_8b  8b  100000 100k "Qwen3 8B"
run_ltd_ablation qwen3_14b 14b 100000 100k "Qwen3 14B"

echo ""
echo "############################################################"
echo "# ALL LTD ABLATIONS COMPLETE at $(date -Iseconds)"
echo "############################################################"
