#!/usr/bin/env bash
# Neil's LTD ablations only. Runs in parallel with run_neil_supervised.sh.
#
#   8b,  50k: co-op + not co-op
#   14b, 50k: co-op + not co-op
#
# Resume-safe. Billing: MODAL_PROFILE=neil.

set -euo pipefail
export MODAL_PROFILE=neil

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results"
LOG="$REPO_ROOT/neil_ltd_ablations.log"
LTD_SCRIPT="modal_qwen3.py"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Neil LTD ablations started at $(date -Iseconds)"
echo "# MODAL_PROFILE=$MODAL_PROFILE"
echo "############################################################"

modal profile list || true

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

benchmark_done() {
  local remote_path="$1"
  modal volume ls ltd-qwen3-results "$remote_path" 2>/dev/null \
    | grep -q "summary.csv"
}

run_ltd_ablation() {
  local preset="$1"       # qwen3_8b | qwen3_14b
  local label="$2"        # 8b | 14b
  local timesteps="$3"    # 50000
  local tk_label="$4"     # 50k
  local model_label="$5"  # "Qwen3 8B" | "Qwen3 14B"

  local coop_dir="$RESULTS_ROOT/${label}/LTD/${tk_label}_coop"
  local notcoop_dir="$RESULTS_ROOT/${label}/LTD/${tk_label}_notcoop"
  local staging="$REPO_ROOT/.ltd_staging_neil_${label}_${tk_label}"
  local sentinel="$RESULTS_ROOT/${label}/LTD/.${tk_label}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] LTD $label $tk_label (Neil)"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel exists -> skipping."
    return 0
  fi

  for stage in iter0_size iter0_depth iter1_size iter2_depth iter3_size iter4_depth; do
    if stage_done "$preset" "$stage"; then
      echo "--- stage $stage already done ---"
      continue
    fi
    echo "--- [$(date -Iseconds)] stage=$stage preset=$preset timesteps=$timesteps ---"
    modal run "$LTD_SCRIPT" \
      --action iterative-train \
      --model-preset "$preset" \
      --current-stage "$stage" \
      --size-total-timesteps "$timesteps" \
      --depth-total-timesteps "$timesteps" \
      --batch-size 256 --n-steps 2048 --lr 0.001 \
      --validation-fraction 0.2 --split-seed 42
  done

  if benchmark_done "$preset/iterative/benchmark_outputs"; then
    echo "--- co-op benchmark already done ---"
  else
    echo "--- [$(date -Iseconds)] benchmark co-op ---"
    modal run "$LTD_SCRIPT" \
      --action benchmark-suite --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --model-label "$model_label"
  fi

  if benchmark_done "$preset/iterative/benchmark_outputs_iter0"; then
    echo "--- not-coop benchmark already done ---"
  else
    echo "--- [$(date -Iseconds)] benchmark not co-op ---"
    modal run "$LTD_SCRIPT" \
      --action benchmark-suite --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter0_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter0_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --output-subdir "$preset/iterative/benchmark_outputs_iter0" \
      --model-label "$model_label iter0"
  fi

  echo "--- [$(date -Iseconds)] downloading ---"
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

  find "$coop_dir" "$notcoop_dir" -type f -name "*.zip" -delete
  rm -rf "$staging"

  mkdir -p "$(dirname "$sentinel")"
  date -Iseconds > "$sentinel"

  echo "--- wiping modal volume for $preset LTD paths ---"
  modal volume rm -r ltd-qwen3-results "$preset/iterative" || true

  echo "=== [$(date -Iseconds)] DONE LTD $label $tk_label ==="
}

run_ltd_ablation qwen3_8b  8b  50000 50k "Qwen3 8B"
run_ltd_ablation qwen3_14b 14b 50000 50k "Qwen3 14B"

echo ""
echo "############################################################"
echo "# Neil LTD ABLATIONS COMPLETE at $(date -Iseconds)"
echo "############################################################"
