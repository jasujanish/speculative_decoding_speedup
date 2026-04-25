#!/usr/bin/env bash
# Resumable fire-and-forget script for Neil's ablations (LTD + supervised).
#
# Billing: uses MODAL_PROFILE=neil (token must already be in ~/.modal.toml).
#
# Assignments:
#   14b, 10k: supervised base × {1, 20 ep}, supervised titan × {1, 20 ep}       = 4
#   8b,  50k: LTD (coop + notcoop), supervised base × {1, 20}, titan × {1, 20}  = 6
#   14b, 50k: LTD (coop + notcoop), supervised base × {1, 20}, titan × {1, 20}  = 6
# = 16 ablation entries.
#
# LTD notes:
#   - A co-op / not-coop pair shares one iterative training (6 PPO stages).
#   - co-op uses iter3_size + iter4_depth, not-coop uses iter0_size + iter0_depth.
#   - co-op saves cumulative timing across all 6 stages before the volume wipe.
#
# Supervised notes:
#   - Per (model, variant) the dataset is collected once and reused between
#     1-epoch and 20-epoch runs.
#
# Run in screen/tmux:
#   screen -S neil
#   bash run_ablations_neil.sh

set -euo pipefail

# Route every modal call to Neil's workspace.
export MODAL_PROFILE=neil

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results_neil"
LOG="$REPO_ROOT/neil_ablations.log"
LTD_SCRIPT="modal_qwen3.py"
SUP_SCRIPT="supervised_depth_modal/modal_supervised_depth_qwen3.py"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Neil's ablations started at $(date -Iseconds)"
echo "# MODAL_PROFILE=$MODAL_PROFILE"
echo "############################################################"

# Sanity-check the profile so we don't accidentally charge the wrong account.
modal profile list || true

# --- shared helpers --------------------------------------------------------

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

train_done_sup() {
  local preset="$1"
  local train_subdir="$2"
  modal volume ls ltd-qwen3-results "$preset/supervised_depth/$train_subdir" 2>/dev/null \
    | grep -q "supervised_depth_model.pt"
}

write_ltd_cumulative_summary() {
  local iterative_dir="$1"
  local output_path="$2"

  python - "$iterative_dir" "$output_path" <<'PY'
import json
import sys
from pathlib import Path

iterative_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
stage_names = [
    "iter0_size",
    "iter0_depth",
    "iter1_size",
    "iter2_depth",
    "iter3_size",
    "iter4_depth",
]
summary_keys = [
    "training_time_seconds",
    "validation_time_seconds",
    "training_flops",
    "validation_flops",
]
manifest_keys = [
    "training_target_model_calls",
    "training_draft_model_calls",
    "validation_target_model_calls",
    "validation_draft_model_calls",
]
summary = {
    "event": "cumulative_summary",
    "stage_names": stage_names,
    "stage_count": len(stage_names),
}

for key in summary_keys + manifest_keys:
    summary[key] = 0.0

for stage_name in stage_names:
    stage_dir = iterative_dir / stage_name
    time_path = stage_dir / "time_summary.jsonl"
    manifest_path = stage_dir / "dataset_manifest.json"
    if not time_path.exists():
        raise SystemExit(f"Missing LTD time summary: {time_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing LTD manifest: {manifest_path}")

    with time_path.open("r", encoding="utf-8") as handle:
        time_record = json.loads(handle.readline())
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for key in summary_keys:
        summary[key] += float(time_record.get(key, 0.0))
    for key in manifest_keys:
        summary[key] += float(manifest.get(key, 0.0))

summary["total_time_seconds"] = (
    summary["training_time_seconds"] + summary["validation_time_seconds"]
)
summary["total_target_model_calls"] = (
    summary["training_target_model_calls"] + summary["validation_target_model_calls"]
)
summary["total_draft_model_calls"] = (
    summary["training_draft_model_calls"] + summary["validation_draft_model_calls"]
)
summary["total_flops"] = summary["training_flops"] + summary["validation_flops"]

for key, value in list(summary.items()):
    if isinstance(value, float) and value.is_integer():
        summary[key] = int(value)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(summary))
    handle.write("\n")
PY
}

# --- LTD (one pair = one training + two benchmarks) ------------------------

run_ltd_ablation() {
  local preset="$1"       # qwen3_8b | qwen3_14b
  local label="$2"        # 8b | 14b
  local timesteps="$3"    # e.g. 50000
  local tk_label="$4"     # e.g. 50k
  local model_label="$5"  # "Qwen3 8B" | "Qwen3 14B"

  local coop_dir="$RESULTS_ROOT/${label}_LTD_${tk_label}timesteps_coop"
  local notcoop_dir="$RESULTS_ROOT/${label}_LTD_${tk_label}timesteps_notcoop"
  local staging="$REPO_ROOT/.ltd_staging_neil_${label}_${tk_label}"
  local sentinel="$RESULTS_ROOT/.${label}_LTD_${tk_label}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] LTD $label $tk_label (Neil)"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel $sentinel exists -> skipping."
    return 0
  fi

  for stage in iter0_size iter0_depth iter1_size iter2_depth iter3_size iter4_depth; do
    if stage_done "$preset" "$stage"; then
      echo "--- stage $stage already done, skipping ---"
      continue
    fi
    echo "--- [$(date -Iseconds)] stage=$stage preset=$preset timesteps=$timesteps ---"
    modal run "$LTD_SCRIPT" \
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

  if benchmark_done "$preset/benchmark_outputs"; then
    echo "--- co-op benchmark already done, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmark co-op ---"
    modal run "$LTD_SCRIPT" \
      --action benchmark-suite \
      --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --model-label "$model_label"
  fi

  if benchmark_done "$preset/iterative/benchmark_outputs_iter0"; then
    echo "--- not-coop benchmark already done, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmark not co-op ---"
    modal run "$LTD_SCRIPT" \
      --action benchmark-suite \
      --model-preset "$preset" \
      --token-model-path "$preset/iterative/iter0_size/ppo_speculative_decoder_controller_rebuttal.zip" \
      --depth-model-path "$preset/iterative/iter0_depth/ppo_speculative_decoder_controller_v1_single_action.zip" \
      --output-subdir "$preset/iterative/benchmark_outputs_iter0" \
      --model-label "$model_label iter0"
  fi

  echo "--- [$(date -Iseconds)] downloading LTD results ---"
  rm -rf "$staging"
  mkdir -p "$staging"
  modal volume get ltd-qwen3-results "$preset/iterative"         "$staging/"
  modal volume get ltd-qwen3-results "$preset/benchmark_outputs" "$staging/"

  rm -rf "$coop_dir" "$notcoop_dir"
  mkdir -p "$coop_dir" "$notcoop_dir"
  cp -r "$staging/iterative/iter3_size"  "$coop_dir/"
  cp -r "$staging/iterative/iter4_depth" "$coop_dir/"
  cp -r "$staging/benchmark_outputs"     "$coop_dir/"
  write_ltd_cumulative_summary \
    "$staging/iterative" \
    "$coop_dir/cumulative_time_summary.jsonl"
  cp -r "$staging/iterative/iter0_size"              "$notcoop_dir/"
  cp -r "$staging/iterative/iter0_depth"             "$notcoop_dir/"
  cp -r "$staging/iterative/benchmark_outputs_iter0" "$notcoop_dir/"

  echo "--- pruning .zip policy files locally ---"
  find "$coop_dir" "$notcoop_dir" -type f -name "*.zip" -delete

  rm -rf "$staging"

  mkdir -p "$RESULTS_ROOT"
  date -Iseconds > "$sentinel"

  echo "--- wiping modal volume for $preset LTD paths ---"
  modal volume rm -r ltd-qwen3-results "$preset/iterative"        || true
  modal volume rm -r ltd-qwen3-results "$preset/benchmark_outputs" || true

  echo "=== [$(date -Iseconds)] DONE LTD $label $tk_label (Neil) ==="
}

# --- Supervised ------------------------------------------------------------

run_supervised_ablation() {
  local preset="$1"       # qwen3_8b | qwen3_14b
  local label="$2"        # 8b | 14b
  local variant="$3"      # base | titan
  local timesteps="$4"    # 10000 | 50000
  local tk_label="$5"     # 10k | 50k
  local epochs="$6"       # 1 | 20
  local ep_label="$7"     # 1epoch | 20epochs
  local model_label="$8"  # "Qwen3 8B" | "Qwen3 14B"

  local suffix=""
  local policy_flags=()
  if [[ "$variant" == "titan" ]]; then
    suffix="_titan"
    policy_flags=(--policy-variant titan)
  fi
  local train_subdir="train_t${timesteps}${suffix}"
  local benchmark_subdir="$preset/supervised_depth/benchmark_outputs_${variant}"
  local local_dir="$RESULTS_ROOT/${label}_supervised_${variant}_${tk_label}timesteps_${ep_label}"
  local staging="$REPO_ROOT/.sup_staging_neil_${label}_${variant}_${tk_label}_${ep_label}"
  local sentinel="$RESULTS_ROOT/.${label}_supervised_${variant}_${tk_label}_${ep_label}.done"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] Supervised $label $variant $tk_label $ep_label (Neil)"
  echo "=========================================================="

  if [[ -f "$sentinel" ]]; then
    echo "Sentinel $sentinel exists -> skipping."
    return 0
  fi

  if train_done_sup "$preset" "$train_subdir"; then
    echo "--- train already complete for $train_subdir, skipping ---"
  else
    echo "--- [$(date -Iseconds)] training preset=$preset variant=$variant timesteps=$timesteps epochs=$epochs ---"
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
    echo "--- benchmark already done for $benchmark_subdir, skipping ---"
  else
    echo "--- [$(date -Iseconds)] benchmarking $variant ---"
    modal run "$SUP_SCRIPT" \
      --action benchmark-suite \
      --model-preset "$preset" \
      --depth-model-path "$preset/supervised_depth/$train_subdir/supervised_depth_model.pt" \
      --output-subdir "$benchmark_subdir" \
      --model-label "$model_label $variant $ep_label"
  fi

  echo "--- [$(date -Iseconds)] downloading to $local_dir ---"
  rm -rf "$staging" "$local_dir"
  mkdir -p "$staging" "$local_dir"
  modal volume get ltd-qwen3-results "$preset/supervised_depth/$train_subdir" "$staging/"
  modal volume get ltd-qwen3-results "$benchmark_subdir"                      "$staging/"

  cp -r "$staging/$train_subdir"                   "$local_dir/"
  cp -r "$staging/$(basename "$benchmark_subdir")" "$local_dir/"

  echo "--- pruning intermediate .pt checkpoints ---"
  find "$local_dir" -type f -name "supervised_depth_model_step_*.pt" -delete
  find "$local_dir" -type f -name "supervised_depth_model_best.pt"   -delete

  rm -rf "$staging"

  mkdir -p "$RESULTS_ROOT"
  date -Iseconds > "$sentinel"

  echo "--- wiping train + benchmark outputs on modal (keeping dataset) ---"
  modal volume rm -r ltd-qwen3-results "$preset/supervised_depth/$train_subdir" || true
  modal volume rm -r ltd-qwen3-results "$benchmark_subdir"                      || true

  echo "=== [$(date -Iseconds)] DONE Supervised $label $variant $tk_label $ep_label (Neil) ==="
}

wipe_dataset() {
  local preset="$1"
  local variant="$2"
  local timesteps="$3"
  local suffix=""
  [[ "$variant" == "titan" ]] && suffix="_titan"
  echo "--- wiping dataset on modal: $preset/supervised_depth/dataset_t${timesteps}${suffix} ---"
  modal volume rm -r ltd-qwen3-results "$preset/supervised_depth/dataset_t${timesteps}${suffix}" || true
}

# --- Neil's ablations ------------------------------------------------------

# 14b, 10k supervised (no LTD for this block).
run_supervised_ablation qwen3_14b 14b base  10000 10k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  10000 10k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b base 10000
run_supervised_ablation qwen3_14b 14b titan 10000 10k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b titan 10000 10k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b titan 10000

# 8b, 50k: LTD pair + supervised (both variants × both epoch settings).
run_ltd_ablation qwen3_8b 8b 50000 50k "Qwen3 8B"
run_supervised_ablation qwen3_8b 8b base  50000 50k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b 8b base  50000 50k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b base 50000
run_supervised_ablation qwen3_8b 8b titan 50000 50k 1  1epoch   "Qwen3 8B"
run_supervised_ablation qwen3_8b 8b titan 50000 50k 20 20epochs "Qwen3 8B"
wipe_dataset qwen3_8b titan 50000

# 14b, 50k: LTD pair + supervised (both variants × both epoch settings).
run_ltd_ablation qwen3_14b 14b 50000 50k "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  50000 50k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b base  50000 50k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b base 50000
run_supervised_ablation qwen3_14b 14b titan 50000 50k 1  1epoch   "Qwen3 14B"
run_supervised_ablation qwen3_14b 14b titan 50000 50k 20 20epochs "Qwen3 14B"
wipe_dataset qwen3_14b titan 50000

echo ""
echo "############################################################"
echo "# ALL NEIL ABLATIONS COMPLETE at $(date -Iseconds)"
echo "############################################################"
