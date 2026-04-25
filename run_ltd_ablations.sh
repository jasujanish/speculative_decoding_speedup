#!/usr/bin/env bash
# Resumable fire-and-forget script for LTD ablations.
#
# Each ablation does:
#   1. Six PPO stages: iter0_size, iter0_depth, iter1_size, iter2_depth,
#      iter3_size, iter4_depth.
#   2. Two benchmark-suite runs: co-op (iter3/iter4) and not co-op (iter0/iter0).
#   3. Download results into results/<label>/LTD/<tk>_{coop,notcoop}/.
#   4. Save a cumulative co-op timing summary across all six PPO stages.
#   5. Prune *.zip PPO policy files locally to stay lightweight.
#   6. Wipe the Modal volume and drop a sentinel file so re-runs skip this ablation.
#
# Resume behavior:
#   - A sentinel at results/<label>/LTD/.<tk>.done marks a fully completed ablation
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
  # folder layout. Co-op keeps final artifacts plus a cumulative timing summary
  # for all six stages; not-coop keeps the iter0 artifacts it actually uses.
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
  write_ltd_cumulative_summary \
    "$staging/iterative" \
    "$coop_dir/cumulative_time_summary.jsonl"

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
