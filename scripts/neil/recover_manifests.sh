#!/usr/bin/env bash
# Post-hoc recovery for Neil's supervised-depth dataset manifests.
# Billing: MODAL_PROFILE=neil.
# Cells: 14b 10k × {base,titan}, 8b/14b 50k × {base,titan}.
#
# Re-runs dataset collection (no training, no benchmarking), downloads
# just the manifest, drops it into both _1 and _20 local result folders.
# Resume-safe: skips cells where both destinations already have the manifest.

set -euo pipefail
export MODAL_PROFILE=neil

REPO_ROOT=/home/acheong/Documents/s/mls/speculative_decoding_speedup
RESULTS_ROOT="$REPO_ROOT/results"
LOG="$REPO_ROOT/scripts/neil/recover_manifests.log"
SUP_SCRIPT="supervised_depth_modal/modal_supervised_depth_qwen3.py"

cd "$REPO_ROOT"
source .venv/modal/bin/activate
cd Learning-to-Draft-main

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Manifest recovery (Neil) started at $(date -Iseconds)"
echo "# MODAL_PROFILE=$MODAL_PROFILE"
echo "############################################################"

modal profile list || true

recover_one() {
  local preset="$1"
  local label="$2"
  local timesteps="$3"
  local tk_label="$4"
  local variant="$5"

  local suffix=""
  local method_dir="SupervisedLearning"
  local policy_flags=()
  if [[ "$variant" == "titan" ]]; then
    suffix="_titan"
    method_dir="SupervisedLearningTitan"
    policy_flags=(--policy-variant titan)
  fi
  local dataset_subdir="dataset_t${timesteps}${suffix}"
  local remote_manifest="$preset/supervised_depth/$dataset_subdir/dataset_manifest.json"
  local staging="$REPO_ROOT/.manifest_staging_neil_${label}_${variant}_${tk_label}"

  local dest_1="$RESULTS_ROOT/${label}/${method_dir}/${tk_label}_1/${dataset_subdir}"
  local dest_20="$RESULTS_ROOT/${label}/${method_dir}/${tk_label}_20/${dataset_subdir}"

  echo ""
  echo "=========================================================="
  echo "=== [$(date -Iseconds)] recover $label $variant $tk_label (Neil)"
  echo "=========================================================="

  if [[ -f "$dest_1/dataset_manifest.json" && -f "$dest_20/dataset_manifest.json" ]]; then
    echo "Both destinations already have manifest -> skipping."
    return 0
  fi

  echo "--- [$(date -Iseconds)] collecting dataset on Modal (idempotent) ---"
  modal run "$SUP_SCRIPT" \
    --action collect-depth-dataset \
    --model-preset "$preset" \
    --total-timesteps "$timesteps" \
    "${policy_flags[@]}"

  echo "--- [$(date -Iseconds)] downloading manifest ---"
  rm -rf "$staging"
  mkdir -p "$staging"
  modal volume get ltd-qwen3-results \
    "$remote_manifest" "$staging/dataset_manifest.json"

  mkdir -p "$dest_1" "$dest_20"
  cp "$staging/dataset_manifest.json" "$dest_1/dataset_manifest.json"
  cp "$staging/dataset_manifest.json" "$dest_20/dataset_manifest.json"

  rm -rf "$staging"
  echo "=== [$(date -Iseconds)] DONE $label $variant $tk_label ==="
}

recover_one qwen3_14b 14b 10000  10k  base
recover_one qwen3_14b 14b 10000  10k  titan
recover_one qwen3_8b  8b  50000  50k  base
recover_one qwen3_8b  8b  50000  50k  titan
recover_one qwen3_14b 14b 50000  50k  base
recover_one qwen3_14b 14b 50000  50k  titan

echo ""
echo "############################################################"
echo "# MANIFEST RECOVERY (Neil) COMPLETE at $(date -Iseconds)"
echo "############################################################"
