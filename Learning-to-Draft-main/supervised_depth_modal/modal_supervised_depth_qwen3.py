"""Modal entrypoints for the supervised Qwen3 depth-policy workflow."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any

import modal

from qwen3_model_presets import model_cache_dir, resolve_model_paths


LOCAL_PROJECT_DIR = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_MODELS_DIR = Path("/models")
REMOTE_RESULTS_DIR = Path("/results")
DEFAULT_DATASETS = ("mt_bench", "gsm8k", "alpaca", "qa")
METHOD_LABELS = {
    "baseline": "Baseline",
    "eagle3": "Eagle3",
    "supervised_depth": "Eagle3+SupervisedDepth",
}
VALIDATION_QUESTION_END = 100000
FLOP_ESTIMATE_MULTIPLIER = 2
MODELS_VOLUME = modal.Volume.from_name("ltd-qwen3-models", create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("ltd-qwen3-results", create_if_missing=True)

app = modal.App("ltd-qwen3-supervised-depth-modal")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.56.1",
        "accelerate==0.26.0",
        "fschat==0.2.31",
        "gradio==3.50.2",
        "openai==0.28.0",
        "anthropic==0.5.0",
        "numpy==1.26.4",
        "sentencepiece==0.1.99",
        "protobuf==3.19.0",
        "gymnasium==1.1.1",
        "stable_baselines3==2.7.0",
        "tensorboard",
        "tqdm",
        "huggingface_hub[hf_transfer]==0.34.4",
        "shortuuid",
        "safetensors",
        "modal",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir(str(REMOTE_PROJECT_DIR))
    .add_local_dir(LOCAL_PROJECT_DIR, remote_path=str(REMOTE_PROJECT_DIR))
)


def build_runtime_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment variables.

    Args:
        extra_env: Optional overrides.

    Returns:
        Environment variables for remote subprocesses.
    """
    env = dict(os.environ)
    env["HF_HOME"] = str(REMOTE_MODELS_DIR / ".hf")
    env["PYTHONPATH"] = str(REMOTE_PROJECT_DIR)
    env["TOKENIZERS_PARALLELISM"] = "false"
    if extra_env:
        env.update(extra_env)
    return env


def ensure_dir(path: Path) -> None:
    """Create a directory when it does not already exist.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON payload.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file.

    Args:
        path: JSON file path.
        payload: Serializable payload.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        Parsed JSON records.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a JSONL file.

    Args:
        path: JSONL output path.
        records: JSON records.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file.

    Args:
        path: JSONL file path.
        payload: Record to append.
    """
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def write_time_summary(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one timing summary record to a JSONL file.

    Args:
        path: JSONL output path.
        payload: Timing summary payload.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def answer_manifest_path(answer_path: Path) -> Path:
    """Return the sidecar manifest path for one answer JSONL file.

    Args:
        answer_path: Answer JSONL path.

    Returns:
        Sidecar manifest JSON path.
    """
    return answer_path.with_name(f"{answer_path.name}.manifest.json")


def read_answer_manifest(answer_path: Path) -> dict[str, int]:
    """Read one answer sidecar manifest when present.

    Args:
        answer_path: Answer JSONL path.

    Returns:
        Draft and target model call counts.
    """
    manifest_path = answer_manifest_path(answer_path)
    if not manifest_path.exists():
        return {
            "draft_model_calls": 0,
            "target_model_calls": 0,
        }
    payload = read_json(manifest_path)
    return {
        "draft_model_calls": int(payload.get("draft_model_calls", 0)),
        "target_model_calls": int(payload.get("target_model_calls", 0)),
    }


def estimate_forward_flops(parameter_count: int, forward_calls: int) -> int:
    """Estimate forward-pass FLOPs from parameter count and call count.

    Args:
        parameter_count: Number of model parameters.
        forward_calls: Number of forward calls.

    Returns:
        Approximate floating point operation count.
    """
    return int(FLOP_ESTIMATE_MULTIPLIER * parameter_count * forward_calls)


def resolve_remote_model_paths(model_preset: str) -> tuple[str, str]:
    """Resolve one preset into cached remote model paths.

    Args:
        model_preset: Supported preset.

    Returns:
        Base and Eagle3 model paths inside the Modal volume.
    """
    base_repo_id, ea_repo_id = resolve_model_paths(model_preset=model_preset)
    return (
        str(model_cache_dir(REMOTE_MODELS_DIR, base_repo_id)),
        str(model_cache_dir(REMOTE_MODELS_DIR, ea_repo_id)),
    )


def resolve_results_path(relative_or_absolute_path: str) -> str:
    """Resolve a results path under the mounted results volume.

    Args:
        relative_or_absolute_path: Relative or absolute path.

    Returns:
        Absolute results path.
    """
    if not relative_or_absolute_path:
        return ""
    if relative_or_absolute_path.startswith("/"):
        return relative_or_absolute_path
    return str(REMOTE_RESULTS_DIR / relative_or_absolute_path)


def run_command(command: list[str]) -> None:
    """Run a subprocess in the remote project directory.

    Args:
        command: Command tokens.
    """
    subprocess.run(
        command,
        check=True,
        cwd=str(REMOTE_PROJECT_DIR),
        env=build_runtime_env(),
    )


def create_humaneval_split_files(
    output_root: Path,
    validation_fraction: float,
    split_seed: int,
) -> tuple[str, str]:
    """Create deterministic HumanEval train/validation split files.

    Args:
        output_root: Root artifact directory.
        validation_fraction: Validation fraction.
        split_seed: Shuffle seed.

    Returns:
        Training and validation question file paths.
    """
    source_path = REMOTE_PROJECT_DIR / "eagle" / "data" / "humaneval" / "question.jsonl"
    records = read_jsonl(source_path)
    shuffled = list(records)
    random.Random(split_seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_records = sorted(
        shuffled[:validation_count],
        key=lambda record: int(record["question_id"]),
    )
    training_records = sorted(
        shuffled[validation_count:],
        key=lambda record: int(record["question_id"]),
    )
    train_path = output_root / "split_data" / "humaneval_train" / "question.jsonl"
    validation_path = output_root / "split_data" / "humaneval_val" / "question.jsonl"
    write_jsonl(train_path, training_records)
    write_jsonl(validation_path, validation_records)
    return str(train_path), str(validation_path)


def build_eval_command(
    script_module: str,
    model_preset: str,
    bench_name: str,
    output_subdir: str,
    question_begin: int,
    question_end: int,
    temperature: float,
    num_choices: int,
    total_token: int,
    depth: int,
    question_file: str = "",
    depth_model_path: str = "",
    answer_file_override: str = "",
) -> tuple[list[str], str]:
    """Build a benchmark subprocess command.

    Args:
        script_module: Python module to execute.
        model_preset: Supported preset.
        bench_name: Dataset name.
        output_subdir: Output root under ``/results``.
        question_begin: Begin question index.
        question_end: End question index.
        temperature: Generation temperature.
        num_choices: Number of completions.
        total_token: Fixed verification budget.
        depth: Static draft depth.
        question_file: Optional question JSONL file.
        depth_model_path: Optional supervised depth model path.
        answer_file_override: Optional absolute output file path.

    Returns:
        Command tokens and answer file path.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    if answer_file_override:
        answer_file = Path(answer_file_override)
        ensure_dir(answer_file.parent)
    else:
        output_dir = REMOTE_RESULTS_DIR / output_subdir
        ensure_dir(output_dir)
        answer_file = output_dir / f"{bench_name}.jsonl"
    command = [
        "python",
        "-m",
        script_module,
        "--base-model-path",
        base_model_path,
        "--ea-model-path",
        ea_model_path,
        "--model-id",
        model_preset,
        "--bench-name",
        bench_name,
        "--question-file",
        question_file,
        "--question-begin",
        str(question_begin),
        "--question-end",
        str(question_end),
        "--temperature",
        str(temperature),
        "--num-choices",
        str(num_choices),
        "--answer-file",
        str(answer_file),
        "--total-token",
        str(total_token),
        "--depth",
        str(depth),
    ]
    if depth_model_path:
        command.extend(["--depth-model", resolve_results_path(depth_model_path)])
    return command, str(answer_file)


def extract_speed(record: dict[str, Any]) -> float:
    """Compute throughput for one benchmark record.

    Args:
        record: JSONL record.

    Returns:
        Tokens per second.
    """
    choice = record["choices"][0]
    tokens = float(sum(choice.get("new_tokens", [])))
    wall_time = float(sum(choice.get("wall_time", [])))
    prefill_time = float(sum(choice.get("pre_len_times", [])))
    total_time = wall_time + prefill_time
    if total_time <= 0:
        return 0.0
    return tokens / total_time


def extract_accept_lengths(record: dict[str, Any]) -> list[float]:
    """Extract acceptance lengths from one benchmark record.

    Args:
        record: JSONL record.

    Returns:
        Acceptance lengths.
    """
    return [float(value) for value in record["choices"][0].get("pre_num", [])]


def load_metrics(path: Path) -> tuple[dict[int, float], dict[int, list[float]]]:
    """Load per-question speed and acceptance metrics.

    Args:
        path: JSONL benchmark path.

    Returns:
        Question-id speed map and question-id acceptance-length map.
    """
    data = read_jsonl(path)
    speeds: dict[int, float] = {}
    accept_lengths: dict[int, list[float]] = {}
    for record in data:
        question_id = int(record["question_id"])
        speeds[question_id] = extract_speed(record)
        accept_lengths[question_id] = extract_accept_lengths(record)
    return speeds, accept_lengths


def compute_validation_metrics(baseline_path: Path, method_path: Path) -> tuple[float, float]:
    """Compute validation speedup and acceptance length.

    Args:
        baseline_path: Baseline JSONL path.
        method_path: Method JSONL path.

    Returns:
        Speedup and average acceptance length.
    """
    baseline_speeds, _ = load_metrics(baseline_path)
    method_speeds, accept_lengths = load_metrics(method_path)
    shared_ids = sorted(set(baseline_speeds) & set(method_speeds))
    if not shared_ids:
        raise ValueError(f"No overlapping question ids between {baseline_path} and {method_path}.")
    baseline_mean = mean(baseline_speeds[qid] for qid in shared_ids)
    method_mean = mean(method_speeds[qid] for qid in shared_ids)
    speedup = method_mean / baseline_mean if baseline_mean > 0 else 0.0
    tau_values = [value for qid in shared_ids for value in accept_lengths.get(qid, [])]
    tau = mean(tau_values) if tau_values else 0.0
    return speedup, tau


def parse_checkpoint_step(checkpoint_path: Path) -> int:
    """Extract the optimizer step from a checkpoint filename.

    Args:
        checkpoint_path: Checkpoint path.

    Returns:
        Parsed optimizer step or zero when unavailable.
    """
    stem = checkpoint_path.stem
    if "_step_" not in stem:
        return 0
    try:
        return int(stem.rsplit("_step_", maxsplit=1)[1])
    except ValueError:
        return 0


def list_candidate_checkpoints(stage_dir: Path) -> list[Path]:
    """List candidate supervised-depth checkpoints.

    Args:
        stage_dir: Training output directory.

    Returns:
        Candidate checkpoints sorted by optimizer step.
    """
    step_checkpoints = sorted(
        stage_dir.glob("supervised_depth_model_step_*.pt"),
        key=parse_checkpoint_step,
    )
    best_checkpoint = stage_dir / "supervised_depth_model_best.pt"
    candidates = list(step_checkpoints)
    if best_checkpoint.exists():
        candidates.append(best_checkpoint)
    return candidates


def promote_checkpoint(source_path: Path, destination_path: Path) -> str:
    """Copy a selected checkpoint to a canonical path.

    Args:
        source_path: Selected checkpoint.
        destination_path: Canonical checkpoint path.

    Returns:
        Canonical checkpoint path.
    """
    if source_path.resolve() != destination_path.resolve():
        shutil.copyfile(source_path, destination_path)
    return str(destination_path)


def ensure_validation_baseline(
    model_preset: str,
    validation_question_file: str,
    root_dir: Path,
) -> Path:
    """Run and cache the baseline validation generations.

    Args:
        model_preset: Supported preset.
        validation_question_file: Validation question JSONL file.
        root_dir: Workflow root directory.

    Returns:
        Baseline validation JSONL path.
    """
    baseline_path = root_dir / "validation" / "baseline" / "humaneval.jsonl"
    if baseline_path.exists():
        return baseline_path
    command, _ = build_eval_command(
        script_module="eagle.evaluation.gen_baseline_answer_qwen3",
        model_preset=model_preset,
        bench_name="humaneval",
        output_subdir="",
        question_begin=0,
        question_end=VALIDATION_QUESTION_END,
        temperature=0.0,
        num_choices=1,
        total_token=60,
        depth=8,
        question_file=validation_question_file,
        answer_file_override=str(baseline_path),
    )
    run_command(command)
    return baseline_path


def select_best_supervised_depth_checkpoint(
    model_preset: str,
    stage_dir: Path,
    validation_question_file: str,
    root_dir: Path,
) -> tuple[str, dict[str, int]]:
    """Select the best supervised depth checkpoint by validation speedup.

    Args:
        model_preset: Supported preset.
        stage_dir: Training output directory.
        validation_question_file: Validation question JSONL file.
        root_dir: Workflow root directory.

    Returns:
        Promoted canonical checkpoint path and validation call totals.
    """
    baseline_jsonl_path = root_dir / "validation" / "baseline" / "humaneval.jsonl"
    baseline_existed = baseline_jsonl_path.exists()
    baseline_path = ensure_validation_baseline(
        model_preset=model_preset,
        validation_question_file=validation_question_file,
        root_dir=root_dir,
    )
    baseline_counts = (
        {"draft_model_calls": 0, "target_model_calls": 0}
        if baseline_existed
        else read_answer_manifest(baseline_path)
    )
    validation_dir = stage_dir / "validation"
    ensure_dir(validation_dir)
    metrics_log_path = stage_dir / "validation_metrics.jsonl"
    results: list[dict[str, Any]] = []
    total_validation_draft_calls = baseline_counts["draft_model_calls"]
    total_validation_target_calls = baseline_counts["target_model_calls"]
    for candidate_path in list_candidate_checkpoints(stage_dir):
        answer_path = validation_dir / f"{candidate_path.stem}.jsonl"
        command, _ = build_eval_command(
            script_module="supervised_depth_modal.gen_supervised_depth_answer_qwen3",
            model_preset=model_preset,
            bench_name="humaneval",
            output_subdir="",
            question_begin=0,
            question_end=VALIDATION_QUESTION_END,
            temperature=0.0,
            num_choices=1,
            total_token=60,
            depth=8,
            question_file=validation_question_file,
            depth_model_path=str(candidate_path),
            answer_file_override=str(answer_path),
        )
        run_command(command)
        speedup, tau = compute_validation_metrics(baseline_path, answer_path)
        answer_counts = read_answer_manifest(answer_path)
        total_validation_draft_calls += answer_counts["draft_model_calls"]
        total_validation_target_calls += answer_counts["target_model_calls"]
        result = {
            "checkpoint_path": str(candidate_path),
            "checkpoint_step": parse_checkpoint_step(candidate_path),
            "speedup": speedup,
            "tau": tau,
            "validation_draft_model_calls": answer_counts["draft_model_calls"],
            "validation_target_model_calls": answer_counts["target_model_calls"],
        }
        results.append(result)
        append_jsonl_record(
            metrics_log_path,
            {
                "event": "candidate",
                **result,
            },
        )
    if not results:
        raise ValueError(f"No supervised depth checkpoints found in {stage_dir}.")
    best_result = max(results, key=lambda item: (item["speedup"], item["tau"], item["checkpoint_step"]))
    promoted_path = promote_checkpoint(
        source_path=Path(best_result["checkpoint_path"]),
        destination_path=stage_dir / "supervised_depth_model.pt",
    )
    append_jsonl_record(
        metrics_log_path,
        {
            "event": "selected",
            **best_result,
            "promoted_checkpoint_path": promoted_path,
        },
    )
    write_json(
        stage_dir / "validation_selection.json",
        {
            "baseline_path": str(baseline_path),
            "validation_question_file": validation_question_file,
            "best_result": best_result,
            "candidate_results": results,
            "validation_metrics_log_path": str(metrics_log_path),
            "promoted_checkpoint_path": promoted_path,
        },
    )
    return promoted_path, {
        "draft_model_calls": total_validation_draft_calls,
        "target_model_calls": total_validation_target_calls,
    }


def build_root_dir(model_preset: str, output_subdir: str) -> Path:
    """Build the workflow root directory.

    Args:
        model_preset: Supported preset.
        output_subdir: Optional output override.

    Returns:
        Workflow root directory.
    """
    return REMOTE_RESULTS_DIR / (output_subdir or f"{model_preset}/supervised_depth")


@app.function(
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=60 * 60,
)
def download_models(model_preset: str = "qwen3_8b") -> dict[str, str]:
    """Download one supported model pair into the shared Modal volume.

    Args:
        model_preset: Supported preset.

    Returns:
        Cached base and Eagle3 paths.
    """
    from huggingface_hub import snapshot_download

    base_repo_id, ea_repo_id = resolve_model_paths(model_preset=model_preset)
    for repo_id in (base_repo_id, ea_repo_id):
        local_dir = model_cache_dir(REMOTE_MODELS_DIR, repo_id)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    MODELS_VOLUME.commit()
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    return {
        "base_model_path": base_model_path,
        "ea_model_path": ea_model_path,
    }


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=24 * 60 * 60,
)
def collect_depth_dataset(
    model_preset: str = "qwen3_8b",
    total_timesteps: int = 20000,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    output_subdir: str = "",
) -> dict[str, Any]:
    """Collect a supervised depth dataset under a fixed interaction budget.

    Args:
        model_preset: Supported preset.
        total_timesteps: Environmental interaction budget.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval split seed.
        output_subdir: Optional output override.

    Returns:
        Dataset manifest.
    """
    from supervised_depth_modal.core import collect_supervised_depth_dataset

    root_dir = build_root_dir(model_preset, output_subdir)
    ensure_dir(root_dir)
    train_question_file, _ = create_humaneval_split_files(root_dir, validation_fraction, split_seed)
    dataset_dir = root_dir / f"dataset_t{total_timesteps}"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if int(manifest.get("total_timesteps", -1)) == total_timesteps:
            return manifest
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    manifest = collect_supervised_depth_dataset(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        question_file=train_question_file,
        output_dir=str(dataset_dir),
        total_timesteps=total_timesteps,
    )
    RESULTS_VOLUME.commit()
    return manifest


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=24 * 60 * 60,
)
def train_depth_policy(
    model_preset: str = "qwen3_8b",
    total_timesteps: int = 20000,
    epochs: int = 1,
    checkpoint_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    output_subdir: str = "",
) -> dict[str, Any]:
    """Collect data, train, validate, and promote a supervised depth policy.

    Args:
        model_preset: Supported preset.
        total_timesteps: Environmental interaction budget used for collection.
        epochs: Number of full passes over the supervised training split.
        checkpoint_epochs: Checkpoint interval in epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval split seed.
        output_subdir: Optional output override.

    Returns:
        Training summary with the promoted checkpoint path.
    """
    from supervised_depth_modal.core import (
        collect_supervised_depth_dataset,
        train_supervised_depth_model,
    )

    root_dir = build_root_dir(model_preset, output_subdir)
    ensure_dir(root_dir)
    train_question_file, validation_question_file = create_humaneval_split_files(
        root_dir,
        validation_fraction,
        split_seed,
    )
    del train_question_file
    dataset_dir = root_dir / f"dataset_t{total_timesteps}"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        dataset_manifest = read_json(manifest_path)
    else:
        base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
        dataset_manifest = collect_supervised_depth_dataset(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            question_file=str(root_dir / "split_data" / "humaneval_train" / "question.jsonl"),
            output_dir=str(dataset_dir),
            total_timesteps=total_timesteps,
        )
    dataset_path = dataset_manifest["dataset_path"]
    stage_dir = root_dir / f"train_t{total_timesteps}"
    training_start_time = time.time()
    summary = train_supervised_depth_model(
        dataset_path=dataset_path,
        output_dir=str(stage_dir),
        total_timesteps=total_timesteps,
        epochs=epochs,
        checkpoint_epochs=checkpoint_epochs,
        batch_size=batch_size,
        lr=lr,
    )
    training_time_seconds = time.time() - training_start_time
    validation_start_time = time.time()
    promoted_path, validation_call_counts = select_best_supervised_depth_checkpoint(
        model_preset=model_preset,
        stage_dir=stage_dir,
        validation_question_file=validation_question_file,
        root_dir=root_dir,
    )
    validation_time_seconds = time.time() - validation_start_time
    validation_flops = (
        estimate_forward_flops(
            int(dataset_manifest.get("draft_model_parameters", 0)),
            int(validation_call_counts.get("draft_model_calls", 0)),
        )
        + estimate_forward_flops(
            int(dataset_manifest.get("target_model_parameters", 0)),
            int(validation_call_counts.get("target_model_calls", 0)),
        )
    )
    training_flops = int(summary.get("estimated_training_flops", 0))
    summary["promoted_checkpoint_path"] = promoted_path
    write_json(stage_dir / "training_summary.json", summary)
    write_time_summary(
        stage_dir / "time_summary.jsonl",
        {
            "event": "summary",
            "phase_type": "supervised_depth",
            "total_timesteps": total_timesteps,
            "epochs": epochs,
            "checkpoint_epochs": checkpoint_epochs,
            "optimizer_steps": int(summary.get("optimizer_steps", 0)),
            "total_time_seconds": training_time_seconds + validation_time_seconds,
            "training_time_seconds": training_time_seconds,
            "validation_time_seconds": validation_time_seconds,
            "training_flops": training_flops,
            "validation_flops": validation_flops,
            "total_flops": training_flops + validation_flops,
            "validation_draft_model_calls": int(validation_call_counts.get("draft_model_calls", 0)),
            "validation_target_model_calls": int(validation_call_counts.get("target_model_calls", 0)),
            "flop_estimate_method": (
                "training: 6 * supervised_model_parameters * examples_seen; "
                "validation: 2 * parameter_count * forward_calls"
            ),
        },
    )
    RESULTS_VOLUME.commit()
    return summary


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=8 * 60 * 60,
)
def evaluate_supervised_depth(
    model_preset: str = "qwen3_8b",
    bench_name: str = "gsm8k",
    depth_model_path: str = "",
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
    total_token: int = 60,
    depth: int = 8,
    output_subdir: str = "",
) -> str:
    """Run benchmark generation with the supervised depth policy.

    Args:
        model_preset: Supported preset.
        bench_name: Benchmark dataset.
        depth_model_path: Supervised depth checkpoint path.
        question_begin: Start index.
        question_end: End index.
        temperature: Generation temperature.
        num_choices: Number of completions.
        total_token: Fixed verification size.
        depth: Static draft depth limit.
        output_subdir: Optional output root.

    Returns:
        JSONL answer file path.
    """
    if not depth_model_path:
        raise ValueError("depth_model_path is required for supervised depth evaluation.")
    command, answer_file = build_eval_command(
        script_module="supervised_depth_modal.gen_supervised_depth_answer_qwen3",
        model_preset=model_preset,
        bench_name=bench_name,
        output_subdir=output_subdir or f"{model_preset}/supervised_depth/eval/supervised_depth",
        question_begin=question_begin,
        question_end=question_end,
        temperature=temperature,
        num_choices=num_choices,
        total_token=total_token,
        depth=depth,
        depth_model_path=depth_model_path,
    )
    run_command(command)
    RESULTS_VOLUME.commit()
    return answer_file


def dataset_metrics(
    baseline_path: Path,
    method_path: Path,
    method_name: str,
) -> tuple[float, str]:
    """Compute dataset speedup and acceptance text.

    Args:
        baseline_path: Baseline JSONL path.
        method_path: Method JSONL path.
        method_name: Method identifier.

    Returns:
        Speedup and acceptance-length text.
    """
    baseline_speeds, _ = load_metrics(baseline_path)
    if method_name == "baseline":
        return 1.0, ""
    method_speeds, accept_lengths = load_metrics(method_path)
    shared_ids = sorted(set(baseline_speeds) & set(method_speeds))
    baseline_mean = mean(baseline_speeds[qid] for qid in shared_ids)
    method_mean = mean(method_speeds[qid] for qid in shared_ids)
    speedup = method_mean / baseline_mean if baseline_mean > 0 else 0.0
    tau_values = [value for qid in shared_ids for value in accept_lengths.get(qid, [])]
    tau_text = f"{mean(tau_values):.4f}" if tau_values else ""
    return speedup, tau_text


def summarize_results(
    input_root: Path,
    output_csv: Path,
    model_label: str,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
) -> str:
    """Summarize benchmark outputs into one CSV.

    Args:
        input_root: Benchmark output root.
        output_csv: Output CSV path.
        model_label: Model label for the CSV.
        datasets: Included datasets.
        methods: Included methods.

    Returns:
        Output CSV path.
    """
    ensure_dir(output_csv.parent)
    header = ["model", "method"]
    for dataset in datasets:
        header.extend([f"{dataset}_speedup", f"{dataset}_tau"])
    header.extend(["mean_speedup", "mean_tau"])
    rows: list[list[str]] = []
    for method_name in methods:
        row = [model_label, METHOD_LABELS.get(method_name, method_name)]
        speedups: list[float] = []
        taus: list[float] = []
        for dataset in datasets:
            baseline_path = input_root / "baseline" / f"{dataset}.jsonl"
            method_path = input_root / method_name / f"{dataset}.jsonl"
            speedup, tau_text = dataset_metrics(baseline_path, method_path, method_name)
            row.extend([f"{speedup:.4f}", tau_text])
            speedups.append(speedup)
            if tau_text:
                taus.append(float(tau_text))
        row.extend(
            [
                f"{mean(speedups):.4f}" if speedups else "",
                f"{mean(taus):.4f}" if taus else "",
            ]
        )
        rows.append(row)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(output_csv)


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=24 * 60 * 60,
)
def benchmark_suite(
    model_preset: str = "qwen3_8b",
    depth_model_path: str = "",
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
    total_token: int = 60,
    depth: int = 8,
    output_subdir: str = "",
    model_label: str = "",
) -> dict[str, str]:
    """Run baseline, Eagle3, and supervised-depth benchmarks and summarize them.

    Args:
        model_preset: Supported preset.
        depth_model_path: Supervised depth checkpoint path.
        datasets: Datasets to evaluate.
        question_begin: Start index.
        question_end: End index.
        temperature: Generation temperature.
        num_choices: Number of completions.
        total_token: Fixed verification size.
        depth: Static draft depth limit.
        output_subdir: Optional output root.
        model_label: CSV model label.

    Returns:
        Results root and summary CSV path.
    """
    if not depth_model_path:
        raise ValueError("depth_model_path is required for benchmark_suite.")
    output_root = REMOTE_RESULTS_DIR / (
        output_subdir or f"{model_preset}/supervised_depth/benchmark_outputs"
    )
    for dataset in datasets:
        baseline_command, _ = build_eval_command(
            script_module="eagle.evaluation.gen_baseline_answer_qwen3",
            model_preset=model_preset,
            bench_name=dataset,
            output_subdir=str(output_root.relative_to(REMOTE_RESULTS_DIR) / "baseline"),
            question_begin=question_begin,
            question_end=question_end,
            temperature=temperature,
            num_choices=num_choices,
            total_token=total_token,
            depth=depth,
        )
        run_command(baseline_command)
        eagle_command, _ = build_eval_command(
            script_module="eagle.evaluation.gen_ea_answer_qwen3",
            model_preset=model_preset,
            bench_name=dataset,
            output_subdir=str(output_root.relative_to(REMOTE_RESULTS_DIR) / "eagle3"),
            question_begin=question_begin,
            question_end=question_end,
            temperature=temperature,
            num_choices=num_choices,
            total_token=total_token,
            depth=depth,
        )
        run_command(eagle_command)
        supervised_command, _ = build_eval_command(
            script_module="supervised_depth_modal.gen_supervised_depth_answer_qwen3",
            model_preset=model_preset,
            bench_name=dataset,
            output_subdir=str(output_root.relative_to(REMOTE_RESULTS_DIR) / "supervised_depth"),
            question_begin=question_begin,
            question_end=question_end,
            temperature=temperature,
            num_choices=num_choices,
            total_token=total_token,
            depth=depth,
            depth_model_path=depth_model_path,
        )
        run_command(supervised_command)
    summary_csv = summarize_results(
        input_root=output_root,
        output_csv=output_root / "summary.csv",
        model_label=model_label or model_preset,
        datasets=datasets,
        methods=("baseline", "eagle3", "supervised_depth"),
    )
    RESULTS_VOLUME.commit()
    return {
        "results_root": str(output_root),
        "summary_csv": summary_csv,
    }


@app.local_entrypoint()
def main(
    action: str = "benchmark-suite",
    model_preset: str = "qwen3_8b",
    total_timesteps: int = 20000,
    epochs: int = 1,
    checkpoint_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    depth_model_path: str = "",
    output_subdir: str = "",
    model_label: str = "",
) -> None:
    """Convenience local entrypoint for the supervised-depth workflow.

    Args:
        action: Remote action to execute.
        model_preset: Supported preset.
        total_timesteps: Environmental interaction budget.
        epochs: Number of full passes over the supervised training split.
        checkpoint_epochs: Checkpoint interval in epochs.
        batch_size: Batch size.
        lr: Learning rate.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval split seed.
        depth_model_path: Supervised depth checkpoint path.
        output_subdir: Optional output override.
        model_label: CSV model label.
    """
    if action == "download-models":
        print(download_models.remote(model_preset=model_preset))
        return
    if action == "collect-depth-dataset":
        print(
            collect_depth_dataset.remote(
                model_preset=model_preset,
                total_timesteps=total_timesteps,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
                output_subdir=output_subdir,
            )
        )
        return
    if action == "train-depth":
        print(
            train_depth_policy.remote(
                model_preset=model_preset,
                total_timesteps=total_timesteps,
                epochs=epochs,
                checkpoint_epochs=checkpoint_epochs,
                batch_size=batch_size,
                lr=lr,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
                output_subdir=output_subdir,
            )
        )
        return
    if action == "benchmark-suite":
        print(
            benchmark_suite.remote(
                model_preset=model_preset,
                depth_model_path=depth_model_path,
                output_subdir=output_subdir,
                model_label=model_label,
            )
        )
        return
    raise ValueError(
        "Unsupported action. Use one of: download-models, collect-depth-dataset, "
        "train-depth, benchmark-suite."
    )
