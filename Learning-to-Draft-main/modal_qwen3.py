"""Modal entrypoints for LTD with Qwen3 and AngelSlim Eagle3."""

from __future__ import annotations

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


LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_MODELS_DIR = Path("/models")
REMOTE_RESULTS_DIR = Path("/results")
DEFAULT_DATASETS = ("mt_bench", "gsm8k", "alpaca", "qa")
ITERATIVE_STAGE_ORDER = (
    "iter0_size",
    "iter0_depth",
    "iter1_size",
    "iter2_depth",
    "iter3_size",
    "iter4_depth",
)
VALIDATION_QUESTION_END = 100000
FLOP_ESTIMATE_MULTIPLIER = 2
MODELS_VOLUME = modal.Volume.from_name("ltd-qwen3-models", create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("ltd-qwen3-results", create_if_missing=True)

app = modal.App("ltd-qwen3-modal")
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
    """Build environment variables for Modal subprocesses.

    Args:
        extra_env: Optional extra environment variables.

    Returns:
        The environment variables used for remote subprocess execution.
    """
    env = os.environ.copy()
    env["HF_HOME"] = str(REMOTE_MODELS_DIR / ".hf")
    env["PYTHONPATH"] = str(REMOTE_PROJECT_DIR)
    env["TOKENIZERS_PARALLELISM"] = "false"
    if extra_env:
        env.update(extra_env)
    return env


def resolve_remote_model_paths(model_preset: str) -> tuple[str, str]:
    """Resolve a preset into cached model paths inside the Modal Volume.

    Args:
        model_preset: The supported Qwen3 preset.

    Returns:
        The cached base and Eagle3 model paths inside the models Volume.
    """
    base_repo_id, ea_repo_id = resolve_model_paths(model_preset=model_preset)
    return (
        str(model_cache_dir(REMOTE_MODELS_DIR, base_repo_id)),
        str(model_cache_dir(REMOTE_MODELS_DIR, ea_repo_id)),
    )


def resolve_results_path(relative_or_absolute_path: str) -> str:
    """Resolve a result path into the mounted results Volume.

    Args:
        relative_or_absolute_path: An absolute remote path or a path relative to
            ``/results``.

    Returns:
        The absolute remote path.
    """
    if not relative_or_absolute_path:
        return ""
    if relative_or_absolute_path.startswith("/"):
        return relative_or_absolute_path
    return str(REMOTE_RESULTS_DIR / relative_or_absolute_path)


def run_command(command: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Run a subprocess inside the remote LTD project directory.

    Args:
        command: The command to execute.
        extra_env: Optional extra environment variables.
    """
    subprocess.run(
        command,
        check=True,
        cwd=str(REMOTE_PROJECT_DIR),
        env=build_runtime_env(extra_env=extra_env),
    )


def ensure_results_dir(path: Path) -> None:
    """Create a results directory when needed.

    Args:
        path: The directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: The JSONL file path.

    Returns:
        Parsed JSON records.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file.

    Args:
        path: The JSONL output path.
        records: The records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def create_humaneval_split_files(
    output_root: Path,
    validation_fraction: float,
    split_seed: int,
) -> tuple[str, str]:
    """Create deterministic HumanEval train and validation split files.

    Args:
        output_root: Root directory for the split files.
        validation_fraction: Fraction of examples used for validation.
        split_seed: Random seed for the split.

    Returns:
        The train and validation question file paths.
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


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def update_phase_dataset_manifest(
    stage_dir: Path,
    validation_counts: dict[str, int],
) -> dict[str, Any]:
    """Update one phase dataset manifest with validation call counts.

    Args:
        stage_dir: Phase output directory.
        validation_counts: Validation drafter/target call totals.
    Returns:
        Updated manifest payload, or an empty dict when absent.
    """
    manifest_path = stage_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    training_draft_calls = int(
        payload.get("training_draft_model_calls", payload.get("draft_model_calls", 0))
    )
    training_target_calls = int(
        payload.get("training_target_model_calls", payload.get("target_model_calls", 0))
    )
    validation_draft_calls = int(validation_counts.get("draft_model_calls", 0))
    validation_target_calls = int(validation_counts.get("target_model_calls", 0))
    draft_model_parameters = int(payload.get("draft_model_parameters", 0))
    target_model_parameters = int(payload.get("target_model_parameters", 0))
    training_draft_flops = estimate_forward_flops(draft_model_parameters, training_draft_calls)
    training_target_flops = estimate_forward_flops(target_model_parameters, training_target_calls)
    validation_draft_flops = estimate_forward_flops(draft_model_parameters, validation_draft_calls)
    validation_target_flops = estimate_forward_flops(target_model_parameters, validation_target_calls)
    training_flops = training_draft_flops + training_target_flops
    validation_flops = validation_draft_flops + validation_target_flops
    payload["training_draft_model_calls"] = training_draft_calls
    payload["training_target_model_calls"] = training_target_calls
    payload["validation_draft_model_calls"] = validation_draft_calls
    payload["validation_target_model_calls"] = validation_target_calls
    payload["draft_model_calls"] = training_draft_calls + validation_draft_calls
    payload["target_model_calls"] = training_target_calls + validation_target_calls
    payload["training_draft_flops"] = training_draft_flops
    payload["training_target_flops"] = training_target_flops
    payload["training_flops"] = training_flops
    payload["validation_draft_flops"] = validation_draft_flops
    payload["validation_target_flops"] = validation_target_flops
    payload["validation_flops"] = validation_flops
    payload["total_flops"] = training_flops + validation_flops
    payload["flop_estimate_method"] = "2 * parameter_count * forward_calls"
    write_json(manifest_path, payload)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk.

    Args:
        path: JSON file path.
        payload: Data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file.

    Args:
        path: JSONL file path.
        payload: Record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def write_time_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write one timing summary record to a JSONL file.

    Args:
        path: JSONL output path.
        payload: Timing summary payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def iterative_state_path(root_dir: Path) -> Path:
    """Return the iterative-training state file path.

    Args:
        root_dir: Root directory for iterative-training artifacts.

    Returns:
        The state JSON path.
    """
    return root_dir / "iterative_state.json"


def stage_output_dir(root_dir: Path, stage_name: str) -> Path:
    """Return the output directory for one iterative-training stage.

    Args:
        root_dir: Root directory for iterative-training artifacts.
        stage_name: Stage identifier.

    Returns:
        The stage output directory.
    """
    return root_dir / stage_name


def stage_policy_type(stage_name: str) -> str:
    """Return whether a stage trains the size or depth policy.

    Args:
        stage_name: Stage identifier.

    Returns:
        ``"size"`` or ``"depth"``.
    """
    return "size" if stage_name.endswith("size") else "depth"


def validate_iterative_stage_name(stage_name: str) -> str:
    """Validate one iterative-training stage name.

    Args:
        stage_name: Requested stage identifier.

    Returns:
        The validated stage name.

    Raises:
        ValueError: If the stage name is not supported.
    """
    if stage_name not in ITERATIVE_STAGE_ORDER:
        supported = ", ".join(ITERATIVE_STAGE_ORDER)
        raise ValueError(
            f"Unsupported iterative stage '{stage_name}'. Use one of: {supported}."
        )
    return stage_name


def build_iterative_state(
    model_preset: str,
    dataset_train: str,
    size_total_timesteps: int,
    depth_total_timesteps: int,
    batch_size: int,
    n_steps: int,
    lr: float,
    validation_fraction: float,
    split_seed: int,
    output_subdir: str,
    root_dir: Path,
) -> dict[str, Any]:
    """Build the initial iterative-training state payload.

    Args:
        model_preset: Supported Qwen3 preset.
        dataset_train: Training dataset.
        size_total_timesteps: Size-stage PPO budget.
        depth_total_timesteps: Depth-stage PPO budget.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval train-validation split seed.
        output_subdir: Output subdirectory under ``/results``.
        root_dir: Root directory for iterative-training artifacts.

    Returns:
        The initialized iterative state.
    """
    train_question_file, validation_question_file = create_humaneval_split_files(
        output_root=root_dir,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    return {
        "model_preset": model_preset,
        "dataset_train": dataset_train,
        "size_total_timesteps": size_total_timesteps,
        "depth_total_timesteps": depth_total_timesteps,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "lr": lr,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "output_subdir": output_subdir,
        "results_root": str(root_dir),
        "train_question_file": train_question_file,
        "validation_question_file": validation_question_file,
        "stage_order": list(ITERATIVE_STAGE_ORDER),
        "completed_stages": [],
        "stage_outputs": {},
        "promoted_checkpoints": {},
        "last_completed_stage": "",
        "next_stage": ITERATIVE_STAGE_ORDER[0],
        "done": False,
    }


def rename_stage_identifier(stage_name: str) -> str:
    """Normalize legacy iterative stage names.

    Args:
        stage_name: Stage identifier stored in state.

    Returns:
        The normalized stage identifier.
    """
    if stage_name == "iter3_depth":
        return "iter4_depth"
    return stage_name


def migrate_iterative_state(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate persisted iterative state to the latest stage naming.

    Args:
        state: Persisted iterative-training state.

    Returns:
        The migrated state payload.
    """
    state["stage_order"] = [
        rename_stage_identifier(stage_name)
        for stage_name in state.get("stage_order", ITERATIVE_STAGE_ORDER)
    ]
    state["completed_stages"] = [
        rename_stage_identifier(stage_name)
        for stage_name in state.get("completed_stages", [])
    ]
    state["stage_outputs"] = {
        rename_stage_identifier(stage_name): output_dir
        for stage_name, output_dir in state.get("stage_outputs", {}).items()
    }
    state["promoted_checkpoints"] = {
        rename_stage_identifier(stage_name): checkpoint_path
        for stage_name, checkpoint_path in state.get("promoted_checkpoints", {}).items()
    }
    state["last_completed_stage"] = rename_stage_identifier(
        state.get("last_completed_stage", "")
    )
    state["next_stage"] = rename_stage_identifier(state.get("next_stage", ""))
    if "iter3_depth" in state:
        state["iter4_depth"] = state.pop("iter3_depth")
    if "final_depth_model_path" not in state and "iter4_depth" in state.get(
        "promoted_checkpoints",
        {},
    ):
        state["final_depth_model_path"] = state["promoted_checkpoints"]["iter4_depth"]
    return state


def load_or_create_iterative_state(
    model_preset: str,
    dataset_train: str,
    size_total_timesteps: int,
    depth_total_timesteps: int,
    batch_size: int,
    n_steps: int,
    lr: float,
    validation_fraction: float,
    split_seed: int,
    output_subdir: str,
) -> tuple[Path, dict[str, Any]]:
    """Load or initialize the iterative-training state file.

    Args:
        model_preset: Supported Qwen3 preset.
        dataset_train: Training dataset.
        size_total_timesteps: Size-stage PPO budget.
        depth_total_timesteps: Depth-stage PPO budget.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval train-validation split seed.
        output_subdir: Output subdirectory under ``/results``.

    Returns:
        The root directory and state payload.
    """
    root_dir = REMOTE_RESULTS_DIR / (output_subdir or f"{model_preset}/iterative")
    ensure_results_dir(root_dir)
    state_file = iterative_state_path(root_dir)
    if state_file.exists():
        state = migrate_iterative_state(read_json(state_file))
        expected_values = {
            "model_preset": model_preset,
            "dataset_train": dataset_train,
            "size_total_timesteps": size_total_timesteps,
            "depth_total_timesteps": depth_total_timesteps,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "lr": lr,
            "validation_fraction": validation_fraction,
            "split_seed": split_seed,
            "output_subdir": output_subdir,
        }
        mismatches = [
            key
            for key, value in expected_values.items()
            if state.get(key) != value
        ]
        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(
                "Existing iterative state does not match the requested "
                f"configuration. Mismatched fields: {mismatch_text}."
            )
        write_json(state_file, state)
        return root_dir, state
    state = build_iterative_state(
        model_preset=model_preset,
        dataset_train=dataset_train,
        size_total_timesteps=size_total_timesteps,
        depth_total_timesteps=depth_total_timesteps,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        output_subdir=output_subdir,
        root_dir=root_dir,
    )
    write_json(state_file, state)
    return root_dir, state


def next_pending_stage(state: dict[str, Any]) -> str:
    """Return the next unfinished iterative-training stage.

    Args:
        state: Iterative-training state payload.

    Returns:
        The next stage name, or an empty string when complete.
    """
    completed = set(state.get("completed_stages", []))
    for stage_name in state.get("stage_order", ITERATIVE_STAGE_ORDER):
        if stage_name not in completed:
            return stage_name
    return ""


def expected_promoted_checkpoint(root_dir: Path, stage_name: str) -> str:
    """Return the canonical promoted checkpoint path for a stage.

    Args:
        root_dir: Root directory for iterative-training artifacts.
        stage_name: Stage identifier.

    Returns:
        The canonical checkpoint path.
    """
    output_dir = stage_output_dir(root_dir, stage_name)
    if stage_policy_type(stage_name) == "size":
        return size_policy_checkpoint_path(output_dir)
    return depth_policy_checkpoint_path(output_dir)


def stage_dependencies(state: dict[str, Any], root_dir: Path, stage_name: str) -> tuple[str, str]:
    """Resolve the fixed-policy and resume checkpoints for one stage.

    Args:
        state: Iterative-training state payload.
        root_dir: Root directory for iterative-training artifacts.
        stage_name: Stage identifier.

    Returns:
        A tuple of ``(fixed_partner_checkpoint, resume_checkpoint)``.
    """
    promoted = state.get("promoted_checkpoints", {})
    if stage_name == "iter0_size":
        return "", ""
    if stage_name == "iter0_depth":
        return "", ""
    if stage_name == "iter1_size":
        return promoted["iter0_depth"], promoted["iter0_size"]
    if stage_name == "iter2_depth":
        return promoted["iter1_size"], promoted["iter0_depth"]
    if stage_name == "iter3_size":
        return promoted["iter2_depth"], promoted["iter1_size"]
    if stage_name == "iter4_depth":
        return promoted["iter3_size"], promoted["iter2_depth"]
    raise ValueError(f"Unsupported iterative stage: {stage_name}")


def run_iterative_stage(
    state: dict[str, Any],
    root_dir: Path,
    stage_name: str,
) -> dict[str, Any]:
    """Run one iterative-training stage and update state metadata.

    Args:
        state: Iterative-training state payload.
        root_dir: Root directory for iterative-training artifacts.
        stage_name: Stage identifier to execute.

    Returns:
        The updated state payload.
    """
    output_dir = stage_output_dir(root_dir, stage_name)
    ensure_results_dir(output_dir)
    train_question_file = state["train_question_file"]
    validation_question_file = state["validation_question_file"]
    fixed_checkpoint, resume_checkpoint = stage_dependencies(state, root_dir, stage_name)
    training_start_time = time.time()

    if stage_policy_type(stage_name) == "size":
        run_command(
            build_train_size_command(
                model_preset=state["model_preset"],
                dataset_train=state["dataset_train"],
                total_timesteps=int(state["size_total_timesteps"]),
                batch_size=int(state["batch_size"]),
                n_steps=int(state["n_steps"]),
                lr=float(state["lr"]),
                output_dir=output_dir,
                question_file=train_question_file,
                depth_model_path=fixed_checkpoint,
                rl_checkpoint_path=resume_checkpoint,
            ),
        )
        training_time_seconds = time.time() - training_start_time
        validation_start_time = time.time()
        promoted_checkpoint, validation_call_counts = select_best_size_checkpoint(
            model_preset=state["model_preset"],
            stage_name=stage_name,
            stage_dir=output_dir,
            validation_question_file=validation_question_file,
            root_dir=root_dir,
            fixed_depth_model_path=fixed_checkpoint,
        )
    else:
        run_command(
            build_train_depth_command(
                model_preset=state["model_preset"],
                dataset_train=state["dataset_train"],
                total_timesteps=int(state["depth_total_timesteps"]),
                batch_size=int(state["batch_size"]),
                n_steps=int(state["n_steps"]),
                lr=float(state["lr"]),
                output_dir=output_dir,
                question_file=train_question_file,
                rl_token_model_path=fixed_checkpoint,
                rl_checkpoint_path=resume_checkpoint,
            ),
        )
        training_time_seconds = time.time() - training_start_time
        validation_start_time = time.time()
        promoted_checkpoint, validation_call_counts = select_best_depth_checkpoint(
            model_preset=state["model_preset"],
            stage_name=stage_name,
            stage_dir=output_dir,
            validation_question_file=validation_question_file,
            root_dir=root_dir,
            fixed_token_model_path=fixed_checkpoint,
        )
    phase_manifest = update_phase_dataset_manifest(output_dir, validation_call_counts)
    validation_time_seconds = time.time() - validation_start_time
    write_time_summary(
        output_dir / "time_summary.jsonl",
        {
            "event": "summary",
            "stage_name": stage_name,
            "phase_type": stage_policy_type(stage_name),
            "total_timesteps": int(
                state["size_total_timesteps"]
                if stage_policy_type(stage_name) == "size"
                else state["depth_total_timesteps"]
            ),
            "total_time_seconds": training_time_seconds + validation_time_seconds,
            "training_time_seconds": training_time_seconds,
            "validation_time_seconds": validation_time_seconds,
            "training_flops": int(phase_manifest.get("training_flops", 0)),
            "validation_flops": int(phase_manifest.get("validation_flops", 0)),
            "total_flops": int(phase_manifest.get("total_flops", 0)),
        },
    )

    state.setdefault("completed_stages", []).append(stage_name)
    state.setdefault("stage_outputs", {})[stage_name] = str(output_dir)
    state.setdefault("promoted_checkpoints", {})[stage_name] = promoted_checkpoint
    state["last_completed_stage"] = stage_name
    state["next_stage"] = next_pending_stage(state)
    state["done"] = state["next_stage"] == ""
    if "iter3_size" in state["promoted_checkpoints"]:
        state["final_token_model_path"] = state["promoted_checkpoints"]["iter3_size"]
    if "iter4_depth" in state["promoted_checkpoints"]:
        state["final_depth_model_path"] = state["promoted_checkpoints"]["iter4_depth"]
    return state


def size_policy_checkpoint_path(output_dir: Path) -> str:
    """Return the canonical size-policy checkpoint path for an output directory.

    Args:
        output_dir: The output directory that stores size-policy artifacts.

    Returns:
        The canonical size-policy checkpoint path.
    """
    return str(output_dir / "ppo_speculative_decoder_controller_rebuttal.zip")


def depth_policy_checkpoint_path(output_dir: Path) -> str:
    """Return the canonical depth-policy checkpoint path for an output directory.

    Args:
        output_dir: The output directory that stores depth-policy artifacts.

    Returns:
        The canonical depth-policy checkpoint path.
    """
    return str(output_dir / "ppo_speculative_decoder_controller_v1_single_action.zip")


def first_choice(record: dict[str, Any]) -> dict[str, Any]:
    """Return the first choice entry from a benchmark record.

    Args:
        record: One JSONL record.

    Returns:
        The first choice object.
    """
    return record["choices"][0]


def extract_speed(record: dict[str, Any]) -> float:
    """Compute token throughput for one benchmark record.

    Args:
        record: One JSONL record.

    Returns:
        Tokens per second for the first choice.
    """
    choice = first_choice(record)
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
        record: One JSONL record.

    Returns:
        Acceptance lengths for the first choice.
    """
    choice = first_choice(record)
    return [float(value) for value in choice.get("pre_num", [])]


def load_metrics(path: Path) -> tuple[dict[int, float], dict[int, list[float]]]:
    """Load per-question speed and acceptance statistics from a JSONL file.

    Args:
        path: The JSONL file path.

    Returns:
        A tuple of ``question_id -> speed`` and ``question_id -> acceptance lengths``.
    """
    data = read_jsonl(path)
    speeds: dict[int, float] = {}
    accept_lengths: dict[int, list[float]] = {}
    for record in data:
        question_id = int(record["question_id"])
        speeds[question_id] = extract_speed(record)
        accept_lengths[question_id] = extract_accept_lengths(record)
    return speeds, accept_lengths


def compute_validation_metrics(
    baseline_path: Path,
    method_path: Path,
) -> tuple[float, float]:
    """Compute validation speedup and acceptance length.

    Args:
        baseline_path: Baseline JSONL path.
        method_path: Method JSONL path.

    Returns:
        The validation speedup ratio and acceptance length.
    """
    baseline_speeds, _ = load_metrics(baseline_path)
    method_speeds, accept_lengths_by_qid = load_metrics(method_path)
    shared_ids = sorted(set(baseline_speeds) & set(method_speeds))
    if not shared_ids:
        raise ValueError(
            f"No overlapping question ids between {baseline_path} and {method_path}."
        )
    baseline_mean = mean(baseline_speeds[qid] for qid in shared_ids)
    method_mean = mean(method_speeds[qid] for qid in shared_ids)
    speedup = method_mean / baseline_mean if baseline_mean > 0 else 0.0
    accept_lengths = [
        value
        for qid in shared_ids
        for value in accept_lengths_by_qid.get(qid, [])
    ]
    tau = mean(accept_lengths) if accept_lengths else 0.0
    return speedup, tau


def parse_checkpoint_step(checkpoint_path: Path) -> int:
    """Extract the training step from a checkpoint filename.

    Args:
        checkpoint_path: The checkpoint path.

    Returns:
        The parsed training step, or ``0`` when unavailable.
    """
    stem = checkpoint_path.stem
    if "_step_" not in stem:
        return 0
    try:
        return int(stem.rsplit("_step_", maxsplit=1)[1])
    except ValueError:
        return 0


def checkpoint_step_for_candidate(stage_dir: Path, checkpoint_path: Path) -> int:
    """Resolve the training step for a candidate checkpoint.

    Args:
        stage_dir: Stage output directory.
        checkpoint_path: Candidate checkpoint path.

    Returns:
        The training step associated with the checkpoint.
    """
    parsed_step = parse_checkpoint_step(checkpoint_path)
    if parsed_step > 0:
        return parsed_step
    summary_path = stage_dir / "training_summary.json"
    if not summary_path.exists():
        return 0
    summary = read_json(summary_path)
    return int(summary.get("best_model_step", 0))


def list_candidate_checkpoints(stage_dir: Path) -> list[Path]:
    """List candidate checkpoints for validation selection.

    Args:
        stage_dir: The stage output directory.

    Returns:
        Candidate checkpoint paths sorted by step.
    """
    step_checkpoints = sorted(
        stage_dir.glob("ppo_speculative_decoder_controller_step_*.zip"),
        key=lambda checkpoint_path: checkpoint_step_for_candidate(
            stage_dir,
            checkpoint_path,
        ),
    )
    best_train_checkpoint = stage_dir / "ppo_speculative_decoder_controller_best.zip"
    candidates = list(step_checkpoints)
    if best_train_checkpoint.exists():
        candidates.append(best_train_checkpoint)
    return candidates


def promote_checkpoint(source_path: Path, destination_path: Path) -> str:
    """Copy a selected checkpoint into the canonical destination path.

    Args:
        source_path: The selected checkpoint path.
        destination_path: The canonical destination path.

    Returns:
        The canonical checkpoint path.
    """
    if source_path.resolve() != destination_path.resolve():
        shutil.copyfile(source_path, destination_path)
    return str(destination_path)


def build_train_size_command(
    model_preset: str,
    dataset_train: str,
    total_timesteps: int,
    batch_size: int,
    n_steps: int,
    lr: float,
    output_dir: Path,
    question_file: str = "",
    depth_model_path: str = "",
    rl_checkpoint_path: str = "",
) -> list[str]:
    """Build the subprocess command for size-policy training.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        output_dir: Output directory for this stage.
        question_file: Optional JSONL file used for training prompts.
        depth_model_path: Optional fixed depth-policy checkpoint path.
        rl_checkpoint_path: Optional size-policy resume checkpoint path.

    Returns:
        The subprocess command.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    command = [
        "python",
        "-m",
        "rl.rl_total",
        "--base_model_path",
        base_model_path,
        "--ea_model_path",
        ea_model_path,
        "--data_dir",
        "./eagle/data",
        "--dataset_train",
        dataset_train,
        "--save_path",
        str(output_dir),
        "--question_file",
        question_file,
        "--total_timesteps",
        str(total_timesteps),
        "--batch_size",
        str(batch_size),
        "--n_steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--gamma",
        "0.9",
        "--n_epochs",
        "20",
        "--ent_coef",
        "0.01",
        "--pi_arch",
        "1024",
        "256",
        "--vf_arch",
        "1024",
        "256",
    ]
    if depth_model_path:
        command.extend(
            ["--use_dyn_depth", "--depth_model", resolve_results_path(depth_model_path)]
        )
    if rl_checkpoint_path:
        command.extend(
            ["--rl_checkpoint_path", resolve_results_path(rl_checkpoint_path)]
        )
    return command


def build_train_depth_command(
    model_preset: str,
    dataset_train: str,
    total_timesteps: int,
    batch_size: int,
    n_steps: int,
    lr: float,
    output_dir: Path,
    question_file: str = "",
    rl_token_model_path: str = "",
    rl_checkpoint_path: str = "",
) -> list[str]:
    """Build the subprocess command for depth-policy training.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        output_dir: Output directory for this stage.
        question_file: Optional JSONL file used for training prompts.
        rl_token_model_path: Optional size-policy checkpoint path.
        rl_checkpoint_path: Optional depth-policy resume checkpoint path.

    Returns:
        The subprocess command.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    command = [
        "python",
        "-m",
        "rl.rl_depth",
        "--base_model_path",
        base_model_path,
        "--ea_model_path",
        ea_model_path,
        "--data_dir",
        "./eagle/data",
        "--dataset_train",
        dataset_train,
        "--save_path",
        str(output_dir),
        "--question_file",
        question_file,
        "--total_timesteps",
        str(total_timesteps),
        "--batch_size",
        str(batch_size),
        "--n_steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--gamma",
        "0.999",
        "--n_epochs",
        "20",
        "--ent_coef",
        "0.01",
        "--pi_arch",
        "1024",
        "--vf_arch",
        "1024",
        "256",
    ]
    if rl_token_model_path:
        command.extend(
            ["--rl_token_model_path", resolve_results_path(rl_token_model_path)]
        )
    if rl_checkpoint_path:
        command.extend(
            ["--rl_checkpoint_path", resolve_results_path(rl_checkpoint_path)]
        )
    return command


@app.function(
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=60 * 60,
)
def download_models(model_preset: str = "qwen3_8b") -> dict[str, str]:
    """Download one supported Qwen3 plus AngelSlim model pair to a Volume.

    Args:
        model_preset: The supported model preset.

    Returns:
        The cached remote paths for the base and Eagle3 models.
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
def train_size_policy(
    model_preset: str = "qwen3_8b",
    dataset_train: str = "humaneval",
    total_timesteps: int = 100000,
    batch_size: int = 256,
    n_steps: int = 2048,
    lr: float = 1e-3,
    question_file: str = "",
    depth_model_path: str = "",
    rl_checkpoint_path: str = "",
    save_subdir: str = "",
) -> str:
    """Train the LTD size policy on Modal.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset under ``eagle/data``.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        question_file: Optional JSONL file used for training prompts.
        depth_model_path: Optional fixed depth-policy checkpoint path relative
            to ``/results`` or absolute in the remote container.
        rl_checkpoint_path: Optional size-policy checkpoint path used to resume
            training from an earlier stage.
        save_subdir: Optional results subdirectory under ``/results``.

    Returns:
        The output directory containing the trained policy.
    """
    output_dir = REMOTE_RESULTS_DIR / (save_subdir or f"{model_preset}/size")
    ensure_results_dir(output_dir)
    training_start_time = time.time()
    command = build_train_size_command(
        model_preset=model_preset,
        dataset_train=dataset_train,
        total_timesteps=total_timesteps,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        output_dir=output_dir,
        question_file=question_file,
        depth_model_path=depth_model_path,
        rl_checkpoint_path=rl_checkpoint_path,
    )

    run_command(command)
    training_time_seconds = time.time() - training_start_time
    training_manifest = read_json(output_dir / "dataset_manifest.json")
    training_flops = int(training_manifest.get("training_flops", training_manifest.get("total_flops", 0)))
    write_time_summary(
        output_dir / "time_summary.jsonl",
        {
            "event": "summary",
            "phase_type": "size",
            "total_timesteps": total_timesteps,
            "total_time_seconds": training_time_seconds,
            "training_time_seconds": training_time_seconds,
            "validation_time_seconds": 0.0,
            "training_flops": training_flops,
            "validation_flops": 0,
            "total_flops": training_flops,
        },
    )
    RESULTS_VOLUME.commit()
    return str(output_dir)


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
    dataset_train: str = "humaneval",
    total_timesteps: int = 1000000,
    batch_size: int = 256,
    n_steps: int = 2048,
    lr: float = 1e-3,
    question_file: str = "",
    rl_token_model_path: str = "",
    rl_checkpoint_path: str = "",
    save_subdir: str = "",
) -> str:
    """Train the LTD depth policy on Modal.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset under ``eagle/data``.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        question_file: Optional JSONL file used for training prompts.
        rl_token_model_path: Optional size-policy checkpoint path relative to
            ``/results`` or absolute in the remote container.
        rl_checkpoint_path: Optional depth-policy checkpoint path relative to
            ``/results`` or absolute in the remote container.
        save_subdir: Optional results subdirectory under ``/results``.

    Returns:
        The output directory containing the trained policy.
    """
    output_dir = REMOTE_RESULTS_DIR / (save_subdir or f"{model_preset}/depth")
    ensure_results_dir(output_dir)
    training_start_time = time.time()
    command = build_train_depth_command(
        model_preset=model_preset,
        dataset_train=dataset_train,
        total_timesteps=total_timesteps,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        output_dir=output_dir,
        question_file=question_file,
        rl_token_model_path=rl_token_model_path,
        rl_checkpoint_path=rl_checkpoint_path,
    )

    run_command(command)
    training_time_seconds = time.time() - training_start_time
    training_manifest = read_json(output_dir / "dataset_manifest.json")
    training_flops = int(training_manifest.get("training_flops", training_manifest.get("total_flops", 0)))
    write_time_summary(
        output_dir / "time_summary.jsonl",
        {
            "event": "summary",
            "phase_type": "depth",
            "total_timesteps": total_timesteps,
            "total_time_seconds": training_time_seconds,
            "training_time_seconds": training_time_seconds,
            "validation_time_seconds": 0.0,
            "training_flops": training_flops,
            "validation_flops": 0,
            "total_flops": training_flops,
        },
    )
    RESULTS_VOLUME.commit()
    return str(output_dir)


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=24 * 60 * 60,
)
def iterative_train(
    model_preset: str = "qwen3_8b",
    dataset_train: str = "humaneval",
    current_stage: str = "",
    size_total_timesteps: int = 100000,
    depth_total_timesteps: int = 1000000,
    batch_size: int = 256,
    n_steps: int = 2048,
    lr: float = 1e-3,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    output_subdir: str = "",
) -> dict[str, str]:
    """Run one pending iterative-training stage on Modal.

    Re-running this function advances the workflow by exactly one stage using
    persistent state in the results volume. This keeps long LTD training runs
    split across multiple Modal executions instead of requiring one 24-hour+
    function call.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset under ``eagle/data``.
        current_stage: Explicit iterative-training stage to execute.
        size_total_timesteps: PPO training horizon for size-policy stages.
        depth_total_timesteps: PPO training horizon for depth-policy stages.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        validation_fraction: Fraction of HumanEval used for validation.
        split_seed: Random seed for the HumanEval train-validation split.
        output_subdir: Optional results root under ``/results``.

    Returns:
        The updated iterative-training state.
    """
    if dataset_train != "humaneval":
        raise ValueError(
            "The paper-faithful iterative workflow is implemented only for "
            "dataset_train='humaneval'."
        )
    root_dir, state = load_or_create_iterative_state(
        model_preset=model_preset,
        dataset_train=dataset_train,
        size_total_timesteps=size_total_timesteps,
        depth_total_timesteps=depth_total_timesteps,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        output_subdir=output_subdir,
    )
    if state["done"]:
        RESULTS_VOLUME.commit()
        return state
    expected_stage = next_pending_stage(state)
    if not current_stage:
        raise ValueError(
            "current_stage must be provided explicitly. "
            f"The next pending stage is '{expected_stage}'."
        )
    stage_name = validate_iterative_stage_name(current_stage)
    if stage_name != expected_stage:
        raise ValueError(
            f"Requested current_stage '{stage_name}' does not match the next "
            f"pending stage '{expected_stage}'."
        )
    state = run_iterative_stage(
        state=state,
        root_dir=root_dir,
        stage_name=stage_name,
    )
    write_json(iterative_state_path(root_dir), state)
    RESULTS_VOLUME.commit()
    return state


@app.function(
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=10 * 60,
)
def iterative_status(
    model_preset: str = "qwen3_8b",
    output_subdir: str = "",
) -> dict[str, Any]:
    """Return the persisted iterative-training state.

    Args:
        model_preset: Supported Qwen3 preset.
        output_subdir: Output root under ``/results``.

    Returns:
        The current iterative-training state.
    """
    root_dir = REMOTE_RESULTS_DIR / (output_subdir or f"{model_preset}/iterative")
    state_file = iterative_state_path(root_dir)
    if not state_file.exists():
        return {
            "exists": False,
            "results_root": str(root_dir),
            "next_stage": ITERATIVE_STAGE_ORDER[0],
            "done": False,
        }
    state = read_json(state_file)
    state["exists"] = True
    return state


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
    token_model_path: str = "",
    depth_model_path: str = "",
    answer_file_override: str = "",
) -> tuple[list[str], str]:
    """Build an LTD evaluation subprocess command.

    Args:
        script_module: The Python module to execute.
        model_preset: The supported model preset.
        bench_name: The evaluation benchmark name.
        output_subdir: Results subdirectory under ``/results``.
        question_begin: Start index for the question subset.
        question_end: End index for the question subset.
        temperature: Generation temperature.
        num_choices: Number of generations per prompt.
        total_token: Draft token budget.
        depth: Draft depth.
        question_file: Optional JSONL file used for evaluation prompts.
        token_model_path: Optional size-policy checkpoint path.
        depth_model_path: Optional depth-policy checkpoint path.
        answer_file_override: Optional absolute answer-file path.

    Returns:
        The subprocess command and output file path.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    if answer_file_override:
        answer_file = Path(answer_file_override)
        ensure_results_dir(answer_file.parent)
    else:
        output_dir = REMOTE_RESULTS_DIR / output_subdir
        ensure_results_dir(output_dir)
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
    if token_model_path:
        command.extend(["--use_dyn_token", "--token_model", resolve_results_path(token_model_path)])
    if depth_model_path:
        command.extend(["--use_dyn_depth", "--depth_model", resolve_results_path(depth_model_path)])
    return command, str(answer_file)


def ensure_validation_baseline(
    model_preset: str,
    validation_question_file: str,
    root_dir: Path,
) -> Path:
    """Run and cache the baseline validation set generations.

    Args:
        model_preset: The supported model preset.
        validation_question_file: HumanEval validation split JSONL path.
        root_dir: Root directory for iterative-training artifacts.

    Returns:
        The baseline validation JSONL path.
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


def evaluate_size_candidate(
    model_preset: str,
    candidate_path: Path,
    validation_question_file: str,
    baseline_path: Path,
    stage_dir: Path,
    fixed_depth_model_path: str = "",
) -> dict[str, Any]:
    """Evaluate one size-policy checkpoint on the validation split.

    Args:
        model_preset: The supported model preset.
        candidate_path: The candidate size-policy checkpoint.
        validation_question_file: HumanEval validation split JSONL path.
        baseline_path: Cached baseline validation JSONL path.
        stage_dir: Stage output directory.
        fixed_depth_model_path: Optional fixed depth-policy checkpoint path.

    Returns:
        Validation metrics for the candidate checkpoint.
    """
    validation_dir = stage_dir / "validation"
    ensure_results_dir(validation_dir)
    step = checkpoint_step_for_candidate(stage_dir, candidate_path)
    candidate_name = candidate_path.stem
    if fixed_depth_model_path:
        answer_path = validation_dir / f"{candidate_name}.jsonl"
        command, _ = build_eval_command(
            script_module="eagle.evaluation.gen_ea_answer_qwen3",
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
            token_model_path=str(candidate_path),
            depth_model_path=fixed_depth_model_path,
            answer_file_override=str(answer_path),
        )
        run_command(command)
        speedup, tau = compute_validation_metrics(baseline_path, answer_path)
        answer_counts = read_answer_manifest(answer_path)
        return {
            "checkpoint_path": str(candidate_path),
            "checkpoint_step": step,
            "speedup": speedup,
            "tau": tau,
            "validation_draft_model_calls": answer_counts["draft_model_calls"],
            "validation_target_model_calls": answer_counts["target_model_calls"],
        }

    depth_metrics: list[dict[str, float]] = []
    total_validation_draft_calls = 0
    total_validation_target_calls = 0
    for draft_depth in range(1, 13):
        answer_path = validation_dir / f"{candidate_name}_depth_{draft_depth}.jsonl"
        command, _ = build_eval_command(
            script_module="eagle.evaluation.gen_ea_answer_qwen3",
            model_preset=model_preset,
            bench_name="humaneval",
            output_subdir="",
            question_begin=0,
            question_end=VALIDATION_QUESTION_END,
            temperature=0.0,
            num_choices=1,
            total_token=60,
            depth=draft_depth,
            question_file=validation_question_file,
            token_model_path=str(candidate_path),
            answer_file_override=str(answer_path),
        )
        run_command(command)
        speedup, tau = compute_validation_metrics(baseline_path, answer_path)
        answer_counts = read_answer_manifest(answer_path)
        total_validation_draft_calls += answer_counts["draft_model_calls"]
        total_validation_target_calls += answer_counts["target_model_calls"]
        depth_metrics.append(
            {
                "depth": float(draft_depth),
                "speedup": speedup,
                "tau": tau,
            }
        )

    return {
        "checkpoint_path": str(candidate_path),
        "checkpoint_step": step,
        "speedup": mean(metric["speedup"] for metric in depth_metrics),
        "tau": mean(metric["tau"] for metric in depth_metrics),
        "depth_sweep": depth_metrics,
        "validation_draft_model_calls": total_validation_draft_calls,
        "validation_target_model_calls": total_validation_target_calls,
    }


def evaluate_depth_candidate(
    model_preset: str,
    candidate_path: Path,
    validation_question_file: str,
    baseline_path: Path,
    stage_dir: Path,
    fixed_token_model_path: str = "",
) -> dict[str, Any]:
    """Evaluate one depth-policy checkpoint on the validation split.

    Args:
        model_preset: The supported model preset.
        candidate_path: The candidate depth-policy checkpoint.
        validation_question_file: HumanEval validation split JSONL path.
        baseline_path: Cached baseline validation JSONL path.
        stage_dir: Stage output directory.
        fixed_token_model_path: Optional fixed size-policy checkpoint path.

    Returns:
        Validation metrics for the candidate checkpoint.
    """
    validation_dir = stage_dir / "validation"
    ensure_results_dir(validation_dir)
    answer_path = validation_dir / f"{candidate_path.stem}.jsonl"
    command, _ = build_eval_command(
        script_module="eagle.evaluation.gen_ea_answer_qwen3",
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
        token_model_path=fixed_token_model_path,
        depth_model_path=str(candidate_path),
        answer_file_override=str(answer_path),
    )
    run_command(command)
    speedup, tau = compute_validation_metrics(baseline_path, answer_path)
    answer_counts = read_answer_manifest(answer_path)
    return {
        "checkpoint_path": str(candidate_path),
        "checkpoint_step": checkpoint_step_for_candidate(stage_dir, candidate_path),
        "speedup": speedup,
        "tau": tau,
        "validation_draft_model_calls": answer_counts["draft_model_calls"],
        "validation_target_model_calls": answer_counts["target_model_calls"],
    }


def select_best_size_checkpoint(
    model_preset: str,
    stage_name: str,
    stage_dir: Path,
    validation_question_file: str,
    root_dir: Path,
    fixed_depth_model_path: str = "",
) -> tuple[str, dict[str, int]]:
    """Select the best size-policy checkpoint by validation speedup.

    Args:
        model_preset: The supported model preset.
        stage_name: Human-readable stage name.
        stage_dir: Stage output directory.
        validation_question_file: HumanEval validation split JSONL path.
        root_dir: Root directory for iterative-training artifacts.
        fixed_depth_model_path: Optional fixed depth-policy checkpoint path.

    Returns:
        The promoted canonical checkpoint path and validation call totals.
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
    results: list[dict[str, Any]] = []
    metrics_log_path = stage_dir / "validation_metrics.jsonl"
    total_validation_draft_calls = baseline_counts["draft_model_calls"]
    total_validation_target_calls = baseline_counts["target_model_calls"]
    for candidate_path in list_candidate_checkpoints(stage_dir):
        metrics = evaluate_size_candidate(
            model_preset=model_preset,
            candidate_path=candidate_path,
            validation_question_file=validation_question_file,
            baseline_path=baseline_path,
            stage_dir=stage_dir,
            fixed_depth_model_path=fixed_depth_model_path,
        )
        total_validation_draft_calls += int(metrics.get("validation_draft_model_calls", 0))
        total_validation_target_calls += int(metrics.get("validation_target_model_calls", 0))
        results.append(metrics)
        append_jsonl_record(
            metrics_log_path,
            {
                "event": "candidate",
                "stage_name": stage_name,
                "policy_type": "size",
                **metrics,
            },
        )
    if not results:
        raise ValueError(f"No candidate size checkpoints found in {stage_dir}.")
    best_result = max(
        results,
        key=lambda item: (item["speedup"], item["tau"], item["checkpoint_step"]),
    )

    promoted_path = promote_checkpoint(
        source_path=Path(best_result["checkpoint_path"]),
        destination_path=Path(size_policy_checkpoint_path(stage_dir)),
    )
    append_jsonl_record(
        metrics_log_path,
        {
            "event": "selected",
            "stage_name": stage_name,
            "policy_type": "size",
            **best_result,
            "promoted_checkpoint_path": promoted_path,
        },
    )
    summary_path = stage_dir / "validation_selection.json"
    summary_payload = {
        "stage_name": stage_name,
        "policy_type": "size",
        "baseline_path": str(baseline_path),
        "validation_question_file": validation_question_file,
        "fixed_depth_model_path": fixed_depth_model_path,
        "best_result": best_result,
        "candidate_results": results,
        "validation_metrics_log_path": str(metrics_log_path),
        "promoted_checkpoint_path": promoted_path,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return promoted_path, {
        "draft_model_calls": total_validation_draft_calls,
        "target_model_calls": total_validation_target_calls,
    }


def select_best_depth_checkpoint(
    model_preset: str,
    stage_name: str,
    stage_dir: Path,
    validation_question_file: str,
    root_dir: Path,
    fixed_token_model_path: str = "",
) -> tuple[str, dict[str, int]]:
    """Select the best depth-policy checkpoint by validation speedup.

    Args:
        model_preset: The supported model preset.
        stage_name: Human-readable stage name.
        stage_dir: Stage output directory.
        validation_question_file: HumanEval validation split JSONL path.
        root_dir: Root directory for iterative-training artifacts.
        fixed_token_model_path: Optional fixed size-policy checkpoint path.

    Returns:
        The promoted canonical checkpoint path and validation call totals.
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
    results: list[dict[str, Any]] = []
    metrics_log_path = stage_dir / "validation_metrics.jsonl"
    total_validation_draft_calls = baseline_counts["draft_model_calls"]
    total_validation_target_calls = baseline_counts["target_model_calls"]
    for candidate_path in list_candidate_checkpoints(stage_dir):
        metrics = evaluate_depth_candidate(
            model_preset=model_preset,
            candidate_path=candidate_path,
            validation_question_file=validation_question_file,
            baseline_path=baseline_path,
            stage_dir=stage_dir,
            fixed_token_model_path=fixed_token_model_path,
        )
        total_validation_draft_calls += int(metrics.get("validation_draft_model_calls", 0))
        total_validation_target_calls += int(metrics.get("validation_target_model_calls", 0))
        results.append(metrics)
        append_jsonl_record(
            metrics_log_path,
            {
                "event": "candidate",
                "stage_name": stage_name,
                "policy_type": "depth",
                **metrics,
            },
        )
    if not results:
        raise ValueError(f"No candidate depth checkpoints found in {stage_dir}.")
    best_result = max(
        results,
        key=lambda item: (item["speedup"], item["tau"], item["checkpoint_step"]),
    )

    promoted_path = promote_checkpoint(
        source_path=Path(best_result["checkpoint_path"]),
        destination_path=Path(depth_policy_checkpoint_path(stage_dir)),
    )
    append_jsonl_record(
        metrics_log_path,
        {
            "event": "selected",
            "stage_name": stage_name,
            "policy_type": "depth",
            **best_result,
            "promoted_checkpoint_path": promoted_path,
        },
    )
    summary_path = stage_dir / "validation_selection.json"
    summary_payload = {
        "stage_name": stage_name,
        "policy_type": "depth",
        "baseline_path": str(baseline_path),
        "validation_question_file": validation_question_file,
        "fixed_token_model_path": fixed_token_model_path,
        "best_result": best_result,
        "candidate_results": results,
        "validation_metrics_log_path": str(metrics_log_path),
        "promoted_checkpoint_path": promoted_path,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return promoted_path, {
        "draft_model_calls": total_validation_draft_calls,
        "target_model_calls": total_validation_target_calls,
    }


def summarize_results(
    input_root: str,
    output_csv: str,
    model_label: str,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
) -> str:
    """Run the benchmark summarizer inside the remote project.

    Args:
        input_root: Results root under ``/results`` or an absolute remote path.
        output_csv: Output CSV path under ``/results`` or an absolute remote path.
        model_label: Label written in the CSV model column.
        datasets: Datasets included in the summary.
        methods: Methods included in the summary.

    Returns:
        The absolute output CSV path.
    """
    input_root_path = resolve_results_path(input_root)
    output_csv_path = resolve_results_path(output_csv)
    command = [
        "python",
        "summarize_qwen3_benchmarks.py",
        "--input-root",
        input_root_path,
        "--output-csv",
        output_csv_path,
        "--model-label",
        model_label,
        "--datasets",
        *datasets,
        "--methods",
        *methods,
    ]
    run_command(command)
    RESULTS_VOLUME.commit()
    return output_csv_path


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=8 * 60 * 60,
)
def evaluate_baseline(
    model_preset: str = "qwen3_8b",
    bench_name: str = "gsm8k",
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
) -> str:
    """Run baseline Qwen3 evaluation on Modal.

    Args:
        model_preset: The supported model preset.
        bench_name: The evaluation benchmark name.
        question_begin: Start index for the question subset.
        question_end: End index for the question subset.
        temperature: Generation temperature.
        num_choices: Number of generations per prompt.

    Returns:
        The JSONL output path.
    """
    command, answer_file = build_eval_command(
        script_module="eagle.evaluation.gen_baseline_answer_qwen3",
        model_preset=model_preset,
        bench_name=bench_name,
        output_subdir=f"{model_preset}/eval/baseline",
        question_begin=question_begin,
        question_end=question_end,
        temperature=temperature,
        num_choices=num_choices,
        total_token=60,
        depth=8,
    )
    run_command(command)
    RESULTS_VOLUME.commit()
    return answer_file


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=8 * 60 * 60,
)
def evaluate_eagle3(
    model_preset: str = "qwen3_8b",
    bench_name: str = "gsm8k",
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
    total_token: int = 60,
    depth: int = 8,
) -> str:
    """Run Eagle3 evaluation on Modal.

    Args:
        model_preset: The supported model preset.
        bench_name: The evaluation benchmark name.
        question_begin: Start index for the question subset.
        question_end: End index for the question subset.
        temperature: Generation temperature.
        num_choices: Number of generations per prompt.
        total_token: Draft token budget.
        depth: Draft depth.

    Returns:
        The JSONL output path.
    """
    command, answer_file = build_eval_command(
        script_module="eagle.evaluation.gen_ea_answer_qwen3",
        model_preset=model_preset,
        bench_name=bench_name,
        output_subdir=f"{model_preset}/eval/eagle3",
        question_begin=question_begin,
        question_end=question_end,
        temperature=temperature,
        num_choices=num_choices,
        total_token=total_token,
        depth=depth,
    )
    run_command(command)
    RESULTS_VOLUME.commit()
    return answer_file


@app.function(
    gpu="H100!",
    image=image,
    volumes={
        REMOTE_MODELS_DIR.as_posix(): MODELS_VOLUME,
        REMOTE_RESULTS_DIR.as_posix(): RESULTS_VOLUME,
    },
    timeout=8 * 60 * 60,
)
def evaluate_ltd(
    model_preset: str = "qwen3_8b",
    bench_name: str = "gsm8k",
    token_model_path: str = "",
    depth_model_path: str = "",
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
    total_token: int = 60,
    depth: int = 8,
) -> str:
    """Run LTD evaluation on Modal with trained policies.

    Args:
        model_preset: The supported model preset.
        bench_name: The evaluation benchmark name.
        token_model_path: Size-policy checkpoint path relative to ``/results``.
        depth_model_path: Depth-policy checkpoint path relative to ``/results``.
        question_begin: Start index for the question subset.
        question_end: End index for the question subset.
        temperature: Generation temperature.
        num_choices: Number of generations per prompt.
        total_token: Draft token budget.
        depth: Draft depth.

    Returns:
        The JSONL output path.
    """
    if not token_model_path or not depth_model_path:
        raise ValueError(
            "Both token_model_path and depth_model_path must be provided for LTD evaluation."
        )
    command, answer_file = build_eval_command(
        script_module="eagle.evaluation.gen_ea_answer_qwen3",
        model_preset=model_preset,
        bench_name=bench_name,
        output_subdir=f"{model_preset}/eval/ltd",
        question_begin=question_begin,
        question_end=question_end,
        temperature=temperature,
        num_choices=num_choices,
        total_token=total_token,
        depth=depth,
        token_model_path=token_model_path,
        depth_model_path=depth_model_path,
    )
    run_command(command)
    RESULTS_VOLUME.commit()
    return answer_file


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
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    question_begin: int = 0,
    question_end: int = 80,
    temperature: float = 0.0,
    num_choices: int = 1,
    total_token: int = 60,
    depth: int = 8,
    include_baseline: bool = True,
    include_eagle3: bool = True,
    include_ltd: bool = True,
    token_model_path: str = "",
    depth_model_path: str = "",
    output_subdir: str = "",
    model_label: str = "",
) -> dict[str, str]:
    """Run baseline, Eagle3, and LTD benchmarks and summarize them.

    Args:
        model_preset: Supported Qwen3 preset.
        datasets: Datasets to evaluate.
        question_begin: Question start index.
        question_end: Question end index.
        temperature: Generation temperature.
        num_choices: Number of generations per prompt.
        total_token: Draft token budget.
        depth: Draft depth.
        include_baseline: Whether to run the baseline method.
        include_eagle3: Whether to run the Eagle3 method.
        include_ltd: Whether to run the LTD method.
        token_model_path: Size-policy checkpoint path for LTD.
        depth_model_path: Depth-policy checkpoint path for LTD.
        output_subdir: Output root under ``/results``.
        model_label: Model label for the summary CSV.

    Returns:
        A mapping containing the results root and summary CSV path.
    """
    if not include_baseline and (include_eagle3 or include_ltd):
        raise ValueError(
            "include_baseline must stay enabled when computing speedups for Eagle3 or LTD."
        )
    output_root = output_subdir or f"{model_preset}/benchmark_outputs"
    methods: list[str] = []

    for dataset in datasets:
        if include_baseline:
            command, _ = build_eval_command(
                script_module="eagle.evaluation.gen_baseline_answer_qwen3",
                model_preset=model_preset,
                bench_name=dataset,
                output_subdir=f"{output_root}/baseline",
                question_begin=question_begin,
                question_end=question_end,
                temperature=temperature,
                num_choices=num_choices,
                total_token=total_token,
                depth=depth,
            )
            run_command(command)
        if include_eagle3:
            command, _ = build_eval_command(
                script_module="eagle.evaluation.gen_ea_answer_qwen3",
                model_preset=model_preset,
                bench_name=dataset,
                output_subdir=f"{output_root}/eagle3",
                question_begin=question_begin,
                question_end=question_end,
                temperature=temperature,
                num_choices=num_choices,
                total_token=total_token,
                depth=depth,
            )
            run_command(command)
        if include_ltd:
            if not token_model_path or not depth_model_path:
                raise ValueError(
                    "token_model_path and depth_model_path are required when include_ltd=True."
                )
            command, _ = build_eval_command(
                script_module="eagle.evaluation.gen_ea_answer_qwen3",
                model_preset=model_preset,
                bench_name=dataset,
                output_subdir=f"{output_root}/ltd",
                question_begin=question_begin,
                question_end=question_end,
                temperature=temperature,
                num_choices=num_choices,
                total_token=total_token,
                depth=depth,
                token_model_path=token_model_path,
                depth_model_path=depth_model_path,
            )
            run_command(command)

    if include_baseline:
        methods.append("baseline")
    if include_eagle3:
        methods.append("eagle3")
    if include_ltd:
        methods.append("ltd")

    summary_csv = summarize_results(
        input_root=output_root,
        output_csv=f"{output_root}/summary.csv",
        model_label=model_label or model_preset,
        datasets=datasets,
        methods=tuple(methods),
    )
    return {
        "results_root": resolve_results_path(output_root),
        "summary_csv": summary_csv,
    }


@app.local_entrypoint()
def main(
    action: str = "benchmark-suite",
    model_preset: str = "qwen3_8b",
    token_model_path: str = "",
    depth_model_path: str = "",
    current_stage: str = "",
    dataset_train: str = "humaneval",
    total_timesteps: int = 0,
    size_total_timesteps: int = 100000,
    depth_total_timesteps: int = 1000000,
    batch_size: int = 256,
    n_steps: int = 2048,
    lr: float = 1e-3,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    output_subdir: str = "",
    model_label: str = "",
) -> None:
    """Convenience local entrypoint for `modal run modal_qwen3.py`.

    Args:
        action: Remote action to execute.
        model_preset: Supported Qwen3 preset.
        token_model_path: Size-policy checkpoint path for LTD.
        depth_model_path: Depth-policy checkpoint path for LTD.
        current_stage: Explicit iterative-training stage to execute.
        dataset_train: Training dataset.
        total_timesteps: Shared training horizon override for standalone actions.
        size_total_timesteps: Training horizon override for size-policy stages.
        depth_total_timesteps: Training horizon override for depth-policy stages.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        validation_fraction: HumanEval validation split fraction.
        split_seed: HumanEval train-validation split seed.
        output_subdir: Output root under ``/results``.
        model_label: Model label for the summary CSV.
    """
    if action == "download-models":
        print(download_models.remote(model_preset=model_preset))
        return
    if action == "iterative-status":
        print(
            iterative_status.remote(
                model_preset=model_preset,
                output_subdir=output_subdir,
            )
        )
        return
    if action == "train-size":
        print(
            train_size_policy.remote(
                model_preset=model_preset,
                dataset_train=dataset_train,
                total_timesteps=total_timesteps or size_total_timesteps,
                batch_size=batch_size,
                n_steps=n_steps,
                lr=lr,
                save_subdir=output_subdir,
            )
        )
        return
    if action == "train-depth":
        print(
            train_depth_policy.remote(
                model_preset=model_preset,
                dataset_train=dataset_train,
                total_timesteps=total_timesteps or depth_total_timesteps,
                batch_size=batch_size,
                n_steps=n_steps,
                lr=lr,
                rl_token_model_path=token_model_path,
                save_subdir=output_subdir,
            )
        )
        return
    if action == "iterative-train":
        print(
            iterative_train.remote(
                model_preset=model_preset,
                dataset_train=dataset_train,
                current_stage=current_stage,
                size_total_timesteps=size_total_timesteps,
                depth_total_timesteps=depth_total_timesteps,
                batch_size=batch_size,
                n_steps=n_steps,
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
                token_model_path=token_model_path,
                depth_model_path=depth_model_path,
                output_subdir=output_subdir,
                model_label=model_label,
            )
        )
        return
    raise ValueError(
        "Unsupported action. Use one of: download-models, iterative-status, "
        "train-size, train-depth, iterative-train, benchmark-suite."
    )
