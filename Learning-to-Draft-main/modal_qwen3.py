"""Modal entrypoints for LTD with Qwen3 and AngelSlim Eagle3."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

from qwen3_model_presets import model_cache_dir, resolve_model_paths


LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_MODELS_DIR = Path("/models")
REMOTE_RESULTS_DIR = Path("/results")
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
        "wandb",
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


def build_runtime_env() -> dict[str, str]:
    """Build environment variables for Modal subprocesses.

    Returns:
        The environment variables used for remote subprocess execution.
    """
    env = os.environ.copy()
    env["HF_HOME"] = str(REMOTE_MODELS_DIR / ".hf")
    env["PYTHONPATH"] = str(REMOTE_PROJECT_DIR)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = str(REMOTE_RESULTS_DIR / "wandb")
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


def run_command(command: list[str]) -> None:
    """Run a subprocess inside the remote LTD project directory.

    Args:
        command: The command to execute.
    """
    subprocess.run(
        command,
        check=True,
        cwd=str(REMOTE_PROJECT_DIR),
        env=build_runtime_env(),
    )


def ensure_results_dir(path: Path) -> None:
    """Create a results directory when needed.

    Args:
        path: The directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


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
    batch_size: int = 64,
    n_steps: int = 128,
    lr: float = 3e-4,
    save_subdir: str = "",
    use_wandb: bool = False,
    wandb_project: str = "speculative-decoding-rl",
) -> str:
    """Train the LTD size policy on Modal.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset under ``eagle/data``.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        save_subdir: Optional results subdirectory under ``/results``.
        use_wandb: Whether to enable Weights & Biases logging.
        wandb_project: Weights & Biases project name.

    Returns:
        The output directory containing the trained policy.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    output_dir = REMOTE_RESULTS_DIR / (save_subdir or f"{model_preset}/size")
    ensure_results_dir(output_dir)

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
        "--total_timesteps",
        str(total_timesteps),
        "--batch_size",
        str(batch_size),
        "--n_steps",
        str(n_steps),
        "--lr",
        str(lr),
    ]
    if use_wandb:
        command.extend(["--use_wandb", "--wandb_project", wandb_project])

    run_command(command)
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
    total_timesteps: int = 100000,
    batch_size: int = 64,
    n_steps: int = 128,
    lr: float = 3e-4,
    rl_token_model_path: str = "",
    rl_checkpoint_path: str = "",
    save_subdir: str = "",
    use_wandb: bool = False,
    wandb_project: str = "speculative-decoding-rl",
) -> str:
    """Train the LTD depth policy on Modal.

    Args:
        model_preset: The supported model preset.
        dataset_train: The training dataset under ``eagle/data``.
        total_timesteps: PPO training horizon.
        batch_size: PPO minibatch size.
        n_steps: PPO rollout length.
        lr: PPO learning rate.
        rl_token_model_path: Optional size-policy checkpoint path relative to
            ``/results`` or absolute in the remote container.
        rl_checkpoint_path: Optional depth-policy checkpoint path relative to
            ``/results`` or absolute in the remote container.
        save_subdir: Optional results subdirectory under ``/results``.
        use_wandb: Whether to enable Weights & Biases logging.
        wandb_project: Weights & Biases project name.

    Returns:
        The output directory containing the trained policy.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
    output_dir = REMOTE_RESULTS_DIR / (save_subdir or f"{model_preset}/depth")
    ensure_results_dir(output_dir)

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
        "--total_timesteps",
        str(total_timesteps),
        "--batch_size",
        str(batch_size),
        "--n_steps",
        str(n_steps),
        "--lr",
        str(lr),
    ]
    if rl_token_model_path:
        command.extend(
            ["--rl_token_model_path", resolve_results_path(rl_token_model_path)]
        )
    if rl_checkpoint_path:
        command.extend(
            ["--rl_checkpoint_path", resolve_results_path(rl_checkpoint_path)]
        )
    if use_wandb:
        command.extend(["--use_wandb", "--wandb_project", wandb_project])

    run_command(command)
    RESULTS_VOLUME.commit()
    return str(output_dir)


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
    token_model_path: str = "",
    depth_model_path: str = "",
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
        token_model_path: Optional size-policy checkpoint path.
        depth_model_path: Optional depth-policy checkpoint path.

    Returns:
        The subprocess command and output file path.
    """
    base_model_path, ea_model_path = resolve_remote_model_paths(model_preset)
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
