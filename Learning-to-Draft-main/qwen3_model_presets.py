"""Shared Qwen3 model presets for LTD training and evaluation."""

from __future__ import annotations

from pathlib import Path


MODEL_PRESETS: dict[str, tuple[str, str]] = {
    "qwen3_8b": ("Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3"),
    "qwen3_14b": ("Qwen/Qwen3-14B", "AngelSlim/Qwen3-14B_eagle3"),
}


def list_model_presets() -> list[str]:
    """Return the supported model preset names.

    Returns:
        The supported preset names in declaration order.
    """
    return list(MODEL_PRESETS.keys())


def resolve_model_paths(
    model_preset: str | None,
    base_model_path: str | None = None,
    ea_model_path: str | None = None,
) -> tuple[str, str]:
    """Resolve a preset or explicit model paths into a model pair.

    Args:
        model_preset: The preset name. Must be one of ``MODEL_PRESETS`` when
            provided.
        base_model_path: The explicit base model path or Hugging Face repo ID.
        ea_model_path: The explicit EAGLE draft model path or Hugging Face repo
            ID.

    Returns:
        A ``(base_model_path, ea_model_path)`` tuple.

    Raises:
        ValueError: If the inputs do not describe one supported model pair.
    """
    if model_preset:
        if model_preset not in MODEL_PRESETS:
            raise ValueError(
                f"Unsupported model preset '{model_preset}'. "
                f"Expected one of: {', '.join(list_model_presets())}."
            )
        if base_model_path or ea_model_path:
            raise ValueError(
                "Pass either --model_preset or explicit model paths, not both."
            )
        return MODEL_PRESETS[model_preset]

    if not base_model_path or not ea_model_path:
        raise ValueError(
            "Provide --model_preset or both --base_model_path and "
            "--ea_model_path."
        )

    return base_model_path, ea_model_path


def model_cache_dir(models_root: str | Path, repo_id: str) -> Path:
    """Build the on-disk cache path for a model repo ID.

    Args:
        models_root: Root directory that stores downloaded models.
        repo_id: The Hugging Face repo ID.

    Returns:
        The local path where the repo should be stored.
    """
    return Path(models_root) / repo_id
