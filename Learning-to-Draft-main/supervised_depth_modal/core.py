"""Core utilities for supervised depth-policy data collection and training."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eagle.model.ea_model import EaModel
from eagle.model.kv_cache import KVCache, initialize_past_key_values
from eagle.model.utils import (
    evaluate_posterior,
    initialize_tree,
    reset_past_key_values,
    reset_tree_mode,
    tree_decoding,
)


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share "
    "false information."
)
OBSERVATION_DIM = 129
ENHANCED_OBSERVATION_DIM = 135
MAX_OBS_SCORES = 100
MAX_SEQUENCE_LENGTH = 1748
MAX_GENERATED_TOKENS = 256
DEFAULT_TOTAL_TOKEN = 60
DEFAULT_TOP_K = 10
DEFAULT_MAX_DRAFT_DEPTH = 12
DEFAULT_POLICY_VARIANT = "base"
ENHANCED_POLICY_VARIANT = "enhanced"
DEFAULT_HIDDEN_DIMS = (1024,)
ENHANCED_HIDDEN_DIMS = (1536, 384)
EOS_LOGPROB_FALLBACK = -100.0
FLOP_ESTIMATE_MULTIPLIER = 2
TRAINING_FLOP_ESTIMATE_MULTIPLIER = 6


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
        path: JSONL path.

    Returns:
        Parsed records.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a JSONL file.

    Args:
        path: JSONL output path.
        records: Records to serialize.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSONL record.

    Args:
        path: JSONL file path.
        payload: Record to append.
    """
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def safe_cuda_synchronize() -> None:
    """Synchronize CUDA when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(module: nn.Module) -> int:
    """Count parameters in one torch module.

    Args:
        module: Torch module.

    Returns:
        Number of parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def estimate_forward_flops(parameter_count: int, forward_calls: int) -> int:
    """Estimate forward-pass FLOPs from parameter count and call count.

    Args:
        parameter_count: Number of model parameters.
        forward_calls: Number of forward calls.

    Returns:
        Approximate floating point operation count.
    """
    return int(FLOP_ESTIMATE_MULTIPLIER * parameter_count * forward_calls)


def scores_entropy(scores: torch.Tensor) -> float:
    """Compute entropy over a frontier score vector.

    Args:
        scores: Log-score tensor.

    Returns:
        Entropy of the normalized score distribution.
    """
    if scores.numel() == 0:
        return 0.0
    probs = torch.softmax(scores.float(), dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy.item())


def normalize_policy_variant(policy_variant: str) -> str:
    """Validate and normalize a supervised policy variant.

    Args:
        policy_variant: Policy variant name.

    Returns:
        Normalized policy variant.
    """
    normalized = policy_variant.strip().lower() or DEFAULT_POLICY_VARIANT
    if normalized not in {DEFAULT_POLICY_VARIANT, ENHANCED_POLICY_VARIANT}:
        raise ValueError(
            f"Unsupported policy_variant '{policy_variant}'. Use "
            f"'{DEFAULT_POLICY_VARIANT}' or '{ENHANCED_POLICY_VARIANT}'."
        )
    return normalized


def policy_observation_dim(policy_variant: str) -> int:
    """Return the observation dimension for a policy variant.

    Args:
        policy_variant: Policy variant name.

    Returns:
        Observation dimension.
    """
    if normalize_policy_variant(policy_variant) == ENHANCED_POLICY_VARIANT:
        return ENHANCED_OBSERVATION_DIM
    return OBSERVATION_DIM


def policy_hidden_dims(policy_variant: str, hidden_dim: int = 1024) -> tuple[int, ...]:
    """Return hidden dimensions for a policy variant.

    Args:
        policy_variant: Policy variant name.
        hidden_dim: Base-policy hidden width.

    Returns:
        Hidden dimensions.
    """
    if normalize_policy_variant(policy_variant) == ENHANCED_POLICY_VARIANT:
        return ENHANCED_HIDDEN_DIMS
    return (hidden_dim,)


def score_summary_features(scores: torch.Tensor | None) -> tuple[float, float, float, float]:
    """Summarize a score vector.

    Args:
        scores: Log-score tensor.

    Returns:
        Max, mean, standard deviation, and min score.
    """
    if scores is None or scores.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    scores_float = scores.flatten().detach().float()
    return (
        float(scores_float.max().item()),
        float(scores_float.mean().item()),
        float(scores_float.std(unbiased=False).item()),
        float(scores_float.min().item()),
    )


def resolve_end_token_ids(tokenizer: Any) -> list[int]:
    """Resolve target-token ids that should be treated as EOS-like.

    Args:
        tokenizer: Model tokenizer.

    Returns:
        Unique target-token ids for end-of-sequence style tokens.
    """
    token_ids: list[int] = []
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer else None,
        tokenizer.convert_tokens_to_ids("<|endoftext|>") if tokenizer else None,
    ):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def map_target_token_ids_to_draft_ids(ea_layer: Any, target_token_ids: list[int]) -> list[int]:
    """Map target-token ids to draft-vocabulary ids when needed.

    Args:
        ea_layer: Eagle draft layer.
        target_token_ids: Target-model token ids.

    Returns:
        Draft-vocabulary ids corresponding to the target ids.
    """
    if not target_token_ids:
        return []
    draft_vocab_size = int(ea_layer.config.draft_vocab_size)
    if ea_layer.config.vocab_size == ea_layer.config.draft_vocab_size:
        return [
            token_id
            for token_id in target_token_ids
            if 0 <= token_id < draft_vocab_size
        ]
    d2t = ea_layer.d2t.detach().cpu().long()
    draft_ids = torch.arange(d2t.numel(), dtype=torch.long)
    mapped_target_ids = draft_ids + d2t
    matched_ids: list[int] = []
    for target_token_id in target_token_ids:
        matches = draft_ids[mapped_target_ids == int(target_token_id)].tolist()
        for match in matches:
            if 0 <= int(match) < draft_vocab_size and int(match) not in matched_ids:
                matched_ids.append(int(match))
    return matched_ids


def end_token_logprob(log_probs: torch.Tensor, draft_token_ids: list[int]) -> float:
    """Read the largest EOS-like log probability from drafter log probabilities.

    Args:
        log_probs: Draft-layer log-probability tensor.
        draft_token_ids: Draft-vocabulary ids for EOS-like tokens.

    Returns:
        Maximum EOS-like log probability, or a fallback when unavailable.
    """
    if not draft_token_ids or log_probs.numel() == 0:
        return EOS_LOGPROB_FALLBACK
    vocab_size = log_probs.shape[-1]
    valid_ids = [token_id for token_id in draft_token_ids if 0 <= token_id < vocab_size]
    if not valid_ids:
        return EOS_LOGPROB_FALLBACK
    selected = log_probs.detach().float().reshape(-1, vocab_size)[:, valid_ids]
    return float(selected.max().item())


def load_prompt_input_ids(
    tokenizer: Any,
    question_file: str,
) -> list[torch.Tensor]:
    """Load prompt input ids from a question file.

    Args:
        tokenizer: Model tokenizer.
        question_file: JSONL prompt file.

    Returns:
        Tokenized prompts that fit within the project sequence-length limit.
    """
    input_ids_list: list[torch.Tensor] = []
    with open(question_file, "r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["turns"][0]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if input_ids.shape[1] <= MAX_SEQUENCE_LENGTH:
                input_ids_list.append(input_ids)
    if not input_ids_list:
        raise ValueError(f"No prompts loaded from {question_file}.")
    return input_ids_list


class SupervisedDepthModel(nn.Module):
    """MLP depth regressor that predicts one-step throughput delta."""

    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 1024,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        policy_variant: str = DEFAULT_POLICY_VARIANT,
    ) -> None:
        """Initialize the regressor.

        Args:
            input_dim: Observation dimension.
            hidden_dim: Hidden width.
            hidden_dims: Optional hidden layer widths.
            policy_variant: Policy variant name.
        """
        super().__init__()
        self.input_dim = input_dim
        self.policy_variant = normalize_policy_variant(policy_variant)
        self.hidden_dims = tuple(hidden_dims or (hidden_dim,))
        layers: list[nn.Module] = []
        current_dim = input_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.ReLU())
            current_dim = width
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            obs: Observation tensor.

        Returns:
            Predicted throughput delta.
        """
        return self.network(obs).squeeze(-1)


def save_supervised_depth_checkpoint(
    path: Path,
    model: SupervisedDepthModel,
    metadata: dict[str, Any],
) -> None:
    """Save a supervised depth checkpoint.

    Args:
        path: Checkpoint path.
        model: Depth model.
        metadata: Serializable metadata.
    """
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_supervised_depth_checkpoint(
    model_path: str,
    device: str = "cuda",
) -> SupervisedDepthModel:
    """Load a supervised depth checkpoint.

    Args:
        model_path: Checkpoint path.
        device: Torch device.

    Returns:
        Loaded depth regressor.
    """
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    policy_variant = normalize_policy_variant(
        str(metadata.get("policy_variant", DEFAULT_POLICY_VARIANT))
    )
    hidden_dims = metadata.get("hidden_dims")
    if hidden_dims is None:
        hidden_dims = [int(metadata.get("hidden_dim", 1024))]
    model = SupervisedDepthModel(
        input_dim=int(metadata.get("input_dim", policy_observation_dim(policy_variant))),
        hidden_dims=tuple(int(width) for width in hidden_dims),
        policy_variant=policy_variant,
    )
    model.metadata = metadata
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@dataclass
class KVCopyLayout:
    """Mapping from cache tensors to grouped KV blocks."""

    block_index: int
    slice_index: int
    current_index: int


class DepthStateCollector:
    """Collect timestep-limited supervised depth-policy training data."""

    def __init__(
        self,
        model: EaModel,
        input_ids_list: list[torch.Tensor],
        total_token_limit: int = DEFAULT_TOTAL_TOKEN,
        max_draft_depth: int = DEFAULT_MAX_DRAFT_DEPTH,
        policy_variant: str = DEFAULT_POLICY_VARIANT,
    ) -> None:
        """Initialize the collector.

        Args:
            model: Loaded Eagle3 model.
            input_ids_list: Training prompts.
            total_token_limit: Fixed verification size.
            max_draft_depth: Maximum draft depth explored during collection.
            policy_variant: Supervised policy variant.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.policy_variant = normalize_policy_variant(policy_variant)
        self.observation_dim = policy_observation_dim(self.policy_variant)
        self.input_ids_list = [tensor.to(self.device) for tensor in input_ids_list]
        self.total_token_limit = total_token_limit
        self.max_draft_depth = max_draft_depth
        self.ea_layer_top_k = self.model.ea_layer.top_k
        self.eos_draft_token_ids = map_target_token_ids_to_draft_ids(
            self.model.ea_layer,
            resolve_end_token_ids(self.model.get_tokenizer()),
        )
        self.current_input_ids: torch.Tensor | None = None
        self.input_len = 0
        self.new_token_count = 0
        self.finished_overall_generation = True
        self.total_draft_model_calls = 0
        self.total_target_model_calls = 0
        self.episodes_started = 0
        self.examples_collected = 0
        self.prompt_rng = random.Random(42)
        self.kv_layout: list[list[KVCopyLayout]] = []
        self._build_kv_layout_cache = False
        self.previous_entropy_for_obs = 0.0
        self.current_entropy_for_obs = 0.0
        self.eos_logprob_for_obs = EOS_LOGPROB_FALLBACK

    def _build_kv_layout(self) -> None:
        """Build a stable mapping for cloning KV cache state."""
        if self._build_kv_layout_cache:
            return
        layout: list[list[KVCopyLayout]] = []
        for layer_index, pair in enumerate(self.past_key_values):
            layer_layout: list[KVCopyLayout] = []
            for key_value_index, cache in enumerate(pair):
                matched_block = -1
                matched_slice = -1
                for block_index, block in enumerate(self.past_key_values_data):
                    if cache.data.untyped_storage().data_ptr() != block.untyped_storage().data_ptr():
                        continue
                    matched_block = block_index
                    slice_numel = cache.data.numel()
                    matched_slice = cache.data.storage_offset() // slice_numel
                    break
                if matched_block < 0 or matched_slice < 0:
                    raise ValueError("Failed to map KV cache tensors for cloning.")
                layer_layout.append(
                    KVCopyLayout(
                        block_index=matched_block,
                        slice_index=matched_slice,
                        current_index=layer_index * 2 + key_value_index,
                    )
                )
            layout.append(layer_layout)
        self.kv_layout = layout
        self._build_kv_layout_cache = True

    def _clone_kv_state(
        self,
    ) -> tuple[list[list[KVCache]], list[torch.Tensor], torch.Tensor]:
        """Clone the current KV cache state.

        Returns:
            Cloned ``past_key_values``, ``past_key_values_data`` and ``current_length_data``.
        """
        self._build_kv_layout()
        cloned_blocks = [block.clone() for block in self.past_key_values_data]
        current_length_clone = self.current_length_data.clone()
        cloned_past_key_values: list[list[KVCache]] = []
        for layer_layout in self.kv_layout:
            cloned_pair: list[KVCache] = []
            for mapping in layer_layout:
                cloned_pair.append(
                    KVCache(
                        cloned_blocks[mapping.block_index][mapping.slice_index],
                        current_length_clone[mapping.current_index],
                    )
                )
            cloned_past_key_values.append(cloned_pair)
        return cloned_past_key_values, cloned_blocks, current_length_clone

    def _prepare_for_next_topk_cycle(
        self,
        accepted_hidden_state_base: torch.Tensor,
        next_token_sampled: torch.Tensor,
    ) -> None:
        """Prepare drafting buffers for the next speculative cycle.

        Args:
            accepted_hidden_state_base: Verified hidden states from the target model.
            next_token_sampled: Next root token sampled by the target model.
        """
        self.time = 0.0
        safe_cuda_synchronize()
        begin_time = time.time()
        self.hidden_states_for_topk_ea_layer = accepted_hidden_state_base
        self.input_ids_for_topk_first_pass = torch.cat(
            (self.current_input_ids, next_token_sampled.to(self.current_input_ids.device)),
            dim=1,
        )
        self.current_sample_token_for_topk = self.input_ids_for_topk_first_pass[:, -1]
        self.scores_list: list[torch.Tensor] = []
        self.parents_list: list[torch.Tensor] = []
        self.ss_token_list: list[torch.Tensor] = []
        input_ids_first_iter = self.input_ids_for_topk_first_pass[:, 1:]
        self.len_posi_for_topk_loop = input_ids_first_iter.shape[1]

        self.model.ea_layer.reset()
        if hasattr(self.model.ea_layer, "stable_kv") and self.model.ea_layer.stable_kv is not None:
            kv_len = self.model.ea_layer.stable_kv[0][0].shape[2]
            out_hidden, past_key_values_ealayer = self.model.ea_layer(
                self.hidden_states_for_topk_ea_layer,
                input_ids=input_ids_first_iter[:, kv_len:],
                past_key_values=self.model.ea_layer.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values_ealayer = self.model.ea_layer(
                self.hidden_states_for_topk_ea_layer,
                input_ids=input_ids_first_iter,
                use_cache=True,
            )
        self.total_draft_model_calls += 1
        self.model.ea_layer.stable_kv = past_key_values_ealayer
        self.current_past_key_values_ealayer = past_key_values_ealayer

        last_hidden = out_hidden[:, -1]
        last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(last_hidden))
        last_p = self.model.ea_layer.logsoftmax(last_headout)
        top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        current_scores = topk_p[0]
        self.scores_list.append(current_scores[None])
        self.current_scores_for_topk_loop_obs = current_scores
        self.cu_scores_for_obs = None
        self.previous_entropy_for_obs = 0.0
        self.current_entropy_for_obs = scores_entropy(current_scores)
        self.eos_logprob_for_obs = end_token_logprob(last_p, self.eos_draft_token_ids)
        self.parents_list.append(torch.zeros(1, dtype=torch.long, device=current_scores.device))
        if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
            self.ss_token_list.append(topk_index)
            input_ids_for_next_depth_iter = topk_index
        else:
            mapped_tokens = topk_index + self.model.ea_layer.d2t[topk_index]
            self.ss_token_list.append(mapped_tokens)
            input_ids_for_next_depth_iter = mapped_tokens
        self.current_input_ids_for_topk_depth_iter = input_ids_for_next_depth_iter
        self.current_input_hidden_for_topk_depth_iter = last_hidden[None].repeat(
            1, self.ea_layer_top_k, 1
        )
        self.current_tree_mask_for_topk_loop = self.model.ea_layer.tree_mask_init.clone().to(self.device)
        self.current_topk_cs_index_for_loop = torch.arange(
            self.ea_layer_top_k,
            device=self.model.ea_layer.embed_tokens.weight.device,
        )
        self.cnet_step = 0
        safe_cuda_synchronize()
        self.time += time.time() - begin_time

    def _start_new_generation(self) -> None:
        """Reset generation state and prime the next speculative cycle."""
        self.episodes_started += 1
        self.current_episode_rewards: list[float] = []
        self.current_input_ids = self.prompt_rng.choice(self.input_ids_list).clone().to(self.device)
        self.input_len = self.current_input_ids.shape[1]
        self.new_token_count = 0
        self.model.ea_layer.reset_kv()

        if hasattr(self.model, "past_key_values") and self.model.past_key_values is not None:
            reset_past_key_values(self.model.past_key_values)
            self.past_key_values = self.model.past_key_values
            self.past_key_values_data = self.model.past_key_values_data
            self.current_length_data = self.model.current_length_data
            self.current_length_data.zero_()
        else:
            (
                self.past_key_values,
                self.past_key_values_data,
                self.current_length_data,
            ) = initialize_past_key_values(self.model.base_model, max_length=2048)
            self.model.past_key_values = self.past_key_values
            self.model.past_key_values_data = self.past_key_values_data
            self.model.current_length_data = self.current_length_data
            self._build_kv_layout_cache = False

        reset_tree_mode(self.model)
        with torch.no_grad():
            draft_tokens, retrieve_indices_init, tree_mask_init, tree_position_ids_init, _, _, _ = initialize_tree(
                self.current_input_ids,
                self.model,
                self.past_key_values,
                None,
            )
            self.model.base_model.model.tree_mask = tree_mask_init.to(self.device)
            logits_verify, hidden_state_new_verify, _ = tree_decoding(
                self.model,
                draft_tokens.to(self.device),
                self.past_key_values,
                tree_position_ids_init.to(self.device),
                self.current_input_ids,
                retrieve_indices_init.to(self.device),
            )
            self.total_target_model_calls += 1
            padding = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
            padded_draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = padded_draft_tokens[0, retrieve_indices_init.to(self.device)]
            best_candidate_idx, accept_length, sample_p = evaluate_posterior(
                logits_verify,
                candidates,
                None,
            )
            prev_input_len = self.current_input_ids.shape[1]
            select_indices = retrieve_indices_init[best_candidate_idx, : accept_length + 1] + prev_input_len
            accepted_tokens = candidates[best_candidate_idx, : accept_length + 1]
            self.current_input_ids = torch.cat(
                (self.current_input_ids, accepted_tokens.unsqueeze(0).to(self.current_input_ids.device)),
                dim=-1,
            )
            for cache_block in self.past_key_values_data:
                tgt = cache_block[..., select_indices.to(cache_block.device), :]
                dst = cache_block[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
            self.current_length_data.fill_(self.current_input_ids.shape[1])
            retrieve_hidden_state_new = hidden_state_new_verify[:, retrieve_indices_init]
            accepted_hidden_state_base = retrieve_hidden_state_new[:, best_candidate_idx, : accept_length + 1]
            next_token_sampled = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)
        self.new_token_count += accept_length + 1
        self.finished_overall_generation = False
        self._prepare_for_next_topk_cycle(accepted_hidden_state_base, next_token_sampled)

    def _expand_one_level(self) -> None:
        """Expand the current draft tree by one depth level."""
        safe_cuda_synchronize()
        start_time = time.time()
        self.model.ea_layer.tree_mask = self.current_tree_mask_for_topk_loop
        current_position_ids = self.len_posi_for_topk_loop + self.model.ea_layer.position_ids.to(self.device)
        out_hidden, past_key_values_ealayer_new = self.model.ea_layer(
            self.current_input_hidden_for_topk_depth_iter,
            input_ids=self.current_input_ids_for_topk_depth_iter,
            past_key_values=self.current_past_key_values_ealayer,
            position_ids=current_position_ids,
            use_cache=True,
        )
        self.total_draft_model_calls += 1
        self.len_posi_for_topk_loop += 1
        self.current_past_key_values_ealayer = past_key_values_ealayer_new

        bias1 = self.ea_layer_top_k if self.cnet_step > 0 else 0
        bias2 = max(0, self.cnet_step - 1)
        bias = 1 + self.ea_layer_top_k * self.ea_layer_top_k * bias2 + bias1
        parents = self.current_topk_cs_index_for_loop + bias
        self.parents_list.append(parents)

        last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(out_hidden[0]))
        last_p = self.model.ea_layer.logsoftmax(last_headout)
        top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        cu_scores = topk_p + self.current_scores_for_topk_loop_obs[:, None]
        topk_cs = torch.topk(cu_scores.view(-1), self.ea_layer_top_k, dim=-1)
        topk_cs_index_new, topk_cs_p_new = topk_cs.indices, topk_cs.values
        self.cu_scores_for_obs = cu_scores.flatten()
        self.previous_entropy_for_obs = self.current_entropy_for_obs
        self.current_entropy_for_obs = scores_entropy(self.cu_scores_for_obs)
        self.eos_logprob_for_obs = end_token_logprob(last_p, self.eos_draft_token_ids)
        self.current_scores_for_topk_loop_obs = topk_cs_p_new
        self.current_topk_cs_index_for_loop = topk_cs_index_new

        out_ids = (topk_cs_index_new // self.ea_layer_top_k).to(self.current_tree_mask_for_topk_loop.device)
        self.current_input_hidden_for_topk_depth_iter = out_hidden[:, out_ids]
        next_input_ids_val = topk_index.view(-1)[topk_cs_index_new][None]
        if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
            self.ss_token_list.append(topk_index)
            self.current_input_ids_for_topk_depth_iter = next_input_ids_val
        else:
            mapped_tokens = next_input_ids_val + self.model.ea_layer.d2t[next_input_ids_val.squeeze()].unsqueeze(0)
            self.ss_token_list.append(topk_index + self.model.ea_layer.d2t[topk_index.squeeze()])
            self.current_input_ids_for_topk_depth_iter = mapped_tokens
        self.scores_list.append(cu_scores)
        if (
            self.current_tree_mask_for_topk_loop.shape[2] > 0
            and out_ids.numel() > 0
            and out_ids.max() < self.current_tree_mask_for_topk_loop.shape[2]
        ):
            self.current_tree_mask_for_topk_loop = torch.cat(
                (
                    self.current_tree_mask_for_topk_loop[:, :, out_ids],
                    self.model.ea_layer.tree_mask_init.clone().to(self.device),
                ),
                dim=3,
            )
        self.cnet_step += 1
        safe_cuda_synchronize()
        self.time += time.time() - start_time

    def _build_observation(self) -> np.ndarray:
        """Build the supervised depth observation.

        Returns:
            Numpy observation vector.
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        position_ids = self.current_input_ids.shape[1] / 1000.0
        draft_position_ids = self.cnet_step / 10.0
        if self.cu_scores_for_obs is not None:
            scores = self.cu_scores_for_obs.flatten().detach().float().cpu().numpy()
            obs[: min(MAX_OBS_SCORES, len(scores))] = scores[:MAX_OBS_SCORES]
        obs[100:114] = position_ids
        obs[114:128] = draft_position_ids
        if self.cu_scores_for_obs is not None:
            obs[128] = scores_entropy(self.cu_scores_for_obs.flatten())
        if self.policy_variant == ENHANCED_POLICY_VARIANT:
            summary_scores = self.cu_scores_for_obs
            if summary_scores is None:
                summary_scores = self.current_scores_for_topk_loop_obs
            score_max, score_mean, score_std, score_min = score_summary_features(summary_scores)
            obs[129] = score_max
            obs[130] = score_mean
            obs[131] = score_std
            obs[132] = score_min
            obs[133] = self.current_entropy_for_obs - self.previous_entropy_for_obs
            obs[134] = self.eos_logprob_for_obs
        return obs

    def _finalize_tree(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finalize the current draft tree for target verification.

        Returns:
            Draft tokens, tree mask, tree position ids, and retrieve indices.
        """
        scores_flat = torch.cat(self.scores_list, dim=0).view(-1)
        tokens_flat = torch.cat(self.ss_token_list, dim=0).view(-1)
        total_tokens = min(tokens_flat.shape[0], self.total_token_limit)
        top_scores_indices = torch.topk(scores_flat, total_tokens, dim=-1).indices
        top_scores_indices_sorted = torch.sort(top_scores_indices).values

        draft_tokens_flat = tokens_flat[top_scores_indices_sorted]
        finalized_draft_tokens = torch.cat(
            (self.current_sample_token_for_topk.to(self.device), draft_tokens_flat),
            dim=0,
        ).unsqueeze(0)
        draft_parents_flat = torch.cat(self.parents_list, dim=0)[
            top_scores_indices_sorted // self.ea_layer_top_k
        ].long()
        mask_index = torch.searchsorted(top_scores_indices_sorted, draft_parents_flat - 1, right=False)
        mask_index[draft_parents_flat == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask_bool = torch.eye(total_tokens + 1, device=self.device).bool()
        tree_mask_bool[:, 0] = True
        for token_index in range(total_tokens):
            parent_index = mask_index_list[token_index] if token_index < len(mask_index_list) else 0
            tree_mask_bool[token_index + 1].add_(tree_mask_bool[parent_index])
        tree_position_ids = torch.sum(tree_mask_bool.int(), dim=1) - 1

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        leaf_num = total_tokens - (len(noleaf_index) - 1)
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for node_index in range(total_tokens + 1):
            if node_index in noleaf_index:
                continue
            current_id = node_index
            depth = position_ids_list[node_index]
            for depth_index in reversed(range(depth + 1)):
                retrieve_indices[rid][depth_index] = current_id
                current_id = mask_index_list[current_id - 1] if current_id > 0 else -1
            rid += 1
        return (
            finalized_draft_tokens,
            tree_mask_bool.float()[None, None],
            tree_position_ids,
            torch.tensor(retrieve_indices, dtype=torch.long),
        )

    def _verify_current_tree(
        self,
        apply_update: bool,
    ) -> dict[str, Any]:
        """Verify the current draft tree.

        Args:
            apply_update: Whether to mutate the live decoding state.

        Returns:
            Verification metrics including throughput and acceptance length.
        """
        if apply_update:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_input_ids = self.current_input_ids
        else:
            past_key_values, past_key_values_data, current_length_data = self._clone_kv_state()
            current_input_ids = self.current_input_ids.clone()

        finalized_draft_tokens, tree_mask, tree_position_ids, retrieve_indices = self._finalize_tree()
        self.model.base_model.model.tree_mask = tree_mask.to(self.device)

        safe_cuda_synchronize()
        start_time = time.time()
        logits_verify, hidden_state_new_verify, _ = tree_decoding(
            self.model,
            finalized_draft_tokens.to(self.device),
            past_key_values,
            tree_position_ids.to(self.device),
            current_input_ids,
            retrieve_indices.to(self.device),
        )
        self.total_target_model_calls += 1
        safe_cuda_synchronize()
        verify_time = time.time() - start_time

        padding = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
        padded_draft_tokens = torch.cat((finalized_draft_tokens, padding), dim=1)
        candidates = padded_draft_tokens[0, retrieve_indices.to(self.device)]
        best_candidate_idx, accept_length, sample_p = evaluate_posterior(
            logits_verify,
            candidates,
            None,
        )
        accepted_tokens_sequence = candidates[best_candidate_idx, : accept_length + 1]
        prev_input_len = current_input_ids.shape[1]
        updated_input_ids = torch.cat(
            (current_input_ids, accepted_tokens_sequence.unsqueeze(0).to(current_input_ids.device)),
            dim=-1,
        )
        select_indices = retrieve_indices[best_candidate_idx, : accept_length + 1] + prev_input_len
        for cache_block in past_key_values_data:
            tgt = cache_block[..., select_indices.to(cache_block.device), :]
            dst = cache_block[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
        current_length_data.fill_(updated_input_ids.shape[1])

        total_time = self.time + verify_time
        throughput = (accept_length + 1) / (total_time * 100.0 + 1e-6)

        if apply_update:
            self.current_input_ids = updated_input_ids
            self.new_token_count += accepted_tokens_sequence.shape[0]
            retrieve_hidden_state_new = hidden_state_new_verify[:, retrieve_indices.to(hidden_state_new_verify.device)]
            self.accepted_hidden_state_base_for_next_topk = retrieve_hidden_state_new[
                :,
                best_candidate_idx,
                : accept_length + 1,
            ]
            self.next_token_sampled_for_next_topk = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)
            generated_tokens = self.current_input_ids[0, self.input_len :].tolist()
            stop_token_id = (
                self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if self.model.tokenizer
                else -1
            )
            if (stop_token_id != -1 and stop_token_id in generated_tokens) or (
                self.model.tokenizer and self.model.tokenizer.eos_token_id in generated_tokens
            ):
                self.finished_overall_generation = True
            if self.current_input_ids.shape[1] >= MAX_SEQUENCE_LENGTH:
                self.finished_overall_generation = True
            if self.new_token_count >= MAX_GENERATED_TOKENS:
                self.finished_overall_generation = True

        return {
            "throughput": float(throughput),
            "accept_length": float(accept_length + 1),
            "verify_time": float(verify_time),
            "current_seq_len": int(updated_input_ids.shape[1]),
        }

    def collect(self, total_timesteps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Collect a timestep-limited supervised dataset.

        Args:
            total_timesteps: Maximum number of depth-decision states to collect.

        Returns:
            Dataset records and collection metadata.
        """
        records: list[dict[str, Any]] = []
        with tqdm(
            total=total_timesteps,
            desc="Collecting supervised depth data",
            unit="step",
            dynamic_ncols=True,
        ) as progress_bar:
            while len(records) < total_timesteps:
                if self.finished_overall_generation or self.current_input_ids is None:
                    self._start_new_generation()

                observation = self._build_observation()
                decision_depth = self.cnet_step
                current_metrics = self._verify_current_tree(apply_update=False)
                if self.cnet_step < self.max_draft_depth:
                    self._expand_one_level()
                    next_metrics = self._verify_current_tree(apply_update=False)
                else:
                    next_metrics = current_metrics

                delta = float(next_metrics["throughput"] - current_metrics["throughput"])
                record = {
                    "observation": observation.tolist(),
                    "target_delta": delta,
                    "continue_label": int(delta > 0.0),
                    "draft_depth": int(decision_depth),
                    "context_length": int(self.current_input_ids.shape[1]),
                    "entropy": float(observation[128]),
                    "stop_throughput": float(current_metrics["throughput"]),
                    "next_throughput": float(next_metrics["throughput"]),
                    "stop_accept_length": float(current_metrics["accept_length"]),
                    "next_accept_length": float(next_metrics["accept_length"]),
                }
                if self.policy_variant == ENHANCED_POLICY_VARIANT:
                    record.update(
                        {
                            "score_max": float(observation[129]),
                            "score_mean": float(observation[130]),
                            "score_std": float(observation[131]),
                            "score_min": float(observation[132]),
                            "entropy_delta": float(observation[133]),
                            "eos_logprob_from_last_drafter_logits": float(observation[134]),
                        }
                    )
                records.append(record)
                self.examples_collected += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    depth=int(self.cnet_step),
                    episodes=int(self.episodes_started),
                    draft_calls=int(self.total_draft_model_calls),
                    target_calls=int(self.total_target_model_calls),
                )

                if self.cnet_step >= self.max_draft_depth:
                    self._verify_current_tree(apply_update=True)
                    if self.finished_overall_generation:
                        self.current_input_ids = None
                    else:
                        self._prepare_for_next_topk_cycle(
                            self.accepted_hidden_state_base_for_next_topk,
                            self.next_token_sampled_for_next_topk,
                        )

        metadata = {
            "total_timesteps": total_timesteps,
            "collected_examples": len(records),
            "episodes_started": self.episodes_started,
            "draft_model_calls": self.total_draft_model_calls,
            "target_model_calls": self.total_target_model_calls,
            "fixed_total_token": self.total_token_limit,
            "max_draft_depth": self.max_draft_depth,
            "observation_dim": self.observation_dim,
            "policy_variant": self.policy_variant,
        }
        return records, metadata


def collect_supervised_depth_dataset(
    base_model_path: str,
    ea_model_path: str,
    question_file: str,
    output_dir: str,
    total_timesteps: int,
    total_token: int = DEFAULT_TOTAL_TOKEN,
    max_draft_depth: int = DEFAULT_MAX_DRAFT_DEPTH,
    policy_variant: str = DEFAULT_POLICY_VARIANT,
) -> dict[str, Any]:
    """Collect a supervised depth dataset and persist it to disk.

    Args:
        base_model_path: Base model path or repo id.
        ea_model_path: Eagle3 model path or repo id.
        question_file: Training prompt JSONL.
        output_dir: Dataset output directory.
        total_timesteps: Maximum number of collected decision states.
        total_token: Fixed verification size.
        max_draft_depth: Maximum explored draft depth.
        policy_variant: Supervised policy variant.

    Returns:
        Dataset manifest payload.
    """
    policy_variant = normalize_policy_variant(policy_variant)
    collection_start_time = time.time()
    output_root = Path(output_dir)
    ensure_dir(output_root)
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=total_token,
        depth=5,
        top_k=DEFAULT_TOP_K,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_eagle3=True,
        use_dyn_len=False,
        use_dyn_token=False,
    ).to("cuda")
    model.eval()
    tokenizer = model.get_tokenizer()
    input_ids_list = load_prompt_input_ids(tokenizer, question_file)
    collector = DepthStateCollector(
        model=model,
        input_ids_list=input_ids_list,
        total_token_limit=total_token,
        max_draft_depth=max_draft_depth,
        policy_variant=policy_variant,
    )
    records, metadata = collector.collect(total_timesteps=total_timesteps)
    target_model_parameters = count_parameters(model.base_model)
    draft_model_parameters = count_parameters(model.ea_layer)
    target_flops = estimate_forward_flops(
        target_model_parameters,
        int(metadata["target_model_calls"]),
    )
    draft_flops = estimate_forward_flops(
        draft_model_parameters,
        int(metadata["draft_model_calls"]),
    )
    dataset_path = output_root / "dataset.jsonl"
    write_jsonl(dataset_path, records)
    manifest = {
        **metadata,
        "dataset_path": str(dataset_path),
        "question_file": question_file,
        "base_model_path": base_model_path,
        "ea_model_path": ea_model_path,
        "total_collection_time_seconds": time.time() - collection_start_time,
        "target_model_parameters": target_model_parameters,
        "draft_model_parameters": draft_model_parameters,
        "collection_target_flops": target_flops,
        "collection_draft_flops": draft_flops,
        "collection_total_flops": target_flops + draft_flops,
        "flop_estimate_method": "2 * parameter_count * forward_calls",
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def _dataset_to_tensors(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert JSON records into tensors.

    Args:
        records: Dataset records.

    Returns:
        Observation and target tensors.
    """
    observations = torch.tensor(
        [record["observation"] for record in records],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [record["target_delta"] for record in records],
        dtype=torch.float32,
    )
    return observations, targets


def _build_data_loaders(
    dataset_path: str,
    batch_size: int,
    validation_fraction: float,
    split_seed: int,
    expected_input_dim: int | None = None,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Build train and validation loaders.

    Args:
        dataset_path: Dataset JSONL path.
        batch_size: Minibatch size.
        validation_fraction: Validation fraction.
        split_seed: Shuffle seed.
        expected_input_dim: Optional expected observation width.

    Returns:
        Train loader, validation loader, train size and validation size.
    """
    records = read_jsonl(Path(dataset_path))
    observations, targets = _dataset_to_tensors(records)
    if expected_input_dim is not None and observations.shape[1] != expected_input_dim:
        raise ValueError(
            f"Dataset observation dimension {observations.shape[1]} does not match "
            f"expected dimension {expected_input_dim}."
        )
    indices = list(range(len(records)))
    random.Random(split_seed).shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if not train_indices:
        train_indices = validation_indices
    train_dataset = TensorDataset(observations[train_indices], targets[train_indices])
    validation_dataset = TensorDataset(
        observations[validation_indices],
        targets[validation_indices],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, validation_loader, len(train_indices), len(validation_indices)


def _validation_metrics(
    model: SupervisedDepthModel,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    """Compute validation metrics.

    Args:
        model: Depth regressor.
        data_loader: Validation loader.
        device: Torch device.
        loss_fn: Regression loss.

    Returns:
        Validation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_sign_correct = 0
    with torch.no_grad():
        for observations, targets in data_loader:
            observations = observations.to(device)
            targets = targets.to(device)
            predictions = model(observations)
            loss = loss_fn(predictions, targets)
            total_loss += float(loss.item()) * observations.shape[0]
            total_examples += observations.shape[0]
            total_sign_correct += int(((predictions > 0) == (targets > 0)).sum().item())
    if total_examples == 0:
        return {"loss": 0.0, "sign_accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "sign_accuracy": total_sign_correct / total_examples,
    }


def train_supervised_depth_model(
    dataset_path: str,
    output_dir: str,
    total_timesteps: int,
    epochs: int,
    checkpoint_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
    hidden_dim: int = 1024,
    policy_variant: str = DEFAULT_POLICY_VARIANT,
) -> dict[str, Any]:
    """Train the supervised depth regressor.

    Args:
        dataset_path: Dataset JSONL path.
        output_dir: Training output directory.
        total_timesteps: Environmental-interaction budget tied to the dataset.
        epochs: Number of full passes over the training split.
        checkpoint_epochs: Checkpoint interval in epochs.
        batch_size: Batch size.
        lr: Learning rate.
        validation_fraction: Example-level validation split fraction.
        split_seed: Validation split seed.
        hidden_dim: Hidden layer width.
        policy_variant: Supervised policy variant.

    Returns:
        Training summary payload.
    """
    policy_variant = normalize_policy_variant(policy_variant)
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if checkpoint_epochs < 1:
        raise ValueError("checkpoint_epochs must be at least 1.")
    output_root = Path(output_dir)
    ensure_dir(output_root)
    train_loader, validation_loader, train_size, validation_size = _build_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        expected_input_dim=policy_observation_dim(policy_variant),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = policy_observation_dim(policy_variant)
    hidden_dims = policy_hidden_dims(policy_variant, hidden_dim=hidden_dim)
    model = SupervisedDepthModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        policy_variant=policy_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    best_validation_loss = math.inf
    best_model_path = output_root / "supervised_depth_model_best.pt"
    metrics_log_path = output_root / "training_metrics.jsonl"
    model_parameter_count = count_parameters(model)
    train_examples_seen = 0
    train_batches_per_epoch = len(train_loader)
    total_optimizer_steps = epochs * train_batches_per_epoch
    optimizer_step = 0

    for epoch in range(1, epochs + 1):
        for batch_index, (observations, targets) in enumerate(train_loader, start=1):
            optimizer_step += 1
            train_examples_seen += observations.shape[0]
            observations = observations.to(device)
            targets = targets.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            should_checkpoint_epoch = (
                batch_index == train_batches_per_epoch
                and (epoch % checkpoint_epochs == 0 or epoch == epochs)
            )
            if optimizer_step == 1 or should_checkpoint_epoch:
                validation_metrics = _validation_metrics(
                    model=model,
                    data_loader=validation_loader,
                    device=device,
                    loss_fn=loss_fn,
                )
                append_jsonl_record(
                    metrics_log_path,
                    {
                        "event": "step",
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "train_loss": float(loss.item()),
                        "validation_loss": float(validation_metrics["loss"]),
                        "validation_sign_accuracy": float(validation_metrics["sign_accuracy"]),
                    },
                )
                checkpoint_path = output_root / f"supervised_depth_model_step_{optimizer_step}.pt"
                save_supervised_depth_checkpoint(
                    checkpoint_path,
                    model,
                    {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dims[0],
                        "hidden_dims": list(hidden_dims),
                        "policy_variant": policy_variant,
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "total_timesteps": total_timesteps,
                    },
                )
                append_jsonl_record(
                    metrics_log_path,
                    {
                        "event": "checkpoint",
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )
                if validation_metrics["loss"] < best_validation_loss:
                    best_validation_loss = validation_metrics["loss"]
                    save_supervised_depth_checkpoint(
                        best_model_path,
                        model,
                        {
                            "input_dim": input_dim,
                            "hidden_dim": hidden_dims[0],
                            "hidden_dims": list(hidden_dims),
                            "policy_variant": policy_variant,
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "total_timesteps": total_timesteps,
                        },
                    )
                    append_jsonl_record(
                        metrics_log_path,
                        {
                            "event": "best_model",
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "best_validation_loss": float(best_validation_loss),
                            "best_model_path": str(best_model_path),
                        },
                    )

    final_model_path = output_root / "supervised_depth_model.pt"
    if best_model_path.exists():
        final_model_path.write_bytes(best_model_path.read_bytes())
    else:
        save_supervised_depth_checkpoint(
            final_model_path,
            model,
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dims[0],
                "hidden_dims": list(hidden_dims),
                "policy_variant": policy_variant,
                "epoch": epochs,
                "optimizer_step": total_optimizer_steps,
                "total_timesteps": total_timesteps,
            },
        )
    summary = {
        "dataset_path": dataset_path,
        "policy_variant": policy_variant,
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "total_timesteps": total_timesteps,
        "epochs": epochs,
        "optimizer_steps": total_optimizer_steps,
        "train_batches_per_epoch": train_batches_per_epoch,
        "checkpoint_epochs": checkpoint_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "checkpoint_frequency": f"every_{checkpoint_epochs}_epochs",
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "train_examples": train_size,
        "validation_examples": validation_size,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else "",
        "metrics_log_path": str(metrics_log_path),
        "model_parameters": model_parameter_count,
        "training_examples_seen": train_examples_seen,
        "estimated_training_flops": int(
            TRAINING_FLOP_ESTIMATE_MULTIPLIER * model_parameter_count * train_examples_seen
        ),
        "training_flop_estimate_method": "6 * supervised_model_parameters * training_examples_seen",
    }
    write_json(output_root / "training_summary.json", summary)
    return summary
