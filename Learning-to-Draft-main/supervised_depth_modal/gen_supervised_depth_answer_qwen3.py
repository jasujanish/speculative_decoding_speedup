"""Generate Qwen3 answers with a supervised depth controller."""

from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path
from typing import Any

import shortuuid
import torch
from accelerate.utils import set_seed
from fastchat.llm_judge.common import load_questions
from qwen3_model_presets import list_model_presets, resolve_model_paths
from tqdm import tqdm

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor
from supervised_depth_modal.core import load_supervised_depth_checkpoint, scores_entropy


set_seed(0)


def answer_manifest_path(answer_file: str) -> Path:
    """Return the sidecar manifest path for an answer file.

    Args:
        answer_file: JSONL answer file path.

    Returns:
        JSON manifest path.
    """
    answer_path = Path(os.path.expanduser(answer_file))
    return answer_path.with_name(f"{answer_path.name}.manifest.json")


def patch_depth_runner(model: EaModel, depth_model_path: str) -> None:
    """Attach a supervised depth model to Eagle3 tree growth.

    Args:
        model: Loaded Eagle3 model.
        depth_model_path: Supervised depth checkpoint path.
    """
    depth_model = load_supervised_depth_checkpoint(depth_model_path, device="cuda")
    model.dyn_depth_ffn = depth_model

    @torch.no_grad()
    def topk_generate_with_supervised_depth(
        self: Any,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        head: Any,
        logits_processor: Any,
        depth: int = -1,
        total_token: int = -1,
        obs_tensor: torch.Tensor | None = None,
        policy_net: Any = None,
        dyn_token_ffn: Any = None,
        dyn_depth_ffn: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
        """Run tree growth while consulting the supervised controller at each depth.

        Args:
            hidden_states: Verified hidden states from the target model.
            input_ids: Prefix tokens for the next speculative cycle.
            head: Unused compatibility argument.
            logits_processor: Optional logits processor.
            depth: Requested maximum draft depth.
            total_token: Verification-token budget.
            obs_tensor: Optional token-policy observation buffer.
            policy_net: Optional legacy token policy.
            dyn_token_ffn: Optional dynamic token controller.
            dyn_depth_ffn: Supervised depth controller.

        Returns:
            Draft tokens, retrieve indices, tree mask, tree position ids, controller
            overhead time, realized draft depth, and realized verification width.
        """
        del head
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = total_token if total_token != -1 else self.total_tokens
        depth = depth if depth != -1 else self.depth
        top_k = self.top_k
        sample_token = input_ids[:, -1]
        rl_time = 0.0
        scores_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []
        ss_token: list[torch.Tensor] = []

        input_ids = input_ids[:, 1:].to(hidden_states.device)
        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                use_cache=True,
            )
        if hasattr(self, "stats_draft_model_calls"):
            self.stats_draft_model_calls += 1
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]
        last_headout = self.lm_head(self.norm(last_hidden))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        input_ids_shape_len = input_ids.shape[1] / 1000.0
        cnet_step = 0

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            if hasattr(self, "stats_draft_model_calls"):
                self.stats_draft_model_calls += 1
            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = (topk_cs_index // top_k).to(tree_mask.device)
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            decision_start = time.time()
            cnet_step += 1
            if dyn_depth_ffn is not None and i != depth - 1:
                if self._run_dyn_depth_on_cpu(cu_scores, cnet_step, input_ids_shape_len, dyn_depth_ffn):
                    rl_time += time.time() - decision_start
                    break
            rl_time += time.time() - decision_start

        if dyn_token_ffn is not None and obs_tensor is not None:
            scores_obs = torch.cat(scores_list, dim=0).view(-1)
            obs_tensor[0 : scores_obs.shape[0]] = scores_obs
            obs_tensor[1239:1268].fill_(cnet_step / 10.0)
            with torch.inference_mode():
                logits = dyn_token_ffn(obs_tensor.unsqueeze(0))
                actions = torch.argmax(logits, dim=1)
            total_tokens = (actions[0] + 1) * 10

        scores_list_tensor = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        total_tokens = min(ss_token_list.shape[0], total_tokens)
        top_scores = torch.topk(scores_list_tensor, total_tokens, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for token_index in range(total_tokens):
            tree_mask[token_index + 1].add_(tree_mask[mask_index_list[token_index]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for token_index in range(total_tokens + 1):
            if token_index in noleaf_index:
                continue
            current_id = token_index
            current_depth = position_ids_list[token_index]
            for depth_index in reversed(range(current_depth + 1)):
                retrieve_indices[rid][depth_index] = current_id
                current_id = mask_index_list[current_id - 1]
            rid += 1

        if logits_processor is not None:
            max_item = total_tokens + 5

            def custom_sort(values: list[int]) -> list[int]:
                """Sort retrieve rows while pushing masked entries to the end."""
                return [value if value >= 0 else max_item for value in values]

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices_tensor = torch.tensor(retrieve_indices, dtype=torch.long)
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        return (
            draft_tokens,
            retrieve_indices_tensor,
            tree_mask,
            tree_position_ids,
            rl_time,
            cnet_step,
            total_tokens,
        )

    def run_dyn_depth_on_cpu(
        self: Any,
        scores_cpu: torch.Tensor,
        current_depth: int,
        context_length: int,
        dyn_depth_ffn: torch.nn.Module,
    ) -> bool:
        """Return whether drafting should stop at the next depth check.

        Args:
            scores_cpu: Frontier score tensor.
            current_depth: Current draft depth.
            context_length: Current prefix length.
            dyn_depth_ffn: Loaded supervised regressor.

        Returns:
            Whether expansion should stop.
        """
        scores_flat = scores_cpu.flatten().detach().float().cpu()
        obs = torch.zeros(129, dtype=torch.float32)
        obs[: min(100, scores_flat.numel())] = scores_flat[:100]
        obs[100:114] = float(context_length) / 1000.0
        obs[114:128] = float(current_depth) / 10.0
        obs[128] = scores_entropy(scores_flat)
        with torch.inference_mode():
            delta = dyn_depth_ffn(obs.unsqueeze(0).to("cuda"))
        return bool(delta.item() <= 0.0)

    model.ea_layer._run_dyn_depth_on_cpu = types.MethodType(
        run_dyn_depth_on_cpu,
        model.ea_layer,
    )
    model.ea_layer.topK_genrate = types.MethodType(
        topk_generate_with_supervised_depth,
        model.ea_layer,
    )


def run_eval(
    base_model_path: str,
    ea_model_path: str,
    model_id: str,
    question_file: str,
    question_begin: int,
    question_end: int,
    answer_file: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    num_gpus_total: int,
    max_gpu_memory: str | None,
    temperature: float,
    args: argparse.Namespace,
) -> None:
    """Run answer generation for a prompt subset.

    Args:
        base_model_path: Base model path.
        ea_model_path: Eagle3 model path.
        model_id: Output model identifier.
        question_file: Benchmark question file.
        question_begin: Start index.
        question_end: End index.
        answer_file: JSONL answer file.
        max_new_token: Unused compatibility argument.
        num_choices: Number of completions per prompt.
        num_gpus_per_model: GPUs per model shard.
        num_gpus_total: Total GPUs.
        max_gpu_memory: Unused compatibility argument.
        temperature: Generation temperature.
        args: Parsed CLI args.
    """
    del max_new_token, max_gpu_memory
    questions = load_questions(question_file, question_begin, question_end)
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1
    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers
    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    handles = []
    for start in range(0, len(questions), chunk_size):
        handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[start : start + chunk_size],
                answer_file,
                num_choices,
                temperature,
                args,
            )
        )
    if use_ray:
        ray.get(handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path: str,
    ea_model_path: str,
    model_id: str,
    questions: list[dict[str, object]],
    answer_file: str,
    num_choices: int,
    temperature: float,
    args: argparse.Namespace,
) -> None:
    """Generate answers for one question shard.

    Args:
        base_model_path: Base model path.
        ea_model_path: Eagle3 model path.
        model_id: Output model identifier.
        questions: Benchmark questions.
        answer_file: JSONL output path.
        num_choices: Number of completions per prompt.
        temperature: Generation temperature.
        args: Parsed CLI args.
    """
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_dyn_len=False,
        use_dyn_token=False,
        use_rl=False,
        device_map="auto",
    )
    if args.depth_model:
        patch_depth_runner(model, args.depth_model)
    model.stats_target_model_calls = 0
    model.ea_layer.stats_draft_model_calls = 0
    tokenizer = model.get_tokenizer()
    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
    model.eval()

    warmup_question = questions[0]
    for _ in range(3):
        torch.manual_seed(0)
        messages: list[dict[str, str]] = []
        for turn in warmup_question["turns"]:
            messages.append({"role": "user", "content": turn})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            input_ids = [input_ids[0][:1790]]
            torch.cuda.synchronize()
            model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
            )
            torch.cuda.synchronize()
    print("Warmup done")

    for question in tqdm(questions):
        choices = []
        for choice_index in range(num_choices):
            torch.manual_seed(choice_index)
            messages = []
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            pre_len_times = []
            dtimes = []
            dyn_tokens = []
            dyn_depths = []
            pre_num = []
            for turn in question["turns"]:
                messages.append({"role": "user", "content": turn})
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                if len(input_ids[0]) > 1790:
                    input_ids = [input_ids[0][:1790]]
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, pre_len_time, pre_num = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    pre_len=True,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                total_time -= pre_len_time
                output_ids = output_ids[0][len(input_ids[0]) :]
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                ]
                stop_positions = [i for i, token_id in enumerate(output_ids) if token_id in stop_token_ids]
                if stop_positions:
                    output_ids = output_ids[: stop_positions[0]]
                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for token in special_token:
                            output = output.replace(token, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                pre_len_times.append(pre_len_time)
                dtimes.extend(model.dtimes)
                dyn_tokens.extend(model.dyn_tokens)
                dyn_depths.extend(model.cnet_steps)
                messages.append({"role": "assistant", "content": output})
            choices.append(
                {
                    "index": choice_index,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "pre_len_times": pre_len_times,
                    "pre_num": pre_num,
                    "dtimes": dtimes,
                    "dyn_tokens": dyn_tokens,
                    "dyn_depths": dyn_depths,
                }
            )
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "question_id": question["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": choices,
                        "tstamp": time.time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest = {
        "answer_file": os.path.expanduser(answer_file),
        "question_count": len(questions),
        "draft_model_calls": int(getattr(model.ea_layer, "stats_draft_model_calls", 0)),
        "target_model_calls": int(getattr(model, "stats_target_model_calls", 0)),
    }
    answer_manifest_path(answer_file).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def reorg_answer_file(answer_file: str) -> None:
    """Sort and deduplicate answer records by question id.

    Args:
        answer_file: JSONL answer file.
    """
    answers: dict[int, str] = {}
    with open(answer_file, "r", encoding="utf-8") as handle:
        for line in handle:
            answers[int(json.loads(line)["question_id"])] = line
    with open(answer_file, "w", encoding="utf-8") as handle:
        for question_id in sorted(answers):
            handle.write(answers[question_id])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-preset", choices=list_model_presets(), default=None)
    parser.add_argument("--ea-model-path", type=str, default="")
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--model-id", type=str, default="qwen3-supervised-depth")
    parser.add_argument("--bench-name", type=str, default="gsm8k")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=80)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--depth-model", type=str, default="")
    args = parser.parse_args()
    args.base_model_path, args.ea_model_path = resolve_model_paths(
        model_preset=args.model_preset,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    args.model_id = (
        f"{args.model_id}-temperature{args.temperature}"
        f"token{args.total_token}depth{args.depth}choices{args.num_choices}"
    )
    if args.depth_model:
        args.model_id += os.path.splitext(os.path.basename(args.depth_model))[0]
    question_file = args.question_file or f"{Path(__file__).resolve().parents[1]}/eagle/data/{args.bench_name}/question.jsonl"
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()
    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args,
    )
    reorg_answer_file(args.answer_file)
