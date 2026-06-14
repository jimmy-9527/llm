import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional
from unittest.mock import patch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_utils import tokenize_prompt_and_output

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm.worker import worker as _vllm_worker_module

# vLLM's UniProcExecutor hardcodes local_rank=0, so Worker.init_device builds
# its device as cuda:{local_rank}=cuda:0 even when device_config targets another
# GPU. Two consequences:
#   1. It captures its memory baseline snapshot on cuda:0. If cuda:0 is busy
#      (e.g. another process), the baseline is wrong and vLLM derives a negative
#      non_torch_memory, over-estimating KV-cache space and OOMing on the real
#      device.
#   2. xformers then creates attn_bias on cuda:0 while Q/K/V live on the
#      configured device.
# Fix by pointing local_rank at the configured device *before* init runs, so
# device selection, the baseline snapshot, and the distributed group all use the
# right GPU; the trailing set_device keeps current_device() consistent too.
_orig_init_device = _vllm_worker_module.Worker.init_device
def _patched_init_device(self):
    target = self.device_config.device
    if getattr(target, "type", None) == "cuda" and target.index is not None:
        self.local_rank = target.index
    _orig_init_device(self)
    torch.cuda.set_device(self.device_config.device)
_vllm_worker_module.Worker.init_device = _patched_init_device


@dataclass
class EvalRow:
    idx: int
    problem_id: Optional[str]
    prompt: str
    ground_truth: Any
    response: str
    reward: float
    format_reward: float
    answer_reward: float
    category: str  # "F1A1", "F1A0", "F0A0"


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.float16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_question(ex: Dict[str, Any]) -> str:
    for k in ["problem", "question", "prompt"]:
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    raise KeyError(f"Cannot find question field in example keys={list(ex.keys())}")


def get_ground_truth(ex: Dict[str, Any]) -> Any:
    for k in ["answer", "ground_truth", "target"]:
        if k in ex:
            return ex[k]
    raise KeyError(f"Cannot find answer field in example keys={list(ex.keys())}")


def build_prompts_and_gts(
    data,
    prompt_file: str,
    max_examples: int = 0,
    return_uids: bool = False,
) -> tuple:
    """
    Build r1-zero prompts and ground truths from examples.

    Args:
      data: either a path to a .jsonl file, or an already-loaded list of example dicts.
      prompt_file: path to the prompt template file.
      max_examples: if > 0, truncate to the first N examples.
      return_uids: if True, also return a list of each example's "unique_id".

    Returns:
      (prompts, gts) or, when return_uids is True, (prompts, gts, uids).
    """
    prompt_template = read_text(prompt_file)
    if isinstance(data, str):
        data = load_jsonl(data)
    if max_examples and max_examples > 0:
        data = data[:max_examples]

    prompts, gts, uids = [], [], []
    for ex in data:
        prompts.append(format_r1_zero_prompt(prompt_template, get_question(ex)))
        gts.append(get_ground_truth(ex))
        uids.append(ex.get("unique_id", ""))
    if return_uids:
        return prompts, gts, uids
    return prompts, gts


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_r1_zero_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.format(question=question)


def categorize(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "F1A1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "F1A0"
    return "F0A0"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[Any],
    eval_sampling_params: SamplingParams,
    request_batch_size: int = 64,
) -> List[EvalRow]:
    assert len(prompts) == len(ground_truths)

    rows: List[EvalRow] = []
    idx_base = 0

    for start in range(0, len(prompts), request_batch_size):
        end = min(len(prompts), start + request_batch_size)
        batch_prompts = prompts[start:end]
        batch_gts = ground_truths[start:end]

        outputs = vllm_model.generate(batch_prompts, eval_sampling_params)

        for i, out in enumerate(outputs):
            prompt = out.prompt
            full_response = out.outputs[0].text

            gt = batch_gts[i]
            scores = reward_fn(full_response, gt)

            fr = float(scores.get("format_reward", 0.0))
            ar = float(scores.get("answer_reward", 0.0))
            rr = float(scores.get("reward", 0.0))

            rows.append(
                EvalRow(
                    idx=idx_base + i,
                    problem_id=None,
                    prompt=prompt,
                    ground_truth=gt,
                    response=full_response,
                    reward=rr,
                    format_reward=fr,
                    answer_reward=ar,
                    category=categorize(fr, ar),
                )
            )

        idx_base += end - start

    return rows


def write_jsonl(path: str, rows: List[EvalRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def summarize(rows: List[EvalRow]) -> Dict[str, Any]:
    n = len(rows)
    c = {"F1A1": 0, "F1A0": 0, "F0A0": 0}
    for r in rows:
        c[r.category] += 1

    return {
        "n": n,
        "counts": c,
        "format_rate": sum(r.format_reward for r in rows) / n if n else 0.0,
        "answer_accuracy": sum(r.answer_reward for r in rows) / n if n else 0.0,
        "avg_reward": sum(r.reward for r in rows) / n if n else 0.0,
    }


def sample_examples(rows: List[EvalRow], category: str, k: int = 10) -> List[EvalRow]:
    return [r for r in rows if r.category == category][:k]


def eval_policy_with_vllm(
    *,
    policy: torch.nn.Module,
    llm: LLM,
    eval_prompts: List[str],
    eval_gts: List[Any],
    eval_sampling_params: SamplingParams,
    request_batch_size: int = 64,
    reward_fn: Callable[[str, Any], Dict[str, float]] = r1_zero_reward_fn,
) -> Dict[str, Any]:
    """
    Load the current policy weights into the vLLM instance, evaluate on the
    given prompts, and return an eval/* metrics dict.

    Leaves the policy in eval() mode; callers that resume training should call
    policy.train() afterwards.
    """
    policy.eval()
    with torch.no_grad():
        load_policy_into_vllm_instance(policy, llm)
        rows = evaluate_vllm(
            vllm_model=llm,
            reward_fn=reward_fn,
            prompts=eval_prompts,
            ground_truths=eval_gts,
            eval_sampling_params=eval_sampling_params,
            request_batch_size=request_batch_size,
        )
    s = summarize(rows)
    return {
        "eval/accuracy": s["answer_accuracy"],
        "eval/format_rate": s["format_rate"],
        "eval/avg_reward": s["avg_reward"],
        "eval/n": s["n"],
    }


def filter_correct_sft_samples(data_path: str, out_path: str) -> Dict[str, Any]:
    all_examples = load_jsonl(data_path)
    kept = []
    total = len(all_examples)
    for ex in all_examples:
        gt = ex.get("answer") or ex.get("ground_truth")
        if gt is None:
            raise RuntimeError(
                "sft.jsonl does not contain ground-truth fields (answer/ground_truth). "
            )
        scores = r1_zero_reward_fn(ex["response"], gt)
        if float(scores.get("answer_reward", 0.0)) >= 1.0:
            kept.append(ex)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in kept:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {"filtered/kept": len(kept), "filtered/total": total}


def collate_fn(batch, tokenizer):
    prompts = [x[0] for x in batch]
    outputs = [x[1] for x in batch]
    return tokenize_prompt_and_output(prompts, outputs, tokenizer)


def log_event(
    log_path: Path,
    step: int,
    micro_idx: int,
    opt_step: int,
    event: Dict[str, Any],
    *,
    also_print: bool = True,
) -> None:
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "time": time.time(),
        "step": step,
        "micro_idx": micro_idx,
        "opt_step": opt_step,
        **event,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()

    if also_print:
        if "msg" in event:
            print(event["msg"])
        else:
            print(payload)
