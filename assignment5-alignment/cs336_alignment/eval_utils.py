import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Dict, Any, Optional

from vllm import LLM, SamplingParams


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
