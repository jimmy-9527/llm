import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval_utils import (
    EvalRow,
    load_jsonl,
    read_text,
    format_r1_zero_prompt,
    evaluate_vllm,
    write_jsonl,
    summarize,
    sample_examples,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--data", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")
    ap.add_argument("--out_dir", default="runs/math_baseline")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=int, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    prompt_template = read_text(args.prompt_file)

    examples = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

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

    prompts, gts = [], []
    for ex in examples:
        prompts.append(format_r1_zero_prompt(prompt_template, get_question(ex)))
        gts.append(get_ground_truth(ex))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(
        model=args.model,
        dtype="float16",
    )

    rows = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=gts,
        eval_sampling_params=sampling_params,
        request_batch_size=args.batch_size,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(str(out_dir / "predictions.jsonl"), rows)

    summary = summarize(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    samples = {
        "F1A1": [asdict(r) for r in sample_examples(rows, "F1A1", 10)],
        "F1A0": [asdict(r) for r in sample_examples(rows, "F1A0", 10)],
        "F0A0": [asdict(r) for r in sample_examples(rows, "F0A0", 10)],
    }
    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("Saved to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
