import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from typing import Dict, Any
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.utils import (
    evaluate_vllm, load_jsonl, summarize,
    init_vllm, load_policy_into_vllm_instance, build_prompts_and_gts,
    filter_correct_sft_samples, collate_fn, log_event,
)
from cs336_alignment.sft_utils import get_response_log_probs, sft_microbatch_train_step, log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from vllm import SamplingParams


class SFTDataset(Dataset):
    def __init__(self, path: str, limit: int = 0, seed: int = 0):
        self.data = load_jsonl(path)
        if limit and limit > 0:
            rnd = random.Random(seed)
            rnd.shuffle(self.data)
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["prompt"], ex["response"], ex
    



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--sft_path", default="data/MATH/sft.jsonl")
    ap.add_argument("--val_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")

    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--vllm_device", default="cuda:1")

    ap.add_argument("--train_samples", type=int, default=0, help="0 means full dataset")
    ap.add_argument("--filter_correct", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/sft_experiment")
    ap.add_argument("--eval_max_examples", type=int, default=500)
    args = ap.parse_args()

    # logging
    run_dir = Path(args.out_dir) / f"samples{args.train_samples or 'full'}_{'filtered' if args.filter_correct else 'all'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    opt_step = 0  # counts optimizer updates
    step = 0
    micro_idx = 0    

    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # tokenizer/model on train device
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(args.train_device)
    policy.gradient_checkpointing_enable()
    policy.train()

    # vLLM on eval device
    llm = init_vllm(args.model_id, device=args.vllm_device, seed=args.seed)

    eval_prompts, eval_gts = build_prompts_and_gts(
        args.val_path, args.prompt_file, max_examples=args.eval_max_examples
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # optionally filter dataset
    data_path = args.sft_path
    if args.filter_correct:
        filtered_path = str(Path(args.out_dir) / "filtered_sft.jsonl")
        stats = filter_correct_sft_samples(args.sft_path, filtered_path)
        log_event(log_path, step, micro_idx, opt_step, {"type": "filter_stats", "stats": stats, "msg": f"Filter stats: {stats}"})
        data_path = filtered_path

    dataset = SFTDataset(data_path, limit=args.train_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, foreach=False, fused=False)

    # training loop
    opt.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in loader:
            step += 1
            micro_idx += 1

            input_ids = batch["input_ids"].to(args.train_device)
            labels = batch["labels"].to(args.train_device)
            response_mask = batch["response_mask"].to(args.train_device)

            # get per-token log_probs (B, T)
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = out["log_probs"]            

            # microbatch train step: does backward inside
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_acc_steps,
                normalize_constant=1.0,
            )

            # optimizer step each grad_acc_steps
            if micro_idx % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    log_event(log_path, step, micro_idx, opt_step, {"type": "train_loss", "loss": float(loss.detach())}, also_print=False)

            # periodic eval
            if step % args.eval_interval == 0:
                policy.eval()
                with torch.no_grad():
                    load_policy_into_vllm_instance(policy, llm)
                    rows = evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        ground_truths=eval_gts,
                        eval_sampling_params=eval_sampling_params,
                        request_batch_size=64,
                    )

                # generation log records
                # gen_log = log_generations(
                #     model=policy,
                #     tokenizer=tokenizer,
                #     prompts=eval_prompts[:8],
                #     ground_truths=eval_gts[:8],
                #     reward_fn=r1_zero_reward_fn,
                #     num_log=8,
                #     step=step,
                #     stop_str="</answer>",
                #     max_new_tokens=512,
                #     temperature=0.0,
                # )

                # log_event({"type": "gen_stats", "gen_stats": gen_log["stats"], "msg": f"gen stats: {gen_log['stats']}"})

                s = summarize(rows)
                metrics = {
                    "eval/accuracy": s["answer_accuracy"],
                    "eval/format_rate": s["format_rate"],
                    "eval/avg_reward": s["avg_reward"],
                    "eval/n": s["n"],
                }
                log_event(log_path, step, micro_idx, opt_step, {"type": "eval_metrics", "loss": float(loss.detach()), "metrics": metrics,
                        "msg": f"[step={step}] loss={float(loss.detach()):.4f} {metrics}"})
                policy.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # save
    policy.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    log_event(log_path, step, micro_idx, opt_step, {"type": "save", "out_dir": str(run_dir), "msg": f"Saved: {run_dir}"})


# uv run python scripts/sft_experiment.py \
#   --train_samples 128 \
#   --max_steps 20 \
#   --eval_interval 10 \
#   --eval_max_examples 32 \
#   --micro_batch_size 1 \
#   --grad_acc_steps 2 \
#   --lr 2e-5
if __name__ == "__main__":
    main()
