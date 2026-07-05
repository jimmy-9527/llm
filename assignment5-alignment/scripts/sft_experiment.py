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
    load_jsonl, init_vllm, build_prompts_and_gts,
    filter_correct_sft_samples, collate_fn, make_logger,
    eval_policy_with_vllm,
)
from cs336_alignment.sft_utils import get_response_log_probs, sft_microbatch_train_step

from vllm import SamplingParams


class SFTDataset(Dataset):
    """Prompt/response pairs for SFT, from a JSONL path or an in-memory list.

    Provide exactly one of:
      - ``path``: load a JSONL file, optionally subsampled to ``limit`` items.
      - ``items``: already-loaded dicts (e.g. EI-kept rollouts).
    Each item must have ``prompt`` and ``response`` fields.
    """
    def __init__(self, path: str = None, limit: int = 0, seed: int = 0, *, items=None):
        if items is not None:
            self.data = list(items)
        elif path is not None:
            self.data = load_jsonl(path)
            if limit and limit > 0:
                rnd = random.Random(seed)
                rnd.shuffle(self.data)
                self.data = self.data[:limit]
        else:
            raise ValueError("SFTDataset requires either `path` or `items`")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["prompt"], ex["response"], ex


def load_policy_and_tokenizer(model_id: str, device: str, dtype=torch.bfloat16):
    """Load a tokenizer + causal-LM policy for full SFT on ``device``.

    NOTE: train in bf16, not fp16. Pure-fp16 full fine-tuning is unstable here:
    AdamW eps=1e-8 underflows to 0 in fp16 and fp16 log-softmax over the 151k
    vocab overflows, producing NaN losses within a few steps. bf16 has fp32's
    exponent range (no overflow, eps representable) and is supported on the T4.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    policy.gradient_checkpointing_enable()
    policy.train()
    return tokenizer, policy


def make_sft_optimizer(policy, lr: float):
    """Single-tensor AdamW (foreach/fused disabled), shared by the SFT and EI loops."""
    return torch.optim.AdamW(policy.parameters(), lr=lr, foreach=False, fused=False)


def clip_and_step(policy, opt, max_norm: float = 1.0):
    """Grad-clip, optimizer step, then zero grads. Shared by the SFT and EI loops."""
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)


def sft_forward_backward(
    policy, batch, device, *, grad_acc_steps, normalize_constant=1.0, return_entropy=False
):
    """Run one SFT micro-batch: move to ``device``, forward, masked-NLL backward.

    ``sft_microbatch_train_step`` scales the loss by 1/grad_acc_steps and calls
    backward() internally, so the caller only owns optimizer stepping.

    normalize_constant: divisor for the summed per-token NLL. Pass a float (e.g.
      1.0), or ``None`` to normalize by the number of response tokens in the batch
      (a per-token-mean NLL, which keeps gradients O(1) rather than O(resp_len)).
    return_entropy: if True, also return the response-masked mean token entropy
      (for logging); otherwise the second return value is None.

    Returns ``(loss, avg_entropy_or_None)``.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    out = get_response_log_probs(
        policy, input_ids, labels, return_token_entropy=return_entropy
    )

    if normalize_constant is None:
        normalize_constant = float(torch.clamp(response_mask.sum(), min=1.0).detach().cpu())

    loss, _ = sft_microbatch_train_step(
        policy_log_probs=out["log_probs"],
        response_mask=response_mask,
        gradient_accumulation_steps=grad_acc_steps,
        normalize_constant=normalize_constant,
    )

    avg_entropy = None
    if return_entropy:
        with torch.no_grad():
            te = out["token_entropy"]
            m = response_mask.to(te.dtype)
            denom = torch.clamp(m.sum(), min=1.0)
            avg_entropy = float((te * m).sum().cpu() / denom.cpu())

    return loss, avg_entropy



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
    log_event = make_logger(run_dir / "log.jsonl")

    opt_step = 0  # counts optimizer updates
    step = 0
    micro_idx = 0    

    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # tokenizer/model on train device
    tokenizer, policy = load_policy_and_tokenizer(args.model_id, args.train_device)

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
        log_event({"type": "filter_stats", "stats": stats, "msg": f"Filter stats: {stats}"}, step=step, micro_idx=micro_idx, opt_step=opt_step)
        data_path = filtered_path

    dataset = SFTDataset(data_path, limit=args.train_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    opt = make_sft_optimizer(policy, args.lr)

    # training loop
    opt.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in loader:
            step += 1
            micro_idx += 1

            # forward + masked-NLL backward (backward happens inside)
            loss, _ = sft_forward_backward(
                policy, batch, args.train_device,
                grad_acc_steps=args.grad_acc_steps, normalize_constant=1.0,
            )

            # optimizer step each grad_acc_steps
            if micro_idx % args.grad_acc_steps == 0:
                clip_and_step(policy, opt)
                opt_step += 1
                if opt_step % 10 == 0:
                    log_event({"type": "train_loss", "loss": float(loss.detach())}, step=step, micro_idx=micro_idx, opt_step=opt_step, also_print=False)

            # periodic eval
            if step % args.eval_interval == 0:
                metrics = eval_policy_with_vllm(
                    policy=policy,
                    llm=llm,
                    eval_prompts=eval_prompts,
                    eval_gts=eval_gts,
                    eval_sampling_params=eval_sampling_params,
                    request_batch_size=64,
                )
                log_event({"type": "eval_metrics", "loss": float(loss.detach()), "metrics": metrics,
                        "msg": f"[step={step}] loss={float(loss.detach()):.4f} {metrics}"}, step=step, micro_idx=micro_idx, opt_step=opt_step)
                policy.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # save
    policy.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    log_event({"type": "save", "out_dir": str(run_dir), "msg": f"Saved: {run_dir}"}, step=step, micro_idx=micro_idx, opt_step=opt_step)


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
