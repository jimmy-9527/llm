import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    load_jsonl,
    collate_fn,
    build_prompts_and_gts,
    eval_policy_with_vllm,
    make_logger,
    write_jsonl,
)

# Reuse the SFT sweep's dataset / model-loading / optimizer / train-step helpers.
sys.path.insert(0, os.path.dirname(__file__))
from sft_experiment import (
    SFTDataset,
    load_policy_and_tokenizer,
    make_sft_optimizer,
    sft_forward_backward,
    clip_and_step,
)


def rollout_and_filter_correct(
    *,
    llm: LLM,
    prompts: List[str],
    gts: List[Any],
    uids: List[str],
    ei_step: int,
    G: int,
    sampling_params: SamplingParams,
    request_batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    For each prompt, generate G candidates via vLLM (n=G), score each candidate,
    and keep only correct ones (answer_reward==1).
    Returns:
      kept_sft_items: list of dicts with keys prompt/response (+meta)
      stats: dict with counts
    """
    assert len(prompts) == len(gts) == len(uids)

    kept: List[Dict[str, Any]] = []
    total_gen = 0
    total_correct = 0
    total_format_ok = 0

    # vLLM batch generate
    for start in range(0, len(prompts), request_batch_size):
        end = min(len(prompts), start + request_batch_size)
        batch_prompts = prompts[start:end]
        batch_gts = gts[start:end]
        batch_uids = uids[start:end]

        outputs = llm.generate(batch_prompts, sampling_params)

        # outputs aligned with prompts
        for i, out in enumerate(outputs):
            prompt = out.prompt
            gt = batch_gts[i]
            uid = batch_uids[i]

            # out.outputs is a list of length n=G
            for j, cand in enumerate(out.outputs):
                resp = cand.text
                scores = r1_zero_reward_fn(resp, gt)
                fr = float(scores.get("format_reward", 0.0))
                ar = float(scores.get("answer_reward", 0.0))
                rr = float(scores.get("reward", 0.0))

                total_gen += 1
                if fr >= 1.0:
                    total_format_ok += 1
                if ar >= 1.0:
                    total_correct += 1
                    kept.append(
                        {
                            "prompt": prompt,
                            "response": resp,
                            "answer": gt,
                            "unique_id": uid,
                            "ei_step": ei_step,
                            "rollout_idx": j,
                            "reward": rr,
                            "format_reward": fr,
                            "answer_reward": ar,
                        }
                    )

    stats = {
        "ei_step": ei_step,
        "num_questions": len(prompts),
        "G": G,
        "total_generations": total_gen,
        "num_correct_trajs": total_correct,
        "format_ok_trajs": total_format_ok,
        "kept_sft_size": len(kept),
        "kept_per_question": (len(kept) / max(1, len(prompts))),
    }
    return kept, stats


def train_sft_on_items(
    *,
    policy: torch.nn.Module,
    tokenizer,
    items: List[Dict[str, Any]],
    device: str,
    lr: float,
    micro_batch_size: int,
    grad_acc_steps: int,
    epochs: int,
    max_train_steps: int,
    log_event,
    ei_step: int,
    train_log_interval_opt_steps: int = 10,
) -> Dict[str, Any]:
    """
    Runs SFT training on prompt/response items for some epochs.
    Logs:
      - train/loss
      - train/avg_token_entropy (masked on response tokens)
    Returns training summary stats.
    """
    if len(items) == 0:
        return {"ei_step": ei_step, "skipped": True, "reason": "no_kept_items"}

    dataset = SFTDataset(items=items)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    opt = make_sft_optimizer(policy, lr)
    policy.train()

    step = 0
    micro_idx = 0
    opt_step = 0

    opt.zero_grad(set_to_none=True)

    # running stats
    ent_sum = 0.0
    ent_count = 0

    for ep in range(epochs):
        for batch in loader:
            step += 1
            micro_idx += 1

            # forward + masked-NLL backward; normalize_constant=None => per-token
            # mean NLL (keeps gradients O(1) rather than O(response_len)).
            loss, avg_ent = sft_forward_backward(
                policy, batch, device,
                grad_acc_steps=grad_acc_steps,
                normalize_constant=None,
                return_entropy=True,
            )
            ent_sum += avg_ent
            ent_count += 1

            if micro_idx % grad_acc_steps == 0:
                clip_and_step(policy, opt)
                opt_step += 1

                # periodic log
                if opt_step % train_log_interval_opt_steps == 0:
                    avg_ent_running = ent_sum / max(1, ent_count)
                    log_event(
                        {
                            "type": "train_step",
                            "ei_step": ei_step,
                            "epoch": ep,
                            "opt_step": opt_step,
                            "micro_step": step,
                            "loss": float(loss.detach().cpu()),
                            "train/avg_token_entropy": avg_ent_running,
                            "msg": f"[EI {ei_step}] ep={ep} opt_step={opt_step} loss={float(loss.detach()):.4f} avg_ent={avg_ent_running:.4f}",
                        },
                        also_print=True,
                    )
                    ent_sum = 0.0
                    ent_count = 0

            if max_train_steps > 0 and step >= max_train_steps:
                break
        if max_train_steps > 0 and step >= max_train_steps:
            break

    return {
        "ei_step": ei_step,
        "epochs": epochs,
        "train_micro_steps": step,
        "train_opt_steps": opt_step,
        "num_items": len(items),
        "skipped": False,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", default="data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--train_path", default="data/MATH/train.jsonl")
    ap.add_argument("--val_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")

    ap.add_argument("--train_device", default="cuda:2")
    ap.add_argument("--vllm_device", default="cuda:3")

    ap.add_argument("--out_dir", default="runs/expert_iteration")
    ap.add_argument("--seed", type=int, default=0)

    # EI hyperparams
    ap.add_argument("--n_ei_steps", type=int, default=5)
    ap.add_argument("--D_i", type=int, default=512, help="number of questions sampled per EI step")
    ap.add_argument("--G", type=int, default=2, help="rollouts per question")
    ap.add_argument("--epochs", type=int, default=1, help="SFT epochs per EI step")
    ap.add_argument("--max_train_steps_per_ei", type=int, default=0, help="0 means no cap")

    # SFT optimizer params
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)

    # vLLM sampling params for rollout
    ap.add_argument("--sampling_temperature", type=float, default=1.0)
    ap.add_argument("--sampling_top_p", type=float, default=1.0)
    ap.add_argument("--sampling_max_tokens", type=int, default=256)
    ap.add_argument("--sampling_min_tokens", type=int, default=4)

    # eval params
    ap.add_argument("--eval_max_examples", type=int, default=500)
    ap.add_argument("--eval_request_batch_size", type=int, default=64)
    ap.add_argument("--rollout_request_batch_size", type=int, default=32)
    ap.add_argument("--save_each_ei_step", action="store_true")

    args = ap.parse_args()

    # run folder
    run_name = f"ei_G{args.G}_E{args.epochs}_D{args.D_i}_seed{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    log_event = make_logger(log_path)

    # seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    log_event(
        {
            "type": "config",
            "args": vars(args),
            "run_dir": str(run_dir),
            "msg": f"Run dir: {run_dir}",
        }
    )

    # load data
    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)

    # build eval prompts
    eval_prompts, eval_gts = build_prompts_and_gts(
        val_data, args.prompt_file, max_examples=args.eval_max_examples
    )

    # init tokenizer/policy (HF) on train_device
    tokenizer, policy = load_policy_and_tokenizer(args.model_id, args.train_device)

    # init vLLM on vllm_device
    llm = init_vllm(args.model_id, device=args.vllm_device, seed=args.seed)

    # sampling params for rollout (n=G)
    rollout_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.G,
        seed=args.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # eval sampling params
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # initial eval
    init_metrics = eval_policy_with_vllm(
        policy=policy,
        llm=llm,
        eval_prompts=eval_prompts,
        eval_gts=eval_gts,
        eval_sampling_params=eval_sampling_params,
        request_batch_size=args.eval_request_batch_size,
    )
    log_event({"type": "eval_metrics", "ei_step": 0, "metrics": init_metrics, "msg": f"[EI 0] {init_metrics}"})

    # EI loop
    rng = random.Random(args.seed)

    for ei_step in range(1, args.n_ei_steps + 1):
        # sample D_i questions
        rng.shuffle(train_data)
        sampled = train_data[: args.D_i]

        batch_prompts, batch_gts, batch_uids = build_prompts_and_gts(
            sampled, args.prompt_file, return_uids=True
        )

        # load policy -> vLLM and rollout
        policy.eval()
        with torch.no_grad():
            load_policy_into_vllm_instance(policy, llm)
        
        kept_items, rollout_stats = rollout_and_filter_correct(
            llm=llm,
            prompts=batch_prompts,
            gts=batch_gts,
            uids=batch_uids,
            ei_step=ei_step,
            G=args.G,
            sampling_params=rollout_sampling_params,
            request_batch_size=args.rollout_request_batch_size,
        )

        # save EI dataset for this step
        ei_data_path = run_dir / f"ei_step_{ei_step:02d}_kept.jsonl"
        write_jsonl(str(ei_data_path), kept_items)

        log_event(
            {
                "type": "rollout_stats",
                "ei_step": ei_step,
                "stats": rollout_stats,
                "ei_data_path": str(ei_data_path),
                "msg": f"[EI {ei_step}] rollout stats: {rollout_stats}",
            }
        )

        # train SFT on kept items
        policy.train()
        train_summary = train_sft_on_items(
            policy=policy,
            tokenizer=tokenizer,
            items=kept_items,
            device=args.train_device,
            lr=args.lr,
            micro_batch_size=args.micro_batch_size,
            grad_acc_steps=args.grad_acc_steps,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps_per_ei,
            log_event=log_event,
            ei_step=ei_step,
            train_log_interval_opt_steps=10,            
        )
        log_event({"type": "train_summary", "ei_step": ei_step, "summary": train_summary,
                   "msg": f"[EI {ei_step}] train summary: {train_summary}"})
        
        # eval after EI step
        metrics = eval_policy_with_vllm(
            policy=policy,
            llm=llm,
            eval_prompts=eval_prompts,
            eval_gts=eval_gts,
            eval_sampling_params=eval_sampling_params,
            request_batch_size=args.eval_request_batch_size,
        )
        log_event({"type": "eval_metrics", "ei_step": ei_step, "metrics": metrics,
                   "msg": f"[EI {ei_step}] {metrics}"})

        # save model
        if args.save_each_ei_step:
            step_dir = run_dir / f"model_ei_step_{ei_step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(step_dir))
            tokenizer.save_pretrained(str(step_dir))
            log_event({"type": "save", "ei_step": ei_step, "out_dir": str(step_dir),
                       "msg": f"[EI {ei_step}] Saved model: {step_dir}"})
            
        # return to train mode for next step
        policy.train()

    # final save
    final_dir = run_dir / "model_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log_event({"type": "save", "ei_step": args.n_ei_steps, "out_dir": str(final_dir),
               "msg": f"Saved final model: {final_dir}"})


# uv run python scripts/expert_iteration_experiment.py \
#   --n_ei_steps 1 \
#   --D_i 32 \
#   --G 2 \
#   --epochs 1 \
#   --sampling_max_tokens 128 \
#   --eval_max_examples 32 \
#   --max_train_steps_per_ei 50 \
#   --micro_batch_size 1 \
#   --grad_acc_steps 2 \
#   --save_each_ei_step
if __name__ == "__main__":
    main()
