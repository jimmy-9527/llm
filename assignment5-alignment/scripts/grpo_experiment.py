import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Reduce allocator fragmentation on the (small) policy GPU; reclaims the
# reserved-but-unallocated blocks that otherwise trigger OOM near capacity.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
import random
from pathlib import Path
from enum import Enum

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import SamplingParams

from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from cs336_alignment.utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    build_prompts_and_gts,
    eval_policy_with_vllm,
)


app = typer.Typer()


class LossType(str, Enum):
    no_baseline = "no_baseline"
    reinforce_with_baseline = "reinforce_with_baseline"
    grpo_clip = "grpo_clip"
    grpo_no_clip = "grpo_no_clip"


@app.command()
def main(
    model_id: str = "data/models/Qwen2.5-Math-1.5B",
    train_path: str = "data/MATH/train.jsonl",
    val_path: str = "data/MATH/validation.jsonl",
    prompt_file: str = "cs336_alignment/prompts/r1_zero.prompt",
    seed: int = 0,

    # ===== GRPO hypers =====
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: LossType = LossType.reinforce_with_baseline,
    use_std_normalization: bool = True,
    cliprange: float = 0.2,

    # ===== logging / eval =====
    eval_interval: int = 10,
    eval_max_examples: int = 1024,
    log_dir: str = "runs/grpo",
    save_interval: int = 50,
    stop_at: str = "</answer>",
    # Per-token entropy logging needs two extra (B, T, vocab) tensors per
    # microbatch; off by default to keep the policy GPU from OOMing.
    log_token_entropy: bool = False,
):
    torch.manual_seed(seed)
    random.seed(seed)

    # -------- sanity checks (handout suggested) --------
    assert train_batch_size % gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size >= group_size, \
        "train_batch_size must be >= group_size"

    assert rollout_batch_size % train_batch_size == 0, \
        "For simplicity, require rollout_batch_size divisible by train_batch_size"
    n_optimizer_updates_per_epoch = rollout_batch_size // train_batch_size

    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "train_log.jsonl"

    # -------- load data --------
    train_prompts, train_gts = build_prompts_and_gts(train_path, prompt_file)
    val_prompts, val_gts = build_prompts_and_gts(val_path, prompt_file)

    # -------- init models --------
    # Pin the policy to an explicit device: init_vllm sets the *current* CUDA
    # device to the vLLM GPU as a side effect, so relying on a bare .cuda()
    # afterwards would scatter policy tensors onto the wrong device.
    policy_device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(policy_device)
    # Trade compute for memory: the policy GPU is small, so checkpoint
    # activations to make room for the (B, T, vocab) logits spike during scoring.
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.train()

    # Full-FT AdamW keeps two fp32-equivalent moments per param (~6GB for 1.5B),
    # which alone overflows a 15GB card. 8-bit Adam stores those moments in int8
    # (~1.5GB) while preserving Adam dynamics; fall back to torch AdamW if bnb is
    # unavailable.
    try:
        from bitsandbytes.optim import PagedAdamW8bit
        optimizer = PagedAdamW8bit(
            policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
        )
        print("[optimizer] using bitsandbytes PagedAdamW8bit (8-bit optimizer states)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
        )
        print("[optimizer] bitsandbytes unavailable; using torch.optim.AdamW (fp16 states)")

    llm = init_vllm(model_id=model_id, device="cuda:1", seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    load_policy_into_vllm_instance(policy, llm)

    # -------- training loop --------
    global_step = 0
    t0 = time.time()
    loss_type = loss_type.value

    for grpo_step in range(n_grpo_steps):
        # ========== 1) sample prompts ==========
        # on-policy: rollout is regenerated at every step
        idxs = [random.randrange(0, len(train_prompts)) for _ in range(n_prompts_per_rollout_batch)]
        batch_prompts = [train_prompts[i] for i in idxs]
        batch_gts_prompt = [train_gts[i] for i in idxs]

        # ========== 2) rollout via vLLM ==========
        sp = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            n=group_size,
            stop=[stop_at],
            # r1_zero_reward_fn requires the closing </answer> tag in the response;
            # vLLM strips stop strings unless we keep them.
            include_stop_str_in_output=True,
        )
        outs = llm.generate(batch_prompts, sp)

        rollout_prompts = []
        rollout_responses = []
        repeated_gts = []
        for p, gt, out in zip(batch_prompts, batch_gts_prompt, outs):
            # out.outputs is a list with length group_size
            for o in out.outputs:
                rollout_prompts.append(p)
                rollout_responses.append(o.text)
                repeated_gts.append(gt)

        assert len(rollout_responses) == rollout_batch_size

        # ========== 3) compute group-normalized rewards (advantages) ==========
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # ========== 4) tokenize prompt+response for scoring ==========
        toks = tokenize_prompt_and_output(
            prompt_strs=rollout_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        # toks: input_ids, labels, response_mask
        input_ids = toks["input_ids"].to(policy_device)
        labels = toks["labels"].to(policy_device)
        response_mask = toks["response_mask"].to(policy_device)

        # ========== 5) (optional) old_log_probs for off-policy grpo_clip ==========
        old_log_probs = None
        if loss_type == "grpo_clip" or epochs_per_rollout_batch > 1 or train_batch_size != rollout_batch_size:
            # typical off-policy case: multiple epochs / multiple updates
            with torch.inference_mode():
                scored_old = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                old_log_probs = scored_old["log_probs"].detach()  # (B, T)
                # disable gradients for old policy logprobs
                old_log_probs.requires_grad_(False)
        
        # ========== 6) gradient updates on this rollout batch ==========
        perm = torch.randperm(rollout_batch_size, device=input_ids.device)

        # move reward/advantage to GPU and reorder according to perm
        advantages_gpu = advantages.to(policy_device)[perm].unsqueeze(-1)      # (B, 1)
        raw_rewards_gpu = raw_rewards.to(policy_device)[perm].unsqueeze(-1)    # (B, 1)
        input_ids = input_ids[perm]
        labels = labels[perm]
        response_mask = response_mask[perm]
        if old_log_probs is not None:
            old_log_probs = old_log_probs.to(policy_device)[perm]

        # actual optimization
        policy.train()
        for epoch in range(epochs_per_rollout_batch):
            for upd in range(n_optimizer_updates_per_epoch):
                start = upd * train_batch_size
                end = start + train_batch_size

                mb_input_ids = input_ids[start:end]
                mb_labels = labels[start:end]
                mb_mask = response_mask[start:end]
                mb_adv = advantages_gpu[start:end]
                mb_raw = raw_rewards_gpu[start:end]
                mb_old = old_log_probs[start:end] if old_log_probs is not None else None

                # one train_batch is split into gradient_accumulation_steps microbatches
                optimizer.zero_grad(set_to_none=True)

                # used for logging
                loss_accum = 0.0
                entropies = []

                for k in range(gradient_accumulation_steps):
                    ms = k * micro_train_batch_size
                    me = ms + micro_train_batch_size

                    micro_input_ids = mb_input_ids[ms:me]
                    micro_labels = mb_labels[ms:me]
                    micro_mask = mb_mask[ms:me]
                    micro_adv = mb_adv[ms:me]
                    micro_raw = mb_raw[ms:me]
                    micro_old = mb_old[ms:me] if mb_old is not None else None

                    scored = get_response_log_probs(
                        model=policy,
                        input_ids=micro_input_ids,
                        labels=micro_labels,
                        return_token_entropy=log_token_entropy,
                    )
                    policy_log_probs = scored["log_probs"]          # (microB, T)

                    # training step (includes masked_mean + /grad_acc_steps scaling + backward)
                    micro_loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=micro_raw if loss_type == "no_baseline" else None,
                        advantages=micro_adv if loss_type != "no_baseline" else None,
                        old_log_probs=micro_old if loss_type == "grpo_clip" else None,
                        cliprange=cliprange if loss_type == "grpo_clip" else None,
                    )

                    loss_accum += float(micro_loss.detach().cpu())
                    # average token entropy (only over response tokens), if enabled
                    if log_token_entropy:
                        ent = masked_mean(scored["token_entropy"].detach(), micro_mask, dim=None)
                        entropies.append(float(ent.cpu()))

                # gradient clipping + step
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1

                # ===== logging =====
                log_obj = {
                    "learning_rate": learning_rate,
                    "grpo_step": grpo_step,
                    "global_step": global_step,
                    "epoch": epoch,
                    "update_in_epoch": upd,
                    "loss": loss_accum,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "train_reward": float(mb_raw.mean().detach().cpu()),
                    "train_adv": float(mb_adv.mean().detach().cpu()),
                    "wall_time_sec": time.time() - t0,
                }
                if entropies:
                    log_obj["token_entropy"] = sum(entropies) / len(entropies)
                # clip fraction (if recorded in grpo_clip metadata)
                if "clip_fraction" in meta:
                    log_obj["clip_fraction"] = float(meta["clip_fraction"].detach().cpu())

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj) + "\n")                

                # ===== periodic eval =====
                if global_step % eval_interval == 0:
                    eval_sp = SamplingParams(
                        temperature=0.0,          # greedy decoding for eval
                        min_tokens=sampling_min_tokens,
                        max_tokens=sampling_max_tokens,
                        stop=[stop_at],
                        include_stop_str_in_output=True,
                    )
                    # eval_policy_with_vllm syncs the policy into vLLM and leaves it in eval()
                    val_metrics = eval_policy_with_vllm(
                        policy=policy,
                        llm=llm,
                        eval_prompts=val_prompts[:eval_max_examples],
                        eval_gts=val_gts[:eval_max_examples],
                        eval_sampling_params=eval_sp,
                        reward_fn=r1_zero_reward_fn,
                    )
                    policy.train()
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"learning_rate": learning_rate, "global_step": global_step, **val_metrics}) + "\n")

                # ===== periodic save =====
                if global_step % save_interval == 0:
                    save_dir = Path(log_dir) / f"ckpt_step_{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

        # before next rollout: synchronize latest policy weights into vLLM
        load_policy_into_vllm_instance(policy, llm)


if __name__ == "__main__":
    app()