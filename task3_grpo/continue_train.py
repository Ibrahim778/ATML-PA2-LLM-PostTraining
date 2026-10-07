"""Task 3 Step 1: GRPO continuation from the supplied midpoint (no critic).

Each update:
  1. take `prompts_per_update` prompts (fixed seeded order, identical across forks) and sample K completions each
  2. score every completion with the frozen RM; old / reference log-probs on the sampled tokens
  3. group-relative advantages (one scalar per completion, normalised within its prompt group)
  4. mask completions that hit max_completion_length (config), then `policy_epochs` clipped GRPO steps
  5. log reward, KL, within-group reward std, uninformative-group fraction, loss, grad norm, entropy, length
     plus per-completion gradient weights (used by the normalisation study)

    python -m task3_grpo.continue_train
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task2_ppo.continue_train import disable_dropout, kv_cache_on, masked_entropy_from_logits, prompt_order, upcast_trainable
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

STD_TOL = 1e-6  # same tolerance as the released advantage helper's eps


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # never cut the assistant header off a long prompt
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)      # otherwise ratio != 1 before any step
    upcast_trainable(policy)     # fp32 LoRA weights: fp16 + AdamW eps underflows to NaN
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


@torch.no_grad()
def collect_group_rollout(bundle, prompts, cfg):
    """K completions per prompt in one batch; group_ids[i] = index of the prompt that produced completion i."""
    policy, tok, K = bundle["policy"], bundle["tokenizer"], int(cfg["num_generations"])
    g = cfg["generation"]
    batch_prompts = [p for p in prompts for _ in range(K)]
    group_ids = torch.arange(len(prompts)).repeat_interleave(K)
    with kv_cache_on(policy):
        gen = batch_generate(policy, tok, batch_prompts, max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(cfg["max_completion_length"]), temperature=float(g["temperature"]),
                             top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
    # batch_generate uses inference_mode; clone so the update can backprop through these token ids.
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        gen[k] = gen[k].clone()
    args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
    mask = gen["response_mask"]
    old_logp, logits = response_token_logprobs(policy, *args)
    entropy = float(masked_entropy_from_logits(logits, mask))
    del logits
    with reference_mode(policy):
        ref_logp, _ = response_token_logprobs(policy, *args)
    rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], batch_prompts, gen["responses"]).float()
    return {**gen, "batch_prompts": batch_prompts, "group_ids": group_ids.to(mask.device), "rewards": rewards.to(mask.device),
            "old_logp": old_logp.detach(), "ref_logp": ref_logp.detach(), "rollout_entropy": entropy}


def group_stats(rewards, group_ids):
    """Within-group population std per group, uninformative = std <= tolerance."""
    stds = []
    for gid in torch.unique(group_ids):
        r = rewards[group_ids == gid]
        stds.append(float(r.std(unbiased=False)))
    stds = np.array(stds)
    return {"group_reward_std_mean": float(stds.mean()), "uninformative_group_frac": float((stds <= STD_TOL).mean())}


def gradient_weights(adv, token_mask, loss_type, max_len):
    """Per-completion share of the policy-gradient signal implied by the normalisation.

    At ratio ~ 1 the gradient of the policy term w.r.t. each token log-prob is A_k / (denom_k * N):
      grpo    denom_k = |o_k| (realised length)   dr_grpo  denom_k = max_completion_length
    Returns per-token weight and per-sequence total weight (|A_k| * n_k / denom_k / N).
    """
    n_tok = token_mask.sum(-1)
    N = adv.shape[0]
    denom = n_tok.clamp_min(1.0) if loss_type == "grpo" else torch.full_like(n_tok, float(max_len))
    per_token = adv.abs() / denom / N
    per_token = torch.where(n_tok > 0, per_token, torch.zeros_like(per_token))
    return per_token, per_token * n_tok


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, opt = bundle["policy"], bundle["optimizer"]
    n_updates, ppu, K = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["num_generations"])
    eps, beta, max_len = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    res_dir = repo_path(cfg["results_dir"])
    log_path, comp_path = res_dir / f"{run_name}_train_log.jsonl", res_dir / f"{run_name}_completions.jsonl"
    for p in (log_path, comp_path):
        if p.exists():
            p.unlink()

    rows, order = bundle["prompt_rows"], prompt_order(len(bundle["prompt_rows"]), int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    total_tokens = 0
    print(f"[{run_name}] GRPO continuation: updates={n_updates} prompts/update={ppu} K={K} loss_type={loss_type} "
          f"eps={eps} beta={beta} mask_truncated={cfg['mask_truncated_completions']}")

    pbar = tqdm(range(1, n_updates + 1), desc=f"GRPO[{run_name}]", unit="upd", dynamic_ncols=True)
    for u in pbar:
        idxs = [order[((u - 1) * ppu + j) % len(order)] for j in range(ppu)]
        ro = collect_group_rollout(bundle, [prompt_messages(rows[i]) for i in idxs], cfg)
        args = (ro["sequences"], ro["attention_mask"], ro["prompt_width"], ro["response_ids"])
        mask = ro["response_mask"]
        token_mask = mask_truncated_sequences(mask, ro["truncated"]) if cfg["mask_truncated_completions"] else mask
        adv = group_relative_advantages(ro["rewards"], ro["group_ids"])

        epochs = []
        for _ in range(int(cfg["policy_epochs"])):
            new_logp, _ = response_token_logprobs(policy, *args)
            loss, info = grpo_policy_loss(new_logp, ro["old_logp"], adv, token_mask, ro["ref_logp"], eps, beta,
                                          loss_type=loss_type, max_completion_length=max_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(gn)) and float(token_mask.sum()) > 0
            if ok:  # skip non-finite steps and steps where every completion was masked
                opt.step()
            opt.zero_grad(set_to_none=True)
            epochs.append({"loss": float(loss.detach()), "grad_norm": float(gn), "step_skipped": not ok,
                           **{k: float(v) for k, v in info.items()}})

        lengths = list(ro["response_lengths"])
        total_tokens += int(sum(lengths))
        w_tok, w_seq = gradient_weights(adv.detach(), token_mask, loss_type, max_len)
        rec = {
            "run": run_name, "update": u, "prompt_indices": idxs, "loss_type": loss_type, "K": K,
            "reward_mean": float(ro["rewards"].mean()), "reward_max": float(ro["rewards"].max()),
            **group_stats(ro["rewards"], ro["group_ids"]),
            "kl_ref_per_token": float(sampled_kl(ro["old_logp"], ro["ref_logp"], mask)),   # course helper, all valid tokens
            "kl_ref_per_sequence": float(((ro["old_logp"] - ro["ref_logp"]) * mask).sum(-1).mean()),
            "entropy": ro["rollout_entropy"],
            "response_length": float(np.mean(lengths)), "response_length_std": float(np.std(lengths)),
            "truncation_rate": float(np.mean(ro["truncated"])),
            "masked_completion_frac": float((token_mask.sum(-1) == 0).float().mean()),
            "advantage_abs_mean": float(adv.abs().mean()),
            "generated_tokens_cum": total_tokens,
            **epochs[0],
            "elapsed_s": elapsed(),
        }
        append_jsonl(log_path, rec)
        n_tok = token_mask.sum(-1)
        for i in range(len(lengths)):
            append_jsonl(comp_path, {
                "update": u, "group": int(ro["group_ids"][i]), "prompt_index": idxs[int(ro["group_ids"][i])],
                "prompt": ro["batch_prompts"][i][-1]["content"], "response": ro["responses"][i],
                "reward": float(ro["rewards"][i]), "advantage": float(adv[i]), "length": int(lengths[i]),
                "truncated": bool(ro["truncated"][i]), "masked": bool(n_tok[i] == 0),
                "grad_weight_per_token": float(w_tok[i]), "grad_weight_sequence": float(w_seq[i])})
        pbar.set_postfix(R=f"{rec['reward_mean']:.2f}", gstd=f"{rec['group_reward_std_mean']:.2f}",
                         KL=f"{rec['kl_ref_per_token']:.4f}", L=f"{rec['loss']:.3f}", gn=f"{rec['grad_norm']:.2f}",
                         len=int(rec["response_length"]))
        del ro, adv, token_mask
    pbar.close()

    policy.save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    summary = {
        "run": run_name, "config_path": config_path, "loss_type": loss_type, "updates": n_updates,
        "prompts_per_update": ppu, "num_generations": K, "policy_epochs": int(cfg["policy_epochs"]),
        "clip_epsilon": eps, "kl_beta": beta, "learning_rate": float(cfg["learning_rate"]),
        "max_completion_length": max_len, "mask_truncated_completions": bool(cfg["mask_truncated_completions"]),
        "seed": int(cfg["seed"]), "generated_tokens_total": total_tokens, "wall_clock_s": elapsed(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "adapter_path": str(out), "train_log": str(log_path), "completions_log": str(comp_path),
    }
    save_json(res_dir / f"{run_name}_train_summary.json", summary)
    print(f"[{run_name}] done: {n_updates} updates, {total_tokens} generated tokens, "
          f"{summary['wall_clock_s']:.0f}s, peak VRAM {summary['peak_vram_gb']} GB -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()