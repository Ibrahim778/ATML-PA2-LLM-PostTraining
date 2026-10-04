"""Task 2 Step 1: PPO continuation from the supplied midpoint (policy LoRA + matched critic).

Each update:
  1. sample prompts (fixed seeded order, identical across forks) and generate on-policy responses
  2. score with the frozen RM (+ missing-EOS penalty), compute old / reference log-probs and old values
  3. KL-shaped token rewards -> GAE advantages / returns
  4. `ppo_epochs` passes of clipped policy loss + value MSE, separate optimizers
  5. log reward, KL, losses, entropy, grad norms, clip fraction, length, critic diagnostics

    python -m task2_ppo.continue_train                       # standard 20-update continuation
"""
from __future__ import annotations

import argparse
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # never cut the assistant header off a long prompt
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ----------------------------------------------------------------------------------------------
# Helpers (also used by analyze_clipping.py)
# ----------------------------------------------------------------------------------------------
def disable_dropout(model):
    """LoRA dropout would make pi_new != pi_old even before any update, inflating ratios/clip fraction."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


@contextmanager
def kv_cache_on(model):
    """load_policy(trainable=True) disables the KV cache; re-enable it for fast generation only."""
    cfgs = {id(c): c for c in (getattr(model, "config", None), getattr(getattr(model, "base_model", None), "config", None)) if c is not None}
    old = {k: c.use_cache for k, c in cfgs.items()}
    for c in cfgs.values():
        c.use_cache = True
    try:
        yield
    finally:
        for k, c in cfgs.items():
            c.use_cache = old[k]


def masked_entropy_from_logits(logits, mask):
    logp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return masked_mean(ent, mask)


def response_values(value_model, sequences, attention_mask, prompt_width, n_steps):
    """V(s_t) for each response step t: the value read at the position just before token t."""
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1: prompt_width - 1 + n_steps].float()


def explained_variance(pred, target, mask):
    m = mask.bool()
    y, p = target[m], pred[m]
    var_y = y.var(unbiased=False)
    return float(1.0 - (y - p).var(unbiased=False) / var_y) if var_y > 0 else float("nan")


def prompt_order(n_rows, seed):
    """Same prompt sequence for every run/fork -> matched comparisons."""
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    return order


@torch.no_grad()
def collect_rollout(bundle, prompts, cfg):
    policy, value_model, tok = bundle["policy"], bundle["value_model"], bundle["tokenizer"]
    g = cfg["generation"]
    with kv_cache_on(policy):
        gen = batch_generate(policy, tok, prompts, max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(cfg["max_response_length"]), temperature=float(g["temperature"]),
                             top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
    args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
    mask = gen["response_mask"]
    old_logp, logits = response_token_logprobs(policy, *args)
    entropy = masked_entropy_from_logits(logits, mask)
    del logits
    with reference_mode(policy):
        ref_logp, _ = response_token_logprobs(policy, *args)

    rm_score = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], prompts, gen["responses"],
                                  max_length=int(cfg["reward_max_length"])).to(mask.device)
    no_eos = torch.tensor([not e for e in gen["terminated_with_eos"]], device=mask.device, dtype=torch.float32)
    task_reward = rm_score - float(cfg["missing_eos_penalty"]) * no_eos

    old_values = response_values(value_model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], mask.shape[1])
    return {**gen, "old_logp": old_logp.detach(), "ref_logp": ref_logp.detach(), "old_values": old_values.detach(),
            "rm_score": rm_score, "task_reward": task_reward, "rollout_entropy": float(entropy)}


def ppo_update(bundle, ro, advantages, returns, cfg, eps):
    """`ppo_epochs` passes over the rollout batch. Returns per-epoch diagnostics."""
    policy, value_model = bundle["policy"], bundle["value_model"]
    args = (ro["sequences"], ro["attention_mask"], ro["prompt_width"], ro["response_ids"])
    mask, max_norm = ro["response_mask"], float(cfg["max_grad_norm"])
    out = []
    for _ in range(int(cfg["ppo_epochs"])):
        new_logp, _ = response_token_logprobs(policy, *args)
        p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, ro["old_logp"], advantages, mask, eps=eps)
        bundle["policy_optimizer"].zero_grad(set_to_none=True)
        p_loss.backward()
        p_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm)
        bundle["policy_optimizer"].step()

        new_v = response_values(value_model, ro["sequences"], ro["attention_mask"], ro["prompt_width"], mask.shape[1])
        v_loss = value_mse_loss(new_v, returns, mask)
        bundle["value_optimizer"].zero_grad(set_to_none=True)
        (float(cfg["value_coef"]) * v_loss).backward()
        v_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm)
        bundle["value_optimizer"].step()

        log_r = torch.log(ratio.clamp_min(1e-12))
        out.append({
            "policy_loss": float(p_loss), "value_loss": float(v_loss),
            "clip_fraction": float(clip_frac),
            "approx_kl_old_new": float(masked_mean((ratio - 1) - log_r, mask)),   # k3 estimator
            "ratio_mean": float(masked_mean(ratio, mask)),
            "ratio_max": float(ratio[mask.bool()].max()), "ratio_min": float(ratio[mask.bool()].min()),
            "policy_grad_norm": float(p_gn), "value_grad_norm": float(v_gn),
        })
    return out


# ----------------------------------------------------------------------------------------------
def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    n_updates, ppu = int(cfg["updates"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    res_dir = repo_path(cfg["results_dir"])
    log_path, ro_path = res_dir / f"{run_name}_train_log.jsonl", res_dir / f"{run_name}_rollouts.jsonl"
    for p in (log_path, ro_path):
        if p.exists():
            p.unlink()

    disable_dropout(bundle["policy"])
    disable_dropout(bundle["value_model"])
    rows, order = bundle["prompt_rows"], prompt_order(len(bundle["prompt_rows"]), int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    total_tokens = 0
    print(f"[{run_name}] PPO continuation: updates={n_updates} prompts/update={ppu} eps={eps} beta_kl={beta}")

    pbar = tqdm(range(1, n_updates + 1), desc=f"PPO[{run_name}]", unit="upd", dynamic_ncols=True)
    for u in pbar:
        idxs = [order[((u - 1) * ppu + j) % len(order)] for j in range(ppu)]
        prompts = [prompt_messages(rows[i]) for i in idxs]

        ro = collect_rollout(bundle, prompts, cfg)
        mask = ro["response_mask"]
        rewards = shaped_rewards(ro["task_reward"], ro["old_logp"], ro["ref_logp"], mask, beta)
        adv, returns = compute_gae(rewards, ro["old_values"], mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
        adv_n = normalize_advantages(adv, mask)

        epochs = ppo_update(bundle, ro, adv_n, returns, cfg, eps)
        lengths = ro["response_lengths"]
        total_tokens += int(sum(lengths))
        kl_tok = ((ro["old_logp"] - ro["ref_logp"]) * mask)
        rec = {
            "run": run_name, "update": u, "prompt_indices": idxs, "clip_epsilon": eps, "kl_beta": beta,
            "reward_rm": float(ro["rm_score"].mean()),            # learned reward model score
            "reward_task": float(ro["task_reward"].mean()),       # incl. missing-EOS penalty
            "reward_shaped_total": float((rewards * mask).sum(-1).mean()),
            "kl_ref_per_token": float(sampled_kl(ro["old_logp"], ro["ref_logp"], mask)),   # course helper
            "kl_ref_per_sequence": float(kl_tok.sum(-1).mean()),
            "entropy": ro["rollout_entropy"],
            "response_length": float(np.mean(lengths)),
            "eos_rate": float(np.mean(ro["terminated_with_eos"])),
            "truncation_rate": float(np.mean(ro["truncated"])),
            "value_mean": float(masked_mean(ro["old_values"], mask)),
            "return_mean": float(masked_mean(returns, mask)),
            "advantage_mean_raw": float(masked_mean(adv, mask)),
            "critic_explained_variance": explained_variance(ro["old_values"], returns, mask),
            "generated_tokens_cum": total_tokens,
            # first-epoch values are the "on-batch" view; last-epoch shows how far the update moved
            **{k: v for k, v in epochs[0].items()},
            **{f"{k}_last_epoch": v for k, v in epochs[-1].items()},
            "clip_fraction_mean": float(np.mean([e["clip_fraction"] for e in epochs])),
            "elapsed_s": elapsed(),
        }
        append_jsonl(log_path, rec)
        for i, p in enumerate(prompts):
            append_jsonl(ro_path, {"update": u, "prompt": p[-1]["content"], "response": ro["responses"][i],
                                   "rm_score": float(ro["rm_score"][i]), "length": int(lengths[i]),
                                   "eos": bool(ro["terminated_with_eos"][i])})
        pbar.set_postfix(R=f"{rec['reward_rm']:.2f}", KL=f"{rec['kl_ref_per_token']:.4f}",
                         pl=f"{rec['policy_loss']:.3f}", vl=f"{rec['value_loss']:.3f}",
                         clip=f"{rec['clip_fraction_mean']:.3f}", len=int(rec["response_length"]))
        del ro, rewards, adv, adv_n, returns
    pbar.close()

    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    bundle["value_model"].save_pretrained(str(out) + "_value")
    summary = {
        "run": run_name, "config_path": config_path, "updates": n_updates, "prompts_per_update": ppu,
        "ppo_epochs": int(cfg["ppo_epochs"]), "clip_epsilon": eps, "kl_beta": beta,
        "policy_learning_rate": float(cfg["policy_learning_rate"]), "seed": int(cfg["seed"]),
        "generated_tokens_total": total_tokens,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "adapter_path": str(out), "value_path": str(out) + "_value", "train_log": str(log_path),
    }
    save_json(res_dir / f"{run_name}_train_summary.json", summary)
    print(f"[{run_name}] done: {n_updates} updates, {total_tokens} generated tokens, "
          f"{summary['wall_clock_s']:.0f}s, peak VRAM {summary['peak_vram_gb']} GB -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()