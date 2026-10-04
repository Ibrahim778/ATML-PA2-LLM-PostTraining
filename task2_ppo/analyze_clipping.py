"""Task 2 Step 2: clipping study.

Part A, cached batch (immediate geometry). Rebuild the fixed cached rollout batch, compute its advantages/returns,
then for each epsilon start from a fresh copy of the midpoint and run the configured `ppo_epochs` on that one
batch. Before every epoch (and after the last step) measure, over valid response tokens:
  * clip fraction: ratio outside [1-eps, 1+eps]  (manual definition)
  * affected-token fraction: tokens whose surrogate gradient is removed by clipping,
    i.e. (ratio > 1+eps and A > 0) or (ratio < 1-eps and A < 0)
  * unclipped surrogate mean(ratio * A) and the clipped objective from task2_ppo.ppo.ppo_policy_loss
At epoch 1 the policy equals the rollout policy (ratio ~ 1), so the interesting numbers are from epoch 2 on.

Part B, matched short forks. For each epsilon, cfg["fork_updates"] continuation updates from the same midpoint,
same prompts/seed/beta_KL, then the common held-out evaluation and stability statistics.

    python -m task2_ppo.analyze_clipping
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, load_value_model, trainable_parameters
from task2_ppo.ablate_kl import ensure_midpoint_eval, run_fork
from task2_ppo.continue_train import disable_dropout, response_values
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

MICRO_BATCH = 4  # sequences per forward pass (memory); losses are token-weighted so results don't depend on it


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _first(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _vec(x):
    return torch.as_tensor(x, dtype=torch.float32).flatten()


# ----------------------------------------------------------------------------------------------
# Part A: rebuild the cached batch
# ----------------------------------------------------------------------------------------------
def build_cached_batch(rows, cfg, tok):
    """Left-padded prompt | right-padded response tensors, matching common.generation.batch_generate's layout."""
    pool = read_jsonl(cfg["paths"]["rl_prompt_train"])
    prompts, p_ids, r_ids, old, ref, notes = [], [], [], [], [], []
    for row in rows:
        msgs = row.get("messages") if isinstance(row.get("messages"), list) else (
            row["prompt"] if isinstance(row.get("prompt"), list) else prompt_messages(pool[int(row["source_index"])]))
        prompts.append(msgs)
        pid = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)[-int(cfg["max_prompt_length"]):]
        o, rf = _vec(row["old_logprobs"]), _vec(row["ref_logprobs"])
        rid = _first(row, "response_ids", "response_token_ids")
        if rid is None:
            rid = tok(row["response"], add_special_tokens=False)["input_ids"]
            if len(rid) == len(o) - 1:            # cached log-probs include the EOS token
                rid = rid + [tok.eos_token_id]
        rid = [int(t) for t in (rid.tolist() if torch.is_tensor(rid) else rid)]
        n = min(len(rid), len(o), len(rf))
        if not (len(rid) == len(o) == len(rf)):
            notes.append({"source_index": int(row["source_index"]), "response_tokens": len(rid),
                          "old_logprobs": len(o), "ref_logprobs": len(rf), "used": n})
        p_ids.append(pid); r_ids.append(rid[:n]); old.append(o[:n]); ref.append(rf[:n])

    pw, rw, pad = max(map(len, p_ids)), max(map(len, r_ids)), tok.pad_token_id
    B = len(rows)
    seq = torch.full((B, pw + rw), pad, dtype=torch.long)
    attn = torch.zeros((B, pw + rw), dtype=torch.long)
    rids = torch.full((B, rw), pad, dtype=torch.long)
    mask, old_t, ref_t = torch.zeros(B, rw), torch.zeros(B, rw), torch.zeros(B, rw)
    for i in range(B):
        lp, lr = len(p_ids[i]), len(r_ids[i])
        seq[i, pw - lp:pw] = torch.tensor(p_ids[i]); seq[i, pw:pw + lr] = torch.tensor(r_ids[i])
        attn[i, pw - lp:pw + lr] = 1
        rids[i, :lr] = torch.tensor(r_ids[i]); mask[i, :lr] = 1
        old_t[i, :lr] = old[i]; ref_t[i, :lr] = ref[i]
    return {"prompts": prompts, "sequences": seq, "attention_mask": attn, "prompt_width": pw, "response_ids": rids,
            "response_mask": mask, "old_logp": old_t, "ref_logp": ref_t, "length_mismatches": notes}


def _micro(batch, s, device):
    sl = slice(s, s + MICRO_BATCH)
    return (batch["sequences"][sl].to(device), batch["attention_mask"][sl].to(device), batch["prompt_width"],
            batch["response_ids"][sl].to(device)), sl


@torch.no_grad()
def cached_advantages(batch, rows, cfg, tok, device):
    """Use cached reward/values/advantages when the cache provides them; otherwise recompute with the midpoint critic + RM."""
    B, mask = len(rows), batch["response_mask"]
    adv_cached = [_first(r, "advantages") for r in rows]
    ret_cached = [_first(r, "returns") for r in rows]
    source = {}
    if all(a is not None for a in adv_cached) and all(r is not None for r in ret_cached):
        adv, ret = torch.zeros_like(mask), torch.zeros_like(mask)
        for i in range(B):
            n = int(mask[i].sum())
            adv[i, :n], ret[i, :n] = _vec(adv_cached[i])[:n], _vec(ret_cached[i])[:n]
        source["advantages"] = "cache"
    else:
        rew = [_first(r, "task_reward", "reward", "rm_score", "score") for r in rows]
        if all(x is not None for x in rew):
            task_reward = torch.tensor([float(x) for x in rew])
            source["reward"] = "cache"
        else:
            rm, rm_tok = load_reward_model(cfg)
            scores = score_reward_pairs(rm, rm_tok, batch["prompts"], [r["response"] for r in rows],
                                        max_length=int(cfg["reward_max_length"])).cpu()
            eos = []
            for i, r in enumerate(rows):
                flag = _first(r, "terminated_with_eos", "eos", "has_eos")
                last_tok = batch["response_ids"][i, int(mask[i].sum()) - 1]
                eos.append(bool(flag) if flag is not None else bool(last_tok == tok.eos_token_id))
            eos = torch.tensor(eos)
            task_reward = scores - float(cfg["missing_eos_penalty"]) * (~eos).float()
            source["reward"] = "reward model (recomputed)"
            del rm; clear_gpu()

        vals = [_first(r, "values", "old_values", "value") for r in rows]
        if all(v is not None for v in vals):
            values = torch.zeros_like(mask)
            for i in range(B):
                n = int(mask[i].sum()); values[i, :n] = _vec(vals[i])[:n]
            source["values"] = "cache"
        else:
            vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
            values = torch.zeros_like(mask)
            for s in range(0, B, MICRO_BATCH):
                (seq, attn, pw, rid), sl = _micro(batch, s, device)
                values[sl] = response_values(vm, seq, attn, pw, rid.shape[1]).cpu()
            values *= mask
            source["values"] = "midpoint critic (recomputed)"
            del vm; clear_gpu()

        rewards = shaped_rewards(task_reward, batch["old_logp"], batch["ref_logp"], mask, float(cfg["kl_beta"]))
        adv, ret = compute_gae(rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
        source["advantages"] = "GAE from shaped rewards"
    return normalize_advantages(adv, mask), ret, source


def geometry(ratio, adv, mask, eps):
    m = mask.bool()
    r, a = ratio[m], adv[m]
    outside = (r < 1 - eps) | (r > 1 + eps)
    affected = ((r > 1 + eps) & (a > 0)) | ((r < 1 - eps) & (a < 0))
    return {"clip_fraction": float(outside.float().mean()), "affected_token_fraction": float(affected.float().mean()),
            "unclipped_surrogate": float((r * a).mean()), "ratio_mean": float(r.mean()),
            "ratio_min": float(r.min()), "ratio_max": float(r.max())}


def cached_batch_study(config_path, cfg, eps_values):
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows), "| cache keys:", sorted(rows[0].keys()))
    batch = build_cached_batch(rows, cfg, tok)
    if batch["length_mismatches"]:
        print(f"WARNING: {len(batch['length_mismatches'])} rows had response/log-prob length mismatches (truncated to the shorter).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv, ret, source = cached_advantages(batch, rows, cfg, tok, device)
    mask, old = batch["response_mask"], batch["old_logp"]
    total_tok = float(mask.sum())

    def forward_ratios(policy, grad, eps):
        """Per-token new log-probs for the whole batch (micro-batched); optionally backprop the PPO loss."""
        new = torch.zeros_like(mask)
        loss_total = 0.0
        for s in range(0, len(rows), MICRO_BATCH):
            args, sl = _micro(batch, s, device)
            with torch.set_grad_enabled(grad):
                nl, _ = response_token_logprobs(policy, *args)
                if grad:
                    m = mask[sl].to(device)
                    loss, _, _ = ppo_policy_loss(nl, old[sl].to(device), adv[sl].to(device), m, eps=eps)
                    w = float(m.sum()) / total_tok           # token-weighted == one masked mean over the batch
                    (loss * w).backward()
                    loss_total += float(loss) * w
            new[sl] = nl.detach().float().cpu()
        return torch.exp(new - old) * mask + (1 - mask), loss_total

    results = {"n_rows": len(rows), "n_tokens": int(total_tok), "advantage_source": source,
               "length_mismatches": batch["length_mismatches"], "epsilons": {}}
    for eps in eps_values:
        set_seed(int(cfg["seed"]))
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy)
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        per_epoch = []
        for ep in tqdm(range(1, int(cfg["ppo_epochs"]) + 1), desc=f"cached[eps={eps}]", dynamic_ncols=True):
            opt.zero_grad(set_to_none=True)
            ratio, loss = forward_ratios(policy, grad=True, eps=eps)
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            opt.step()
            per_epoch.append({"epoch": ep, "clipped_objective_loss": loss, "grad_norm": float(gn), **geometry(ratio, adv, mask, eps)})
        ratio_after, _ = forward_ratios(policy, grad=False, eps=eps)
        results["epsilons"][str(eps)] = {"before_each_epoch": per_epoch, "after_last_step": geometry(ratio_after, adv, mask, eps)}
        del policy, opt
        clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps_values = [float(e) for e in cfg["clip_values"]]
    print("Required epsilon values:", eps_values)
    res_dir = repo_path(cfg["results_dir"])

    # Part A
    cached = cached_batch_study(args.config, cfg, eps_values)
    save_json(res_dir / "clipping_cached_batch.json", cached)
    print("\neps    epoch  clip_frac  affected  ratio[min,max]")
    for e, d in cached["epsilons"].items():
        for row in d["before_each_epoch"] + [{"epoch": "post", **d["after_last_step"]}]:
            print(f"{e:<6} {row['epoch']!s:<6} {row['clip_fraction']:>8.4f} {row['affected_token_fraction']:>9.4f}  "
                  f"[{row['ratio_min']:.3f}, {row['ratio_max']:.3f}]")

    # Part B
    ensure_midpoint_eval(args.config, cfg)
    forks = [run_fork(args.config, cfg, f"clip_{e:g}", clip_epsilon=e) for e in eps_values]
    save_json(res_dir / "clipping_forks_summary.json", {"forks": forks})
    print("\neps    heldout RM   KL/tok    len     train clip_frac  max_step_KL  frac_grad_clipped")
    for f in forks:
        h, s = f["heldout"], f["stability"]
        print(f"{f['clip_epsilon']:<6} {h['rm_score_mean']:>8.3f} {h['kl_per_token']:>9.4f} {h['length_mean']:>7.1f}"
              f"   {np.mean(f['trajectories']['clip_fraction_mean']):>10.4f}   {s['max_update_approx_kl']:>10.5f}"
              f"   {s['frac_updates_grad_clipped']:>8.2f}")
    print(f"saved -> {res_dir}/clipping_cached_batch.json, clipping_forks_summary.json")


if __name__ == "__main__":
    main()