"""Task 1 evaluation: held-out DPO loss/accuracy, sampled KL, reward-model score, length stats and
qualitative candidates. Step 3 (length strata, word limits) lives in analyze_length.py and reuses these helpers.

Every condition (SFT baseline, standard DPO, beta forks, length-balanced) is evaluated with the same
prompts, decoding settings, generation cap and seed, so results are directly comparable.

Examples
--------
python -m task1_dpo.evaluate --name sft                                             # no adapter = reference
python -m task1_dpo.evaluate --name standard --adapter outputs/task1_dpo/standard
python -m task1_dpo.evaluate --name beta_0.03 --adapter outputs/task1_dpo/beta_0.03   # beta read from its train summary
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import preference_accuracy, word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


# ----------------------------------------------------------------------------------------------
# Loading / small utils
# ----------------------------------------------------------------------------------------------
def load_eval_tokenizer(cfg):
    tok = load_tokenizer(cfg["base_model"])
    tok.truncation_side = "left"  # if a prompt is too long, drop old context, never the assistant header
    return tok


def load_models(cfg, adapter: str | None, load_rm: bool = True):
    # adapter=None -> plain SFT model; reference_mode() is then a no-op and KL/margins are exactly 0.
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg) if load_rm else (None, None)
    return policy, rm, rm_tok


def run_beta(cfg, name):
    """beta the run was trained with (from its train summary), so beta forks get the right held-out loss."""
    summ = repo_path(cfg["results_dir"]) / f"{name}_train_summary.json"
    return float(load_json(summ)["beta"]) if summ.exists() else float(cfg["beta"])


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def stats(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(a.min()), "max": float(a.max())}


# ----------------------------------------------------------------------------------------------
# 1) Held-out preference metrics (teacher-forced, no generation)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def score_pairs(model, tokenizer, rows, cfg, batch_size, desc="pairs"):
    """Summed response log-probs of chosen/rejected under policy and reference, one record per pair."""
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])))
    device = next(model.parameters()).device
    model.eval()
    per_pair, idx = [], 0
    for chosen, rejected in tqdm(loader, desc=f"logp[{desc}]", unit="batch", dynamic_ncols=True):
        chosen, rejected = _to(chosen, device), _to(rejected, device)
        pol_c, _, mc = response_sequence_logprobs(model, chosen)
        pol_r, _, mr = response_sequence_logprobs(model, rejected)
        with reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)
        for i in range(pol_c.shape[0]):
            row = rows[idx]
            p = {"idx": idx, "prompt_id": row.get("prompt_id", row.get("id")),
                 "policy_logp_chosen": float(pol_c[i]), "policy_logp_rejected": float(pol_r[i]),
                 "ref_logp_chosen": float(ref_c[i]), "ref_logp_rejected": float(ref_r[i]),
                 "len_chosen": int(mc[i].sum()), "len_rejected": int(mr[i].sum())}
            p["chosen_logratio"] = p["policy_logp_chosen"] - p["ref_logp_chosen"]
            p["rejected_logratio"] = p["policy_logp_rejected"] - p["ref_logp_rejected"]
            p["margin"] = p["chosen_logratio"] - p["rejected_logratio"]  # m_theta from the manual
            per_pair.append(p)
            idx += 1
    return per_pair


def summarize_pairs(ps, beta):
    """Held-out preference metrics for any subset of per-pair records (overall or one stratum)."""
    if not ps:
        return {"n": 0}
    t = lambda k: torch.tensor([p[k] for p in ps], dtype=torch.float64)
    loss, _ = dpo_loss(t("policy_logp_chosen"), t("policy_logp_rejected"),
                       t("ref_logp_chosen"), t("ref_logp_rejected"), beta)
    return {
        "n": len(ps),
        "dpo_loss": float(loss),  # via the (fixed) objective in dpo.py
        # Accuracy on log-RATIOS, i.e. fraction with m_theta > 0 (manual definition).
        "preference_accuracy": preference_accuracy(t("chosen_logratio"), t("rejected_logratio")),
        # Diagnostic only: raw policy likelihood ranking (length-confounded, NOT the manual metric).
        "raw_policy_logp_accuracy": preference_accuracy(t("policy_logp_chosen"), t("policy_logp_rejected")),
        "margin": stats([p["margin"] for p in ps]),
        "implicit_reward_chosen_mean": float(beta * t("chosen_logratio").mean()),
        "implicit_reward_rejected_mean": float(beta * t("rejected_logratio").mean()),
    }


# ----------------------------------------------------------------------------------------------
# 2) Generation: reward score, sampled KL, length stats
# ----------------------------------------------------------------------------------------------
def generate_and_score(model, tokenizer, rm, rm_tok, prompts, cfg, batch_size, use_reference=False, desc="gen"):
    """Generate one response per prompt (policy, or reference if use_reference) and score it.

    KL uses the released sampled-response estimator on the policy's own samples: per-token
    log pi_theta - log pi_ref over valid response tokens, token-weighted mean over the whole set
    (equivalent to common.metrics.sampled_kl on the concatenated set). Per-sequence sums are also stored.
    """
    g = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"]) - max_new
    set_seed(int(cfg["seed"]))  # identical decoding RNG across conditions

    records, kl_num, kl_den = [], 0.0, 0.0
    for s in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch", dynamic_ncols=True):
        chunk = prompts[s:s + batch_size]
        with reference_mode(model) if use_reference else nullcontext():
            gen = batch_generate(model, tokenizer, chunk, max_prompt_length=max_prompt, max_new_tokens=max_new,
                                 temperature=float(g["temperature"]), top_p=float(g["top_p"]),
                                 do_sample=bool(g["do_sample"]))
        with torch.no_grad():
            if use_reference:
                kl_seq = torch.zeros(len(chunk))
            else:
                args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
                pol_tok, _ = response_token_logprobs(model, *args)
                with reference_mode(model):
                    ref_tok, _ = response_token_logprobs(model, *args)
                diff = (pol_tok - ref_tok) * gen["response_mask"]
                kl_seq = diff.sum(-1).float().cpu()
                kl_num += float(diff.sum())
                kl_den += float(gen["response_mask"].sum())
            rewards = score_reward_pairs(rm, rm_tok, chunk, gen["responses"]).cpu()

        for i, resp in enumerate(gen["responses"]):
            records.append({
                "prompt": chunk[i][-1]["content"] if chunk[i] else "",
                "response": resp,
                "reward": float(rewards[i]),
                "kl_seq": float(kl_seq[i]),
                "length_tokens": int(gen["response_lengths"][i]),
                "length_words": word_count(resp),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "truncated": bool(gen["truncated"][i]),
            })
        del gen
    summary = {
        "n": len(records),
        "reward": stats([r["reward"] for r in records]),
        "kl_per_token": (kl_num / kl_den) if kl_den else 0.0,
        "kl_per_sequence": stats([r["kl_seq"] for r in records]),
        "length_tokens": stats([r["length_tokens"] for r in records]),
        "length_words": stats([r["length_words"] for r in records]),
        "truncated_frac": float(np.mean([r["truncated"] for r in records])) if records else 0.0,
        "eos_frac": float(np.mean([r["terminated_with_eos"] for r in records])) if records else 0.0,
    }
    return summary, records


# ----------------------------------------------------------------------------------------------
# 3) Qualitative candidates (RQ3: stronger reward signal but worse response?)
# ----------------------------------------------------------------------------------------------
def qualitative_candidates(pol_recs, ref_recs, k=5):
    out = {}
    if ref_recs:
        paired = [{"prompt": p["prompt"], "policy_response": p["response"], "reference_response": r["response"],
                   "policy_reward": p["reward"], "reference_reward": r["reward"],
                   "reward_gain": p["reward"] - r["reward"],
                   "policy_len": p["length_tokens"], "reference_len": r["length_tokens"],
                   "policy_truncated": p["truncated"]}
                  for p, r in zip(pol_recs, ref_recs)]
        # Reward went up, but response got much longer or hit the cap -> inspect for verbosity / quality loss.
        cands = [x for x in paired if x["reward_gain"] > 0
                 and (x["policy_truncated"] or x["policy_len"] > 1.5 * max(1, x["reference_len"]))]
        out["reward_up_but_longer_or_truncated"] = sorted(cands, key=lambda x: -x["reward_gain"])[:k]
        out["largest_reward_gain"] = sorted(paired, key=lambda x: -x["reward_gain"])[:k]
    out["high_reward_but_truncated"] = sorted([r for r in pol_recs if r["truncated"]], key=lambda r: -r["reward"])[:k]
    return out


# ----------------------------------------------------------------------------------------------
def evaluate_run(config_path: str, adapter: str | None, name: str) -> dict:
    """Evaluate one condition on the standard held-out set; returns the results dict (also saved to disk)."""
    timer = wall_timer()
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    tok = load_eval_tokenizer(cfg)
    model, rm, rm_tok = load_models(cfg, adapter)
    beta, bs = run_beta(cfg, name), int(cfg["batch_size"])
    res_dir = repo_path(cfg["results_dir"])
    results = {"name": name, "adapter": adapter, "beta_for_loss": beta, "n_pairs": len(rows), "seed": int(cfg["seed"]),
               "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"])}}

    per_pair = score_pairs(model, tok, rows, cfg, bs, desc=name)
    results["preference"] = summarize_pairs(per_pair, beta)
    write_jsonl(res_dir / f"{name}_pairs.jsonl", per_pair)

    prompts = [prompt_messages_from_preference(r) for r in rows]
    results["generation"], pol_recs = generate_and_score(model, tok, rm, rm_tok, prompts, cfg, bs, desc=f"gen[{name}]")
    write_jsonl(res_dir / f"{name}_generations.jsonl", pol_recs)

    ref_recs = []
    if adapter:  # reference generations on the same prompts, for qualitative comparison
        results["reference_generation"], ref_recs = generate_and_score(
            model, tok, rm, rm_tok, prompts, cfg, bs, use_reference=True, desc="gen[reference]")
        write_jsonl(res_dir / f"{name}_reference_generations.jsonl", ref_recs)
    save_json(res_dir / f"{name}_qualitative_candidates.json", qualitative_candidates(pol_recs, ref_recs))

    results["wall_clock_s"] = timer()
    save_json(res_dir / f"{name}_eval.json", results)

    o, g = results["preference"], results["generation"]
    print(f"\n[{name}] pairs={o['n']} loss={o['dpo_loss']:.4f} acc={o['preference_accuracy']:.3f} "
          f"margin={o['margin']['mean']:.3f} (raw-logp acc={o['raw_policy_logp_accuracy']:.3f})")
    print(f"   reward={g['reward']['mean']:.3f}±{g['reward']['std']:.3f}  KL/token={g['kl_per_token']:.4f}  "
          f"KL/seq={g['kl_per_sequence']['mean']:.2f}  len={g['length_tokens']['mean']:.1f}±{g['length_tokens']['std']:.1f} "
          f"trunc={g['truncated_frac']:.2f}")
    print(f"   saved -> {res_dir}/{name}_eval.json  ({results['wall_clock_s']:.0f}s)")
    del model, rm
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", help="LoRA adapter dir; omit to evaluate the SFT reference itself")
    ap.add_argument("--name", default="standard", help="run name (matches train.py --run-name)")
    args = ap.parse_args()
    evaluate_run(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()