"""Task 3 Step 3: canonical GRPO vs Dr. GRPO normalisation.

Two short continuations (cfg["fork_updates"]) from the identical midpoint with the same prompts, seed, K, reward,
beta, epsilon and generation settings; only `loss_type` differs. Then the common held-out evaluation.

Length-conditioned gradient statistics (from each run's per-completion log). At ratio ~ 1 the policy-term gradient
gives every token of completion k the weight |A_k| / (denom_k * N), with denom_k = |o_k| (grpo) or
max_completion_length (dr_grpo). We report, for unmasked completions:
  * corr(length, per-token weight) and corr(length, per-sequence weight = per-token weight * length)
  * share of total gradient weight going to "long" completions, long = length > L_split, where L_split is the
    median length of all unmasked completions pooled over both runs (one threshold, defined once)
  * per-length-tercile means of per-token weight, sequence weight, |advantage|, reward
  * update 1 separately: both runs sample identical completions there (same seed, same starting weights), so it
    isolates the pure normalisation effect on identical data.

    python -m task3_grpo.compare_normalization
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_run

LOSS_TYPES = ("grpo", "dr_grpo")
TRAJ_KEYS = ("reward_mean", "kl_ref_per_token", "entropy", "response_length", "group_reward_std_mean",
             "uninformative_group_frac", "loss", "grad_norm", "masked_completion_frac")


def ensure_midpoint_eval(config_path, cfg):
    if not (repo_path(cfg["results_dir"]) / "midpoint_eval.json").exists():
        evaluate_run(config_path, cfg["paths"]["grpo_midpoint_policy"], "midpoint")


def run_fork(config_path, cfg, loss_type):
    name = f"norm_{loss_type}"
    out = str(Path(cfg["output"]).parent / name)
    if not (repo_path(out) / "adapter_config.json").exists():
        run_grpo(config_path, output=out, updates=int(cfg["fork_updates"]), loss_type=loss_type, run_name=name)
        clear_gpu()
    else:
        print(f"[{name}] adapter exists at {out}; skipping training")
    ev = evaluate_run(config_path, out, name)
    res_dir = repo_path(cfg["results_dir"])
    return name, ev, read_jsonl(res_dir / f"{name}_train_log.jsonl"), read_jsonl(res_dir / f"{name}_completions.jsonl"), \
        load_json(res_dir / f"{name}_train_summary.json")


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def length_stats(comps, l_split, l_terciles):
    c = [x for x in comps if not x["masked"]]
    if not c:
        return {"n": 0}
    L = np.array([x["length"] for x in c], float)
    wt = np.array([x["grad_weight_per_token"] for x in c])
    ws = np.array([x["grad_weight_sequence"] for x in c])
    long = L > l_split
    out = {
        "n": len(c),
        "corr_length_vs_per_token_weight": _corr(L, wt),
        "corr_length_vs_sequence_weight": _corr(L, ws),
        "long_share_of_gradient_weight": float(ws[long].sum() / ws.sum()) if ws.sum() > 0 else float("nan"),
        "long_share_of_completions": float(long.mean()),
        "per_token_weight_long_over_short": float(wt[long].mean() / wt[~long].mean()) if long.any() and (~long).any() and wt[~long].mean() > 0 else float("nan"),
        "by_length_tercile": {},
    }
    edges = [-np.inf, *l_terciles, np.inf]
    for name, lo, hi in zip(("short", "mid", "long"), edges[:-1], edges[1:]):
        sel = (L > lo) & (L <= hi)
        if sel.any():
            out["by_length_tercile"][name] = {
                "n": int(sel.sum()), "mean_length": float(L[sel].mean()),
                "per_token_weight": float(wt[sel].mean()), "sequence_weight": float(ws[sel].mean()),
                "abs_advantage": float(np.mean([abs(x["advantage"]) for x, s in zip(c, sel) if s])),
                "reward": float(np.mean([x["reward"] for x, s in zip(c, sel) if s])),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    ensure_midpoint_eval(args.config, cfg)
    runs = {lt: run_fork(args.config, cfg, lt) for lt in LOSS_TYPES}

    pooled = [x["length"] for lt in LOSS_TYPES for x in runs[lt][3] if not x["masked"]]
    l_split = float(np.median(pooled))
    l_terc = [float(v) for v in np.quantile(pooled, [1 / 3, 2 / 3])]

    summary = {"length_split_median": l_split, "length_tercile_cutpoints": l_terc,
               "midpoint": load_json(repo_path(cfg["results_dir"]) / "midpoint_eval.json"), "conditions": {}}
    for lt in LOSS_TYPES:
        name, ev, log, comps, train = runs[lt]
        summary["conditions"][lt] = {
            "name": name, "updates": len(log), "generated_tokens": train["generated_tokens_total"],
            "wall_clock_s": train["wall_clock_s"], "peak_vram_gb": train["peak_vram_gb"],
            "heldout": {"rm_score_mean": ev["rm_score"]["mean"], "rm_score_std": ev["rm_score"]["std"],
                        "kl_per_token": ev["kl_per_token"], "entropy_per_token": ev["entropy_per_token"],
                        "length_mean": ev["length_tokens"]["mean"], "length_std": ev["length_tokens"]["std"],
                        "eos_rate": ev["eos_rate"], "truncation_rate": ev["truncation_rate"]},
            "length_conditioned_all_updates": length_stats(comps, l_split, l_terc),
            "length_conditioned_update1": length_stats([x for x in comps if x["update"] == 1], l_split, l_terc),
            "trajectories": {k: [x[k] for x in log] for k in TRAJ_KEYS},
        }
    out = repo_path(cfg["results_dir"]) / "normalization_summary.json"
    save_json(out, summary)

    print(f"\nlong = length > {l_split:.0f} tokens (pooled median)")
    print("loss_type  heldout RM   KL/tok    len     | corr(len, w_tok)  corr(len, w_seq)  long share of grad  (update 1)")
    for lt in LOSS_TYPES:
        c = summary["conditions"][lt]
        h, a, u1 = c["heldout"], c["length_conditioned_all_updates"], c["length_conditioned_update1"]
        print(f"{lt:<9} {h['rm_score_mean']:>9.3f} {h['kl_per_token']:>9.5f} {h['length_mean']:>7.1f}   | "
              f"{a.get('corr_length_vs_per_token_weight', float('nan')):>15.3f} {a.get('corr_length_vs_sequence_weight', float('nan')):>17.3f}"
              f" {a.get('long_share_of_gradient_weight', float('nan')):>18.3f}  ({u1.get('long_share_of_gradient_weight', float('nan')):.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()