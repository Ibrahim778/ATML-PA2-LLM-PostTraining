"""Task 2 Step 3: KL-pressure / reward-overoptimization study.

For each beta_KL in cfg["kl_values"]: a short continuation (cfg["fork_updates"]) from the exact same midpoint
(policy + critic), same prompt sequence, seed and clip epsilon; then the common held-out evaluation.

    python -m task2_ppo.ablate_kl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_run

TRAJ_KEYS = ("reward_rm", "kl_ref_per_token", "entropy", "response_length", "policy_loss", "value_loss",
             "clip_fraction_mean", "policy_grad_norm", "approx_kl_old_new_last_epoch", "critic_explained_variance")


def ensure_midpoint_eval(config_path, cfg):
    """Baseline row + reference generations for qualitative comparisons (evaluated once)."""
    if not (repo_path(cfg["results_dir"]) / "midpoint_eval.json").exists():
        evaluate_run(config_path, cfg["paths"]["ppo_midpoint_policy"], "midpoint")


def stability_stats(log, max_grad_norm):
    """Clearly-defined stability statistics over a fork's updates."""
    r = np.array([x["reward_rm"] for x in log])
    return {
        "reward_std_over_updates": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "max_update_approx_kl": float(max(x["approx_kl_old_new_last_epoch"] for x in log)),   # largest single policy step
        "mean_ratio_max": float(np.mean([x["ratio_max_last_epoch"] for x in log])),
        "frac_updates_grad_clipped": float(np.mean([x["policy_grad_norm"] > max_grad_norm for x in log])),
        "policy_loss_std": float(np.std([x["policy_loss"] for x in log], ddof=1)) if len(log) > 1 else 0.0,
    }


def run_fork(config_path, cfg, name, **overrides):
    """Train one short fork from the midpoint (skipped if its adapter exists), evaluate it, summarize."""
    out = str(Path(cfg["output"]).parent / name)
    if not (repo_path(out) / "adapter_config.json").exists():
        run_ppo(config_path, output=out, updates=int(cfg["fork_updates"]), run_name=name, **overrides)
        clear_gpu()
    else:
        print(f"[{name}] adapter exists at {out}; skipping training")
    ev = evaluate_run(config_path, out, name)
    res_dir = repo_path(cfg["results_dir"])
    log = read_jsonl(res_dir / f"{name}_train_log.jsonl")
    train = load_json(res_dir / f"{name}_train_summary.json")
    return {
        "name": name, **overrides, "updates": len(log), "generated_tokens": train["generated_tokens_total"],
        "wall_clock_s": train["wall_clock_s"], "peak_vram_gb": train["peak_vram_gb"],
        "heldout": {"rm_score_mean": ev["rm_score"]["mean"], "rm_score_std": ev["rm_score"]["std"],
                    "task_reward_mean": ev["task_reward"]["mean"], "kl_per_token": ev["kl_per_token"],
                    "entropy_per_token": ev["entropy_per_token"], "length_mean": ev["length_tokens"]["mean"],
                    "length_std": ev["length_tokens"]["std"], "eos_rate": ev["eos_rate"],
                    "truncation_rate": ev["truncation_rate"]},
        "stability": stability_stats(log, float(cfg["max_grad_norm"])),
        "trajectories": {k: [x[k] for x in log] for k in TRAJ_KEYS},
    }


def first_divergence(test, ref, k=1.0):
    """First update where `test` departs from the matched `ref` trajectory by > k * std(ref). None if never."""
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    scale = ref.std(ddof=1) if len(ref) > 1 and ref.std(ddof=1) > 0 else max(abs(ref).mean(), 1e-8)
    hit = np.nonzero(np.abs(test - ref) > k * scale)[0]
    return int(hit[0]) + 1 if hit.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    ensure_midpoint_eval(args.config, cfg)
    forks = [run_fork(args.config, cfg, f"kl_{b:g}", kl_beta=float(b)) for b in cfg["kl_values"]]

    # RQ2: with matched prompts, when does each observable of a weaker-KL fork first separate from the
    # reference-config fork (beta = cfg["kl_beta"])? Heuristic threshold: 1 std of the reference trajectory.
    ref = next(f for f in forks if f["kl_beta"] == float(cfg["kl_beta"]))
    for f in forks:
        f["first_divergence_vs_reference_beta"] = None if f is ref else {
            k: first_divergence(f["trajectories"][k], ref["trajectories"][k])
            for k in ("reward_rm", "kl_ref_per_token", "entropy", "response_length")}

    out = repo_path(cfg["results_dir"]) / "kl_ablation_summary.json"
    save_json(out, {"midpoint": load_json(repo_path(cfg["results_dir"]) / "midpoint_eval.json"), "forks": forks})
    print("\nbeta_KL  heldout RM   KL/tok    H/tok   len     | train reward(first->last)   KL(first->last)")
    for f in forks:
        h, t = f["heldout"], f["trajectories"]
        print(f"{f['kl_beta']:<8} {h['rm_score_mean']:>8.3f} {h['kl_per_token']:>9.4f} {h['entropy_per_token']:>7.3f} "
              f"{h['length_mean']:>6.1f}   | {t['reward_rm'][0]:.3f}->{t['reward_rm'][-1]:.3f}"
              f"            {t['kl_ref_per_token'][0]:.4f}->{t['kl_ref_per_token'][-1]:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()