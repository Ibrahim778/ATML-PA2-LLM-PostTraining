"""Task 1 Step 2: beta sweep.

For each beta in cfg["betas"]: train a fresh LoRA fork from the original initialization on the first
cfg["short_ablation_examples"] pairs (same data, seed, optimizer, LoRA), then evaluate it under the
common protocol. Only beta changes between forks.

    python -m task1_dpo.ablate_beta
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_run
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", n)

    rows = []
    for beta in cfg["betas"]:
        name = f"beta_{beta:g}"
        adapter = f"outputs/task1_dpo/{name}"
        train = run_training(args.config, run_name=name, output_path=adapter, beta=float(beta), max_examples=n)
        clear_gpu()  # free the training model before loading for evaluation
        res = evaluate_run(args.config, adapter, name)
        p, g = res["preference"], res["generation"]
        rows.append({
            "beta": float(beta), "examples": n, "optimizer_updates": train["optimizer_updates"],
            "heldout_dpo_loss": p["dpo_loss"], "preference_accuracy": p["preference_accuracy"],
            "margin_mean": p["margin"]["mean"],
            "kl_per_token": g["kl_per_token"], "kl_per_sequence_mean": g["kl_per_sequence"]["mean"],
            "reward_mean": g["reward"]["mean"], "reward_std": g["reward"]["std"],
            "length_mean": g["length_tokens"]["mean"], "length_std": g["length_tokens"]["std"],
            "truncated_frac": g["truncated_frac"],
        })
        clear_gpu()

    out = repo_path(cfg["results_dir"]) / "beta_sweep_summary.json"
    save_json(out, rows)
    print("\nbeta    acc    loss    KL/tok   reward   len")
    for r in rows:
        print(f"{r['beta']:<6} {r['preference_accuracy']:.3f}  {r['heldout_dpo_loss']:.4f}  "
              f"{r['kl_per_token']:.4f}  {r['reward_mean']:.3f}  {r['length_mean']:.1f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()