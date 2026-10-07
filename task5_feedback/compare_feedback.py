from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

POLICIES = ("sft", "rlvr", "rlaif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    od = repo_path(cfg["results_dir"]) / "task5_feedback"
    need = {k: od / f for k, f in (("gsm", "gsm_eval.json"), ("transfer", "transfer_eval.json"),
                                    ("diag", "diagnostics_summary.json"))}
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Run evaluate_math (--dataset gsm and transfer) and score_perturbations first; missing: "
                                + ", ".join(missing))
    gsm, tr, diag = (load_json(need[k]) for k in ("gsm", "transfer", "diag"))

    # In-domain vs out-of-domain per policy
    policies = {}
    for p in POLICIES:
        g, t = gsm["policies"][p], tr["policies"][p]
        policies[p] = {
            "gsm_accuracy": g["exact_accuracy"], "transfer_accuracy": t["exact_accuracy"],
            "accuracy_drop": g["exact_accuracy"] - t["exact_accuracy"],
            "relative_drop": (g["exact_accuracy"] - t["exact_accuracy"]) / g["exact_accuracy"] if g["exact_accuracy"] else float("nan"),
            "gsm_format": g["format_compliance"], "transfer_format": t["format_compliance"],
            "gsm_length": g["length_mean"], "transfer_length": t["length_mean"],
            "gsm_failure_types": g["failure_types"], "transfer_failure_types": t["failure_types"],
        }
        if p != "sft":
            for ds, src in (("gsm", gsm), ("transfer", tr)):
                pw = src["pairwise"][f"{p}_vs_sft"]
                policies[p][f"{ds}_ai_win_rate_vs_sft"] = pw["win_rate_a"]
                policies[p][f"{ds}_ai_wtl_vs_sft"] = [pw["wins_a"], pw["ties"], pw["losses_a"]]
                policies[p][f"{ds}_accuracy_gain_vs_sft"] = src["policies"][p]["exact_accuracy"] - src["policies"]["sft"]["exact_accuracy"]
            policies[p]["transfer_ai_win_rate_drop"] = policies[p]["gsm_ai_win_rate_vs_sft"] - policies[p]["transfer_ai_win_rate_vs_sft"]
            gi = policies[p]["gsm_accuracy_gain_vs_sft"]
            policies[p]["gain_retained_out_of_domain"] = (policies[p]["transfer_accuracy_gain_vs_sft"] / gi) if gi else float("nan")

    # Feedback-source comparison (coverage, noise, exploitability, cost) from the evidence above.
    P = diag["pairs"]
    pw_gsm = [gsm["pairwise"][f"{p}_vs_sft"] for p in ("rlvr", "rlaif")]
    sources = {
        "coverage": {
            "rlvr_tie_rate_reasoning_only": P["reasoning_corruption"]["rlvr"]["tie_rate"],
            "rlaif_distinguishes_reasoning_only": 1 - P["reasoning_corruption"]["rlaif"]["tie_rate"],
            "gsm_judge_distinguishes_verifier_ties": [x["judge_distinguishes_verifier_ties"] for x in pw_gsm],
        },
        "noise": {
            "rlaif_order_inconsistency": {k: 1 - v["rlaif_order_consistency"] for k, v in P.items()},
            "gsm_judge_disagrees_with_verifier_on_decisive_pairs": [1 - x["judge_agrees_with_verifier"] for x in pw_gsm],
            "rlvr_verifier_mismatches_vs_expected": diag["verifier_mismatches_vs_expected"],
        },
        "exploitability": {
            "filler_wrong_preference": diag["filler_susceptibility"],
            "gold_distractor_wrong_preference": {m: P["gold_distractor"][m]["wrong_preference_rate"] for m in ("rlvr", "rlaif")},
            "rlaif_group_reward_clean_vs_filler": [diag["rlaif_group_reward_by_variant"]["clean_correct"]["mean"],
                                                   diag["rlaif_group_reward_by_variant"]["persuasive_filler_correct"]["mean"]],
        },
        "cost": {
            "rlvr": "regex + numeric compare; negligible, no model call",
            "rlaif_seconds_per_judge_call_gsm": gsm["judge_cost"]["seconds_per_new_call"],
            "rlaif_judge_calls_per_group_of_K": "K(K-1)/2 (6 for K=4) per prompt per training step",
        },
        "S_reason": diag["S_reason"], "S_outcome": diag["S_outcome"],
    }
    out = od / "feedback_comparison.json"
    save_json(out, {"policies": policies, "feedback_sources": sources})

    print("policy  GSM acc  SVAMP acc  drop    GSM win/SFT  SVAMP win/SFT  GSM len  SVAMP len")
    for p in POLICIES:
        x = policies[p]
        w = lambda k: f"{x[k]:.3f}" if k in x else "  -  "
        print(f"{p:<7} {x['gsm_accuracy']:.3f}    {x['transfer_accuracy']:.3f}      {x['accuracy_drop']:+.3f}  "
              f"{w('gsm_ai_win_rate_vs_sft'):>10}  {w('transfer_ai_win_rate_vs_sft'):>12}  {x['gsm_length']:>7.1f}  {x['transfer_length']:>9.1f}")
    print(f"\nS_reason  rlvr={sources['S_reason']['rlvr']:.3f} rlaif={sources['S_reason']['rlaif']:.3f}   "
          f"S_outcome rlvr={sources['S_outcome']['rlvr']:.3f} rlaif={sources['S_outcome']['rlaif']:.3f}")
    print(f"filler wrong-preference: {sources['exploitability']['filler_wrong_preference']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()