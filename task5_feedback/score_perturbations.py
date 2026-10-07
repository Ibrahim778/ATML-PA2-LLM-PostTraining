from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


# Controlled pairs: (diagnostically better, worse, what changes, family).
# Reasoning pairs hold the final answer correct and degrade reasoning; outcome pairs hold reasoning ~fixed and change
# the designated final. Filler: same correct content + irrelevant persuasive text; the clean version counts as
# "better" (filler should never help), and a tie is the ideal invariant behaviour.
PAIRS = {
    "reasoning_corruption": ("clean_correct", "corrupt_reasoning_correct_final", "reasoning"),
    "wrong_final": ("clean_correct", "good_reasoning_wrong_final", "outcome"),
    "persuasive_filler": ("clean_correct", "persuasive_filler_correct", "style"),
    "gold_distractor": ("clean_correct", "gold_distractor_wrong_final", "outcome"),
}
VARIANT_ORDER = ("clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final")


def rates(outcomes):
    """outcomes: list of +1 (prefers better), 0 (tie), -1 (prefers worse); fractional values allowed."""
    o = np.asarray(outcomes, float)
    return {"n": int(len(o)), "better_rate": float((o > 0).mean()) if len(o) else float("nan"),
            "tie_rate": float((o == 0).mean()) if len(o) else float("nan"),
            "wrong_preference_rate": float((o < 0).mean()) if len(o) else float("nan")}


def judge_pair(judge, q, better, worse):
    """Both presentation orders; the class's hash-based orientation balancing applies inside each call.

    Returns per-order outcomes (+1 better / 0 tie / -1 worse) and whether the two orders agree.
    """
    p1 = judge.compare(q, better, worse)            # A = better
    p2 = judge.compare(q, worse, better)            # A = worse
    o1 = {"A": 1, "B": -1, "TIE": 0}[p1]
    o2 = {"A": -1, "B": 1, "TIE": 0}[p2]
    return o1, o2, o1 == o2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")
    od = repo_path(cfg["results_dir"]) / "task5_feedback"
    od.mkdir(parents=True, exist_ok=True)

    # Sanity: the verifier must reproduce the staff-validated expected exact reward on every response.
    mism = [(pid, v) for pid, g in groups.items() for v, r in g.items()
            if float(r.get("expected_exact_reward", exact_reward(r["response"], r["gold_final"])))
            != exact_reward(r["response"], str(r["gold_final"]))]
    if mism:
        print(f"WARNING: verifier disagrees with expected_exact_reward on {len(mism)} responses: {mism[:5]}")

    judge = PairwiseAIJudge(cfg, od / "pairwise_judge_cache.json")
    n0, t0 = len(judge.cache), time.perf_counter()
    records = []
    per_pair = {name: {"rlvr": [], "rlaif": [], "rlaif_order_consistent": []} for name in PAIRS}
    group_rewards = {v: [] for v in VARIANT_ORDER}
    for pid, g in tqdm(sorted(groups.items()), desc="diagnostics", unit="problem", dynamic_ncols=True):
        q, gold = g["clean_correct"]["question"], str(g["clean_correct"]["gold_final"])
        for name, (bv, wv, fam) in PAIRS.items():
            b, w = g[bv]["response"], g[wv]["response"]
            rb, rw = exact_reward(b, gold), exact_reward(w, gold)
            v_out = int(np.sign(rb - rw))
            o1, o2, consistent = judge_pair(judge, q, b, w)
            per_pair[name]["rlvr"].append(v_out)
            per_pair[name]["rlaif"] += [o1, o2]          # both orders count, each with weight 1/2 of the pair
            per_pair[name]["rlaif_order_consistent"].append(consistent)
            records.append({"problem_id": pid, "pair": name, "family": fam, "better": bv, "worse": wv,
                            "rlvr_better": rb, "rlvr_worse": rw, "rlvr_outcome": v_out,
                            "rlaif_outcome_better_first": o1, "rlaif_outcome_worse_first": o2,
                            "rlaif_order_consistent": consistent})
        # Direct-RLAIF reward over all five variants as one group: (wins + 0.5 ties) / (K - 1).
        gr = judge.group_rewards(q, [g[v]["response"] for v in VARIANT_ORDER])
        for v, r in zip(VARIANT_ORDER, gr):
            group_rewards[v].append(r)

    summary = {"n_problems": len(groups), "pairs": {}, "verifier_mismatches_vs_expected": len(mism),
               "judge_cost": {"new_judge_calls": len(judge.cache) - n0, "seconds": time.perf_counter() - t0}}
    for name, (bv, wv, fam) in PAIRS.items():
        d = per_pair[name]
        summary["pairs"][name] = {"better": bv, "worse": wv, "family": fam,
                                  "rlvr": rates(d["rlvr"]), "rlaif": rates(d["rlaif"]),
                                  "rlaif_order_consistency": float(np.mean(d["rlaif_order_consistent"]))}
    P = summary["pairs"]
    summary["S_reason"] = {"rlvr": P["reasoning_corruption"]["rlvr"]["better_rate"],
                           "rlaif": P["reasoning_corruption"]["rlaif"]["better_rate"]}
    summary["S_outcome"] = {"rlvr": P["wrong_final"]["rlvr"]["better_rate"],
                            "rlaif": P["wrong_final"]["rlaif"]["better_rate"]}
    outcome_pool = {m: rates(per_pair["wrong_final"][m] + per_pair["gold_distractor"][m]) for m in ("rlvr", "rlaif")}
    summary["S_outcome_incl_distractor"] = {m: outcome_pool[m]["better_rate"] for m in outcome_pool}
    summary["filler_susceptibility"] = {m: P["persuasive_filler"][m]["wrong_preference_rate"] for m in ("rlvr", "rlaif")}
    summary["distractor_robustness"] = {m: P["gold_distractor"][m]["better_rate"] for m in ("rlvr", "rlaif")}
    summary["rlaif_group_reward_by_variant"] = {v: {"mean": float(np.mean(r)), "std": float(np.std(r))}
                                                for v, r in group_rewards.items()}
    summary["rlvr_reward_by_variant"] = {v: float(np.mean([exact_reward(g[v]["response"], str(g[v]["gold_final"]))
                                                           for g in groups.values()])) for v in VARIANT_ORDER}

    # Qualitative: pairs where the two mechanisms disagree.
    disagree = [r for r in records if np.sign(r["rlaif_outcome_better_first"] + r["rlaif_outcome_worse_first"]) != r["rlvr_outcome"]]
    write_jsonl(od / "diagnostic_pair_records.jsonl", records)
    examples = []
    for r in disagree[:12]:
        g = groups[r["problem_id"]]
        examples.append({**r, "question": g["clean_correct"]["question"], "gold_final": str(g["clean_correct"]["gold_final"]),
                         "better_response": g[r["better"]]["response"][-500:], "worse_response": g[r["worse"]]["response"][-500:]})
    summary["n_mechanism_disagreements"] = len(disagree)
    save_json(od / "diagnostics_summary.json", summary)
    save_json(od / "diagnostics_qualitative.json", examples)

    print("\npair                   mechanism  better  tie    wrong   (RLAIF order-consistent)")
    for name, d in P.items():
        for m in ("rlvr", "rlaif"):
            x = d[m]
            extra = f"  ({d['rlaif_order_consistency']:.2f})" if m == "rlaif" else ""
            print(f"{name:<22} {m:<9} {x['better_rate']:.3f}  {x['tie_rate']:.3f}  {x['wrong_preference_rate']:.3f}{extra}")
    print(f"S_reason  rlvr={summary['S_reason']['rlvr']:.3f}  rlaif={summary['S_reason']['rlaif']:.3f}")
    print(f"S_outcome rlvr={summary['S_outcome']['rlvr']:.3f}  rlaif={summary['S_outcome']['rlaif']:.3f}")
    print("RLAIF group reward by variant:", {v: round(x["mean"], 3) for v, x in summary["rlaif_group_reward_by_variant"].items()})
    print(f"saved -> {od}/diagnostics_summary.json")


if __name__ == "__main__":
    main()