"""Task 3 Step 2: group-size study on the supplied K=8 completion/reward cache (no training).

Equal-generation protocol: every prompt contributes exactly its first 8 cached completions, so the total
generation budget is 8 * P for every K. For K in {2, 4, 8} each prompt's 8 completions are partitioned into
8/K disjoint groups of size K. To avoid depending on one arbitrary partition, completions are randomly permuted
within each prompt `N_RESAMPLES` times (seeded) and every metric is averaged over partitions.

Per K (overall and per difficulty bin):
  informative_group_rate     groups whose reward std (population) > STD_TOL (the released helper's tolerance)
  weak_signal_group_rate     groups whose std < WEAK_FRAC * (median within-prompt std over all 8 completions)
  group_reward_std_mean      mean within-group reward std
  centered_signal_var        variance of (r - group mean): magnitude of the relative signal before std-normalisation
  advantage_abs_mean         mean |A| with A = (r - mu_g) / (sigma_g + eps)
  baseline_mse               mean (mu_g - mu_prompt)^2: noise in the group-mean baseline vs the 8-sample prompt mean
  sign_agreement             fraction of completions where sign(r - mu_g) == sign(r - mu_prompt)
  informative_generation_frac  share of the generation budget that lands in informative groups

Difficulty bins (defined once): prompts are split into terciles of their mean cached reward over all 8
completions -> "hard" (lowest third), "medium", "easy" (highest third).

    python -m task3_grpo.analyze_group_size
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

STD_TOL = 1e-6
WEAK_FRAC = 0.1
N_RESAMPLES = 50
BUDGET_PER_PROMPT = 8


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def _reward(row):
    for k in ("reward", "rm_score", "score", "rewards"):
        if k in row and row[k] is not None:
            return float(row[k])
    raise KeyError(f"No reward field in cache row; keys: {sorted(row)}")


def regroup_equal_generation_budget(by_prompt, k: int, rng: np.random.Generator | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt contributes exactly BUDGET_PER_PROMPT completions (its first 8 by generation_index), permuted
    by `rng` if given (else kept in cache order), and split into BUDGET_PER_PROMPT // k disjoint groups.
    Total generations = 8 * P for every k. Returns a list of (prompt_id, [rows]).
    """
    if BUDGET_PER_PROMPT % k:
        raise ValueError(f"K={k} must divide the per-prompt budget {BUDGET_PER_PROMPT}")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:BUDGET_PER_PROMPT]
        idx = rng.permutation(len(comps)) if rng is not None else np.arange(len(comps))
        for s in range(0, BUDGET_PER_PROMPT, k):
            groups.append((pid, [comps[i] for i in idx[s:s + k]]))
    return groups


def group_metrics(groups, prompt_mean, weak_thr):
    stds, centered, adv_abs, base_err, agree, n_inf_gen, n_gen = [], [], [], [], [], 0, 0
    for pid, g in groups:
        r = np.array([_reward(x) for x in g])
        mu, sd = r.mean(), r.std()            # population std, as in training
        stds.append(sd)
        centered.extend(r - mu)
        adv_abs.extend(np.abs((r - mu) / (sd + 1e-6)))
        base_err.append((mu - prompt_mean[pid]) ** 2)
        agree.extend(np.sign(r - mu) == np.sign(r - prompt_mean[pid]))
        n_gen += len(r)
        n_inf_gen += len(r) if sd > STD_TOL else 0
    stds = np.array(stds)
    return {
        "n_groups": len(groups), "total_generations": n_gen,
        "informative_group_rate": float((stds > STD_TOL).mean()),
        "weak_signal_group_rate": float((stds < weak_thr).mean()),
        "group_reward_std_mean": float(stds.mean()),
        "centered_signal_var": float(np.var(centered)),
        "advantage_abs_mean": float(np.mean(adv_abs)),
        "baseline_mse": float(np.mean(base_err)),
        "sign_agreement": float(np.mean(agree)),
        "informative_generation_frac": n_inf_gen / max(1, n_gen),
    }


def average(dicts):
    keys = dicts[0].keys()
    out = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
    out.update({f"{k}_sd_over_partitions": float(np.std([d[k] for d in dicts]))
                for k in ("informative_group_rate", "group_reward_std_mean", "sign_agreement")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    Ks = [int(k) for k in cfg["group_sizes"]]
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", Ks)
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    # Prompt difficulty from all 8 cached completions (defined once, shared by every K).
    prompt_mean = {pid: float(np.mean([_reward(x) for x in c[:BUDGET_PER_PROMPT]])) for pid, c in by_prompt.items()}
    prompt_std8 = {pid: float(np.std([_reward(x) for x in c[:BUDGET_PER_PROMPT]])) for pid, c in by_prompt.items()}
    q1, q2 = np.quantile(list(prompt_mean.values()), [1 / 3, 2 / 3])
    bin_of = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in prompt_mean.items()}
    weak_thr = WEAK_FRAC * float(np.median(list(prompt_std8.values())))

    results = {"n_prompts": len(by_prompt), "budget_per_prompt": BUDGET_PER_PROMPT,
               "total_generations": BUDGET_PER_PROMPT * len(by_prompt), "n_resamples": N_RESAMPLES,
               "std_tolerance": STD_TOL, "weak_signal_threshold": weak_thr,
               "difficulty_rule": "terciles of mean cached reward over 8 completions; hard <= q1 < medium <= q2 < easy",
               "difficulty_cutpoints": [float(q1), float(q2)],
               "bin_sizes": {b: sum(v == b for v in bin_of.values()) for b in ("hard", "medium", "easy")},
               "by_K": {}}

    rng = np.random.default_rng(int(cfg["seed"]))
    for k in Ks:
        parts = [regroup_equal_generation_budget(by_prompt, k)] + \
                [regroup_equal_generation_budget(by_prompt, k, rng) for _ in range(N_RESAMPLES - 1)]
        entry = {"overall": average([group_metrics(p, prompt_mean, weak_thr) for p in parts]), "by_difficulty": {}}
        for b in ("hard", "medium", "easy"):
            entry["by_difficulty"][b] = average([group_metrics([g for g in p if bin_of[g[0]] == b], prompt_mean, weak_thr)
                                                 for p in parts])
        results["by_K"][str(k)] = entry

    # Qualitative: the lowest-signal K=2 groups per difficulty bin (cache order partition).
    qual = {}
    for b in ("hard", "medium", "easy"):
        gs = [g for g in regroup_equal_generation_budget(by_prompt, 2) if bin_of[g[0]] == b]
        gs.sort(key=lambda g: np.std([_reward(x) for x in g[1]]))
        qual[b] = [{"source_index": pid, "prompt_mean_reward_k8": prompt_mean[pid],
                    "completions": [{"reward": _reward(x),
                                     "text": str(x.get("completion", x.get("response", "")))[:400]} for x in g]}
                   for pid, g in gs[:3]]
    results["qualitative_low_signal_K2"] = qual

    out = repo_path(cfg["results_dir"]) / "group_size_analysis.json"
    save_json(out, results)
    print(f"\nDifficulty cutpoints (mean reward): {q1:.3f}, {q2:.3f}; weak-signal threshold std < {weak_thr:.3f}")
    print("K  bin      groups  informative  weak   std_mean  centered_var  baseline_mse  sign_agree")
    for k in Ks:
        e = results["by_K"][str(k)]
        for b, d in [("all", e["overall"])] + list(e["by_difficulty"].items()):
            print(f"{k:<2} {b:<8} {d['n_groups']:>6.0f}  {d['informative_group_rate']:>10.3f}  {d['weak_signal_group_rate']:>5.3f}"
                  f"  {d['group_reward_std_mean']:>8.3f}  {d['centered_signal_var']:>12.3f}  {d['baseline_mse']:>12.3f}"
                  f"  {d['sign_agreement']:>10.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()