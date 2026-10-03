"""Task 1 Step 3: length-confounding study.

1. Dataset diagnostics (property of the DATA): how response length relates to the preference label in the
   standard train set, the length-balanced train set and the length-stratified eval set.
2. Train the length-balanced DPO model from the original initialization (skipped if already trained).
3. Evaluate standard vs length-balanced DPO on the length-stratified held-out set, per stratum
   (preferred-longer / matched / rejected-longer).
4. Generated length and explicit word-limit compliance on the common word-limit prompts
   (property of the POLICY), with the SFT model as a baseline.

    python -m task1_dpo.train --run-name standard        # Step 1 must exist first
    python -m task1_dpo.analyze_length
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common.data import load_yaml, preference_responses, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.metrics import parse_word_limit, safe_corr, word_limit_compliance
from common.models import clear_gpu
from task1_dpo.evaluate import (generate_and_score, load_eval_tokenizer, load_models, run_beta, score_pairs,
                                stats, summarize_pairs)
from task1_dpo.train import run_training

STRATUM_KEYS = ("stratum", "length_stratum", "length_bucket", "bucket", "strata", "length_category", "category")


def detect_stratum_key(rows, required=True):
    for k in STRATUM_KEYS:
        if rows and k in rows[0]:
            return k
    if required:
        raise KeyError(f"No stratum field found (tried {STRATUM_KEYS}). Row keys: {list(rows[0])}")
    return None


# ----------------------------------------------------------------------------------------------
# 1) Dataset diagnostics: length correlation in the preference data itself
# ----------------------------------------------------------------------------------------------
def dataset_length_stats(rows, tokenizer, stratum_key=None):
    lc, lr, strata = [], [], []
    for row in rows:
        yc, yr = preference_responses(row)
        lc.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]))
        lr.append(len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
        strata.append(str(row.get(stratum_key)) if stratum_key else None)
    lc, lr = np.array(lc), np.array(lr)
    diff = lc - lr
    out = {
        "n": len(rows),
        "frac_chosen_longer": float((lc > lr).mean()),
        "frac_rejected_longer": float((lr > lc).mean()),
        "len_chosen": stats(lc), "len_rejected": stats(lr),
        "len_diff_chosen_minus_rejected": stats(diff),
        # Every pair appears both ways (chosen-first label 1, rejected-first label 0) -> correlation between
        # signed length difference and the preference label. Positive = "longer tends to be preferred".
        "corr_length_diff_vs_label": safe_corr(np.concatenate([diff, -diff]),
                                               np.concatenate([np.ones_like(diff), np.zeros_like(diff)])),
    }
    if stratum_key:
        out["stratum_counts"] = dict(Counter(strata))
    return out


# ----------------------------------------------------------------------------------------------
# 3) + 4) Per-condition evaluation
# ----------------------------------------------------------------------------------------------
def word_limit_eval(model, tok, rm, rm_tok, cfg, batch_size, desc):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    summary, recs = generate_and_score(model, tok, rm, rm_tok, [prompt_messages(r) for r in rows], cfg,
                                       batch_size, desc=desc)
    for r, row in zip(recs, rows):
        r["prompt_id"] = row.get("prompt_id")
        r["word_limit"] = parse_word_limit(r["prompt"])
        r["compliant"] = word_limit_compliance(r["prompt"], r["response"])
    comp = [r["compliant"] for r in recs if r["compliant"] is not None]
    summary["compliance_rate"] = float(np.mean(comp)) if comp else None
    summary["n_with_parsed_limit"] = len(comp)
    summary["words_over_limit"] = stats([max(0, r["length_words"] - r["word_limit"]) for r in recs if r["word_limit"]])
    return summary, recs


def evaluate_condition(cfg, name, adapter, strat_rows, stratum_key, tok, res_dir, do_strata=True):
    model, rm, rm_tok = load_models(cfg, adapter)
    bs = int(cfg["batch_size"])
    out = {"name": name, "adapter": adapter}

    if do_strata:
        beta = run_beta(cfg, name)
        per_pair = score_pairs(model, tok, strat_rows, cfg, bs, desc=f"strata[{name}]")
        groups = defaultdict(list)
        for p, row in zip(per_pair, strat_rows):
            p["stratum"] = str(row[stratum_key])
            groups[p["stratum"]].append(p)
        out["beta_for_loss"] = beta
        out["overall"] = summarize_pairs(per_pair, beta)
        out["by_stratum"] = {k: summarize_pairs(v, beta) for k, v in sorted(groups.items())}
        write_jsonl(res_dir / f"length_{name}_strata_pairs.jsonl", per_pair)

    out["word_limit"], wl_recs = word_limit_eval(model, tok, rm, rm_tok, cfg, bs, desc=f"wordlimit[{name}]")
    write_jsonl(res_dir / f"length_{name}_word_limit.jsonl", wl_recs)
    out["word_limit_violations"] = sorted([r for r in wl_recs if r["compliant"] == 0.0],
                                          key=lambda r: -(r["length_words"] - r["word_limit"]))[:5]
    del model, rm
    clear_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res_dir = repo_path(cfg["results_dir"])
    tok = load_eval_tokenizer(cfg)

    std_train = read_jsonl(cfg["paths"]["dpo_standard_train"])
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    skey = detect_stratum_key(stratified)
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified), "| stratum field:", skey)

    # 1) data properties
    report = {"stratum_key": skey, "dataset": {
        "standard_train": dataset_length_stats(std_train, tok),
        "length_balanced_train": dataset_length_stats(balanced, tok, detect_stratum_key(balanced, required=False)),
        "length_stratified_eval": dataset_length_stats(stratified, tok, skey),
    }}

    # 2) train the length-balanced condition (same recipe, different data)
    std_adapter, bal_adapter = cfg["standard_output"], cfg["length_output"]
    if not (repo_path(std_adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Standard DPO adapter not found at {std_adapter}; run `python -m task1_dpo.train` first.")
    if not (repo_path(bal_adapter) / "adapter_config.json").exists():
        run_training(args.config, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=bal_adapter)
        clear_gpu()
    else:
        print(f"Length-balanced adapter found at {bal_adapter}; skipping training.")

    # 3) + 4) evaluate
    report["conditions"] = {
        "sft": evaluate_condition(cfg, "sft", None, stratified, skey, tok, res_dir, do_strata=False),
        "standard": evaluate_condition(cfg, "standard", std_adapter, stratified, skey, tok, res_dir),
        "length_balanced": evaluate_condition(cfg, "length_balanced", bal_adapter, stratified, skey, tok, res_dir),
    }
    save_json(res_dir / "length_analysis.json", report)

    # summary
    print("\nDataset: frac chosen longer | corr(len diff, label)")
    for k, d in report["dataset"].items():
        print(f"  {k:<24} {d['frac_chosen_longer']:.3f} | {d['corr_length_diff_vs_label']:+.3f}")
    strata = sorted(report["conditions"]["standard"]["by_stratum"])
    print("\nPreference accuracy by stratum:")
    print("  " + f"{'condition':<16}" + "".join(f"{s:>20}" for s in strata) + f"{'overall':>10}")
    for name in ("standard", "length_balanced"):
        c = report["conditions"][name]
        print("  " + f"{name:<16}" + "".join(f"{c['by_stratum'][s]['preference_accuracy']:>20.3f}" for s in strata)
              + f"{c['overall']['preference_accuracy']:>10.3f}")
    print("\nWord-limit prompts: compliance | mean words | mean tokens")
    for name, c in report["conditions"].items():
        w = c["word_limit"]
        print(f"  {name:<16} {w['compliance_rate']:.2f} | {w['length_words']['mean']:.1f} | {w['length_tokens']['mean']:.1f}")
    print(f"\nsaved -> {res_dir / 'length_analysis.json'}")


if __name__ == "__main__":
    main()