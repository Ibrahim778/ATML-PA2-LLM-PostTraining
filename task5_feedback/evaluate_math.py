"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) or the fixed SVAMP subset (transfer).

For each policy: one greedy response per problem (same decoding, cap `math_max_new_tokens`, batch size and prompt
order for every policy). Then:
  exact accuracy      exact_reward(response, gold_final) (the supplied `#### <number>` verifier)
  format compliance   a designated `#### <number>` final was parsed (regardless of correctness)
  length              generated tokens; truncation at the cap
  failure type        correct / no_final (format failure; split by truncated) / wrong_final, plus whether the
                      gold number appears in the text even though the designated final is wrong
  AI pairwise         the fixed PairwiseAIJudge compares each RL policy's response with SFT's on the same problem
                      (and RLVR vs RLAIF); win = 1, tie = 0.5, loss = 0
  verifier-judge agreement
                      on pairs where exactly one response is verifier-correct: does the judge prefer it? On pairs the
                      verifier ties (both right / both wrong): how often does the judge still pick one?

    python -m task5_feedback.evaluate_math --dataset gsm
    python -m task5_feedback.evaluate_math --dataset transfer
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

PAIRINGS = (("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif"))


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def out_dir(cfg):
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def judge_cache_path(cfg):
    return out_dir(cfg) / "pairwise_judge_cache.json"   # shared by every Task 5 script


def row_id(row, i):
    return str(row.get("prompt_id", row.get("source_index", i)))


def gold_in_text(text, gold):
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text))
    return any(numerically_equal(n.replace(",", ""), gold) for n in nums)


def generate_policy(cfg, rows, tok, name, batch_size):
    model = load_frozen_policy(cfg, name)
    max_new = int(cfg["math_max_new_tokens"])
    recs = []
    for s in tqdm(range(0, len(rows), batch_size), desc=f"gen[{name}]", unit="batch", dynamic_ncols=True):
        chunk = rows[s:s + batch_size]
        gen = batch_generate(model, tok, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                             max_new_tokens=max_new, temperature=0.0, top_p=1.0, do_sample=False)
        for j, r in enumerate(chunk):
            resp, gold = gen["responses"][j], str(r["gold_final"])
            pred = extract_designated_final(resp)
            correct = exact_reward(resp, gold)
            truncated = bool(gen["truncated"][j])
            if correct:
                failure = "correct"
            elif pred is None:
                failure = "no_final_truncated" if truncated else "no_final"
            else:
                failure = "wrong_final"
            recs.append({"id": row_id(r, s + j), "policy": name, "question": r["question"], "gold_final": gold,
                         "response": resp, "pred_final": pred, "correct": correct, "format_ok": pred is not None,
                         "length_tokens": int(gen["response_lengths"][j]), "truncated": truncated,
                         "failure_type": failure,
                         "gold_number_in_text": gold_in_text(resp, gold) if not correct else True})
    del model
    clear_gpu()
    return recs


def policy_metrics(recs):
    L = np.array([r["length_tokens"] for r in recs], float)
    ft = {}
    for r in recs:
        ft[r["failure_type"]] = ft.get(r["failure_type"], 0) + 1
    wrong = [r for r in recs if r["failure_type"] == "wrong_final"]
    return {
        "n": len(recs),
        "exact_accuracy": float(np.mean([r["correct"] for r in recs])),
        "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
        "length_mean": float(L.mean()), "length_std": float(L.std(ddof=1)) if len(L) > 1 else 0.0,
        "length_median": float(np.median(L)), "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
        "failure_types": ft,
        "wrong_final_with_gold_in_text": sum(r["gold_number_in_text"] for r in wrong),
    }


def pairwise(judge, by_policy, a, b):
    """Judge a's response against b's on every problem. Returns summary + per-problem records."""
    out, A, B = [], by_policy[a], by_policy[b]
    for ra, rb in tqdm(list(zip(A, B)), desc=f"judge[{a} vs {b}]", unit="pair", dynamic_ncols=True):
        pref = judge.compare(ra["question"], ra["response"], rb["response"])   # orientation balanced inside
        score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
        out.append({"id": ra["id"], "a": a, "b": b, "judge": pref, "score_a": score,
                    "a_correct": ra["correct"], "b_correct": rb["correct"]})
    s = np.array([x["score_a"] for x in out])
    # Verifier-judge agreement
    decisive = [x for x in out if x["a_correct"] != x["b_correct"]]
    agree = [x for x in decisive if (x["judge"] == "A") == bool(x["a_correct"]) and x["judge"] != "TIE"]
    tied_by_verifier = [x for x in out if x["a_correct"] == x["b_correct"]]
    return {
        "a": a, "b": b, "n": len(out),
        "win_rate_a": float(s.mean()),                       # wins + 0.5 ties
        "wins_a": int((s == 1).sum()), "ties": int((s == 0.5).sum()), "losses_a": int((s == 0).sum()),
        "verifier_decisive_pairs": len(decisive),
        "judge_agrees_with_verifier": len(agree) / max(1, len(decisive)),
        "judge_ties_on_verifier_decisive": sum(x["judge"] == "TIE" for x in decisive) / max(1, len(decisive)),
        "verifier_tied_pairs": len(tied_by_verifier),
        "judge_distinguishes_verifier_ties": sum(x["judge"] != "TIE" for x in tied_by_verifier) / max(1, len(tied_by_verifier)),
    }, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    tok.truncation_side = "left"
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    od, bs = out_dir(cfg), int(cfg.get("eval_batch_size", 8))

    by_policy = {}
    for name in policy_specs(cfg):
        path = od / f"{args.dataset}_generated_{name}.jsonl"
        if path.exists() and len(read_jsonl(path)) == len(rows):
            by_policy[name] = read_jsonl(path)
            print(f"[{name}] loaded cached generations from {path}")
        else:
            by_policy[name] = generate_policy(cfg, rows, tok, name, bs)
            write_jsonl(path, by_policy[name])
    results = {"dataset": args.dataset, "n": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]),
               "policies": {p: policy_metrics(r) for p, r in by_policy.items()}}

    judge = PairwiseAIJudge(cfg, judge_cache_path(cfg))
    n_cached, t0 = len(judge.cache), time.perf_counter()
    results["pairwise"], all_judgments = {}, []
    for a, b in PAIRINGS:
        summ, recs = pairwise(judge, by_policy, a, b)
        results["pairwise"][f"{a}_vs_{b}"] = summ
        all_judgments += recs
    new_calls = len(judge.cache) - n_cached
    results["judge_cost"] = {"new_judge_calls": new_calls, "seconds": time.perf_counter() - t0,
                             "seconds_per_new_call": (time.perf_counter() - t0) / max(1, new_calls)}
    write_jsonl(od / f"{args.dataset}_judgments.jsonl", all_judgments)
    save_json(od / f"{args.dataset}_eval.json", results)

    print(f"\n[{args.dataset}] policy  acc    format  len    trunc  wrong-final(gold in text)")
    for p, m in results["policies"].items():
        print(f"           {p:<6} {m['exact_accuracy']:.3f}  {m['format_compliance']:.3f}  {m['length_mean']:>5.1f}  "
              f"{m['truncation_rate']:.2f}   {m['failure_types'].get('wrong_final', 0)}({m['wrong_final_with_gold_in_text']})")
    for k, s in results["pairwise"].items():
        print(f"  {k:<14} win={s['win_rate_a']:.3f} (W/T/L {s['wins_a']}/{s['ties']}/{s['losses_a']})  "
              f"judge agrees w/ verifier {s['judge_agrees_with_verifier']:.2f} on {s['verifier_decisive_pairs']} decisive pairs")
    print(f"saved -> {od}/{args.dataset}_eval.json")


if __name__ == "__main__":
    main()