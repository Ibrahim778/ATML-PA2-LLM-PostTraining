"""Task 2 held-out evaluation (common protocol for the midpoint, standard run and every fork).

Fixed eval prompt pool, fixed seed and decoding, generation cap `eval_max_response_length`.
Reports RM score, sampled KL from the reference, token entropy, response length / EOS / truncation,
and saves every generation. If the midpoint has been evaluated, also writes qualitative candidates
comparing this policy against the midpoint on the same prompts.

    python -m task2_ppo.evaluate --name midpoint --adapter checkpoints/ppo_midpoint_policy   # baseline first
    python -m task2_ppo.evaluate --name standard --adapter outputs/task2_ppo/standard
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    tok = load_tokenizer(cfg["base_model"])
    tok.truncation_side = "left"
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": tok,
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def stats(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(med), "iqr": float(q3 - q1), "min": float(a.min()), "max": float(a.max())}


@torch.no_grad()
def per_sequence_kl_entropy(model, gen):
    """Per sequence (one at a time to bound memory): summed/count KL vs reference and summed token entropy."""
    out = []
    for i in range(gen["sequences"].shape[0]):
        seq, attn = gen["sequences"][i:i + 1], gen["attention_mask"][i:i + 1]
        rids, mask = gen["response_ids"][i:i + 1], gen["response_mask"][i:i + 1]
        n = int(mask.sum())
        pol, logits = response_token_logprobs(model, seq, attn, gen["prompt_width"], rids)
        logp = torch.log_softmax(logits.float(), dim=-1)
        ent = (-(logp.exp() * logp).sum(-1) * mask).sum()
        del logits, logp
        with reference_mode(model):
            ref, _ = response_token_logprobs(model, seq, attn, gen["prompt_width"], rids)
        out.append({"kl_sum": float(((pol - ref) * mask).sum()), "entropy_sum": float(ent), "n_tokens": n})
    return out


def evaluate_run(config_path: str, adapter: str, name: str) -> dict:
    timer = wall_timer()
    b = load_evaluation_bundle(config_path, adapter)
    cfg, rows, tok, model = b["cfg"], b["rows"], b["tokenizer"], b["policy"]
    rm, rm_tok = b["reward"]
    g, bs = cfg["generation"], int(cfg.get("eval_batch_size", 4))
    max_new = int(cfg["eval_max_response_length"])
    prompts = [prompt_messages(r) for r in rows]
    set_seed(int(cfg["seed"]))  # identical decoding RNG for every condition

    recs = []
    for s in tqdm(range(0, len(prompts), bs), desc=f"eval[{name}]", unit="batch", dynamic_ncols=True):
        chunk = prompts[s:s + bs]
        gen = batch_generate(model, tok, chunk, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=max_new,
                             temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
        seq_stats = per_sequence_kl_entropy(model, gen)
        with torch.no_grad():
            scores = score_reward_pairs(rm, rm_tok, chunk, gen["responses"], max_length=int(cfg["reward_max_length"])).cpu()
        for i, resp in enumerate(gen["responses"]):
            eos = bool(gen["terminated_with_eos"][i])
            recs.append({"prompt_index": s + i, "prompt": chunk[i][-1]["content"], "response": resp,
                         "rm_score": float(scores[i]),
                         "task_reward": float(scores[i]) - (0.0 if eos else float(cfg["missing_eos_penalty"])),
                         "length_tokens": int(gen["response_lengths"][i]), "eos": eos,
                         "truncated": bool(gen["truncated"][i]), **seq_stats[i]})
        del gen

    n_tok = sum(r["n_tokens"] for r in recs)
    results = {
        "name": name, "adapter": adapter, "n_prompts": len(recs), "seed": int(cfg["seed"]),
        "decoding": {**g, "max_new_tokens": max_new},
        "rm_score": stats([r["rm_score"] for r in recs]),
        "task_reward": stats([r["task_reward"] for r in recs]),
        # Same convention as training: token-weighted mean over valid response tokens (common.metrics.sampled_kl).
        "kl_per_token": sum(r["kl_sum"] for r in recs) / max(1, n_tok),
        "kl_per_sequence": stats([r["kl_sum"] for r in recs]),
        "entropy_per_token": sum(r["entropy_sum"] for r in recs) / max(1, n_tok),
        "length_tokens": stats([r["length_tokens"] for r in recs]),
        "eos_rate": float(np.mean([r["eos"] for r in recs])),
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
    }
    res_dir = repo_path(cfg["results_dir"])
    write_jsonl(res_dir / f"{name}_eval_generations.jsonl", recs)

    mid_path = res_dir / "midpoint_eval_generations.jsonl"
    if name != "midpoint" and mid_path.exists():
        save_json(res_dir / f"{name}_qualitative_candidates.json", qualitative_candidates(recs, read_jsonl(mid_path)))

    results["wall_clock_s"] = timer()
    save_json(res_dir / f"{name}_eval.json", results)
    print(f"\n[{name}] RM={results['rm_score']['mean']:.3f}±{results['rm_score']['std']:.3f}  "
          f"KL/tok={results['kl_per_token']:.4f}  H/tok={results['entropy_per_token']:.3f}  "
          f"len={results['length_tokens']['mean']:.1f}±{results['length_tokens']['std']:.1f}  "
          f"eos={results['eos_rate']:.2f} trunc={results['truncation_rate']:.2f}  -> {res_dir}/{name}_eval.json")
    del model, rm, b
    clear_gpu()
    return results


def qualitative_candidates(recs, mid_recs, k=5):
    """Same prompt, this policy vs the midpoint. Read these to pick 'reward and quality agree / disagree' examples."""
    paired = []
    for r, m in zip(recs, mid_recs):
        paired.append({"prompt": r["prompt"], "response": r["response"], "midpoint_response": m["response"],
                       "rm_score": r["rm_score"], "midpoint_rm_score": m["rm_score"],
                       "reward_gain": r["rm_score"] - m["rm_score"],
                       "length": r["length_tokens"], "midpoint_length": m["length_tokens"],
                       "truncated": r["truncated"], "kl_sum": r["kl_sum"]})
    suspicious = [p for p in paired if p["reward_gain"] > 0 and (p["truncated"] or p["length"] > 1.5 * max(1, p["midpoint_length"]))]
    return {
        "largest_reward_gain": sorted(paired, key=lambda p: -p["reward_gain"])[:k],
        "reward_up_but_longer_or_truncated": sorted(suspicious, key=lambda p: -p["reward_gain"])[:k],
        "largest_reward_drop": sorted(paired, key=lambda p: p["reward_gain"])[:k],
        "highest_kl": sorted(paired, key=lambda p: -p["kl_sum"])[:k],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_run(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()