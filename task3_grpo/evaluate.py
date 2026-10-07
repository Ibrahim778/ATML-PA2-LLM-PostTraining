"""Task 3 held-out evaluation (common protocol for the GRPO midpoint, standard run and normalisation forks).

Same protocol as Task 2: fixed eval prompt pool, fixed seed and decoding. Generation cap is
`eval_max_response_length` if set, else `cache_generation_cap` (768, the same cap PPO evaluates at).
Reports RM score, sampled KL from the reference, token entropy, length / EOS / truncation, saves every
generation, and (once the midpoint is evaluated) qualitative candidates vs the midpoint.

    python -m task3_grpo.evaluate --name midpoint --adapter checkpoints/grpo_midpoint_policy   # baseline first
    python -m task3_grpo.evaluate --name standard --adapter outputs/task3_grpo/standard
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task2_ppo.evaluate import per_sequence_kl_entropy, qualitative_candidates, stats


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


def evaluate_run(config_path: str, adapter: str, name: str) -> dict:
    timer = wall_timer()
    b = load_evaluation_bundle(config_path, adapter)
    cfg, rows, tok, model = b["cfg"], b["rows"], b["tokenizer"], b["policy"]
    rm, rm_tok = b["reward"]
    g, bs = cfg["generation"], int(cfg.get("eval_batch_size", 4))
    max_new = int(cfg.get("eval_max_response_length", cfg["cache_generation_cap"]))
    rm_len = int(cfg.get("reward_max_length", 1280))
    prompts = [prompt_messages(r) for r in rows]
    set_seed(int(cfg["seed"]))  # identical decoding RNG for every condition

    recs = []
    for s in tqdm(range(0, len(prompts), bs), desc=f"eval[{name}]", unit="batch", dynamic_ncols=True):
        chunk = prompts[s:s + bs]
        gen = batch_generate(model, tok, chunk, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=max_new,
                             temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
        seq_stats = per_sequence_kl_entropy(model, gen)
        with torch.no_grad():
            scores = score_reward_pairs(rm, rm_tok, chunk, gen["responses"], max_length=rm_len).cpu()
        for i, resp in enumerate(gen["responses"]):
            recs.append({"prompt_index": s + i, "prompt": chunk[i][-1]["content"], "response": resp,
                         "rm_score": float(scores[i]), "length_tokens": int(gen["response_lengths"][i]),
                         "eos": bool(gen["terminated_with_eos"][i]), "truncated": bool(gen["truncated"][i]),
                         **seq_stats[i]})
        del gen

    n_tok = sum(r["n_tokens"] for r in recs)
    results = {
        "name": name, "adapter": adapter, "n_prompts": len(recs), "seed": int(cfg["seed"]),
        "decoding": {**g, "max_new_tokens": max_new},
        "rm_score": stats([r["rm_score"] for r in recs]),
        "kl_per_token": sum(r["kl_sum"] for r in recs) / max(1, n_tok),   # token-weighted, same as training
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
          f"KL/tok={results['kl_per_token']:.5f}  H/tok={results['entropy_per_token']:.3f}  "
          f"len={results['length_tokens']['mean']:.1f}±{results['length_tokens']['std']:.1f}  "
          f"eos={results['eos_rate']:.2f} trunc={results['truncation_rate']:.2f}  -> {res_dir}/{name}_eval.json")
    del model, rm, b
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_run(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()