from __future__ import annotations

import argparse

import pandas as pd
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in tqdm(range(0, len(df), batch_size), desc=f"xstest[{policy_name}]", unit="batch", dynamic_ncols=True):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    del model
    clear_gpu()
    return records


def output_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task4_safety"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    df = load_xstest(cfg)
    print("XSTest rows:", len(df))

    # Task 4 uses exactly these four frozen checkpoints; fail early if one is missing.
    for name, adapter in policy_specs(cfg).items():
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{name} adapter not found at {adapter}; train/restore the standard run first.")

    outdir = output_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    expected_ids = [int(x) for x in df["xstest_id"]]
    for name in policy_specs(cfg):
        path = outdir / f"generated_{name}.jsonl"
        if path.exists() and [r["xstest_id"] for r in read_jsonl(path)] == expected_ids:
            print(f"[{name}] {path} already complete; skipping")
            continue
        # Same deterministic decoding (greedy), cap and batch size for every policy; fixed CSV prompt order.
        records = generate_for_policy(cfg, name)
        assert [r["xstest_id"] for r in records] == expected_ids, "prompt order changed"
        write_jsonl(path, records)
        n_tok = [r["response_tokens"] for r in records]
        print(f"[{name}] saved {len(records)} responses -> {path}  (mean {sum(n_tok) / len(n_tok):.1f} tokens, "
              f"{sum(t >= int(cfg['safety_max_new_tokens']) for t in n_tok)} hit the {cfg['safety_max_new_tokens']}-token cap)")


if __name__ == "__main__":
    main()