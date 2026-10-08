from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


POLICIES = ("sft", "dpo", "ppo", "grpo")
LABEL_HELP = "SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Blind labelling sheet: exactly the fixed 60 examples (one response per audited prompt). Each prompt is paired
    # with ONE policy's response, assigned round-robin within each prompt class after a seeded shuffle, so every
    # policy contributes ~equally (15 rows: 7-8 SAFE + 7-8 UNSAFE). Rows are shuffled, the policy name is hidden
    # (key written separately) and no AI labels appear anywhere in the sheet.
    sheet_path, key_path = outdir / "manual_audit_sheet.csv", outdir / "manual_audit_key.csv"
    if sheet_path.exists():
        existing = pd.read_csv(sheet_path, keep_default_na=False)
        if (existing["manual_label"].astype(str).str.strip() != "").any():
            print(f"{sheet_path} already contains manual labels; not overwriting.")
            return
    responses = {}
    for policy in POLICIES:
        path = outdir / f"generated_{policy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; generate all four policies before building the audit sheet")
        responses[policy] = {r["xstest_id"]: r for r in read_jsonl(path)}
    rng = np.random.default_rng(int(cfg["seed"]))
    rows = []
    for label in ("SAFE", "UNSAFE"):
        class_ids = [i for i in ids if responses["sft"][i]["benchmark_class"] == label]
        class_ids = [class_ids[j] for j in rng.permutation(len(class_ids))]
        offset = int(rng.integers(len(POLICIES)))  # rotate so the 8-vs-7 split differs between classes
        for j, i in enumerate(class_ids):
            policy = POLICIES[(j + offset) % len(POLICIES)]
            r = responses[policy][i]
            rows.append({"xstest_id": i, "policy": policy, "benchmark_class": r["benchmark_class"], "type": r["type"],
                         "prompt": r["prompt"], "response": r["response"]})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df.insert(0, "audit_row", range(len(df)))
    df[["audit_row", "policy", "xstest_id"]].to_csv(key_path, index=False)
    sheet = df.drop(columns=["policy"])
    sheet["manual_label"] = ""
    sheet["notes"] = ""
    sheet.to_csv(sheet_path, index=False)
    counts = df.groupby(["policy", "benchmark_class"]).size().unstack(fill_value=0)
    print(f"Wrote blind audit sheet ({len(sheet)} rows, one response per audited prompt): {sheet_path}")
    print("Responses per policy x class:\n" + counts.to_string())
    print(f"Fill `manual_label` with one of: {LABEL_HELP}")
    print("Label without opening judged_*.jsonl first. Key (row -> policy) kept separately in", key_path)


if __name__ == "__main__":
    main()