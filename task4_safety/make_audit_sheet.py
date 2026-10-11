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

    # Blind labelling sheet: the fixed 60 prompt IDs joined to EVERY policy's response (60 x 4 = 240 rows), shuffled,
    # policy name hidden (key written separately), no AI labels anywhere. If a sheet already exists, its manual labels
    # and notes are carried over for every (prompt, policy) pair whose response text is unchanged, so re-running this
    # never loses labelling work.
    sheet_path, key_path = outdir / "manual_audit_sheet.csv", outdir / "manual_audit_key.csv"
    carry = {}
    if sheet_path.exists() and key_path.exists():
        old = pd.read_csv(sheet_path, keep_default_na=False).merge(pd.read_csv(key_path)[["audit_row", "policy"]],
                                                                    on="audit_row")
        for r in old.itertuples():
            if str(r.manual_label).strip():
                carry[(int(r.xstest_id), r.policy)] = (r.response, r.manual_label, r.notes)
    rows = []
    for policy in POLICIES:
        path = outdir / f"generated_{policy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; generate all four policies before building the audit sheet")
        by_id = {r["xstest_id"]: r for r in read_jsonl(path)}
        for i in ids:
            r = by_id[i]
            label, notes = "", ""
            prev = carry.get((i, policy))
            if prev and prev[0] == r["response"]:
                label, notes = prev[1], prev[2]
            rows.append({"xstest_id": i, "policy": policy, "benchmark_class": r["benchmark_class"], "type": r["type"],
                         "prompt": r["prompt"], "response": r["response"], "manual_label": label, "notes": notes})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df.insert(0, "audit_row", range(len(df)))
    df[["audit_row", "policy", "xstest_id"]].to_csv(key_path, index=False)
    df.drop(columns=["policy"]).to_csv(sheet_path, index=False)
    kept = int((df.manual_label.astype(str).str.strip() != "").sum())
    print(f"Wrote blind audit sheet ({len(df)} rows = {len(ids)} prompts x {len(POLICIES)} policies): {sheet_path}")
    print(f"Carried over {kept} existing manual labels; {len(df) - kept} rows left to label.")
    print(f"Fill `manual_label` with one of: {LABEL_HELP}")
    print("Label without opening judged_*.jsonl first. Key (row -> policy) kept separately in", key_path)


if __name__ == "__main__":
    main()