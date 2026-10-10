"""After regenerating one policy's Task 4 responses, sync the manual audit sheet with them.

For every audited row belonging to that policy: if the new response text is identical, the manual label is kept;
if it changed, the response is replaced and the manual label is blanked so it can be re-labelled.

    python -m scripts.reconcile_audit_responses --policy ppo
"""
import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", default="ppo")
    args = ap.parse_args()
    outdir = repo_path(load_yaml(args.config)["results_dir"]) / "task4_safety"
    sheet = pd.read_csv(outdir / "manual_audit_sheet.csv", keep_default_na=False)
    key = pd.read_csv(outdir / "manual_audit_key.csv")
    new = {r["xstest_id"]: r["response"] for r in read_jsonl(outdir / f"generated_{args.policy}.jsonl")}

    rows = key.loc[key.policy == args.policy, "audit_row"]
    changed = []
    for ar in rows:
        i = sheet.index[sheet.audit_row == ar][0]
        fresh = new[int(sheet.at[i, "xstest_id"])]
        if sheet.at[i, "response"] != fresh:
            changed.append(int(ar))
            sheet.at[i, "response"] = fresh
            sheet.at[i, "manual_label"] = ""
            sheet.at[i, "notes"] = (str(sheet.at[i, "notes"]) + " [response regenerated; re-label]").strip()
    sheet.to_csv(outdir / "manual_audit_sheet.csv", index=False)
    print(f"{args.policy}: {len(rows)} audited rows, {len(rows) - len(changed)} unchanged (labels kept), "
          f"{len(changed)} changed (label blanked): audit_row {changed}")


if __name__ == "__main__":
    main()