from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    tokenizer, beta = bundle["tokenizer"], bundle["beta"]
    accum = int(cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = int(cfg.get("epochs", 1))
    device = next(model.parameters()).device

    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # fresh log per run

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    micro_per_epoch = len(loader)
    total_updates = epochs * math.ceil(micro_per_epoch / accum)
    print(f"[{run_name}] beta={beta} examples={len(bundle['rows'])} micro_batches/epoch={micro_per_epoch} "
          f"accum={accum} -> optimizer_updates={total_updates}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = 0
    window = defaultdict(list)  # per-update aggregation over micro-batches

    pbar = tqdm(total=epochs * micro_per_epoch, desc=f"DPO[{run_name}]", unit="batch", dynamic_ncols=True)
    for epoch in range(epochs):
        for micro_idx, (chosen, rejected) in enumerate(loader):
            pbar.update(1)
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

            # Frozen reference: same network with the LoRA adapter disabled, no gradients.
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)

            # Trainable policy (adapter enabled).
            pol_c, _, mask_c = response_sequence_logprobs(model, chosen)
            pol_r, _, mask_r = response_sequence_logprobs(model, rejected)

            loss, stats = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()

            # Diagnostics computed directly from the manual's definitions (independent of dpo_loss internals).
            with torch.no_grad():
                chosen_ratio = (pol_c - ref_c).detach()
                rejected_ratio = (pol_r - ref_r).detach()
                margin = chosen_ratio - rejected_ratio
                window["loss"].append(loss.item())
                window["pref_acc"].append((margin > 0).float().mean().item())
                window["margin"].append(margin.mean().item())
                window["reward_chosen"].append((beta * chosen_ratio).mean().item())
                window["reward_rejected"].append((beta * rejected_ratio).mean().item())
                window["policy_logp_chosen"].append(pol_c.detach().mean().item())
                window["policy_logp_rejected"].append(pol_r.detach().mean().item())
                window["len_chosen"].append(mask_c.sum(-1).mean().item())
                window["len_rejected"].append(mask_r.sum(-1).mean().item())
                for k, v in stats.items():
                    window[f"lossfn_{k}"].append(float(v))

            is_last_micro = micro_idx + 1 == micro_per_epoch
            if (micro_idx + 1) % accum == 0 or is_last_micro:
                # Tail of the epoch: rescale grads so a partial window has the same effective scale.
                n_in_window = len(window["loss"])
                if n_in_window < accum:
                    for p in trainable_parameters(model):
                        if p.grad is not None:
                            p.grad.mul_(accum / n_in_window)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

                record = {"run": run_name, "beta": beta, "epoch": epoch, "update": update,
                          "examples_seen": min((micro_idx + 1) * int(cfg["batch_size"]), len(bundle["rows"]))
                          + epoch * len(bundle["rows"]),
                          "grad_norm": float(grad_norm), "elapsed_s": elapsed()}
                record.update({k: sum(v) / len(v) for k, v in window.items()})
                append_jsonl(log_path, record)
                window = defaultdict(list)

                pbar.set_postfix(
                    upd=f"{update}/{total_updates}",
                    loss=f"{record['loss']:.4f}",
                    acc=f"{record['pref_acc']:.3f}",
                    margin=f"{record['margin']:.3f}",
                    gnorm=f"{record['grad_norm']:.2f}",
                )
    pbar.close()

    # Save LoRA adapter only (small) + exact run metadata.
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {
        "run": run_name,
        "config_path": config_path,
        "dataset_path": str(dataset_path or cfg["paths"]["dpo_standard_train"]),
        "num_examples": len(bundle["rows"]),
        "max_examples": max_examples,
        "beta": beta,
        "seed": int(cfg["seed"]),
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "epochs": epochs,
        "optimizer_updates": update,
        "lora": cfg["lora"],
        "wall_clock_s": elapsed(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "adapter_path": str(output),
        "train_log": str(log_path),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print(f"[{run_name}] done: {update} updates in {summary['wall_clock_s']:.0f}s, adapter -> {output}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()