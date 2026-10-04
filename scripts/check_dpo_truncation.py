"""Count DPO pairs whose encoding changes under the upstream truncation fix (commit 076e7cf).

A pair is unaffected if prompt + response + EOS fits in max_sequence_length: old and new encodings are then identical.

    python -m scripts.check_dpo_truncation
"""
from common.data import load_yaml, preference_responses, prompt_messages_from_preference, read_jsonl
from common.models import load_tokenizer

cfg = load_yaml("configs/dpo.yaml")
tok = load_tokenizer(cfg["base_model"])
L = int(cfg["max_sequence_length"])

for key in ("dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"):
    rows = read_jsonl(cfg["paths"][key])
    affected = prompt_too_long = 0
    for row in rows:
        p = len(tok.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        if p >= L:
            prompt_too_long += 1  # new code raises ValueError on these
            continue
        if any(p + len(tok(y, add_special_tokens=False)["input_ids"]) + 1 > L for y in preference_responses(row)):
            affected += 1
    print(f"{key:<22} rows={len(rows):>6}  pairs changed={affected:>5} ({affected / len(rows):.1%})  "
          f"prompt alone >= {L} (would crash)={prompt_too_long}")