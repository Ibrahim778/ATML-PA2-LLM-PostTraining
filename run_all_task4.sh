python -m task4_safety.generate_responses   # 4 policies x 450 prompts, greedy, 256-token cap
python -m task4_safety.judge_responses      # Qwen2.5-3B judge, 1,800 calls; resumable
python -m task4_safety.make_audit_sheet     # writes the blind labelling sheet
#  -> fill manual_label in results/task4_safety/manual_audit_sheet.csv
python -m task4_safety.evaluate_safety      # all aggregates; re-run after labelling