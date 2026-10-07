python -m task5_feedback.evaluate_math --dataset gsm        # Step 1: 300 GSM8K problems
python -m task5_feedback.evaluate_math --dataset transfer   # Step 3: 100 SVAMP problems
python -m task5_feedback.score_perturbations                # Step 2: 100 diagnostic responses
python -m task5_feedback.compare_feedback                   # final combined comparison