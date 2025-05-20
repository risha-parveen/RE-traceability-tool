import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
import os
from args import get_eval_args

args = get_eval_args()
res_file = os.path.join(args.output_dir, "raw_res.csv")

df = pd.read_csv(res_file)

# Extract prediction probabilities and ground truth
y_scores = df['pred'].tolist()
y_true = df['label'].tolist()

# Compute PR curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

# Compute F1 scores for each threshold
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
best_f1 = f1s[best_idx]

# Optional: Round for cleaner logging
best_threshold_rounded = round(best_threshold, 4)
best_f1_rounded = round(best_f1, 4)

# Print to console
print(f"Best F1: {best_f1_rounded} at threshold {best_threshold_rounded}")

# Write to summary.txt in same folder
summary_path = os.path.join(os.path.dirname(res_file), "summary.txt")
with open(summary_path, "a") as f:
    f.write(f"\nBest F1: {best_f1_rounded} at threshold {best_threshold_rounded}\n")

print(f"Summary written to: {summary_path}")
