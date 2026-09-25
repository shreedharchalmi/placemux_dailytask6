import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2. Data Ingestion & Stratified Split
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Target encoding: 0 = Malignant, 1 = Benign
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.1765, stratify=y_dev, random_state=SEED
)

# 3. Model Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=SEED, max_iter=1000, solver="lbfgs"))
])

pipeline.fit(X_train, y_train)

# 4. Probability Extraction for Malignant Class (0)
# pipeline.classes_ gives array([0, 1]) -> Index 0 corresponds to Malignant
probs_val = pipeline.predict_proba(X_val)
prob_malignant = probs_val[:, 0]
y_val_malignant = (y_val == 0).astype(int) # 1 if malignant, 0 if benign

# 5. Threshold Optimization & Cost Analysis
# Cost matrix: False Negative = 10 units, False Positive = 1 unit
thresholds = np.linspace(0.05, 0.95, 100)
cost_records = []

for t in thresholds:
    pred_malignant = (prob_malignant >= t).astype(int)
    cm = confusion_matrix(y_val_malignant, pred_malignant)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fn * 10) + (fp * 1)
    rec = recall_score(y_val_malignant, pred_malignant, zero_division=0)
    prec = precision_score(y_val_malignant, pred_malignant, zero_division=0)
    acc = accuracy_score(y_val_malignant, pred_malignant)
    
    cost_records.append({
        "threshold": t,
        "cost": total_cost,
        "fn": fn,
        "fp": fp,
        "recall": rec,
        "precision": prec,
        "accuracy": acc
    })

cost_df = pd.DataFrame(cost_records)
optimal_idx = cost_df["cost"].idxmin()
optimal_row = cost_df.loc[optimal_idx]
optimal_threshold = optimal_row["threshold"]

print(f"Optimal Threshold Selected: {optimal_threshold:.2f}")
print(f"Minimum Total Cost: {optimal_row['cost']} (FN: {int(optimal_row['fn'])}, FP: {int(optimal_row['fp'])})")

# 6. Comparative Evaluation
# Default 0.50 Threshold
preds_default_mal = (prob_malignant >= 0.50).astype(int)
# Optimal Threshold
preds_opt_mal = (prob_malignant >= optimal_threshold).astype(int)

cm_default = confusion_matrix(y_val_malignant, preds_default_mal)
cm_optimal = confusion_matrix(y_val_malignant, preds_opt_mal)

print("\n=== Confusion Matrix at 0.50 Default Cutoff ===")
print("TN (Benign ok):", cm_default[0,0], "FP (Benign flagged):", cm_default[0,1])
print("FN (Malignant missed):", cm_default[1,0], "TP (Malignant caught):", cm_default[1,1])

print(f"\n=== Confusion Matrix at {optimal_threshold:.2f} Cost-Optimized Cutoff ===")
print("TN (Benign ok):", cm_optimal[0,0], "FP (Benign flagged):", cm_optimal[0,1])
print("FN (Malignant missed):", cm_optimal[1,0], "TP (Malignant caught):", cm_optimal[1,1])

# 7. Generate Evaluation Diagnostic Visuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Precision-Recall Curve
precisions, recalls, pr_thresh = precision_recall_curve(y_val_malignant, prob_malignant)
ax1.plot(recalls, precisions, color="blue", lw=2, label="PR Curve")
ax1.set_title("Precision-Recall Curve (Malignant Detection)")
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Threshold Cost Curve
ax2.plot(cost_df["threshold"], cost_df["cost"], color="red", lw=2, label="Cost Curve")
ax2.axvline(optimal_threshold, color="black", linestyle="--", label=f"Optimal ({optimal_threshold:.2f})")
ax2.set_title("Total Cost vs. Decision Threshold")
ax2.set_xlabel("Malignant Decision Threshold")
ax2.set_ylabel("Weighted Cost (FN*10 + FP*1)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("binary_decision_analysis.png")
print("\nPlot saved successfully as binary_decision_analysis.png")

# 8. Record Metrics to JSON
summary_output = {
    "task": "Task 6 - The Binary Decision",
    "default_threshold_0_50": {
        "accuracy": float(accuracy_score(y_val_malignant, preds_default_mal)),
        "recall_malignant": float(recall_score(y_val_malignant, preds_default_mal)),
        "precision_malignant": float(precision_score(y_val_malignant, preds_default_mal)),
        "false_negatives": int(cm_default[1, 0]),
        "false_positives": int(cm_default[0, 1])
    },
    "cost_optimized_threshold": {
        "threshold": float(optimal_threshold),
        "accuracy": float(optimal_row["accuracy"]),
        "recall_malignant": float(optimal_row["recall"]),
        "precision_malignant": float(optimal_row["precision"]),
        "false_negatives": int(optimal_row["fn"]),
        "false_positives": int(optimal_row["fp"])
    }
}

with open("task6_metrics_summary.json", "w") as f:
    json.dump(summary_output, f, indent=2)
print("Saved summary metrics to task6_metrics_summary.json")