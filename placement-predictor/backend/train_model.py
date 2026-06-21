"""
Trains and evaluates models for the Placement Predictor.

Pipeline:
1. Load + split data
2. Preprocess (one-hot encode categoricals, scale numerics) via sklearn Pipeline
3. Train multiple models, compare with cross-validation
4. Pick best model, evaluate on held-out test set (accuracy, precision, recall, F1, ROC-AUC)
5. Save the trained pipeline + metrics + feature importance for the API and frontend
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/placement-predictor/data/placement_data.csv")

FEATURES_NUM = [
    "cgpa", "ssc_percentage", "hsc_percentage", "internships",
    "projects", "certifications", "dsa_score", "communication_score",
    "backlogs", "extra_curricular",
]
FEATURES_CAT = ["branch", "specialization"]
TARGET = "placed"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), FEATURES_CAT),
    ]
)

# ---------------------------------------------------------------------------
# 3. Compare candidate models with stratified 5-fold CV (on training data only)
# ---------------------------------------------------------------------------
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

print("=" * 60)
print("CROSS-VALIDATION COMPARISON (5-fold, ROC-AUC)")
print("=" * 60)
for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std()}
    print(f"{name:20s}  AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
print(f"\nBest model by CV: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Train best model on full training set, evaluate on held-out test set
# ---------------------------------------------------------------------------
best_pipeline = Pipeline([("prep", preprocessor), ("clf", candidates[best_model_name])])
best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "model_name": best_model_name,
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "cv_comparison": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cv_results.items()},
}

cm = confusion_matrix(y_test, y_pred)
metrics["confusion_matrix"] = {
    "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
}

fpr, tpr, _ = roc_curve(y_test, y_proba)
idx = np.linspace(0, len(fpr) - 1, min(30, len(fpr))).astype(int)
metrics["roc_curve"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}

print("\n" + "=" * 60)
print(f"TEST SET PERFORMANCE — {best_model_name}")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))
print("Confusion Matrix:")
print(cm)
print(f"ROC-AUC: {metrics['roc_auc']}")

# ---------------------------------------------------------------------------
# 5. Feature importance (for RandomForest/GradientBoosting) or coefficients (LogReg)
# ---------------------------------------------------------------------------
feature_names = (
    FEATURES_NUM
    + list(best_pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
)
clf = best_pipeline.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
else:
    importances = np.zeros(len(feature_names))

importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
metrics["feature_importance"] = importance_df.to_dict(orient="records")

print("\nTop 5 features:")
print(importance_df.head())

# ---------------------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------------------
joblib.dump(best_pipeline, "/home/claude/placement-predictor/backend/model.joblib")

with open("/home/claude/placement-predictor/backend/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

branches_sorted = sorted(df["branch"].unique().tolist())
specializations_sorted = sorted(df["specialization"].unique().tolist())

schema = {
    "numeric_features": {
        "cgpa": {"min": 5.0, "max": 10.0, "step": 0.01, "default": 7.5, "label": "CGPA"},
        "ssc_percentage": {"min": 40, "max": 100, "step": 0.5, "default": 80, "label": "SSC %"},
        "hsc_percentage": {"min": 40, "max": 100, "step": 0.5, "default": 78, "label": "HSC %"},
        "internships": {"min": 0, "max": 5, "step": 1, "default": 1, "label": "Internships"},
        "projects": {"min": 0, "max": 8, "step": 1, "default": 2, "label": "Projects"},
        "certifications": {"min": 0, "max": 6, "step": 1, "default": 1, "label": "Certifications"},
        "dsa_score": {"min": 0, "max": 100, "step": 1, "default": 60, "label": "DSA / Coding Test Score"},
        "communication_score": {"min": 0, "max": 100, "step": 1, "default": 65, "label": "Communication Score"},
        "backlogs": {"min": 0, "max": 5, "step": 1, "default": 0, "label": "Active Backlogs"},
        "extra_curricular": {"min": 0, "max": 1, "step": 1, "default": 0, "label": "Extra-curricular Active (0/1)"},
    },
    "categorical_features": {
        "branch": branches_sorted,
        "specialization": specializations_sorted,
    },
}
with open("/home/claude/placement-predictor/backend/schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\nSaved: model.joblib, metrics.json, schema.json")
