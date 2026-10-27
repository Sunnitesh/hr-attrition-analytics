"""Stage 2 - Leak-free training of the 10-feature Random Forest.

Run:  python train_model.py
Steps (no information from the test set is used before the final evaluation):
  1. Train/test split FIRST (80/20, stratified, random_state=42)
  2. Scaler is fit on training data only (inside each CV fold as well)
  3. Decision threshold is chosen with 5-fold stratified CV on the TRAIN set only
  4. Model is evaluated ONCE on the held-out test set
Outputs: hr_attrition_model_10.pkl, hr_attrition_scaler_10.pkl, model_info.json
"""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
FEATURES = ["Age", "DistanceFromHome", "EnvironmentSatisfaction", "JobSatisfaction",
            "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears",
            "WorkLifeBalance", "YearsAtCompany", "OverTime"]
SEED = 42


def new_rf():
    return RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")


def main():
    # Data comes from the SQL database (run setup_db.py first)
    con = sqlite3.connect(BASE / "hr.db")
    df = pd.read_sql_query(
        "SELECT " + ", ".join(FEATURES) + ", Attrition FROM employees", con)
    con.close()
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    y = (df["Attrition"] == "Yes").astype(int)
    X = df[FEATURES]

    # 1. split BEFORE any scaling / threshold selection
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # 3. threshold from 5-fold CV on the training set only (scaler refit inside each fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_pipeline(StandardScaler(), new_rf()),
                            X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    best_t, best_score = 0.5, -1
    for t in np.arange(0.10, 0.61, 0.01):
        p = (oof >= t).astype(int)
        prec, rec = precision_score(y_tr, p, zero_division=0), recall_score(y_tr, p)
        f2 = 0 if prec + rec == 0 else 5 * prec * rec / (4 * prec + rec)  # F2: favours recall
        if f2 > best_score:
            best_t, best_score = round(float(t), 2), f2

    # 2. scaler fit on training data only, then final model on the training set
    scaler = StandardScaler().fit(X_tr)
    model = new_rf().fit(scaler.transform(X_tr), y_tr)

    # 4. single evaluation on the untouched test set
    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    pred = (proba >= best_t).astype(int)
    metrics = {
        "threshold": best_t,
        "test_size": int(len(y_te)),
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "recall": round(recall_score(y_te, pred), 3),
        "precision": round(precision_score(y_te, pred), 3),
        "f1": round(f1_score(y_te, pred), 3),
        "roc_auc": round(roc_auc_score(y_te, proba), 3),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "feature_importance": dict(zip(FEATURES, [round(float(v), 4) for v in model.feature_importances_])),
    }

    joblib.dump(model, BASE / "hr_attrition_model_10.pkl")
    joblib.dump(scaler, BASE / "hr_attrition_scaler_10.pkl")
    (BASE / "model_info.json").write_text(json.dumps(metrics, indent=2))

    print(f"Threshold chosen by 5-fold CV on training data: {best_t}")
    for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]:
        print(f"  test {k:10s}: {metrics[k]}")
    print("  confusion matrix [[TN FP] [FN TP]]:", metrics["confusion_matrix"])
    print("Saved model, scaler and model_info.json")


if __name__ == "__main__":
    main()
