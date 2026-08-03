#!/usr/bin/env python3
"""
Subject-wise evaluation report for the early hypoxia warning models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from features import FIRMWARE_FEATURES, FULL_FEATURES
from labeling import RISK_LABELS, naive_baseline_predict

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PROCESSED = ROOT_DIR / "dataset" / "processed"
METRICS_DIR = BASE_DIR / "metrics"


def main() -> None:
    labeled_path = PROCESSED / "labeled_windows.csv"
    if not labeled_path.exists():
        raise SystemExit("Run train_model.py first to produce labeled_windows.csv")

    labeled = pd.read_csv(labeled_path)
    gbm = joblib.load(BASE_DIR / "trek_guardian_gbm.pkl")
    tree = joblib.load(BASE_DIR / "trek_guardian_model.pkl")
    le = joblib.load(BASE_DIR / "label_encoder.pkl")

    for col in FULL_FEATURES:
        if col not in labeled.columns:
            labeled[col] = 0.0
    labeled[FULL_FEATURES] = labeled[FULL_FEATURES].fillna(0.0)

    y_true = le.transform(labeled["risk"])
    X_full = labeled[FULL_FEATURES].to_numpy(dtype=float)
    X_fw = labeled[FIRMWARE_FEATURES].to_numpy(dtype=float)

    gbm_pred = gbm.predict(X_full)
    tree_pred = tree.predict(X_fw)
    base_pred = le.transform(naive_baseline_predict(labeled["spo2"].to_numpy(dtype=float)))

    report = {
        "gbm": classification_report(
            y_true, gbm_pred, target_names=list(le.classes_), output_dict=True, zero_division=0
        ),
        "tree": classification_report(
            y_true, tree_pred, target_names=list(le.classes_), output_dict=True, zero_division=0
        ),
        "baseline": classification_report(
            y_true, base_pred, target_names=list(le.classes_), output_dict=True, zero_division=0
        ),
        "macro_f1": {
            "gbm": float(f1_score(y_true, gbm_pred, average="macro", zero_division=0)),
            "tree": float(f1_score(y_true, tree_pred, average="macro", zero_division=0)),
            "baseline": float(f1_score(y_true, base_pred, average="macro", zero_division=0)),
        },
    }

    # Per-subject F1 for GBM
    per_subject = {}
    for sid, idx in labeled.groupby("subject_id").groups.items():
        mask = labeled.index.isin(idx)
        yt = y_true[mask]
        yp = gbm_pred[mask]
        per_subject[sid] = float(f1_score(yt, yp, average="macro", zero_division=0))
    report["per_subject_gbm_macro_f1"] = per_subject

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "holdout_style_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, pred, title in zip(
        axes,
        [gbm_pred, tree_pred, base_pred],
        ["GBM (full features)", "Distilled Tree", "Naive SpO2 baseline"],
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            pred,
            display_labels=list(le.classes_),
            ax=ax,
            colorbar=False,
            xticks_rotation=45,
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    # Lead-time proxy: among CRITICAL labels, fraction where model alerts HIGH/CRITICAL
    # while current SpO2 is still >= 93 (true early warning)
    early_mask = (labeled["risk"] == "CRITICAL") & (labeled["spo2"] >= 93)
    if early_mask.any():
        gbm_early = np.isin(le.inverse_transform(gbm_pred[early_mask]), ["HIGH", "CRITICAL"]).mean()
        tree_early = np.isin(le.inverse_transform(tree_pred[early_mask]), ["HIGH", "CRITICAL"]).mean()
        base_early = np.isin(le.inverse_transform(base_pred[early_mask]), ["HIGH", "CRITICAL"]).mean()
        lead = {
            "n_early_critical_cases": int(early_mask.sum()),
            "gbm_alert_rate": float(gbm_early),
            "tree_alert_rate": float(tree_early),
            "baseline_alert_rate": float(base_early),
        }
        (METRICS_DIR / "lead_time_proxy.json").write_text(json.dumps(lead, indent=2), encoding="utf-8")
        print("[LEAD-TIME PROXY]", json.dumps(lead, indent=2))

    print("[OK] Wrote evaluation artifacts to", METRICS_DIR)
    print(json.dumps(report["macro_f1"], indent=2))


if __name__ == "__main__":
    main()
