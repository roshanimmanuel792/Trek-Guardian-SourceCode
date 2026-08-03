#!/usr/bin/env python3
"""
Trek Guardian — Research-grade early hypoxia warning training.

- Real data only (fails if processed vitals missing)
- Future-window labels (non-circular)
- Subject-wise GroupKFold
- HistGradientBoosting (cloud) + distilled DecisionTree (firmware)
- Naive current-SpO2 baseline comparison
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_sample_weight

from features import FIRMWARE_FEATURES, FULL_FEATURES, build_features_for_all_subjects
from labeling import RISK_LABELS, attach_future_labels, naive_baseline_predict

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PROCESSED = ROOT_DIR / "dataset" / "processed"
METRICS_DIR = BASE_DIR / "metrics"

WINDOW_SEC = 45
STEP_SEC = 5
HORIZON_SEC = 90
TREE_MAX_DEPTH = 7
RANDOM_STATE = 42


def load_vitals() -> pd.DataFrame:
    parquet = PROCESSED / "vitals_1hz.parquet"
    csv = PROCESSED / "vitals_1hz.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise SystemExit(
        "Processed real dataset not found.\n"
        "Run:\n"
        "  python download_datasets.py\n"
        "  python preprocess.py\n"
        "Synthetic fallback is intentionally disabled."
    )


def _macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def evaluate_split(name: str, y_true, y_pred, label_encoder: LabelEncoder) -> dict:
    labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": _macro_f1(y_true, y_pred, labels),
        "classification_report": report,
        "confusion_matrix": cm,
        "label_order": list(label_encoder.classes_),
    }


def subject_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spo2_current: np.ndarray,
    label_encoder: LabelEncoder,
    n_splits: int = 5,
) -> dict:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise SystemExit("Need at least 2 subjects for GroupKFold validation.")

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    gbm_f1s, tree_f1s, base_f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sw = compute_sample_weight("balanced", y_train)
        gbm = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
        gbm.fit(X_train, y_train, sample_weight=sw)
        gbm_pred = gbm.predict(X_test)

        tree = DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        tree.fit(X_train, y_train)
        tree_pred = tree.predict(X_test)

        base_pred_str = naive_baseline_predict(spo2_current[test_idx])
        base_pred = label_encoder.transform(base_pred_str)

        gbm_m = evaluate_split(f"gbm_fold_{fold}", y_test, gbm_pred, label_encoder)
        tree_m = evaluate_split(f"tree_fold_{fold}", y_test, tree_pred, label_encoder)
        base_m = evaluate_split(f"baseline_fold_{fold}", y_test, base_pred, label_encoder)

        gbm_f1s.append(gbm_m["macro_f1"])
        tree_f1s.append(tree_m["macro_f1"])
        base_f1s.append(base_m["macro_f1"])
        fold_metrics.append({"fold": fold, "gbm": gbm_m, "tree": tree_m, "baseline": base_m})
        print(
            f"[FOLD {fold}] GBM F1={gbm_m['macro_f1']:.3f} | "
            f"Tree F1={tree_m['macro_f1']:.3f} | Baseline F1={base_m['macro_f1']:.3f}"
        )

    return {
        "n_splits": n_splits,
        "gbm_macro_f1_mean": float(np.mean(gbm_f1s)),
        "gbm_macro_f1_std": float(np.std(gbm_f1s)),
        "tree_macro_f1_mean": float(np.mean(tree_f1s)),
        "tree_macro_f1_std": float(np.std(tree_f1s)),
        "baseline_macro_f1_mean": float(np.mean(base_f1s)),
        "baseline_macro_f1_std": float(np.std(base_f1s)),
        "beats_baseline_gbm": float(np.mean(gbm_f1s)) > float(np.mean(base_f1s)),
        "beats_baseline_tree": float(np.mean(tree_f1s)) > float(np.mean(base_f1s)),
        "folds": fold_metrics,
    }


def train_final_models(
    X_full: np.ndarray,
    X_fw: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
):
    sw = compute_sample_weight("balanced", y)
    gbm = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=RANDOM_STATE,
    )
    gbm.fit(X_full, y, sample_weight=sw)

    # Knowledge distillation: tree fits GBM predicted labels on firmware features
    soft_targets = gbm.predict(X_full)
    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(X_fw, soft_targets)

    return gbm, tree


def main() -> None:
    print("=" * 60)
    print("Trek Guardian — Early Hypoxia Warning Training")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    vitals = load_vitals()
    print(f"[INFO] Loaded vitals: {len(vitals)} rows, {vitals['subject_id'].nunique()} subjects")

    features = build_features_for_all_subjects(vitals, window_sec=WINDOW_SEC, step_sec=STEP_SEC)
    if features.empty:
        raise SystemExit("No feature windows produced.")

    labeled = attach_future_labels(features, vitals, horizon_sec=HORIZON_SEC)
    print(f"[INFO] Labeled windows: {len(labeled)}")
    print("[INFO] Risk distribution:")
    print(labeled["risk"].value_counts())

    # Fill missing RR
    for col in FULL_FEATURES:
        if col not in labeled.columns:
            labeled[col] = 0.0
    labeled[FULL_FEATURES] = labeled[FULL_FEATURES].fillna(0.0)

    label_encoder = LabelEncoder()
    label_encoder.fit(RISK_LABELS)
    # Keep only known labels
    labeled = labeled[labeled["risk"].isin(RISK_LABELS)].copy()
    y = label_encoder.transform(labeled["risk"])
    groups = labeled["subject_id"].to_numpy()
    spo2_current = labeled["spo2"].to_numpy(dtype=float)

    X_full = labeled[FULL_FEATURES].to_numpy(dtype=float)
    X_fw = labeled[FIRMWARE_FEATURES].to_numpy(dtype=float)

    print("\n[INFO] Subject-wise GroupKFold on FULL features (GBM/Tree proxy)...")
    cv_full = subject_group_cv(X_full, y, groups, spo2_current, label_encoder)

    print("\n[INFO] Subject-wise GroupKFold on FIRMWARE features (deployable tree)...")
    cv_fw = subject_group_cv(X_fw, y, groups, spo2_current, label_encoder)

    print("\n[INFO] Fitting final models on all data...")
    gbm, tree = train_final_models(X_full, X_fw, y, label_encoder)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(gbm, BASE_DIR / "trek_guardian_gbm.pkl")
    joblib.dump(tree, BASE_DIR / "trek_guardian_model.pkl")
    joblib.dump(label_encoder, BASE_DIR / "label_encoder.pkl")

    # Also save feature schema for the Flask /predict endpoint
    schema = {
        "full_features": FULL_FEATURES,
        "firmware_features": FIRMWARE_FEATURES,
        "window_sec": WINDOW_SEC,
        "horizon_sec": HORIZON_SEC,
        "task": "early_hypoxia_warning",
        "labeling": "future_window_spo2_severity",
    }
    (BASE_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    rules_text = export_text(tree, feature_names=FIRMWARE_FEATURES)
    (BASE_DIR / "model_rules.txt").write_text(
        "Trek Guardian — Distilled Decision Tree Rules\n" + "=" * 50 + "\n\n" + rules_text,
        encoding="utf-8",
    )

    # Feature importances for tree
    importances = {
        name: float(imp) for name, imp in zip(FIRMWARE_FEATURES, tree.feature_importances_)
    }

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(labeled)),
        "n_subjects": int(labeled["subject_id"].nunique()),
        "risk_distribution": labeled["risk"].value_counts().to_dict(),
        "cv_full_features": {
            k: v for k, v in cv_full.items() if k != "folds"
        },
        "cv_firmware_features": {
            k: v for k, v in cv_fw.items() if k != "folds"
        },
        "tree_feature_importances": importances,
        "schema": schema,
        "notes": [
            "Labels come from future SpO2 horizon, not current thresholds alone.",
            "Naive baseline uses current SpO2 thresholds only.",
            "Chamber datasets (Harespod/HAPP) ≠ Himalayan field AMS labels.",
        ],
    }
    # Keep fold details in a separate heavier file
    (METRICS_DIR / "cv_folds.json").write_text(
        json.dumps({"full": cv_full, "firmware": cv_fw}, indent=2),
        encoding="utf-8",
    )
    (METRICS_DIR / "training_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save labeled feature table for evaluate.py / analysis
    labeled.to_csv(PROCESSED / "labeled_windows.csv", index=False)

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"GBM  GroupKFold macro-F1:  {cv_full['gbm_macro_f1_mean']:.3f} ± {cv_full['gbm_macro_f1_std']:.3f}")
    print(f"Tree GroupKFold macro-F1: {cv_fw['tree_macro_f1_mean']:.3f} ± {cv_fw['tree_macro_f1_std']:.3f}")
    print(f"Baseline macro-F1:        {cv_fw['baseline_macro_f1_mean']:.3f} ± {cv_fw['baseline_macro_f1_std']:.3f}")
    print(f"GBM beats baseline:  {cv_full['beats_baseline_gbm']}")
    print(f"Tree beats baseline: {cv_fw['beats_baseline_tree']}")
    print("\nNext: python export_rules.py")


if __name__ == "__main__":
    main()
