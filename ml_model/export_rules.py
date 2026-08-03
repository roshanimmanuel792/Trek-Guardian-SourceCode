#!/usr/bin/env python3
"""
Export distilled Decision Tree to ESP8266 C++ (firmware/ml_model.cpp).

Uses firmware-compatible features:
  altitude, spo2, heartRate, spo2_ewma, hr_ewma, spo2_slope, hr_slope
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import _tree

from features import FIRMWARE_FEATURES

BASE_DIR = Path(__file__).resolve().parent
FIRMWARE_DIR = BASE_DIR.parent / "firmware"


def get_tree_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, depth, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = rule.copy()
            left_rule.append(f"{name} <= {threshold:.4f}")
            recurse(tree_.children_left[node], depth + 1, left_rule)
            right_rule = rule.copy()
            right_rule.append(f"{name} > {threshold:.4f}")
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            class_counts = tree_.value[node][0]
            class_idx = int(np.argmax(class_counts))
            class_name = class_names[class_idx]
            confidence = float(class_counts[class_idx] / class_counts.sum())
            rules.append(
                {
                    "conditions": rule,
                    "prediction": class_name,
                    "confidence": confidence,
                    "samples": int(class_counts.sum()),
                }
            )

    recurse(0, 0, [])
    return rules


def tree_to_cpp(clf, class_names, indent="    "):
    tree = clf.tree_

    def generate(node, depth=1):
        pad = indent * depth
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            class_idx = int(np.argmax(tree.value[node][0]))
            return f'{pad}return "{class_names[class_idx]}";\n'

        feature = FIRMWARE_FEATURES[tree.feature[node]]
        threshold = tree.threshold[node]
        left = generate(tree.children_left[node], depth + 1)
        right = generate(tree.children_right[node], depth + 1)
        code = f"{pad}if ({feature} <= {threshold:.4f}) {{\n"
        code += left
        code += f"{pad}}} else {{\n"
        code += right
        code += f"{pad}}}\n"
        return code

    return generate(0)


def export_header(path: Path) -> None:
    content = f'''/**
 * Trek Guardian - Embedded ML Model
 * Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
 *
 * Early hypoxia warning — distilled Decision Tree.
 * Features must match firmware EWMA state in trek_guardian_main.ino.
 */

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <Arduino.h>

#define MODEL_VERSION "2.0.0"
#define NUM_FEATURES 7
#define EWMA_ALPHA 0.15f

// Firmware features (order matters for documentation only):
// altitude, spo2, heartRate, spo2_ewma, hr_ewma, spo2_slope, hr_slope

String predictRisk(
  float altitude,
  float spo2,
  float heartRate,
  float spo2_ewma,
  float hr_ewma,
  float spo2_slope,
  float hr_slope
);

#endif
'''
    path.write_text(content, encoding="utf-8")
    print(f"[INFO] Wrote {path}")


def export_source(clf, class_names, path: Path) -> None:
    body = tree_to_cpp(clf, class_names)
    content = f'''/**
 * Trek Guardian - Embedded ML Model Source
 * Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
 * Distilled Decision Tree for early hypoxia warning.
 */

#include "ml_model.h"

String predictRisk(
  float altitude,
  float spo2,
  float heartRate,
  float spo2_ewma,
  float hr_ewma,
  float spo2_slope,
  float hr_slope
) {{
{body}}}
'''
    path.write_text(content, encoding="utf-8")
    print(f"[INFO] Wrote {path}")


def main() -> None:
    print("=" * 60)
    print("Trek Guardian — Export Distilled Tree to Firmware")
    print("=" * 60)

    model_path = BASE_DIR / "trek_guardian_model.pkl"
    encoder_path = BASE_DIR / "label_encoder.pkl"
    if not model_path.exists():
        raise SystemExit("Model not found. Run train_model.py first.")

    clf = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    class_names = list(label_encoder.classes_)
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Features: {FIRMWARE_FEATURES}")

    rules = get_tree_rules(clf, FIRMWARE_FEATURES, class_names)
    print(f"[INFO] Leaf rules: {len(rules)}")

    export_header(FIRMWARE_DIR / "ml_model.h")
    export_source(clf, class_names, FIRMWARE_DIR / "ml_model.cpp")

    model_json = {
        "model_type": "DecisionTreeClassifier",
        "task": "early_hypoxia_warning",
        "feature_names": FIRMWARE_FEATURES,
        "class_names": class_names,
        "max_depth": int(clf.get_depth()),
        "n_leaves": int(clf.get_n_leaves()),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (BASE_DIR / "model.json").write_text(json.dumps(model_json, indent=2), encoding="utf-8")

    with open(BASE_DIR / "rules_summary.txt", "w", encoding="utf-8") as f:
        f.write("Trek Guardian — Distilled Tree Rules Summary\n")
        f.write("=" * 60 + "\n")
        for i, rule in enumerate(rules, 1):
            f.write(f"\n--- Rule {i} ---\n")
            f.write(f"Conditions: {' AND '.join(rule['conditions'])}\n")
            f.write(f"Prediction: {rule['prediction']}\n")
            f.write(f"Confidence: {rule['confidence']*100:.1f}%\n")
            f.write(f"Samples: {rule['samples']}\n")

    print("[DONE] Firmware ML sources updated. Rebuild/upload the Arduino sketch.")


if __name__ == "__main__":
    main()
