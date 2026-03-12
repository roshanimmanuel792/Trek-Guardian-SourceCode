#!/usr/bin/env python3
"""
Trek Guardian - Decision Tree to C++ Export Script
===================================================

This script converts a trained Decision Tree model to optimized C++ code
suitable for embedded systems (ESP8266).

The exported code uses simple if-else statements for fast inference
with minimal memory footprint.

Author: Trek Guardian Team
License: MIT
"""

import os
import json
import joblib
import numpy as np
from sklearn.tree import _tree
from datetime import datetime


FEATURE_NAMES = ['altitude', 'spo2', 'heartRate']
RISK_LABELS = ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']


def get_tree_rules(tree, feature_names, class_names):
    """
    Extract decision rules from a trained Decision Tree.
    
    Returns a list of rule dictionaries.
    """
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
            left_rule.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], depth + 1, left_rule)
            
            right_rule = rule.copy()
            right_rule.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            class_counts = tree_.value[node][0]
            class_idx = np.argmax(class_counts)
            class_name = class_names[class_idx]
            confidence = class_counts[class_idx] / class_counts.sum()
            
            rules.append({
                'conditions': rule,
                'prediction': class_name,
                'confidence': float(confidence),
                'samples': int(class_counts.sum())
            })
    
    recurse(0, 0, [])
    return rules


def tree_to_cpp_rules(clf, class_names):
    """
    Convert Decision Tree to C++ if-else statements.
    
    Optimized for embedded systems with minimal memory usage.
    """
    tree = clf.tree_
    rules = []
    
    def generate_code(node, depth=0, indent="    "):
        indent_str = indent * depth
        
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            class_idx = np.argmax(tree.value[node][0])
            return f'{indent_str}return "{class_names[class_idx]}";'
        
        feature = FEATURE_NAMES[tree.feature[node]]
        threshold = tree.threshold[node]
        
        left_code = generate_code(tree.children_left[node], depth + 1, indent)
        right_code = generate_code(tree.children_right[node], depth + 1, indent)
        
        cpp_code = f'{indent_str}if ({feature} <= {threshold:.2f}) {{\n'
        cpp_code += f'{indent_str}    {left_code.replace(indent, indent + "    ")}\n'
        cpp_code += f'{indent_str}}} else {{\n'
        cpp_code += f'{indent_str}    {right_code.replace(indent, indent + "    ")}\n'
        cpp_code += f'{indent_str}}}'
        
        return cpp_code
    
    return generate_code(0)


def export_to_cpp_header(output_path, model_name="TrekGuardianML"):
    """Export C++ header file with model configuration."""
    
    header_content = f'''/**
 * Trek Guardian - Embedded ML Model
 * =================================
 * Decision Tree classifier exported for ESP8266
 * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * This is an optimized version for embedded systems.
 * Uses simple if-else logic for fast inference.
 */

#ifndef {model_name}_H
#define {model_name}_H

#include <Arduino.h>

#define MODEL_VERSION "1.0.0"
#define NUM_FEATURES 3

// Feature names for reference
// 0: altitude (meters)
// 1: spo2 (percentage)
// 2: heartRate (BPM)

const char* predictRisk(float altitude, float spo2, float heartRate);

#endif
'''
    
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"[INFO] C++ header exported: {output_path}")
    return output_path


def export_to_cpp_source(clf, class_names, output_path):
    """Export C++ source file with decision tree implementation."""
    
    tree_code = tree_to_cpp_rules(clf, class_names)
    
    source_content = f'''/**
 * Trek Guardian - Embedded ML Model Source
 * ========================================
 * Decision Tree inference implementation
 */

#include "ml_model.h"

/**
 * Predict risk level based on altitude, SpO2, and heart rate
 * 
 * @param altitude   - Current altitude in meters
 * @param spo2       - Blood oxygen saturation (0-100%)
 * @param heartRate  - Heart rate in BPM
 * @return Risk level as String: "LOW", "MODERATE", "HIGH", or "CRITICAL"
 */
String predictRisk(float altitude, float spo2, float heartRate) {{
{tree_code}
}}
'''
    
    with open(output_path, 'w') as f:
        f.write(source_content)
    
    print(f"[INFO] C++ source exported: {output_path}")


def export_json_model(clf, label_encoder, output_path):
    """Export model as JSON for web dashboard integration."""
    
    tree = clf.tree_
    
    model_data = {
        "model_type": "DecisionTreeClassifier",
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_names": list(label_encoder.classes_),
        "max_depth": clf.max_depth,
        "n_classes": clf.n_classes_,
        "tree": {
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "left_child": tree.children_left.tolist(),
            "right_child": tree.children_right.tolist(),
            "value": tree.value.tolist()
        },
        "exported_at": datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"[INFO] JSON model exported: {output_path}")


def export_rules_summary(rules, output_path):
    """Export human-readable rules summary."""
    
    with open(output_path, 'w') as f:
        f.write("Trek Guardian - Decision Tree Rules Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, rule in enumerate(rules, 1):
            f.write(f"\n--- Rule {i} ---\n")
            f.write(f"Conditions: {' AND '.join(rule['conditions'])}\n")
            f.write(f"Prediction: {rule['prediction']}\n")
            f.write(f"Confidence: {rule['confidence']*100:.1f}%\n")
            f.write(f"Samples: {rule['samples']}\n")
    
    print(f"[INFO] Rules summary exported: {output_path}")


def main():
    """Main export pipeline."""
    print("=" * 60)
    print("Trek Guardian - Model Export Pipeline")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'trek_guardian_model.pkl')
    encoder_path = os.path.join(base_dir, 'label_encoder.pkl')
    
    if not os.path.exists(model_path):
        print("[ERROR] Model file not found!")
        print("[ERROR] Please run train_model.py first to train the model.")
        return
    
    print("[INFO] Loading trained model...")
    clf = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    class_names = list(label_encoder.classes_)
    print(f"[INFO] Classes: {class_names}")
    
    print("\n[INFO] Extracting decision rules...")
    rules = get_tree_rules(clf.tree_, FEATURE_NAMES, class_names)
    print(f"[INFO] Extracted {len(rules)} leaf rules")
    
    print("\n[INFO] Exporting to various formats...")
    
    firmware_dir = os.path.join(base_dir, '..', 'firmware')
    export_to_cpp_header(os.path.join(firmware_dir, 'ml_model.h'))
    
    cpp_source = os.path.join(firmware_dir, 'ml_model.cpp')
    tree_code = tree_to_cpp_rules(clf, class_names)
    with open(cpp_source, 'w') as f:
        f.write(f'''/**
 * Trek Guardian - Embedded ML Model Source
 * ========================================
 * Decision Tree inference implementation
 * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

#include "ml_model.h"

String predictRisk(float altitude, float spo2, float heartRate) {{
{tree_code}
}}
''')
    print(f"[INFO] C++ source exported: {cpp_source}")
    
    export_json_model(clf, label_encoder, os.path.join(base_dir, 'model.json'))
    export_rules_summary(rules, os.path.join(base_dir, 'rules_summary.txt'))
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - firmware/ml_model.h")
    print(f"  - firmware/ml_model.cpp")
    print(f"  - ml_model/model.json")
    print(f"  - ml_model/rules_summary.txt")
    print("\nThe firmware is now ready for upload to ESP8266!")


if __name__ == "__main__":
    main()
