#!/usr/bin/env python3
"""
Trek Guardian - Decision Tree Training Script
==============================================

This script trains a Decision Tree classifier on altitude-SpO2-heart rate data
to predict hypoxia risk levels for high-altitude trekkers.

Author: Trek Guardian Team
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json
from datetime import datetime


RISK_LABELS = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
FEATURE_NAMES = ['altitude', 'spo2', 'heartRate']


def load_dataset(filepath):
    """Load and preprocess the altitude-SpO2 dataset."""
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    df = df.dropna()
    print(f"[INFO] After dropping NaN: {df.shape}")
    
    return df


def create_risk_labels(df):
    """
    Create risk labels based on clinical thresholds for high-altitude hypoxia.
    
    Risk Levels:
    - CRITICAL: SpO2 < 88% AND Altitude > 3800m
    - HIGH:     SpO2 < 90% AND Altitude > 3200m
    - MODERATE: SpO2 < 93% AND Altitude > 2500m
    - LOW:      All other conditions
    """
    print("[INFO] Creating risk labels based on clinical thresholds...")
    
    def classify_risk(row):
        altitude = row['altitude']
        spo2 = row['spo2']
        
        if spo2 < 88 and altitude > 3800:
            return 'CRITICAL'
        elif spo2 < 90 and altitude > 3200:
            return 'HIGH'
        elif spo2 < 93 and altitude > 2500:
            return 'MODERATE'
        else:
            return 'LOW'
    
    df['risk'] = df.apply(classify_risk, axis=1)
    
    print(f"[INFO] Risk distribution:")
    print(df['risk'].value_counts())
    
    return df


def train_model(df, max_depth=5, random_state=42):
    """Train a Decision Tree classifier."""
    print(f"\n[INFO] Training Decision Tree (max_depth={max_depth})...")
    
    X = df[FEATURE_NAMES].values
    y = df['risk'].values
    
    label_encoder = LabelEncoder()
    label_encoder.fit(RISK_LABELS)
    y_encoded = label_encoder.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )
    
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[INFO] Model Accuracy: {accuracy:.4f}")
    
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5)
    print(f"[INFO] Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    print("\n[INFO] Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    print("\n[INFO] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return clf, label_encoder


def analyze_feature_importance(clf):
    """Analyze and display feature importance."""
    print("\n[INFO] Feature Importance:")
    importance = clf.feature_importances_
    for name, imp in zip(FEATURE_NAMES, importance):
        print(f"  {name}: {imp:.4f}")


def export_model_rules(clf, label_encoder, output_path):
    """Export model rules to readable format."""
    print(f"\n[INFO] Exporting model rules to {output_path}...")
    
    rules_text = export_text(clf, feature_names=FEATURE_NAMES)
    
    with open(output_path, 'w') as f:
        f.write("Trek Guardian - Decision Tree Rules\n")
        f.write("=" * 50 + "\n\n")
        f.write(rules_text)
    
    print(f"[INFO] Rules exported successfully!")


def save_model(clf, label_encoder, output_dir):
    """Save the trained model and encoder."""
    print(f"\n[INFO] Saving model to {output_dir}...")
    
    model_path = os.path.join(output_dir, 'trek_guardian_model.pkl')
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    
    joblib.dump(clf, model_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Label encoder saved to: {encoder_path}")
    
    return model_path, encoder_path


def generate_sample_dataset(output_path, n_samples=1000):
    """Generate synthetic dataset for demonstration purposes."""
    print(f"\n[INFO] Generating synthetic dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    altitudes = np.random.uniform(500, 6000, n_samples)
    base_spo2 = 98 - (altitudes / 500)
    spo2_noise = np.random.normal(0, 3, n_samples)
    spo2 = np.clip(base_spo2 + spo2_noise, 70, 100)
    
    base_hr = 70 + (altitudes / 100)
    hr_noise = np.random.normal(0, 10, n_samples)
    heart_rate = np.clip(base_hr + hr_noise, 50, 140)
    
    temperature = 20 - (altitudes / 200) + np.random.normal(0, 2, n_samples)
    humidity = np.random.uniform(20, 80, n_samples)
    
    df = pd.DataFrame({
        'altitude': altitudes,
        'spo2': spo2,
        'heartRate': heart_rate,
        'temperature': temperature,
        'humidity': humidity
    })
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Sample dataset saved to: {output_path}")
    
    return df


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Trek Guardian - Decision Tree Training Pipeline")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'dataset', 'altitude_spo2_dataset.csv')
    output_dir = os.path.join(base_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset not found at {dataset_path}")
        print("[INFO] Generating sample dataset for demonstration...")
        dataset_path = os.path.join(output_dir, 'sample_dataset.csv')
        generate_sample_dataset(dataset_path)
    
    df = load_dataset(dataset_path)
    df = create_risk_labels(df)
    
    clf, label_encoder = train_model(df, max_depth=6)
    analyze_feature_importance(clf)
    
    export_model_rules(clf, label_encoder, os.path.join(output_dir, 'model_rules.txt'))
    
    model_path, encoder_path = save_model(clf, label_encoder, output_dir)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review model accuracy and confusion matrix")
    print(f"2. Run export_rules.py to generate C++ code")
    print(f"3. Integrate the exported rules into ml_model.cpp")


if __name__ == "__main__":
    main()
