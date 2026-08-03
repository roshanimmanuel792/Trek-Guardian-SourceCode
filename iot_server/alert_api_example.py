from flask import Flask, request, jsonify
import datetime
import json
from pathlib import Path

import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent / "ml_model"

_gbm = None
_tree = None
_label_encoder = None
_schema = None


def load_models():
    global _gbm, _tree, _label_encoder, _schema
    if _gbm is not None:
        return

    gbm_path = ML_DIR / "trek_guardian_gbm.pkl"
    tree_path = ML_DIR / "trek_guardian_model.pkl"
    enc_path = ML_DIR / "label_encoder.pkl"
    schema_path = ML_DIR / "feature_schema.json"

    if not gbm_path.exists() or not enc_path.exists():
        raise FileNotFoundError(
            "Trained models not found. Run ml_model/train_model.py first."
        )

    _gbm = joblib.load(gbm_path)
    _tree = joblib.load(tree_path) if tree_path.exists() else None
    _label_encoder = joblib.load(enc_path)
    if schema_path.exists():
        _schema = json.loads(schema_path.read_text(encoding="utf-8"))
    else:
        _schema = {
            "full_features": [
                "altitude", "spo2", "heartRate", "spo2_ewma", "hr_ewma",
                "spo2_slope", "hr_slope", "spo2_min", "spo2_std", "hr_std",
                "hr_spo2_ratio", "ascent_rate", "respiratory_rate",
            ],
            "firmware_features": [
                "altitude", "spo2", "heartRate", "spo2_ewma", "hr_ewma",
                "spo2_slope", "hr_slope",
            ],
        }


def _vector_from_payload(data, feature_names):
    spo2 = float(data.get("spo2", 0))
    hr = float(data.get("heartRate", 0))
    altitude = float(data.get("altitude", 0))
    spo2_ewma = float(data.get("spo2_ewma", spo2))
    hr_ewma = float(data.get("hr_ewma", hr))
    spo2_slope = float(data.get("spo2_slope", 0))
    hr_slope = float(data.get("hr_slope", 0))
    spo2_min = float(data.get("spo2_min", spo2))
    spo2_std = float(data.get("spo2_std", 0))
    hr_std = float(data.get("hr_std", 0))
    ascent_rate = float(data.get("ascent_rate", 0))
    rr = float(data.get("respiratory_rate", 0))

    values = {
        "altitude": altitude,
        "spo2": spo2,
        "heartRate": hr,
        "spo2_ewma": spo2_ewma,
        "hr_ewma": hr_ewma,
        "spo2_slope": spo2_slope,
        "hr_slope": hr_slope,
        "spo2_min": spo2_min,
        "spo2_std": spo2_std,
        "hr_std": hr_std,
        "hr_spo2_ratio": hr / max(spo2, 1.0),
        "ascent_rate": ascent_rate,
        "respiratory_rate": rr,
    }
    return np.array([[values[name] for name in feature_names]], dtype=float)


@app.route("/alert", methods=["POST"])
@app.route("/api/alerts", methods=["POST"])
def receive_alert():
    data = request.json or {}

    spo2 = data.get("spo2")
    altitude = data.get("altitude")
    heartRate = data.get("heartRate")
    latitude = data.get("lat")
    longitude = data.get("lon")
    risk = data.get("risk")

    timestamp = datetime.datetime.now()

    print("----- EMERGENCY ALERT RECEIVED -----")
    print("Time:", timestamp)
    print("SpO2:", spo2)
    print("Altitude:", altitude)
    print("Heart Rate:", heartRate)
    print("Location:", latitude, longitude)
    print("Risk Level:", risk)
    print("-----------------------------------")

    return jsonify({"status": "alert received"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Cloud scoring with the full-feature Gradient Boosting model.
    Accepts either a full feature payload or device vitals (+ optional EWMA fields).
    """
    try:
        load_models()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    data = request.json or {}
    X = _vector_from_payload(data, _schema["full_features"])
    pred_idx = int(_gbm.predict(X)[0])
    risk = _label_encoder.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(_gbm, "predict_proba"):
        proba = _gbm.predict_proba(X)[0]
        confidence = float(np.max(proba))
        class_probs = {
            cls: float(p) for cls, p in zip(_label_encoder.classes_, proba)
        }
    else:
        class_probs = None

    tree_risk = None
    if _tree is not None:
        X_fw = _vector_from_payload(data, _schema["firmware_features"])
        tree_idx = int(_tree.predict(X_fw)[0])
        tree_risk = _label_encoder.inverse_transform([tree_idx])[0]

    return jsonify(
        {
            "risk": risk,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "tree_risk": tree_risk,
            "model": "HistGradientBoostingClassifier",
            "task": "early_hypoxia_warning",
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    models_ready = (ML_DIR / "trek_guardian_gbm.pkl").exists()
    return jsonify({"status": "ok", "models_ready": models_ready}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
