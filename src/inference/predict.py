from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


# Define metadata for the script
EXPECTED_ARTIFACT_VERSION = "baseline_v1"
REQUIRED_METADATA = ['unit_number']
FEATURE_ORDER = ["sensor_4", "sensor_11", "sensor_15"]
ALLOWED_INPUT_KEYS = set(REQUIRED_METADATA + FEATURE_ORDER)

# Configuration for paths
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "artifacts").exists():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise RuntimeError("Could not find root containing /artifacts")
    PROJECT_ROOT = PROJECT_ROOT.parent

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Load artifacts
with open(ARTIFACT_DIR / "feature_schema.json") as f:
    feature_schema = json.load(f)

with open(ARTIFACT_DIR / "metrics.json") as f:
    metrics = json.load(f)

model = joblib.load(ARTIFACT_DIR / "model.joblib")
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")

# Check compatability for artifacts
if feature_schema["artifact_version"] != EXPECTED_ARTIFACT_VERSION:
    raise RuntimeError("Feature schema artifact version mismatch")

if metrics["artifact_version"] != EXPECTED_ARTIFACT_VERSION:
    raise RuntimeError("Metrics artifact version mismatch")

# Inference function
def predict_rul(input_data: dict) -> dict:
    """Run inference on a single engine snapshot
    input_data must contain the exact required features"""

    timestamp = datetime.utcnow().isoformat()

    # Validate schema
    missing_metadata = set(REQUIRED_METADATA) - set(input_data.keys())
    missing_features = set(FEATURE_ORDER) - set(input_data.keys())
    extra = set(input_data.keys()) - ALLOWED_INPUT_KEYS

    if missing_metadata:
        raise ValueError(f"Missing required metadata: {missing_metadata}")

    if missing_features:
        raise ValueError(f'Missing required features: {missing_features}')

    if extra:
        raise ValueError(f"Unexpected extra fields: {extra}")

    # Validate metadata
    unit_number = input_data['unit_number']
    if not isinstance(unit_number, (int, np.integer)):
        raise ValueError('unit_number must be an integer')

    # Create Dataframe with features in order
    X = pd.DataFrame([[input_data[feat] for feat in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # Check datatype and sensor values
    if not np.isfinite(X.values).all():
        raise ValueError("Input contains NaN or non-finite values")

    # Check for anomalies and ranges of sensor values
    extrapolation_risk = False
    input_anomaly = False

    for feature in FEATURE_ORDER:
        stats = feature_schema["training_stats"][feature]
        value = X.at[0, feature]

        if value < stats["min"] or value > stats["max"]:
            extrapolation_risk = True

        if stats["std"] > 0:
            z = (value - stats["mean"]) / stats["std"]
            if abs(z) > 3.0:
                input_anomaly = True

    # Scale and predict
    X_scaled = scaler.transform(X)
    rul_pred = float(model.predict(X_scaled)[0])

    # Create a conservative lower bound
    mae = metrics["validation_metrics"]["mae"]
    k = metrics["decision_policy"]["k"]
    rul_lower = rul_pred - k * mae

    # Decision logic
    if rul_lower > 60:
        risk_band = "GREEN"
        action = "CONTINUE"
    elif rul_lower > 30:
        risk_band = "AMBER"
        action = "INSPECT"
    elif rul_lower > 10:
        risk_band = "RED"
        action = "SCHEDULE_MAINTENANCE"
    else:
        risk_band = "CRITICAL"
        action = "REMOVE_FROM_SERVICE"

    # Write output inference.jsonl
    output = {
        "unit_number": int(unit_number),
        "timestamp": timestamp,
        "artifact_version": EXPECTED_ARTIFACT_VERSION,
        "rul_pred": rul_pred,
        "rul_lower": rul_lower,
        "risk_band": risk_band,
        "recommended_action": action,
        "flags": {
            "schema_error": False,
            "extrapolation_risk": extrapolation_risk,
            "input_anomaly": input_anomaly
        }
    }

    # Log and write inference.jsonl
    log_path = LOG_DIR / "inference.jsonl"
    with open(log_path, "a") as l:
        l.write(json.dumps(output) + "\n")

    return output