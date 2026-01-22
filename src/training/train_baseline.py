from pathlib import Path
import json
import hashlib
from src.data_loader import load_fd001

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configuration
PROJECT_ROOT = Path().resolve()
while not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_processed.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)

ARTIFACT_VERSION = "baseline_v1"
FEATURES = ["sensor_4", "sensor_11", "sensor_15"]
TARGET = "RUL_capped"
GROUP_COL = "unit_number"
RUL_CAP = 125
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_CONSERVATIVE = 1.0

# Load in data
df = load_fd001(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df[GROUP_COL].copy()

# Engine-level train/validation split
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, val_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

# Make sure there is no engine leakage
assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx])), "Engine leakage detected between train and validation sets"

# Fit scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train basline model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Extract model coefficients
model_coefficients = {
    feature: float(coef)
    for feature, coef in zip(FEATURES, model.coef_)
}

intercept = float(model.intercept_)

# Validate performance
y_val_pred = model.predict(X_val_scaled)

mae = mean_absolute_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

# Training statistics for monitoring
training_stats = {}

for feature in FEATURES:
    stats = {
        "min": float(X_train[feature].min()),
        "max": float(X_train[feature].max()),
        "mean": float(X_train[feature].mean()),
        "std": float(X_train[feature].std()),
    }
    training_stats[feature] = stats

# Compute a deterministic hash of the dataset file to track exactly which version of the data this model was trained on
hasher = hashlib.sha256()
with open(DATA_PATH, 'rb') as f:
    hasher.update(f.read())

data_hash = hasher.hexdigest()

# Write artifacts for inference
joblib.dump(model, ARTIFACT_DIR / "model.joblib")
joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")

# feature_schema.json
feature_schema = {
    "artifact_version": ARTIFACT_VERSION,
    "dataset": "CMAPSS",
    "subset": "FD001",
    "rul_cap": RUL_CAP,
    "features": FEATURES,
    "feature_order_enforced": True,
    "training_stats": training_stats,
    "notes": {
        "target": TARGET,
        "input_expectation": "Single timestep snapshot per engine cycle",
        "extra_columns_policy": "HARD_ERROR"
    }
}

with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

# metrics.json
metrics = {
    "artifact_version": ARTIFACT_VERSION,

    "model_family": "LinearRegression",
    "model_name": "OLS",
    "features": FEATURES,
    "target": TARGET,

    "split_strategy": {
        "type": "GroupShuffleSplit",
        "group": GROUP_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    },

    "preprocessing": {
        "rul_definition": "max_cycle - time_in_cycles",
        "rul_cap": RUL_CAP,
        "scaling": "StandardScaler (Fit on X_train only)",
    },

    "validation_metrics": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    },

    "model_coefficients": model_coefficients,
    "intercept": intercept,

    "decision_policy": {
        "lower_bound_method": "rul_lower = rul_pred - k * mae",
        "k": K_CONSERVATIVE
    },

    "data_fingerprint": {
        "processed_csv": str(DATA_PATH),
        "sha256": data_hash
    }
}

with open(ARTIFACT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Artifacts written successfully")