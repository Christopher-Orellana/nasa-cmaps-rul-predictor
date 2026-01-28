# NASA CMAPSS FD001 — Predictive Maintenance System

Predictive maintenance system built on NASA CMAPSS FD001, designed for **systems-oriented** 
and **safety-critical** data roles, It is a **training → artifacts → inference** pipeline made
for gauging engine health and outputting a recommendation and status based on model prediction.

---

## What this system does

- Trains a **baseline** Remaining Useful Life (RUL) predictor using FD001
- Produces **versioned artifacts** (model, scaler, schema, metrics) for reproducible inference
- Runs **inference** with:
  - strict input schema validation (no missing/extra features)
  - feature ordering enforcement
  - numeric validity checks
  - distribution-based anomaly and extrapolation risk flags
  - conservative decision policy using a lower confidence bound
  - structured JSONL logging for audit trails

---

## Dataset and scope

- Dataset: **CMAPSS FD001** (single operating condition, single fault mode)
- Raw data: immutable (not tracked here)
- Processed dataset (single source of truth for all runs):
- `data/processed/fd001_processed.csv`
- Shared loader (all files load the same processed data):
- `src/data_loader.py -> load_fd001(data_path, rul_col='RUL', cap_rul=125)`
- **no sensor transformations**

Selected features (from EDA): `sensor_4`, `sensor_11`, `sensor_15`

---

## Baseline model
Trains a linear regression baseline using an engine-level split and
writes metrics and metadata to `artifacts/`.

- Split: engine-level `GroupShuffleSplit` (leakage prevented and assertions added)
- Scaling: `StandardScaler` fit on train only
- Model: OLS via `LinearRegression`
- Performance (validation):
- MAE ≈ 19 | RMSE ≈ 23 | R² ≈ 0.68
- Remaining error is **structural** (missing time context), not overfitting.

---

## Repository layout

```
nasa-cmaps-pipeline/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_baseline.py
│   └── inference/
│       ├── __init__.py
│       ├── predict.py
│       └── test_predict.py
├── data/processed/fd001_processed.csv
├── artifacts/
│   ├── feature_schema.json
│   ├── metrics.json
│   ├── model.joblib
│   └── scaler.joblib
├── logs/inference.jsonl
├── notebooks/
│   ├── eda.ipynb
│   └── modeling_baseline.ipynb
├── README.md
└── requirements.txt
```
## Artifacts & auditability

Training produces a **versioned artifact bundle** under `artifacts/` that defines the inference contract and enables reproducible, auditable runs.

**Artifacts**
- `model.joblib` *(gitignored)* — trained OLS baseline model  
- `scaler.joblib` *(gitignored)* — train-only `StandardScaler`  
- `feature_schema.json` *(committed)* — locked feature names and ordering  
- `metrics.json` *(committed)* — model metadata and validation results  

**Each artifact bundle contains**
- `artifact_version = "baseline_v1"`
- explicit feature list and target (`RUL_capped`)
- held-out validation metrics
- model coefficients and intercept
- training distribution statistics (mean / std / min / max) used for runtime safety checks
- **SHA256 fingerprint** of `fd001_processed.csv` to guarantee data lineage and reproducibility

Artifacts are treated as immutable once written. All inference runs against an explicit, versioned snapshot.

---

## Inference contract observability

Inference is intentionally **defensive** and assumes inputs
may be malformed, out-of-distribution, or unsafe.

`src/inference/predict.py` enforces:
- exact feature schema (no missing features, no extras)
- strict numeric validity checks (`np.isfinite`)
- extrapolation risk flags when inputs fall outside training min/max
- z-score anomaly detection relative to training mean/std (with zero-std guards)
- conservative lower-bound estimate:  
  `rul_lower = rul_pred - k × MAE`

Operational decisions are based on `rul_lower`, not the point estimate:

- **GREEN** — Continue operation  
- **AMBER** — Inspect  
- **RED** — Schedule maintenance  
- **CRITICAL** — Remove from service  

Every inference outputs a structured JSON record to:

- `logs/inference.jsonl`

providing a complete audit trail for post-hoc analysis.

---

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Train baseline and write artifacts
```bash
python -m src.training.train_baseline
```
### 3. Run a single inference example
```bash
python -m src.inference.test_predict
```
---

## License
MIT License © 2026 Christopher Orellana