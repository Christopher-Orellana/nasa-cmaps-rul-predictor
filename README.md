# NASA CMAPSS FD001 — Predictive Maintenance System

Unplanned engine failures cost organizations millions in repairs, program delays, and downtime. This system predicts Remaining Useful Life (RUL) for jet engines using NASA’s CMAPSS FD001 dataset — the same simulation data used by NASA Glenn Research Center for propulsion health management research. My system outputs risk-banded recommendations to support maintenance triage decisions before failure occurs.

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
- Provides a **Streamlit dashboard** for a engine fleet overview, triage filtering and snapshot
  level inspection.

---

## Dataset and scope
CMAPSS originates from NASA Glenn Research Center’s propulsion health management research program. FD001 represents a single fault mode and single operating condition - a controlled scope chosen to establish a clean, reproducible baseline before extending to multi-condition subsets.
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
├── artifacts/
│   ├── feature_schema.json
│   ├── metrics.json
│   ├── model.joblib
│   └── scaler.joblib
├── dashboard/
│   ├── dashboard_data/
│   │   └── demo_inference.jsonl
│   └── app.py
├── data/
│   └── processed/
│       └── fd001_processed.csv
├── docs/
│   ├── failure_modes.md
│   ├── inference_pipeline.md
│   ├── system_design.md
│   └── training_pipeline.md
├── logs/
│   └── inference.jsonl
├── notebooks/
│   ├── 00_preprocess_fd001.ipynb
│   ├── 01_eda.ipynb
│   └── 02_modeling-baseline.ipynb
├── plots/
│   └── saved EDA figures
├── scripts/
│   ├── generate_sample_logs.py
│   └── run_predict_smoke.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py
│   └── training/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_predict.py
├── .gitignore
├── Damage Propagation Modeling.pdf
├── LICENSE
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
## Dashboard
The final system layer is a **Streamlit decision-support dashboard** built on top of the
inference outputs.

The dashboard is designed for a maintenance analyst, operations analyst, or reliability engineer
to quickly answer:
- Which engines are most at risk?
- Which cases should be prioritized first?
- What is the predicted RUL and conservative lower bound for a selected snapshot?
- What action is recommended?

### Dashboard Inputs

The dashboard demo consumes inference records from:
 - `dashboard/dashboard_data/demo_inference.jsonl`

This demo input follows the same structured outputs produced by the inference layer, which includes:

- `unit_number`
- `timestamp`
- `rul_pred`
- `rul_lower`
- `risk_band`
- `recommended_action`
- `validation flags`

### Dashboard views

The dashboard is organized into three decision-oriented sections:

1. **Fleet Overview**
    - Total number of scored snapshots
    - Average predicted RUL
    - combined summary of risk band, recommended action, and count

2. **Decision / Triage View**
    - Urgency-ranked snapshot list
    - Filtering by risk band
    - Prioritization based on lowest `rul_lower`

3. **Snapshot Detail View**
    - Inspection of one selected scored snapshot
    - Predicted RUL
    - Conservative lower bound (RUL)
    - Risk band
    - Validation flags

It is important to note that the dashboard doesn't call the model directly. It operates as the
decision support layer on top of the structured inference outputs logged.

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
### 3. Run tests
```bash
pytest
```
### 4. Run single snapshot inference smoke check
```bash
python scripts/run_predict_smoke.py
```
### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
---

## License
MIT License © 2026 Christopher Orellana
