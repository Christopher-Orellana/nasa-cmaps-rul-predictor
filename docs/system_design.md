# Baseline RUL Inference System (FD001)

## Purpose
Turn a FD001 baseline model into a small, auditable inference system.
Focus is an operational inference pipeline (inputs → prediction → action), not RMSE gains.

## Scope
- single-timestep inference
- fixed feature set (sensor_4, sensor_11, sensor_15)
- deterministic preprocessing + scaling
- decision logic + basic monitoring

We purposely keep certain things out of scope such as more advanced ML models, Time-Series
models, Deep learning, Dashboards, cloud deployment, or retraining

## Input / Output
**Input**
- Required features: sensor_4, sensor_11, sensor_15
- Numeric, finite values only
- Missing or extra columns --> hard error 

**Output**
Single JSON object:
- rul_pred (point estimate)
- rul_lower (conservative bound)
- risk_band: GREEN | AMBER | RED | CRITICAL
- recommended_action
- flags: schema_error, extrapolation_risk, drift_suspected, input_anomaly

## Artifacts
Saved under `artifacts/`:
- model.joblib
- scaler.joblib
- feature_schema.json 
- metrics.json 

Inference refuses to run if artifacts are inconsistent.

## Inference Flow
1. Load artifacts
2. Validate schema and data types
3. Enforce feature order
4. Check ranges vs training data
5. Scale using train-fitted scaler
6. Predict RUL
7. Compute conservative bound:
   rul_lower = rul_pred − k × MAE_val
8. Map rul_lower → risk band + action
9. output JSON and log record

## Decision Policy
Uses conservative lower bound to avoid optimistic decisions.

Parameters:
- MAE_val ≈ 19
- k = 1.0

Actions (based on rul_lower):
- 60 → GREEN / CONTINUE
- 30–60 → AMBER / INSPECT
- 10–30 → RED / SCHEDULE_MAINTENANCE
- ≤ 10 → CRITICAL / REMOVE_FROM_SERVICE

## Monitoring
- extrapolation_risk: feature outside training min/max
- input_anomaly: large z-score vs training mean/std
- drift_suspected: simple rolling stat deviation

All inferences append a JSONL record to `logs/inference.jsonl`.

## Known Failure Modes
- no temporal context
- FD001-only assumptions
- extrapolation near end-of-life
- noisy or degraded sensors
- limited drift detection in monitoring

## Testing
- schema enforcement
- feature order invariance
- scaler not refit at inference
- artifact version mismatch rejected
- smoke test on small CSV
