# Failure Modes

## 1. Invalid or Corrupted Inputs

- Missing require sensor fields are detected during schema validation and cause inference to be
  rejected.
- Unexpected or extra inputs are detected during schema enforcement and cause inference to be rejected.
- Non-numeric or non-finite values are detected during input validation and cause inference to be rejected.
- Inputs that violate numeric constraints are detected and explicitly flagged as invalid.

## 2. Out-of-Distribution Inputs

- Inputs that fall outside the observed training value ranges are detected and flagged as an
  extrapolation risk.
- Inputs that deviate significantly from the training distribution are detected and flagged as
  an anomaly.
- Inputs exhibiting multiple simultaneous anomaly signals are treated as high risk.
- Out-of-distribution inputs are never classified as low risk or safe.

## 3. Artifact and Configuration Failures

- Missing required artifacts are detected prior to inference and cause execution to be blocked.
- Artifact version mismatches are detected during compatibility checks and cause inference
  to be rejected.
- Feature schema mismatches between input data and artifacts are detected and cause inference
  to be rejected.
- Inconsistent or corrupted artifact metadata is detected and causes the run to fail.

## 4. Model and Assumption Limitations

- The model scope is limited to a single operating condition and single fault mode only, any
  inference out this scope should be considered unreliable.
- Baseline point estimates are not treated as precise failure time and must be interpreted with context.
- Absense of a high-risk signal doesn't imply safety, it only implies consistency with known operating conditions.
