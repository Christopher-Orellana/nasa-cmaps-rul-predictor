# System Overview — NASA CMAPSS FD001 Predictive Maintenance

## 1. Purpose and Scope

### 1.1 Purpose

This system is designed to support conservative maintenance decision-making for turbofan 
engines by estimating remaining useful life under explicitly constrained operating assumptions.
It operates under a single fault mode, single operating condition, and assumes sensor
inputs are consistent with the training distribution, making no extrapolations outside that 
bounded scope. 

The system functions as a decision-support mechanism by mapping estimated RUL 
into discrete risk bands with associated recommended actions, rather than attempting precise 
failure-time prediction. Robustness, interpretability, and auditability are prioritized over
predictive optimization to surface uncertainty and prevent unsafe extrapolation.
When uncertainty cannot be resolved within the system’s validated assumptions, the system is
designed to flag risk and defer judgment rather than produce overconfident outputs.

### 1.2 Intended Use

- Used for offline or near-real-time evaluation of individual engine health snapshots.
- Intended for maintenance planners, not for onboard or real-time control systems.
- Functions strictly as an advisory decision-support system and does not autonomously trigger
  maintenance actions.
- Designed to be evaluated on successive engine snapshots, with decisions informed by
  persistent risk signals rather than single observations.
- Success is defined by conservative identification of unsafe conditions and explicit flagging
  when inputs fall outside validated assumptions.
- Outputs are intended to inform maintenance planning and inspection prioritization, not to 
  provide precise failure timing.

### 1.3 Non-Goals

- The baseline system does not attempt to maximize predictive accuracy through complex modeling, 
  hyperparameter tuning.
- The baseline system does not perform true temporal or sequence-based forecasting, as its
  inference contract operates on individual engine snapshots.
- The baseline system does not function as a real-time, onboard, or flight-critical control
  system, and is not intended for cockpit or autonomous operational use.
- The baseline system does not autonomously trigger maintenance actions, retrain itself,
  adapt online, or modify its behavior without explicit human control.
- The baseline system does not claim generalization beyond the CMAPSS FD001 dataset,
  operating condition, fault mode, or validated sensor feature set.

## 2. System Boundary and Interfaces

### 2.1 System boundary
This section defines what the system is responsible for and what responsibilities are external.

The system is responsible for:
- Accepting validated engine health snapshots containing a fixed set of sensor values.
- Estimating RUL under the constrain assumptions.
- Maps estimates from the model to conservative risk bands and recommended actions.
- Performs input validation, anomaly detection, and extrapolation checks.
- Produces semi-structured audible inference logs.

The system is not responsible for:
- Sensor data collection and sensor calibration.
- Upstream data cleaning.
- Decisions to act on given the recommendation output.
- Real-time control, safety enforcement, or autonomous operation.

### 2.2 Interfaces
Input interface:
- Accepts a single engine health snapshot containing a fixed set of required sensors.
- Requires inputs to be validated through the system's schema and numeric constraints.
- Rejects invalidated inputs such as missing readings, non-finite values, and invalid data types

Training interface:
- Takes a validated engine health screenshot and frozen training artifacts.
- Produces an estimated RUL, conservative lower bound, and a risk classification.
- Produces structured inference metadata and safety flags for auditability.

Output interface:
- Returns risk bands and recommended actions.
- Produces logs suitable for downstream inspection or analysis.


## 3. End-to-End Architecture
This system is structured as a linear, contract-driven pipeline separating training and inference.

Training:
- A processed dataset is loaded through the shared system data loader.
- System-defined preprocessing and dataset constraints are applied deterministically.
- Baseline model is trained under fixed assumptions and evaluated on a validation set.
- Versioned artifacts are produced, including the trained model, preprocessing state,
  feature schema, metrics, and a data fingerprint.

Inference:
- A single engine health snapshot is validated against the frozen feature schema.
- Preprocessing and feature order are consistent with training.
- The trained model produces an RUL estimate, conservative lower bound, and risk band along
  with its respective recommended action.
- Safety flags and semi-structured metadata are produced alongside the prediction.

Inference relies on saved artifacts, not on the environment in which the model was trained,
which keeps behavior consistent and reproducible.

## 4. Contracts and Invariants

### 4.1 Core contracts

- The system operates only on a single, processed dataset with a fixed schema.
- All training and inference paths must consume data through the shared data loader.
- Feature names, order, and target definition are fixed by the artifact schema and must not be
  altered at inference time.
- Training and inference preprocessing must be identical and deterministic.
- Inference behavior must be determined by produced artifacts, not training environments.
- Inputs that violate schema, type, or numeric constraints must be rejected or flagged.
- The baseline model and its artifacts are immutable once we version it.

### 4.2 Invariants

- The system never performs inference without a complete and valid artifact set.
- Feature values are never reordered, imputed, or inferred at inference time.
- Predictions are never returned without accompanying safety flags and metadata.
- Inputs outside the validated training distribution are flagged.
- The system never silently extrapolates beyond its validated assumptions.
- All inference events are logged in a structured format.


## 5. Safety and Auditability

- The system is intentionally conservative and biased toward early risk identification rather
  than delayed failure detection.
- Predictions are not ground truth or built for exact failure times, but as decision support
  signals.
- These signals are meant to be used in context, with emphasis on a persistent signal.
- When inputs violate assumptions, the system prioritizes deferral and flagging rather than
  forcing a prediction.
- Absence of a high-risk signal does not imply safety - but a consistency with known operating
  conditions.
- Final responsibility for operation decisions remains with operators and maintenance planners.