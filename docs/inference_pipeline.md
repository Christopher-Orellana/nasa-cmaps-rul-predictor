# Inference Pipeline

## 1. Inference Inputs and Preconditions

- Inference takes a single engine health snapshot containing a fixed set of required sensor features.
- Inputs must conform to the feature schema defined by the training artifacts.
- All inputs must be numeric, finite, and represent a single point-in-time observation.
- A complete and compatible artifact set must be available before running inference.
- Artifact version compatability must be verified before running inference.
- Inference is not permitted to run if any of these preconditions is violated.

## 2. Inference Execution Flow

- A validated engine health snapshot is accepted into the system.
- Preprocessing consistent with the training pipeline is applied.
- Feature ordering is enforced according to the frozen artifact schema.
- The trained baseline model estimates the remaining useful life.
- A conservative lower bound is computed from the estimate and mapped to a risk classification
  and a recommended action.

## 3. Safety Checks and Risk Signaling

- Input values are checked for schema compliance, numeric validity, and a finite representation.
- Inputs are evaluated against the training distribution to detect extrapolation beyond the model's
  scope.
- Anomaly checks are performed to identify inputs that deviate significantly from expected
  operating conditions.
- Safety flags are raised when assumptions are violated,
- Elevated risk classifications are enforced when safety flags are present, unsafe or out-of-scope
  inputs are never treated as low risk.

## 4. Outputs and Logging

- Inference always produces an estimated RUL, a conservative lower bound, and a corresponding risk
  classification.
- Each inference includes recommended actions aligned with the assigned risk band.
- Safety flags and inference metadata are always returned alongside numerical outputs.
- All inference events are logged in a semi-structured format suitable for audit and review.
- Logs capture enough context to reconstruct the inference decision and assess input validity.
