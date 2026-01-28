# Training Pipeline

## 1. Training Inputs and Assumptions

- Training takes a processed dataset with a fixed schema.
- The dataset represents a single operating mode and a single fault mode.
- Raw sensor data collection and upstream processing are assumed to be and correct prior to
  data ingestion.
- All features used for training are defined and fixed prior to model fitting.
- The target variable is derived deterministically from the processed dataset and is capped.

## 2. Data Preparation and Split Strategy

- All training data is loaded exclusively through the shared system data loader to enforce
  consistent preprocessing.
- Selected features are finalized before any dataset splitting.
- Data is split at the engine level to prevent leakage between training and validation sets.
- Scaling and normalization parameters are fit using training data only.
- Any violation of the engine-split assumptions invalidates the training run.

## 3. Model Training and Evaluation

- A single baseline model is trained under fixed assumptions to establish a stable reference
  for system behavior.
- Model training is performed after data splitting and preprocessing were finalized.
- We evaluate on a held-out validation set to assess how the baseline generalizes to unseen engine
  instances.
- Rather than maximizing predictive accuracy, the baseline performance assessment is for validation
  of the model's direction and stability.
- Once the baseline met stability and safety expectations, we froze the baseline and versioned it.

## 4. Artifacts and Reproducibility

- The training script produces a complete, versioned artifact set that defines inference behavior.
- Artifacts include the trained model state, preprocessing state, feature schema, evaluation
  metadata, and a data fingerprint.
- A cryptographic fingerprint of the processed dataset is recorded to track training data lineage
  and detect unintended changes.
- With the same artifact set and valid inputs, inference behavior is deterministic and reproducible.