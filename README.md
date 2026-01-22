# NASA Jet Engine Predictive Maintenance

## Problem Definition

The NASA CMAPSS dataset contains four subsets (FD001–FD004)  
that differ in operating conditions and fault modes, which leads to  
different engine degradation behaviors.

To keep the modeling scope controlled and interpretable,  
this project focuses on FD001, which represents a single operating  
condition and a single fault mode.

The goal is to predict the Remaining Useful Life (RUL) of aircraft  
engines using multivariate sensor data so maintenance decisions  
can be made before failure occurs.

We treat RUL prediction as a decision-support problem rather than  
a pure regression task.

Prediction errors have unequal consequences:  
**predicting too late risks unexpected engine failure,  
while overestimating RUL leads to premature maintenance and  
unnecessary cost**.

The objective is to balance predictive accuracy, interpretability,  
and practicality while respecting the temporal nature of  
engine degradation and the real tradeoffs involved in  
predictive maintenance.

---

## System Design
This project includes a system-level design for inference, decision logic, and monitoring.

See: docs/system_design.md

---

## Approach Overview

This analysis begins with the FD001 subset, which represents a  
single operating condition and a single fault mode, to reduce  
confounding effects and isolate engine degradation behavior.

Exploratory data analysis is used to study how sensor signals evolve  
over an engine’s lifecycle, with emphasis on identifying sensors that  
exhibit meaningful degradation trends versus those that are largely  
noisy or non-informative.

Modeling then starts with interpretable statistical baselines to guide  
feature selection and build intuition, before extending to more  
flexible machine learning models.

Throughout this process, low-variance or non-informative features are  
removed, and targeted feature engineering is applied to improve both  
predictive performance and interpretability for RUL estimation.

---

## Dataset Description

The dataset used in this project comes from NASA’s C-MAPSS  
(Commercial Modular Aero-Propulsion System Simulation) engine  
degradation simulations.

Each subset contains run-to-failure time series data from multiple  
simulated turbofan engines operating under varying conditions  
and fault modes.

We use the FD001 subset, where each engine is represented as a  
multivariate time series.

Each row corresponds to a single operational cycle including  
multiple sensors and operational settings.

The target variable, **Remaining Useful Life (RUL)**, is computed  
relative to the final observed cycle for each engine.

---

## Artifacts
Training exports versioned artifacts under `artifacts/`:
- `feature_schema.json` inference contract + monitoring stats
- `metrics.json` baseline performance + decision policy
- `model.joblib`, `scaler.joblib`

---
## Repository Structure


```
nasa-cmaps-pipeline/
├── notebooks/              # EDA and modeling notebooks
├── src/                    # Reusable functions and modeling code
├── data/                   # Data directory (not tracked in this repo)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── requirements.txt        # Dependencies for reproducibility
├── README.md               # Project overview and methodology
└── .gitignore              # Files we ignored in this repo
```
----------------------------------------------------------------
## Reproducibility

This project is designed to be fully reproducible using a Python virtual environment. 
All dependencies are specified in `requirements.txt`.

```bash
# Clone the repo
git clone https://github.com/Christopher-Orellana/nasa-cmaps-pipeline.git
cd nasa-cmaps-pipeline

# Create a virtual environment
python -m venv .venv

# Activate the venv:

# For users with macOS / Linux
source .venv/bin/activate

# For users with Windows (PowerShell)
.venv\Scripts\Activate.ps1

# For users with Windows (Command Prompt)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
