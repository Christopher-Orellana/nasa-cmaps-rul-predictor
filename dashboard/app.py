from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Set basic dashboard page settings
st.set_page_config(page_title="NASA CMAPSS Dashboard", layout="wide")

st.title("Predictive Maintenance Decision Dashboard")
st.write("Checkpoint 2: load and display parsed results")

DEMO_PATH = Path(__file__).resolve().parent / "dashboard_data" / "demo_inference.jsonl"


def load_inference_logs(log_path: Path) -> pd.DataFrame:
    """Load JSONL inference records into a pandas DataFrame."""
    if not log_path.exists():
        return pd.DataFrame()

    records = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)

    rename_map = {
        "flags.schema_error": "schema_error",
        "flags.extrapolation_risk": "extrapolation_risk",
        "flags.input_anomaly": "input_anomaly"
    }
    df = df.rename(columns=rename_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    sort_cols = [col for col in ["rul_lower", "timestamp"] if col in df.columns]

    if sort_cols == ["rul_lower", "timestamp"]:
        df = df.sort_values(by=["rul_lower", "timestamp"], ascending=[True, False])
    elif sort_cols == ["rul_lower"]:
        df = df.sort_values(by=["rul_lower"], ascending=True)

    return df.reset_index(drop=True)


df = load_inference_logs(DEMO_PATH)

if df.empty:
    st.warning("No inference records found in dashboard/dashboard_data/demo_inference.jsonl")
else:
    st.subheader("Scored Engine Snapshots")
    st.dataframe(df, use_container_width=True)