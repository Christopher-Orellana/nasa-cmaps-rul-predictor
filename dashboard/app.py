from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Set basic dashboard page settings
st.set_page_config(page_title="NASA CMAPS Dashboard", layout="wide")

st.title("Predictive Maintenance Decision Dashboard")
st.write("Checkpoint 3: Fleet overview and urgency-ranked table")

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

# ---------------Fleet Overview---------------
if df.empty:
    st.warning("No inference records found in dashboard/dashboard_data/demo_inference.jsonl")
else:
    st.subheader("Fleet Overview")

    total_snapshots = len(df)
    avg_rul_pred = df['rul_pred'].mean()

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Total Scored Snapshots", total_snapshots)
    metric_col2.metric("Average Predicted Remaining Useful Life", f"{avg_rul_pred: .2f}")

    # ---------------Summary Count---------------
    summary_col1, summary_col2 = st.columns(2)
    risk_band_order = ["CRITICAL", "RED", "AMBER", "GREEN"]
    action_order = [
        "REMOVE_FROM_SERVICE",
        "SCHEDULE_MAINTENANCE",
        "INSPECT",
        "CONTINUE"
    ]

    risk_counts = (
        df["risk_band"]
        .value_counts()
        .reindex(risk_band_order, fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["risk_band", "count"]

    action_counts = (
        df["recommended_action"]
        .value_counts()
        .reindex(action_order, fill_value=0)
        .reset_index()
    )
    action_counts.columns =  ["recommended_action", "count"]

    with summary_col1:
        st.markdown("**Count by Risk Band**")
        st.dataframe(risk_counts, use_container_width=True, hide_index=True)

    with summary_col2:
        st.markdown("**Count by Recommended Action**")
        st.dataframe(action_counts, use_container_width=True, hide_index=True)

    # --------------- Ranked Table ---------------
    st.subheader("Ranked Engine Snapshots by Urgency")

    display_columns = [
        "unit_number",
        "timestamp",
        'rul_pred',
        'rul_lower',
        'risk_band',
        'recommended_action',
        'schema_error',
        'extrapolation_risk',
        'input_anomaly'
    ]

    available_columns = [col for col in display_columns if col in df.columns]
    ranked_df = df[available_columns].copy()

    if 'rul_pred' in ranked_df.columns:
        ranked_df['rul_pred'] = ranked_df['rul_pred'].round(2)

    if 'rul_lower' in ranked_df.columns:
        ranked_df['rul_lower'] = ranked_df['rul_lower'].round(2)

    st.dataframe(ranked_df, use_container_width=True, hide_index=True)

