from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Set basic dashboard page settings
st.set_page_config(page_title="NASA CMAPSS Dashboard", layout="wide")

st.title("Predictive Maintenance Decision Dashboard")

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
    avg_rul_pred = df["rul_pred"].mean()

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Total Scored Snapshots", total_snapshots)
    metric_col2.metric("Average Predicted Remaining Useful Life", f"{avg_rul_pred:.2f}")

    # --------------- Summary Count ---------------
    risk_band_order = ["CRITICAL", "RED", "AMBER", "GREEN"]
    action_map = {
        "CRITICAL": "REMOVE_FROM_SERVICE",
        "RED": "SCHEDULE_MAINTENANCE",
        "AMBER": "INSPECT",
        "GREEN": "CONTINUE"
    }

    summary_counts = (
        df["risk_band"]
        .value_counts()
        .reindex(risk_band_order, fill_value=0)
        .rename_axis("risk_band")
        .reset_index(name="count")
    )

    summary_counts["recommended_action"] = summary_counts['risk_band']
    # --------------- Ranked Data ---------------
    display_columns = [
        "unit_number",
        "timestamp",
        "rul_pred",
        "rul_lower",
        "risk_band",
        "recommended_action",
        "schema_error",
        "extrapolation_risk",
        "input_anomaly"
    ]

    available_columns = [col for col in display_columns if col in df.columns]
    ranked_df = df[available_columns].copy()

    # --------------- Decision / Triage View ---------------
    st.subheader("Decision / Triage View: Ranked Engine Snapshots by Urgency")

    selected_risk_bands = st.multiselect(
        "Filter by Risk Band",
        options=risk_band_order,
        default=risk_band_order
    )

    triage_df = ranked_df.copy()

    if selected_risk_bands:
        triage_df = triage_df[triage_df["risk_band"].isin(selected_risk_bands)]

    if "rul_pred" in triage_df.columns:
        triage_df["rul_pred"] = triage_df["rul_pred"].round(2)

    if "rul_lower" in triage_df.columns:
        triage_df["rul_lower"] = triage_df["rul_lower"].round(2)

    triage_df = triage_df.reset_index(drop=True)

    st.write(f"Filtered snapshots: {len(triage_df)}")

    # --------------- Ranked Table ---------------
    if triage_df.empty:
        st.info("No engine snapshots match the selected filters.")
    else:
        st.dataframe(triage_df, use_container_width=True, hide_index=True)

    # --------------- Snapshot Detail View ---------------
    st.subheader("Snapshot Detail View")

    detail_df = triage_df.copy()

    if not detail_df.empty:
        if "timestamp" in detail_df.columns:
            detail_df["timestamp_str"] = detail_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            detail_df["timestamp_str"] = "Unknown"

        detail_df["snapshot_label"] = (
                "Unit "
                + detail_df["unit_number"].astype(str)
                + " | "
                + detail_df["timestamp_str"].astype(str)
                + " | "
                + detail_df["risk_band"].astype(str)
                + " | RUL Lower: "
                + detail_df["rul_lower"].astype(str)
        )

        selected_label = st.selectbox(
            "Select a scored engine snapshot",
            detail_df["snapshot_label"].tolist()
        )

        selected_row = detail_df.loc[detail_df["snapshot_label"] == selected_label].iloc[0]

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.subheader("Snapshot Summary")
            st.write(f"**Unit Number:** {selected_row['unit_number']}")
            st.write(f"**Timestamp:** {selected_row['timestamp_str']}")
            st.write(f"**Predicted RUL:** {selected_row['rul_pred']}")
            st.write(f"**Conservative RUL Lower Bound:** {selected_row['rul_lower']}")
            st.write(f"**Risk Band:** {selected_row['risk_band']}")
            st.write(f"**Recommended Action:** {selected_row['recommended_action']}")

        with detail_col2:
            st.subheader("Validation Flags")
            st.write(f"**Schema Error:** {selected_row['schema_error']}")
            st.write(f"**Extrapolation Risk:** {selected_row['extrapolation_risk']}")
            st.write(f"**Input Anomaly:** {selected_row['input_anomaly']}")
    else:
        st.info("No snapshot available for detail view.")