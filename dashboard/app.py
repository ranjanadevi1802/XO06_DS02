import streamlit as st
import pandas as pd
import os


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Data Trust Engine",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "outputs/ml_anomaly_results.csv"
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛡️ Data Trust Engine")

st.markdown(
    "### Intelligent Reliability Assessment for Multi-Channel Measurements"
)

st.divider()


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

avg_trust = df["Trust_Score"].mean()

trusted = (
    df["Final_Status"] == "Trusted"
).mean() * 100

needs_review = (
    df["Final_Status"] == "Needs Review"
).mean() * 100

ml_anomalies = (
    df["ML_Anomaly"] == 1
).sum()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Trust Score",
    f"{avg_trust:.1f}"
)

col2.metric(
    "Trusted",
    f"{trusted:.1f}%"
)

col3.metric(
    "Needs Review",
    f"{needs_review:.1f}%"
)

col4.metric(
    "ML Anomalies",
    ml_anomalies
)


st.divider()


# --------------------------------------------------
# TRUST SCORE DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Trust Score Distribution")

st.bar_chart(
    df["Trust_Score"].value_counts(
        bins=10
    ).sort_index()
)


# --------------------------------------------------
# TWO COLUMNS
# --------------------------------------------------

left, right = st.columns(2)


with left:

    st.subheader("🏭 Reliability by Station")

    station_trust = (
        df.groupby("Station_ID")["Trust_Score"]
        .mean()
        .sort_values()
    )

    st.bar_chart(station_trust)


with right:

    st.subheader("🤖 ML Anomaly Status")

    ml_status = (
        df["ML_Status"]
        .value_counts()
    )

    st.bar_chart(ml_status)


st.divider()


# --------------------------------------------------
# OBSERVATION INVESTIGATION
# --------------------------------------------------

st.subheader("🔎 Observation Investigation")

station_list = sorted(
    df["Station_ID"].dropna().unique()
)

selected_station = st.selectbox(
    "Select Station",
    station_list
)


station_data = df[
    df["Station_ID"] == selected_station
]


selected_index = st.selectbox(
    "Select Observation",
    station_data.index
)


row = df.loc[selected_index]


# --------------------------------------------------
# OBSERVATION DETAILS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Trust Score",
        f"{row['Trust_Score']:.1f}"
    )


with col2:

    st.write("### Trust Category")

    st.write(
        row["Trust_Category"]
    )


with col3:

    st.write("### Final Status")

    st.write(
        row["Final_Status"]
    )


st.divider()


# --------------------------------------------------
# DIAGNOSIS
# --------------------------------------------------

st.subheader("🩺 Diagnosis")

col1, col2 = st.columns(2)


with col1:

    st.write("**Affected Sensors**")

    st.info(
        row["Affected_Sensors"]
    )


with col2:

    st.write("**Missing Sensors**")

    st.info(
        row["Missing_Sensors"]
    )


st.write("**Trust Reason**")

st.warning(
    row["Trust_Reason"]
)


st.write("**ML Detection**")

if row["ML_Anomaly"] == 1:

    st.error(
        "⚠️ Isolation Forest detected an anomaly."
    )

else:

    st.success(
        "✅ ML model considers this observation normal."
    )


# --------------------------------------------------
# SENSOR VALUES
# --------------------------------------------------

st.subheader("📡 Sensor Measurements")

sensor_columns = [
    "CO_Channel",
    "NOx_Channel",
    "NO2_Channel",
    "Ambient_Temperature",
    "Relative_Humidity",
    "Absolute_Humidity"
]

sensor_values = row[
    sensor_columns
].to_frame(
    name="Value"
)

st.dataframe(
    sensor_values,
    use_container_width=True
)