# %%
# ============================================================
# DATA TRUST ENGINE
# Reliability Scoring Module
# ============================================================

import numpy as np
import pandas as pd


# %%
# ============================================================
# CONFIGURATION
# ============================================================

TRUST_THRESHOLDS = {
    "Highly Reliable": 80,
    "Reliable": 60,
    "Uncertain": 40,
}


# %%
# ============================================================
# UTILITY FUNCTION
# ============================================================

def safe_numeric(series):
    """
    Convert values to numeric.
    Invalid values are converted to NaN.
    """
    
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# %%
# ============================================================
# ROBUST ANOMALY SCORE
# ============================================================

def robust_anomaly_score(series):
    """
    Calculate a robust anomaly score using MAD.

    Higher score = more anomalous.
    """

    x = safe_numeric(series)

    median = x.median()

    valid_values = x.dropna()

    if len(valid_values) == 0:
        return pd.Series(
            0.0,
            index=series.index
        )

    mad = np.median(
        np.abs(
            valid_values - median
        )
    )

    if pd.isna(mad) or mad == 0:

        mad = x.std()

    if pd.isna(mad) or mad == 0:

        mad = 1.0

    anomaly = (
        np.abs(x - median) / mad
    )

    return anomaly.replace(
        [np.inf, -np.inf],
        np.nan
    )


# %%
# ============================================================
# ANOMALY → RELIABILITY
# ============================================================

def anomaly_to_reliability(anomaly):
    """
    Convert anomaly score to reliability.

    Low anomaly  → high reliability
    High anomaly → low reliability

    Output range: 0–1
    """

    anomaly = safe_numeric(anomaly)

    reliability = (
        1 / (1 + anomaly)
    )

    return reliability.clip(
        0,
        1
    )


# %%
# ============================================================
# TEMPORAL RELIABILITY
# ============================================================

def create_temporal_reliability(
    df,
    sensor_cols
):
    """
    Calculate temporal reliability.

    Requires:
        <sensor>_rolling_mean

    Creates:
        <sensor>_temporal_anomaly
        <sensor>_temporal_reliability
    """

    df = df.copy()

    for sensor in sensor_cols:

        # ----------------------------------------------------
        # Check sensor exists
        # ----------------------------------------------------

        if sensor not in df.columns:
            continue

        rolling_col = (
            f"{sensor}_rolling_mean"
        )

        # ----------------------------------------------------
        # Check rolling feature exists
        # ----------------------------------------------------

        if rolling_col not in df.columns:
            continue

        current = safe_numeric(
            df[sensor]
        )

        rolling_mean = safe_numeric(
            df[rolling_col]
        )

        # ----------------------------------------------------
        # Difference from recent behaviour
        # ----------------------------------------------------

        deviation = (
            current - rolling_mean
        ).abs()

        # ----------------------------------------------------
        # Robust scaling
        # ----------------------------------------------------

        median = deviation.median()

        valid_deviation = (
            deviation.dropna()
        )

        if len(valid_deviation) == 0:

            mad = 1.0

        else:

            mad = np.median(
                np.abs(
                    valid_deviation
                    - median
                )
            )

        if pd.isna(mad) or mad == 0:

            mad = deviation.std()

        if pd.isna(mad) or mad == 0:

            mad = 1.0

        # ----------------------------------------------------
        # Anomaly
        # ----------------------------------------------------

        anomaly = (
            deviation / mad
        )

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        reliability = (
            1 / (1 + anomaly)
        )

        df[
            f"{sensor}_temporal_anomaly"
        ] = anomaly

        df[
            f"{sensor}_temporal_reliability"
        ] = reliability.clip(
            0,
            1
        )

    return df


# %%
# ============================================================
# RELATIONSHIP / CROSS-SENSOR RELIABILITY
# ============================================================

def create_relationship_reliability(
    df,
    relationship_cols=None
):
    """
    Convert relationship/difference features
    into reliability scores.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Automatically identify relationship features
    # --------------------------------------------------------

    if relationship_cols is None:

        relationship_cols = [
            col
            for col in df.columns
            if (
                "relationship" in col.lower()
                or "difference" in col.lower()
                or "diff" in col.lower()
                or "residual" in col.lower()
            )
        ]

    # --------------------------------------------------------
    # Process each relationship feature
    # --------------------------------------------------------

    for col in relationship_cols:

        if col not in df.columns:
            continue

        values = safe_numeric(
            df[col]
        )

        anomaly = robust_anomaly_score(
            values
        )

        reliability = anomaly_to_reliability(
            anomaly
        )

        df[
            f"{col}_reliability"
        ] = reliability

    return df


# %%
# ============================================================
# OUTLIER RELIABILITY
# ============================================================

def create_outlier_reliability(
    df,
    sensor_cols
):
    """
    Detect statistical outliers using IQR.

    Creates:
        outlier_score
        outlier_reliability
    """

    df = df.copy()

    outlier_flags = pd.DataFrame(
        index=df.index
    )

    valid_sensor_cols = []

    for sensor in sensor_cols:

        if sensor not in df.columns:
            continue

        values = safe_numeric(
            df[sensor]
        )

        q1 = values.quantile(
            0.25
        )

        q3 = values.quantile(
            0.75
        )

        iqr = q3 - q1

        # ----------------------------------------------------
        # Constant / unusable feature
        # ----------------------------------------------------

        if pd.isna(iqr) or iqr == 0:

            outlier_flags[sensor] = 0

        else:

            lower = (
                q1 - 1.5 * iqr
            )

            upper = (
                q3 + 1.5 * iqr
            )

            outlier_flags[sensor] = (
                (values < lower)
                |
                (values > upper)
            ).astype(int)

        valid_sensor_cols.append(
            sensor
        )

    # --------------------------------------------------------
    # Overall outlier score
    # --------------------------------------------------------

    if valid_sensor_cols:

        df["outlier_score"] = (
            outlier_flags[
                valid_sensor_cols
            ]
            .mean(axis=1)
        )

    else:

        df["outlier_score"] = 0.0

    # --------------------------------------------------------
    # Convert to reliability
    # --------------------------------------------------------

    df["outlier_reliability"] = (
        1 - df["outlier_score"]
    ).clip(
        0,
        1
    )

    return df


# %%
# ============================================================
# MISSINGNESS RELIABILITY
# ============================================================

def create_missingness_reliability(
    df,
    sensor_cols
):
    """
    Calculate reliability based on
    missing sensor measurements.
    """

    df = df.copy()

    valid_cols = [
        col
        for col in sensor_cols
        if col in df.columns
    ]

    # --------------------------------------------------------
    # No valid sensor columns
    # --------------------------------------------------------

    if not valid_cols:

        df["missing_count"] = 0

        df["missing_ratio"] = 0.0

        df["missingness_reliability"] = 1.0

        return df

    # --------------------------------------------------------
    # Missing count
    # --------------------------------------------------------

    missing_count = (
        df[valid_cols]
        .isna()
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Missing ratio
    # --------------------------------------------------------

    missing_ratio = (
        missing_count
        / len(valid_cols)
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    df["missing_count"] = (
        missing_count
    )

    df["missing_ratio"] = (
        missing_ratio
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    df["missingness_reliability"] = (
        1 - missing_ratio
    ).clip(
        0,
        1
    )

    return df


# %%
# ============================================================
# TRUST SCORE CALCULATION
# ============================================================

def calculate_trust_score(
    df,
    temporal_weight=0.35,
    relationship_weight=0.30,
    outlier_weight=0.20,
    missingness_weight=0.15
):
    """
    Calculate final Trust Score from 0–100.

    Weights:

        Temporal consistency : 35%
        Relationships        : 30%
        Outlier behaviour    : 20%
        Missingness          : 15%
    """

    df = df.copy()

    # --------------------------------------------------------
    # Temporal reliability
    # --------------------------------------------------------

    temporal_cols = [
        col
        for col in df.columns
        if col.endswith(
            "_temporal_reliability"
        )
    ]

    if temporal_cols:

        temporal_score = (
            df[temporal_cols]
            .mean(axis=1)
        )

    else:

        temporal_score = pd.Series(
            1.0,
            index=df.index
        )

    # --------------------------------------------------------
    # Relationship reliability
    # --------------------------------------------------------

    relationship_cols = [
        col
        for col in df.columns
        if (
            col.endswith(
                "_reliability"
            )
            and not col.endswith(
                "_temporal_reliability"
            )
            and col not in [
                "outlier_reliability",
                "missingness_reliability"
            ]
        )
    ]

    if relationship_cols:

        relationship_score = (
            df[relationship_cols]
            .mean(axis=1)
        )

    else:

        relationship_score = pd.Series(
            1.0,
            index=df.index
        )

    # --------------------------------------------------------
    # Outlier reliability
    # --------------------------------------------------------

    if "outlier_reliability" in df.columns:

        outlier_score = (
            df["outlier_reliability"]
        )

    else:

        outlier_score = pd.Series(
            1.0,
            index=df.index
        )

    # --------------------------------------------------------
    # Missingness reliability
    # --------------------------------------------------------

    if (
        "missingness_reliability"
        in df.columns
    ):

        missingness_score = (
            df[
                "missingness_reliability"
            ]
        )

    else:

        missingness_score = pd.Series(
            1.0,
            index=df.index
        )

    # --------------------------------------------------------
    # Weighted Trust Score
    # --------------------------------------------------------

    trust = (
        temporal_weight
        * temporal_score

        +

        relationship_weight
        * relationship_score

        +

        outlier_weight
        * outlier_score

        +

        missingness_weight
        * missingness_score
    )

    # --------------------------------------------------------
    # Convert 0–1 → 0–100
    # --------------------------------------------------------

    df["trust_score"] = (
        trust * 100
    ).clip(
        0,
        100
    )

    # --------------------------------------------------------
    # Store individual components
    # --------------------------------------------------------

    df["temporal_trust"] = (
        temporal_score * 100
    ).clip(
        0,
        100
    )

    df["relationship_trust"] = (
        relationship_score * 100
    ).clip(
        0,
        100
    )

    df["outlier_trust"] = (
        outlier_score * 100
    ).clip(
        0,
        100
    )

    df["missingness_trust"] = (
        missingness_score * 100
    ).clip(
        0,
        100
    )

    return df


# %%
# ============================================================
# TRUST CATEGORY
# ============================================================

def classify_trust_score(score):
    """
    Convert numerical Trust Score
    into a human-readable category.
    """

    if pd.isna(score):

        return "Unknown"

    if score >= 80:

        return "Highly Reliable"

    elif score >= 60:

        return "Reliable"

    elif score >= 40:

        return "Uncertain"

    else:

        return "Low Trust"


# %%
# ============================================================
# ADD TRUST CATEGORY
# ============================================================

def add_trust_category(df):
    """
    Add trust_category column.
    """

    df = df.copy()

    if "trust_score" not in df.columns:

        raise ValueError(
            "trust_score not found. "
            "Run calculate_trust_score() first."
        )

    df["trust_category"] = (
        df["trust_score"]
        .apply(
            classify_trust_score
        )
    )

    return df


# %%
# ============================================================
# SUSPICIOUS OBSERVATIONS
# ============================================================

def get_suspicious_observations(
    df,
    threshold=40
):
    """
    Return observations below
    the specified Trust Score.
    """

    if "trust_score" not in df.columns:

        raise ValueError(
            "trust_score not found."
        )

    suspicious = (
        df[
            df["trust_score"] < threshold
        ]
        .copy()
        .sort_values(
            "trust_score"
        )
    )

    return suspicious


# %%
# ============================================================
# STATION-LEVEL SUMMARY
# ============================================================

def create_station_summary(
    df,
    station_col="Station_ID"
):
    """
    Create station-level reliability summary.
    """

    if station_col not in df.columns:

        return pd.DataFrame()

    if "trust_score" not in df.columns:

        raise ValueError(
            "trust_score not found."
        )

    summary = (
        df.groupby(
            station_col
        )
        .agg(
            average_trust=(
                "trust_score",
                "mean"
            ),

            minimum_trust=(
                "trust_score",
                "min"
            ),

            maximum_trust=(
                "trust_score",
                "max"
            ),

            observations=(
                "trust_score",
                "count"
            )
        )
        .reset_index()
    )

    summary[
        "reliability_percentage"
    ] = (
        summary["average_trust"]
    )

    summary = summary.sort_values(
        "average_trust"
    )

    return summary


# %%
# ============================================================
# COMPLETE RELIABILITY PIPELINE
# ============================================================

def run_reliability_pipeline(
    df,
    sensor_cols,
    relationship_cols=None
):
    """
    Run the complete Data Trust Engine.

    Pipeline:

        1. Temporal reliability
        2. Relationship reliability
        3. Outlier reliability
        4. Missingness reliability
        5. Trust Score
        6. Trust Category
    """

    result = df.copy()

    print(
        "Starting reliability pipeline..."
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "1/6 Creating temporal reliability..."
    )

    result = create_temporal_reliability(
        result,
        sensor_cols
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "2/6 Creating relationship reliability..."
    )

    result = create_relationship_reliability(
        result,
        relationship_cols
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "3/6 Creating outlier reliability..."
    )

    result = create_outlier_reliability(
        result,
        sensor_cols
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "4/6 Creating missingness reliability..."
    )

    result = create_missingness_reliability(
        result,
        sensor_cols
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "5/6 Calculating Trust Score..."
    )

    result = calculate_trust_score(
        result
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print(
        "6/6 Creating Trust Categories..."
    )

    result = add_trust_category(
        result
    )

    print(
        "✅ Reliability pipeline completed!"
    )

    return result


# %%
# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "✅ reliability_scoring.py loaded successfully."
    )