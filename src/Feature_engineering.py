import pandas as pd
import numpy as np


def create_temporal_features(df, sensor_columns, rolling_window=6):

    df = df.copy()

    # Rolling statistics
    for col in sensor_columns:

        df[f"{col}_RollingMean"] = (
            df.groupby("Station_ID")[col]
            .transform(
                lambda x: x.rolling(
                    rolling_window,
                    min_periods=3
                ).mean()
            )
        )

        df[f"{col}_RollingStd"] = (
            df.groupby("Station_ID")[col]
            .transform(
                lambda x: x.rolling(
                    rolling_window,
                    min_periods=3
                ).std()
            )
        )

    # Temporal deviation
    for col in sensor_columns:

        mean_col = f"{col}_RollingMean"

        df[f"{col}_Deviation"] = (
            df[col] - df[mean_col]
        ).abs()

    # Temporal z-score and anomaly
    for col in sensor_columns:

        mean_col = f"{col}_RollingMean"
        std_col = f"{col}_RollingStd"

        df[f"{col}_ZScore"] = (
            (df[col] - df[mean_col])
            / df[std_col].replace(0, np.nan)
        )

        df[f"{col}_TemporalAnomaly"] = (
            df[f"{col}_ZScore"].abs() > 3
        ).astype(int)

    # Total temporal anomalies
    temporal_anomaly_columns = [
        f"{col}_TemporalAnomaly"
        for col in sensor_columns
    ]

    df["Temporal_Anomaly_Count"] = (
        df[temporal_anomaly_columns].sum(axis=1)
    )

    return df

def create_relationship_features(df, sensor_columns):

    df = df.copy()

    # Correlation-based relationship features
    correlation_matrix = df[sensor_columns].corr()

    print("\n--- SENSOR CORRELATION MATRIX ---")
    print(correlation_matrix.round(2))

    # Relationship deviation using rolling correlations
    for i in range(len(sensor_columns)):
        for j in range(i + 1, len(sensor_columns)):

            col1 = sensor_columns[i]
            col2 = sensor_columns[j]

            relationship_col = (
                f"{col1}_{col2}_Difference"
            )

            # Standardized difference
            mean1 = df.groupby("Station_ID")[col1].transform("mean")
            std1 = df.groupby("Station_ID")[col1].transform("std")

            mean2 = df.groupby("Station_ID")[col2].transform("mean")
            std2 = df.groupby("Station_ID")[col2].transform("std")

            z1 = (df[col1] - mean1) / std1.replace(0, np.nan)
            z2 = (df[col2] - mean2) / std2.replace(0, np.nan)

            df[relationship_col] = (
                z1 - z2
            ).abs()

    return df