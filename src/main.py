# %%
import pandas as pd
import numpy as np
from src.feature_engineering import (
    create_temporal_features,
    create_relationship_features
)

# %%
# 1. LOAD DATA
TRAIN_PATH = "data/development_train.csv"
VALIDATION_PATH = "data/development_validation.csv"

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

print("=" * 60)
print("DATA TRUST ENGINE - INITIAL DATA ANALYSIS")
print("=" * 60)

# %%
# 2. BASIC DATASET INFORMATION
print("\n--- DATASET SHAPE ---")

print("Training data:")
print("Rows:", train_df.shape[0])
print("Columns:", train_df.shape[1])

print("\nValidation data:")
print("Rows:", validation_df.shape[0])
print("Columns:", validation_df.shape[1])

# %%
# 3. COLUMN NAMES
print("\n--- COLUMNS ---")

print("Training columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(validation_df.columns.tolist())

# %%
# 4. CLEAN COLUMN NAMES
train_df.columns = (
    train_df.columns
    .str.strip()
    .str.replace(" ", "_")
)

validation_df.columns = (
    validation_df.columns
    .str.strip()
    .str.replace(" ", "_")
)


# %%
# 5. DATA TYPES
print("\n--- DATA TYPES ---")

print(train_df.dtypes)


# %%
# 6. CONVERT TIMESTAMP
train_df["Timestamp"] = pd.to_datetime(
    train_df["Timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

validation_df["Timestamp"] = pd.to_datetime(
    validation_df["Timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

print("\n--- TIMESTAMP CHECK ---")

print(
    "Invalid training timestamps:",
    train_df["Timestamp"].isna().sum()
)

print(
    "Invalid validation timestamps:",
    validation_df["Timestamp"].isna().sum()
)


# %%
# 7. NUMERIC COLUMNS
numeric_columns = [
    "CO_Channel",
    "NOx_Channel",
    "NO2_Channel",
    "Ambient_Temperature",
    "Relative_Humidity",
    "Absolute_Humidity"
]

for col in numeric_columns:

    train_df[col] = pd.to_numeric(
        train_df[col],
        errors="coerce"
    )

    validation_df[col] = pd.to_numeric(
        validation_df[col],
        errors="coerce"
    )


# %%
# 8. MISSING VALUES
print("\n--- MISSING VALUES: TRAINING ---")
print(train_df.isnull().sum())
print("\n--- MISSING VALUES: VALIDATION ---")
print(validation_df.isnull().sum())


# %%
# 9. MISSING VALUE PERCENTAGE
print("\n--- MISSING VALUE % ---")

train_missing = (
    train_df.isnull().mean() * 100
).round(2)

validation_missing = (
    validation_df.isnull().mean() * 100
).round(2)

print("\nTraining:")
print(train_missing)

print("\nValidation:")
print(validation_missing)


# %%
# 10. DUPLICATE CHECK
print("\n--- DUPLICATES ---")

print(
    "Training duplicate rows:",
    train_df.duplicated().sum()
)

print(
    "Validation duplicate rows:",
    validation_df.duplicated().sum()
)

print(
    "Training duplicate station/timestamps:",
    train_df.duplicated(
        subset=["Station_ID", "Timestamp"]
    ).sum()
)

print(
    "Validation duplicate station/timestamps:",
    validation_df.duplicated(
        subset=["Station_ID", "Timestamp"]
    ).sum()
)


# %%
# 11. STATION INFORMATION
print("\n--- STATIONS ---")

print(
    "Training stations:",
    train_df["Station_ID"].nunique()
)

print(
    "Validation stations:",
    validation_df["Station_ID"].nunique()
)

print("\nTraining station distribution:")
print(train_df["Station_ID"].value_counts())

print("\nValidation station distribution:")
print(validation_df["Station_ID"].value_counts())


# %%
# 12. DATE RANGE
print("\n--- DATE RANGE ---")

print(
    "Training:",
    train_df["Timestamp"].min(),
    "to",
    train_df["Timestamp"].max()
)

print(
    "Validation:",
    validation_df["Timestamp"].min(),
    "to",
    validation_df["Timestamp"].max()
)


# %%
# 13. STATISTICAL SUMMARY
print("\n--- TRAINING STATISTICS ---")

print(
    train_df[numeric_columns].describe()
)


print("\n--- VALIDATION STATISTICS ---")

print(
    validation_df[numeric_columns].describe()
)


# %%
# 14. CORRELATION ANALYSIS


print("\n--- TRAINING CORRELATION ---")

print(
    train_df[numeric_columns].corr().round(2)
)


# %%
# 15. SORT DATA


train_df = train_df.sort_values(
    ["Station_ID", "Timestamp"]
).reset_index(drop=True)

validation_df = validation_df.sort_values(
    ["Station_ID", "Timestamp"]
).reset_index(drop=True)


# %%
# 16. TIME DIFFERENCE


train_df["Time_Diff"] = (
    train_df
    .groupby("Station_ID")["Timestamp"]
    .diff()
)

validation_df["Time_Diff"] = (
    validation_df
    .groupby("Station_ID")["Timestamp"]
    .diff()
)

print("\n--- TIME INTERVALS: TRAINING ---")

print(
    train_df["Time_Diff"]
    .value_counts()
    .head(10)
)


# %%
# 17. BASIC RANGE VALIDATION


print("\n--- RANGE VALIDATION ---")

train_invalid_rh = (
    (train_df["Relative_Humidity"] < 0) |
    (train_df["Relative_Humidity"] > 100)
).sum()

validation_invalid_rh = (
    (validation_df["Relative_Humidity"] < 0) |
    (validation_df["Relative_Humidity"] > 100)
).sum()

print(
    "Invalid RH values - Train:",
    train_invalid_rh
)

print(
    "Invalid RH values - Validation:",
    validation_invalid_rh
)


# %%
# 18. CREATE MISSING INDICATORS


for col in numeric_columns:

    train_df[f"{col}_Missing"] = (
        train_df[col].isna().astype(int)
    )

    validation_df[f"{col}_Missing"] = (
        validation_df[col].isna().astype(int)
    )


# %%
# 19. FINAL OUTPUT
print("\n" + "=" * 60)
print("INITIAL DATA PROFILING COMPLETED")
print("=" * 60)

print("\nTraining preview:")
print(train_df.head())

print("\nValidation preview:")
print(validation_df.head())

# %%
# 19. DETAILED DATA QUALITY CHECK

print("\n" + "=" * 60)
print("DETAILED DATA QUALITY CHECK - TRAINING DATA")
print("=" * 60)

print("\n--- INFINITE VALUES ---")
print(np.isinf(train_df[numeric_columns]).sum())

print("\n--- NEGATIVE VALUES ---")

for col in numeric_columns:
    print(
        col,
        ":",
        (train_df[col] < 0).sum()
    )

print("\n--- UNIQUE VALUES ---")

for col in numeric_columns:
    print(
        col,
        ":",
        train_df[col].nunique()
    )

# %%
# 20. VALIDATION RANGES

VALID_RANGES = {
    "CO_Channel": (0, np.inf),
    "NOx_Channel": (0, np.inf),
    "NO2_Channel": (0, np.inf),
    "Ambient_Temperature": (-50, 60),
    "Relative_Humidity": (0, 100),
    "Absolute_Humidity": (0, np.inf)
}

print("\n--- RANGE VALIDATION ---")

for col, (lower, upper) in VALID_RANGES.items():

    invalid_count = (
        (train_df[col] < lower) |
        (train_df[col] > upper)
    ).sum()

    print(
        f"{col}: {invalid_count} invalid values"
    )

# %%
# 21. RANGE VIOLATION FEATURES

for col, (lower, upper) in VALID_RANGES.items():

    train_df[f"{col}_RangeViolation"] = (
        (train_df[col] < lower) |
        (train_df[col] > upper)
    ).astype(int)


range_columns = [
    col for col in train_df.columns
    if "RangeViolation" in col
]

print("\n--- RANGE VIOLATION COUNTS ---")

print(
    train_df[range_columns].sum()
)

# %%
# 22. SENSOR CHANGE FEATURES

sensor_columns = numeric_columns.copy()

for col in sensor_columns:

    train_df[f"{col}_Diff"] = (
        train_df
        .groupby("Station_ID")[col]
        .diff()
    )

print("\n--- SENSOR CHANGES ---")

print(
    train_df[
        [
            "Station_ID",
            "Timestamp",
            "CO_Channel",
            "CO_Channel_Diff",
            "NOx_Channel",
            "NOx_Channel_Diff"
        ]
    ].head(10)
)

# %%
# 23. SPIKE DETECTION

spike_thresholds = {}

for col in sensor_columns:

    diff_col = f"{col}_Diff"

    threshold = (
        train_df[diff_col]
        .abs()
        .quantile(0.99)
    )

    spike_thresholds[col] = threshold

    train_df[f"{col}_Spike"] = (
        train_df[diff_col].abs() > threshold
    ).astype(int)


print("\n--- SPIKE THRESHOLDS ---")

for col, threshold in spike_thresholds.items():
    print(col, ":", threshold)# %%

# %%
# 24. STUCK SENSOR DETECTION

for col in sensor_columns:

    previous_value = (
        train_df
        .groupby("Station_ID")[col]
        .shift(1)
    )

    same_as_previous = (
        train_df[col] == previous_value
    )

    train_df[f"{col}_Stuck"] = (
        same_as_previous
        .groupby(train_df["Station_ID"])
        .transform(
            lambda x: x.rolling(5).sum() >= 5
        )
        .astype(int)
    )


stuck_columns = [
    col for col in train_df.columns
    if "_Stuck" in col
]

print("\n--- STUCK SENSOR COUNTS ---")

print(
    train_df[stuck_columns].sum()
)# %%

# %%
# 25. TIME GAP DETECTION

EXPECTED_INTERVAL = pd.Timedelta(hours=1)

train_df["Time_Gap"] = (
    train_df["Time_Diff"] != EXPECTED_INTERVAL
).astype(int)

print("\n--- TIME GAPS ---")

print(
    "Unexpected time gaps:",
    train_df["Time_Gap"].sum()
)

# %%
# 26. OBSERVATION QUALITY FEATURES

missing_columns = [
    f"{col}_Missing"
    for col in sensor_columns
]

spike_columns = [
    f"{col}_Spike"
    for col in sensor_columns
]

train_df["Missing_Count"] = (
    train_df[missing_columns]
    .sum(axis=1)
)

train_df["Range_Violation_Count"] = (
    train_df[range_columns]
    .sum(axis=1)
)

train_df["Spike_Count"] = (
    train_df[spike_columns]
    .sum(axis=1)
)

train_df["Stuck_Count"] = (
    train_df[stuck_columns]
    .sum(axis=1)
)


print("\n--- QUALITY FEATURES ---")

print(
    train_df[
        [
            "Missing_Count",
            "Range_Violation_Count",
            "Spike_Count",
            "Stuck_Count",
            "Time_Gap"
        ]
    ].describe()
)


# %%
# 27. INITIAL TRUST SCORE

train_df["Trust_Score"] = 100

train_df["Trust_Score"] -= (
    train_df["Missing_Count"] * 15
)

train_df["Trust_Score"] -= (
    train_df["Range_Violation_Count"] * 20
)

train_df["Trust_Score"] -= (
    train_df["Spike_Count"] * 10
)

train_df["Trust_Score"] -= (
    train_df["Stuck_Count"] * 10
)

train_df["Trust_Score"] -= (
    train_df["Time_Gap"] * 10
)

train_df["Trust_Score"] = (
    train_df["Trust_Score"]
    .clip(0, 100)
)

print("\n--- TRUST SCORE ---")

print(
    train_df["Trust_Score"].describe()
)

# %%
# 28. TRUST CATEGORIES

def trust_category(score):

    if score >= 80:
        return "High Trust"

    elif score >= 50:
        return "Medium Trust"

    else:
        return "Low Trust"


train_df["Trust_Category"] = (
    train_df["Trust_Score"]
    .apply(trust_category)
)

print("\n--- TRUST DISTRIBUTION ---")

print(
    train_df["Trust_Category"]
    .value_counts()
)

# %%
# TEMPORAL FEATURE ENGINEERING

train_df = create_temporal_features(
    train_df,
    sensor_columns
)

validation_df = create_temporal_features(
    validation_df,
    sensor_columns
)

print("\nTemporal features created successfully.")

# %%
