import pandas as pd
import numpy as np

# 1. LOAD DATA
TRAIN_PATH = "data/development_train.csv"
VALIDATION_PATH = "data/development_validation.csv"

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

print("=" * 60)
print("DATA TRUST ENGINE - INITIAL DATA ANALYSIS")
print("=" * 60)


# 2. BASIC DATASET INFORMATION
print("\n--- DATASET SHAPE ---")

print("Training data:")
print("Rows:", train_df.shape[0])
print("Columns:", train_df.shape[1])

print("\nValidation data:")
print("Rows:", validation_df.shape[0])
print("Columns:", validation_df.shape[1])


# 3. COLUMN NAMES
print("\n--- COLUMNS ---")

print("Training columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(validation_df.columns.tolist())


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



# 5. DATA TYPES
print("\n--- DATA TYPES ---")

print(train_df.dtypes)



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



# 8. MISSING VALUES
print("\n--- MISSING VALUES: TRAINING ---")
print(train_df.isnull().sum())
print("\n--- MISSING VALUES: VALIDATION ---")
print(validation_df.isnull().sum())



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



# 13. STATISTICAL SUMMARY
print("\n--- TRAINING STATISTICS ---")

print(
    train_df[numeric_columns].describe()
)


print("\n--- VALIDATION STATISTICS ---")

print(
    validation_df[numeric_columns].describe()
)



# 14. CORRELATION ANALYSIS


print("\n--- TRAINING CORRELATION ---")

print(
    train_df[numeric_columns].corr().round(2)
)



# 15. SORT DATA


train_df = train_df.sort_values(
    ["Station_ID", "Timestamp"]
).reset_index(drop=True)

validation_df = validation_df.sort_values(
    ["Station_ID", "Timestamp"]
).reset_index(drop=True)



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



# 18. CREATE MISSING INDICATORS


for col in numeric_columns:

    train_df[f"{col}_Missing"] = (
        train_df[col].isna().astype(int)
    )

    validation_df[f"{col}_Missing"] = (
        validation_df[col].isna().astype(int)
    )



# 19. FINAL OUTPUT
print("\n" + "=" * 60)
print("INITIAL DATA PROFILING COMPLETED")
print("=" * 60)

print("\nTraining preview:")
print(train_df.head())

print("\nValidation preview:")
print(validation_df.head())