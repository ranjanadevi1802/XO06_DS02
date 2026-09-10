# ============================================================
# STEP 16 - ML Model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import joblib


# ============================================================
# STEP 16 - Load Scored Training Data
# ============================================================

df = pd.read_csv(
    "../outputs/scored_train.csv"
)

print("\nTraining data loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# STEP 17 - Create ML Target
# ============================================================

target = "trust_category"

if target not in df.columns:
    raise ValueError(
        f"Column '{target}' not found in training data."
    )

drop_cols = [
    "trust_category",
    "trust_score"
]

# Remove identifiers
if "Station_ID" in df.columns:
    drop_cols.append("Station_ID")

# Remove timestamp
if "Timestamp" in df.columns:
    drop_cols.append("Timestamp")


X = df.drop(
    columns=drop_cols,
    errors="ignore"
)

y = df[target]


# ============================================================
# STEP 18 - Handle Missing Values
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.select_dtypes(
    include=np.number
)


print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# STEP 19 - Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# STEP 20 - Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# STEP 21 - Evaluate
# ============================================================

y_pred = model.predict(X_test)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# STEP 22 - Save Model
# ============================================================

model_path = "../outputs/model.pkl"

joblib.dump(
    model,
    model_path
)

print(
    "\nModel saved to:",
    model_path
)


# ============================================================
# STEP 23 - Feature Importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print(
    "\nTop 20 important features:"
)

print(
    importance.head(20)
)


# Plot feature importance
plt.figure(
    figsize=(10, 8)
)

importance.head(15).sort_values().plot(
    kind="barh"
)

plt.xlabel(
    "Importance"
)

plt.title(
    "Top Features Influencing Data Reliability"
)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 24 - Validation Dataset
# ============================================================
#
# The source instructions require the same feature-engineering
# and reliability-scoring pipeline to be applied here.
#
# Those functions are not defined in this model.py:
#     create_features()
#     temporal_reliability()
#     SENSOR_COLS
#     reliability_cols
#     classify_trust
#
# Therefore, validation scoring is handled separately unless
# those functions are imported from your feature-engineering
# module.
# ============================================================

validation_path = "../data/development_validation.csv"

try:

    validation = pd.read_csv(
        validation_path
    )

    print(
        "\nValidation dataset loaded successfully."
    )

    print(
        "Validation shape:",
        validation.shape
    )

except FileNotFoundError:

    print(
        "\nValidation dataset not found:"
    )

    print(
        validation_path
    )


# ============================================================
# END
# ============================================================

print(
    "\nML model pipeline completed."
)