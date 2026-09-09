import pandas as pd
import numpy as np
                                                                                                        
RAW_PATH = "surprise_challenge_1.csv"
OUT_PATH = "classified_flagged_observations.csv"

df = pd.read_csv(RAW_PATH)

NUM_COLS = [

    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ---------------------------------------------------------------------------
# STEP 1 — FLAGGING (candidate extremes only, NOT the final decision)
# Per-Type (L/M/H) robust z-score, since operating class can shift the
# normal band slightly. This step only decides "worth investigating",
# never "faulty" vs "real" on its own.
# ---------------------------------------------------------------------------
def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    mad = mad if mad > 1e-9 else s.std()
    return 0.6745 * (s - med) / mad

flag_z = pd.DataFrame(index=df.index)
for col in NUM_COLS:
    flag_z[col] = df.groupby("Type")[col].transform(robust_z)

df["is_flagged"] = (flag_z.abs() > 3.0).any(axis=1)
df["flag_reason"] = flag_z.abs().gt(3.0).apply(
    lambda row: ", ".join(row.index[row]), axis=1
)

# ---------------------------------------------------------------------------
# STEP 2 — EVIDENCE FEATURES (multi-signal, beyond the raw value)
# ---------------------------------------------------------------------------

# (a) Cross-sensor physical-consistency evidence
#     Process temp and Air temp are physically coupled (~+10K in this process).
#     Torque, speed and an implied "power" (torque * angular speed) are coupled
#     by the equipment's operating envelope.
df["temp_gap"] = df["Process temperature [K]"] - df["Air temperature [K]"]
temp_gap_mu, temp_gap_sd = df["temp_gap"].mean(), df["temp_gap"].std()
df["temp_gap_z"] = (df["temp_gap"] - temp_gap_mu) / temp_gap_sd

df["power_proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
power_mu, power_sd = df["power_proxy"].mean(), df["power_proxy"].std()
df["power_z"] = (df["power_proxy"] - power_mu) / power_sd

# A flagged point is "physically consistent" if its coupled-sensor residual
# stays inside a normal band even though one raw value is extreme —
# i.e., the extremity shows up coherently across related sensors rather
# than as an isolated break in a known physical relationship.
df["consistency_break"] = (df["temp_gap_z"].abs() > 3) | (df["power_z"].abs() > 3.5)

# (b) Persistence evidence
#     No explicit timestamp is provided, so row order is used as the best
#     available proxy for reading sequence (documented assumption — swap in
#     a real timestamp/sequence key if the held-out batch provides one).
#     A genuine event tends to show sustained/adjacent extremity; a faulty
#     sensor tends to produce an isolated single-row spike.
WINDOW = 3
persist_hits = pd.DataFrame(index=df.index)
for col in NUM_COLS:
    persist_hits[col] = flag_z[col].abs() > 2.5  # softer band for neighbours

df["neighbour_support"] = 0
for col in NUM_COLS:
    hit = persist_hits[col].astype(int)
    rolled = hit.rolling(window=WINDOW, center=True, min_periods=1).sum()
    df["neighbour_support"] += (rolled >= 2).astype(int)

df["is_persistent"] = df["neighbour_support"] > 0

# (c) Rate-of-change evidence
#     Compare each row to its immediate neighbours (row-order proxy again).
#     A single-row jump-and-return is typical of a sensor glitch; a gradual
#     ramp that stays elevated is typical of a real mechanical/environmental
#     event.
df["roc_score"] = 0.0
for col in NUM_COLS:
    diff_prev = (df[col] - df[col].shift(1)).abs()
    diff_next = (df[col] - df[col].shift(-1)).abs()
    col_std = df[col].std()
    spike = (diff_prev > 2 * col_std) & (diff_next > 2 * col_std)
    df["roc_score"] += spike.astype(float)

df["is_isolated_spike"] = df["roc_score"] > 0

# ---------------------------------------------------------------------------
# STEP 3 — DECISION (combines >=2 independent evidence types; NOT a single
# global threshold on the raw value)
# ---------------------------------------------------------------------------
def classify(row):
    if not row["is_flagged"]:
        return "NORMAL"
    votes_unreliable = 0
    votes_valid = 0

    # cross-sensor evidence
    if row["consistency_break"]:
        votes_unreliable += 1
    else:
        votes_valid += 1

    # persistence evidence
    if row["is_persistent"]:
        votes_valid += 1
    else:
        votes_unreliable += 1

    # rate-of-change evidence
    if row["is_isolated_spike"]:
        votes_unreliable += 1
    else:
        votes_valid += 1

    return "VALID_EXTREME_EVENT" if votes_valid > votes_unreliable else "UNRELIABLE_READING"

df["classification"] = df.apply(classify, axis=1)

flagged = df[df["is_flagged"]].copy()
print(f"Total rows: {len(df)}")
print(f"Flagged as candidate-extreme: {len(flagged)}")
print(flagged["classification"].value_counts())
print()
print(flagged[["Equipment_Record_ID","Type","flag_reason","consistency_break",
               "is_persistent","is_isolated_spike","classification"]].head(15).to_string(index=False))

out_cols = ["Equipment_Record_ID","Type"] + NUM_COLS + [
    "is_flagged","flag_reason","consistency_break","is_persistent",
    "is_isolated_spike","classification"
]
df[out_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")