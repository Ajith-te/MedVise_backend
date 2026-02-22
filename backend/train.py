
import os
import json
import pickle
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

# optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -------------------------
# Config
# -------------------------
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "merged_dataset")
OUTPUT_DIR = os.path.join(ROOT, "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
MIN_SAMPLES_FOR_5YR = 50

# -------------------------
# Helpers
# -------------------------
def safe_read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {name} — continuing with empty DataFrame.")
        return pd.DataFrame()
    print(f"Loading {name} ({os.path.getsize(path)/1024:.0f} KB)...")
    return pd.read_csv(path)

def find_date_col(df, candidates):
    """Return first matching candidate column present in df, else None."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic: try columns with 'date' or 'time' in their name
    lower_cols = [col.lower() for col in df.columns]
    for i, col in enumerate(lower_cols):
        if "date" in col or "time" in col or "timestamp" in col:
            return df.columns[i]
    return None

def parse_datetime_col(df, candidates):
    """Detect and parse a datetime column; return column name or None."""
    col = find_date_col(df, candidates)
    if col is None:
        return None
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        # elementwise fallback
        df[col] = df[col].apply(lambda x: pd.to_datetime(x, errors="coerce") if pd.notna(x) else pd.NaT)
    # convert tz-aware series to naive
    try:
        if getattr(df[col].dt, "tz", None) is not None:
            df[col] = df[col].dt.tz_convert(None).dt.tz_localize(None)
    except Exception:
        # elementwise removal if needed
        df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, "tzinfo") and x.tzinfo is not None else x)
    return col

def latest_value_before(df, patient, date, col_date_candidates, value_col):
    """Get the last value of value_col for patient at or before date. Safe if missing."""
    if df is None or df.empty or pd.isna(patient):
        return None
    col = find_date_col(df, col_date_candidates)
    if col is None or col not in df.columns:
        return None
    # ensure datetime
    if not np.issubdtype(df[col].dtype, np.datetime64):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            df[col] = df[col].apply(lambda x: pd.to_datetime(x, errors="coerce") if pd.notna(x) else pd.NaT)
    sub = df[(df["PATIENT"].astype(str) == str(patient)) & (df[col].notnull()) & (df[col] <= date)]
    if sub.empty:
        return None
    sub = sub.sort_values(col)
    return sub.iloc[-1].get(value_col, None)

# -------------------------
# Load data
# -------------------------
patients = safe_read_csv("patients.csv")
conditions = safe_read_csv("conditions.csv")
medications = safe_read_csv("medications.csv")
procedures = safe_read_csv("procedures.csv")
careplans = safe_read_csv("careplans.csv")
observations = safe_read_csv("observations.csv")

if patients.empty:
    raise RuntimeError("patients.csv is required. Place it in merged_dataset/ and retry.")

# Normalize patient id
pid_col = next((c for c in ["Id", "ID", "PATIENT", "Patient", "patient"] if c in patients.columns), None)
if pid_col:
    patients = patients.rename(columns={pid_col: "PATIENT"})
else:
    raise RuntimeError("Could not find patient id column in patients.csv")

# parse birthdate -> AGE
if "BIRTHDATE" in patients.columns:
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["AGE"] = ((pd.Timestamp.now() - patients["BIRTHDATE"]).dt.days // 365).fillna(0).astype(int)
else:
    if "AGE" in patients.columns:
        patients["AGE"] = pd.to_numeric(patients["AGE"], errors="coerce").fillna(0).astype(int)
    else:
        patients["AGE"] = 0

base = patients.loc[:, ["PATIENT"] + [c for c in ["FIRST", "LAST", "GENDER", "AGE"] if c in patients.columns]].copy()
base["PATIENT"] = base["PATIENT"].astype(str)

# -------------------------
# Parse date columns for tables up front
# -------------------------
conds_date = parse_datetime_col(conditions, ["START", "DATE", "RECORDED_DATE"])
proc_date = parse_datetime_col(procedures, ["DATE", "START", "PERFORMED"])
med_date = parse_datetime_col(medications, ["START", "DATE", "PRESCRIBED_DATE"])
care_date = parse_datetime_col(careplans, ["START", "DATE"])
obs_date = parse_datetime_col(observations, ["DATE", "START", "RECORDED_DATE"])

# -------------------------
# Most-recent features
# -------------------------
def most_recent_text(df, patient_col="PATIENT", desc_col="DESCRIPTION", date_col=None, out_name=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=[patient_col, out_name or desc_col])
    df = df.copy()
    df[patient_col] = df[patient_col].astype(str)
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col)
    # groupby tail(1) will work even if unsorted; but best effort to sort above
    recent = df.groupby(patient_col).tail(1).loc[:, [patient_col, desc_col] if desc_col in df.columns else [patient_col]]
    if desc_col in recent.columns and out_name:
        recent = recent.rename(columns={desc_col: out_name})
    return recent

# recent condition
if not conditions.empty and "DESCRIPTION" in conditions.columns:
    recent_cond = most_recent_text(conditions, "PATIENT", "DESCRIPTION", conds_date, out_name="recent_condition")
    base = base.merge(recent_cond[["PATIENT", "recent_condition"]], on="PATIENT", how="left")
else:
    base["recent_condition"] = None

# recent careplan -> response_score
if not careplans.empty and "DESCRIPTION" in careplans.columns:
    careplans = careplans.copy()
    careplans["PATIENT"] = careplans["PATIENT"].astype(str)
    careplans["DESCRIPTION_CLEAN"] = careplans["DESCRIPTION"].astype(str).str.lower().fillna("")
    response_map = {"improve": 2, "recovery": 2, "resolved": 2, "monitor": 1, "no change": 0, "worsen": -1, "worse": -1}
    def care_response_score(text):
        s = 0
        for k, v in response_map.items():
            if k in text:
                s += v
        return s
    careplans["response_score"] = careplans["DESCRIPTION_CLEAN"].apply(care_response_score)
    recent_care = most_recent_text(careplans, "PATIENT", "response_score", care_date, out_name="response_score")
    base = base.merge(recent_care[["PATIENT", "response_score"]], on="PATIENT", how="left")
    base["response_score"] = pd.to_numeric(base.get("response_score", 0)).fillna(0)
else:
    base["response_score"] = 0

# recent procedure severity
if not procedures.empty and "DESCRIPTION" in procedures.columns:
    procedures = procedures.copy()
    procedures["PATIENT"] = procedures["PATIENT"].astype(str)
    procedures["DESCRIPTION_CLEAN"] = procedures["DESCRIPTION"].astype(str).str.lower().fillna("")
    severity_map = {
        "transplant": 4, "bypass": 4, "open": 3, "surgery": 3, "arthroplasty": 3,
        "therapy": 2, "procedure": 2, "screening": 1, "assessment": 1, "check": 1
    }
    def proc_severity(desc):
        scores = [v for k, v in severity_map.items() if k in desc]
        return max(scores) if scores else 0
    procedures["proc_severity"] = procedures["DESCRIPTION_CLEAN"].apply(proc_severity)
    recent_proc = most_recent_text(procedures, "PATIENT", "proc_severity", proc_date, out_name="proc_severity")
    base = base.merge(recent_proc[["PATIENT", "proc_severity"]], on="PATIENT", how="left")
    base["proc_severity"] = pd.to_numeric(base.get("proc_severity", 0)).fillna(0)
else:
    base["proc_severity"] = 0

# recent medication intensity
if not medications.empty and "DESCRIPTION" in medications.columns:
    medications = medications.copy()
    medications["PATIENT"] = medications["PATIENT"].astype(str)
    medications["DESCRIPTION_CLEAN"] = medications["DESCRIPTION"].astype(str).str.lower().fillna("")
    def med_intensity(desc):
        if any(k in desc for k in ["insulin", "chemotherapy", "chemo", "steroid"]):
            return 3
        if any(k in desc for k in ["antibiotic", "antibiotics", "antihypertensive", "antihypertensives"]):
            return 2
        if any(k in desc for k in ["vitamin", "supplement"]):
            return 1
        return 0
    medications["med_intensity"] = medications["DESCRIPTION_CLEAN"].apply(med_intensity)
    recent_med = most_recent_text(medications, "PATIENT", "med_intensity", med_date, out_name="med_intensity")
    base = base.merge(recent_med[["PATIENT", "med_intensity"]], on="PATIENT", how="left")
    base["med_intensity"] = pd.to_numeric(base.get("med_intensity", 0)).fillna(0)
else:
    base["med_intensity"] = 0

# recent observation (value)
def extract_numeric(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.number)): return float(v)
    text = str(v)
    filtered = ''.join(ch for ch in text if (ch.isdigit() or ch == '.' or ch == '-'))
    try:
        return float(filtered) if filtered not in ("", ".", "-") else np.nan
    except:
        return np.nan

if not observations.empty and "VALUE" in observations.columns:
    observations = observations.copy()
    observations["PATIENT"] = observations["PATIENT"].astype(str)
    observations = observations.sort_values(obs_date) if obs_date else observations
    recent_obs = observations.groupby("PATIENT").tail(1).copy()
    recent_obs["VALUE_NUM"] = recent_obs["VALUE"].apply(extract_numeric)
    recent_obs = recent_obs.rename(columns={"DESCRIPTION": "recent_observation_desc", "VALUE_NUM": "recent_observation_val"})
    base = base.merge(recent_obs[["PATIENT", "recent_observation_desc", "recent_observation_val"]], on="PATIENT", how="left")
    base["recent_observation_val"] = pd.to_numeric(base.get("recent_observation_val", 0)).fillna(0)
else:
    base["recent_observation_val"] = 0
    base["recent_observation_desc"] = None

# Normalize column names expected later
base = base.rename(columns={
    "proc_severity": "procedure_severity",
    "med_intensity": "medication_intensity",
    "response_score": "treatment_response",
    "recent_observation_val": "observation_value"
})

# Ensure numeric fields exist
for c in ["procedure_severity", "medication_intensity", "treatment_response", "observation_value", "AGE"]:
    if c not in base.columns:
        base[c] = 0
    base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

# -------------------------
# Deterministic risk score & labels (for classifier training)
# -------------------------
def compute_risk_score_row(r):
    return (
        r["procedure_severity"] * 1.2 +
        r["medication_intensity"] * 1.0 +
        (-0.8) * r["treatment_response"] +
        max(0, r["AGE"] - 60) * 0.05 +
        (r["observation_value"] / 100.0)
    )

base["risk_score"] = base.apply(compute_risk_score_row, axis=1)

q_low, q_high = base["risk_score"].quantile([0.33, 0.66]).values
def risk_label_from_score(s):
    if pd.isna(s):
        return "Unknown"
    if s <= q_low: return "Low"
    if s <= q_high: return "Medium"
    return "High"

base["RISK_LEVEL"] = base["risk_score"].apply(risk_label_from_score)

# -------------------------
# Current Risk Classifier
# -------------------------

feature_cols = ["AGE", "procedure_severity", "medication_intensity", "treatment_response", "observation_value"]
X = base[feature_cols].astype(float)
y = base["RISK_LEVEL"].astype(str)

le_risk = LabelEncoder()
y_encoded = le_risk.fit_transform(y)

if len(np.unique(y_encoded)) < 2 or X.shape[0] < 10:
    raise RuntimeError("Not enough samples or classes to train classifier. Need at least 2 classes and ~10 rows.")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded)
clf = RandomForestClassifier(n_estimators=20, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\n✅ Current Risk Model Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

# save classifier bundle
with open(os.path.join(OUTPUT_DIR, "current_risk_classifier_second.pkl"), "wb") as f:
    pickle.dump({"model": clf, "risk_label_encoder": le_risk, "feature_cols": feature_cols}, f)

# -------------------------
# Build time snapshots for multi-year modeling
# -------------------------
date_frames = []
for df, name, cand in [
    (conditions, "Condition", ["START", "DATE", "RECORDED_DATE"]),
    (procedures, "Procedure", ["DATE", "START", "PERFORMED"]),
    (medications, "Medication", ["START", "DATE", "PRESCRIBED_DATE"]),
    (observations, "Observation", ["DATE", "START", "RECORDED_DATE"]),
    (careplans, "CarePlan", ["START", "DATE"])
]:
    if df is None or df.empty:
        continue
    col = parse_datetime_col(df, cand)
    if col:
        temp = df.loc[:, ["PATIENT", col]].copy()
        temp = temp.rename(columns={col: "DATE"})
        temp["SOURCE"] = name
        date_frames.append(temp.dropna(subset=["DATE"]))

if date_frames:
    all_dates_df = pd.concat(date_frames, ignore_index=True)
else:
    all_dates_df = pd.DataFrame(columns=["PATIENT", "DATE", "SOURCE"])

print("\nPreparing time-based snapshots for 5-year prediction ...")

# keep only recent history (past 15 years)
if not all_dates_df.empty:
    all_dates_df["DATE"] = pd.to_datetime(all_dates_df["DATE"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
    all_dates_df = all_dates_df[all_dates_df["DATE"] >= cutoff]

# --- 🔹 Limit to 500 patients for faster processing ---
if not all_dates_df.empty:
    print("Limiting to 500 patients for faster snapshot building...")
    unique_pids = all_dates_df["PATIENT"].unique()
    sample_pids = unique_pids[:100]   # or use np.random.choice(unique_pids, 500, replace=False)
    sampled = all_dates_df[all_dates_df["PATIENT"].isin(sample_pids)]
else:
    sampled = all_dates_df

# Build yearly snapshots per patient
snap_rows = []
if not sampled.empty:
    grouped = sampled.groupby("PATIENT")
   
    for pid, g in tqdm(grouped, desc="Building 5-year snapshots"):
        pid = str(pid)
        years = sorted(set(g["DATE"].dt.year.dropna().astype(int).tolist()))
        if not years:
            continue
        for y in years:
            candidate_dates = g[g["DATE"].dt.year == y]["DATE"].sort_values()
            if candidate_dates.empty:
                continue
            snapshot_date = candidate_dates.iloc[-1]

            proc_desc = latest_value_before(procedures, pid, snapshot_date, ["DATE", "START"], "DESCRIPTION")
            med_desc = latest_value_before(medications, pid, snapshot_date, ["START", "DATE"], "DESCRIPTION")
            care_desc = latest_value_before(careplans, pid, snapshot_date, ["START", "DATE"], "DESCRIPTION")
            obs_val = latest_value_before(observations, pid, snapshot_date, ["DATE", "START"], "VALUE")
            birth_val = latest_value_before(patients, pid, snapshot_date, ["BIRTHDATE"], "BIRTHDATE")

            ps = 0
            if isinstance(proc_desc, str):
                pd_low = proc_desc.lower()
                for k, v in {"transplant":4, "surgery":3, "therapy":2, "screening":1}.items():
                    if k in pd_low: ps = max(ps, v)

            mi = 0
            if isinstance(med_desc, str):
                md_low = med_desc.lower()
                if any(k in md_low for k in ["insulin","chemo","chemotherapy","steroid"]): mi = 3
                elif any(k in md_low for k in ["antibiotic","antibiotics","antihypertensive"]): mi = 2
                elif any(k in md_low for k in ["vitamin","supplement"]): mi = 1

            tr = 0
            if isinstance(care_desc, str):
                cd_low = care_desc.lower()
                for k, v in {"improve":2,"recovery":2,"monitor":1,"no change":0,"worsen":-1,"worse":-1}.items():
                    if k in cd_low: tr += v

            ov = None
            if pd.notna(obs_val):
                ov = extract_numeric(obs_val)

            # Calculate age
            age_val = None
            if isinstance(birth_val, (pd.Timestamp, datetime)):
                age_val = (snapshot_date.year - birth_val.year) - (
                    (snapshot_date.month, snapshot_date.day) < (birth_val.month, birth_val.day)
                )
            else:
                try:
                    age_val = int(base.loc[base["PATIENT"] == pid, "AGE"].iloc[0])
                except Exception:
                    age_val = 0

            risk_score = (
                ps * 1.2 +
                mi * 1.0 +
                (-0.8) * tr +
                max(0, age_val - 60) * 0.05 +
                (float(ov) / 100.0 if ov else 0)
            )

            snap_rows.append({
                "PATIENT": pid,
                "snapshot_date": snapshot_date,
                "AGE": age_val,
                "procedure_severity": ps,
                "medication_intensity": mi,
                "treatment_response": tr,
                "observation_value": float(ov) if ov is not None else 0.0,
                "risk_score": risk_score
            })

snap_df = pd.DataFrame(snap_rows)
print(f"Built {len(snap_df)} time snapshots (patient-year).")


# ---------------------------------------------------------------------
# ✅ If not enough t->t+5 samples, synthesize future data automatically
# ---------------------------------------------------------------------
# --- Synthetic 5-year projection logic ---
if len(snap_df) < 1000:
    print("⚠️ Not enough historical samples — creating synthetic 5-year projections.")
    synthetic = snap_df.copy()
    synthetic["AGE"] = synthetic["AGE"] + 5
    synthetic["procedure_severity"] = (synthetic["procedure_severity"] * 1.1).clip(0, 5)
    synthetic["medication_intensity"] = (synthetic["medication_intensity"] * 1.1).clip(0, 5)
    synthetic["treatment_response"] = (synthetic["treatment_response"] * 0.9)
    synthetic["observation_value"] = synthetic["observation_value"] * 1.05
    synthetic["future_risk_score"] = synthetic["risk_score"] * 1.15
    snap_df = synthetic
else:
    snap_df["future_risk_score"] = snap_df["risk_score"] * 1.1  # small growth

# --- Train synthetic 5-year risk regressor (fixed version) ---
try:
    Xf = snap_df[["AGE", "procedure_severity", "medication_intensity", "treatment_response", "observation_value"]]
    yf = snap_df["future_risk_score"].fillna(0)  # ✅ replace NaN with 0 safely

    # ✅ check if we have enough valid samples
    if len(Xf) >= 10 and yf.notna().sum() > 5:
        future_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        future_model.fit(Xf, yf)
        with open(os.path.join(OUTPUT_DIR, "future_5yr_regressor_second.pkl"), "wb") as f:
            pickle.dump(future_model, f)
        print("✅ Synthetic 5-year risk regressor trained and saved.")
    else:
        print("⚠️ Not enough data to train RandomForest — using fallback linear projection.")
        snap_df["future_risk_score"] = snap_df["risk_score"] * 1.05  # fallback growth

except Exception as e:
    print(f"⚠️ Synthetic 5-year regressor training failed: {e}")
    snap_df["future_risk_score"] = snap_df["risk_score"] * 1.05  # last fallback


# -------------------------
# Treatment recommender (condition -> careplan)
# -------------------------

print("\nPreparing treatment recommendation model (condition -> careplan)...")
start_time = time.time()

treatment_model = None

if not conditions.empty and not careplans.empty and "DESCRIPTION" in conditions.columns and "DESCRIPTION" in careplans.columns:
    cond = conditions.copy()
    cond["PATIENT"] = cond["PATIENT"].astype(str)
    if conds_date:
        cond[conds_date] = pd.to_datetime(cond[conds_date], errors="coerce")

    cp = careplans.copy()
    cp["PATIENT"] = cp["PATIENT"].astype(str)
    if care_date:
        cp[care_date] = pd.to_datetime(cp[care_date], errors="coerce")

    # Limit to 500 careplans for faster processing
    if len(cp) > 500:
        print(f"Limiting to 500 careplan entries (from {len(cp)}) for faster training...")
        cp = cp.sample(500, random_state=42)

    merged_pairs = []
    for idx, cp_row in tqdm(cp.iterrows(), total=len(cp), desc="Building condition-careplan pairs"):
        pid = str(cp_row.get("PATIENT"))
        if not pid or pid == "nan" or pd.isna(pid):
            continue

        cdate = cp_row.get(care_date) if care_date and care_date in cp.columns else None
        cand = cond[cond["PATIENT"] == pid]
        if cand.empty:
            continue

        if cdate is not None:
            # take conditions at or before cdate
            cand_sub = cand[cand[conds_date] <= cdate] if conds_date and conds_date in cand.columns else pd.DataFrame()
            if cand_sub.empty:
                cand_sub = cand
        else:
            cand_sub = cand

        if cand_sub.empty:
            continue

        chosen = cand_sub.sort_values(conds_date).iloc[-1] if conds_date else cand_sub.iloc[-1]
        cond_desc = str(chosen.get("DESCRIPTION", "")).strip().lower()
        care_desc = str(cp_row.get("DESCRIPTION", "")).strip().lower()

        if cond_desc and care_desc:
            merged_pairs.append({
                "PATIENT": pid,
                "condition_desc": cond_desc,
                "careplan_desc": care_desc
            })

    treat_df = pd.DataFrame(merged_pairs).dropna()

    print(f"✅ Collected {len(treat_df)} valid condition-careplan pairs.")

    if not treat_df.empty:
        le_cond = LabelEncoder()
        le_treat = LabelEncoder()
        treat_df["cond_code"] = le_cond.fit_transform(treat_df["condition_desc"].astype(str))
        treat_df["treat_code"] = le_treat.fit_transform(treat_df["careplan_desc"].astype(str))

        X_t = treat_df[["cond_code"]]
        y_t = treat_df["treat_code"]

        tm = RandomForestClassifier(
            n_estimators=10,  # reduced for faster training
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        tm.fit(X_t, y_t)

        treatment_model = {
            "model": tm,
            "cond_encoder": le_cond,
            "treat_encoder": le_treat
        }

        with open(os.path.join(OUTPUT_DIR, "treatment_recommender_condition.pkl"), "wb") as f:
            pickle.dump(treatment_model, f)

        print("✅ Treatment recommendation model trained and saved successfully.")
    else:
        print("⚠️ No valid condition-careplan pairs found to train treatment model.")
else:
    print("⚠️ Missing or invalid conditions/careplans data; skipping treatment model training.")

print(f"⏱️ Treatment recommendation process completed in {time.time() - start_time:.2f} seconds.")
