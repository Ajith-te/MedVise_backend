'''import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
DATA_DIR = os.path.join(os.getcwd(), "merged_dataset")
OUTPUT_DIR = os.path.join(os.getcwd(), "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------
def safe_read(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"⚠️ Warning: {filename} not found.")
        return pd.DataFrame()

patients = safe_read("patients.csv")
conditions = safe_read("conditions.csv")
medications = safe_read("medications.csv")
procedures = safe_read("procedures.csv")
careplans = safe_read("careplans.csv")
observations = safe_read("observations.csv")

# -----------------------------------------------------------
# Preprocess & Basic Features
# -----------------------------------------------------------
patients["BIRTHDATE"] = pd.to_datetime(patients.get("BIRTHDATE"), errors="coerce")
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365
base = patients[["Id", "FIRST", "LAST", "GENDER", "AGE"]].rename(columns={"Id": "PATIENT"})

# -----------------------------------------------------------
# 1️⃣ Most Recent Condition
# -----------------------------------------------------------
if not conditions.empty:
    date_col = next((c for c in ["START", "DATE", "RECORDED_DATE"] if c in conditions.columns), None)
    if date_col:
        conditions[date_col] = pd.to_datetime(conditions[date_col], errors="coerce")
        recent_conditions = conditions.sort_values(date_col).groupby("PATIENT").tail(1)
    else:
        recent_conditions = conditions.groupby("PATIENT").tail(1)
    base = base.merge(recent_conditions[["PATIENT", "DESCRIPTION"]], on="PATIENT", how="left")
    base.rename(columns={"DESCRIPTION": "recent_condition"}, inplace=True)
else:
    base["recent_condition"] = np.nan

# -----------------------------------------------------------
# 2️⃣ Treatment Response (Careplans)
# -----------------------------------------------------------
if not careplans.empty:
    careplans["DESCRIPTION"] = careplans["DESCRIPTION"].str.lower().fillna("")
    response_map = {"improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worsen": -1}
    careplans["response_score"] = careplans["DESCRIPTION"].apply(
        lambda x: sum([v for k, v in response_map.items() if k in x])
    )
    response_score = careplans.groupby("PATIENT")["response_score"].mean()
    base["treatment_response"] = base["PATIENT"].map(response_score).fillna(0)
else:
    base["treatment_response"] = 0

# -----------------------------------------------------------
# 3️⃣ Procedure Severity
# -----------------------------------------------------------
if not procedures.empty:
    proc_date_col = next((c for c in ["DATE", "START", "PERFORMED"] if c in procedures.columns), None)
    if proc_date_col:
        procedures[proc_date_col] = pd.to_datetime(procedures[proc_date_col], errors="coerce")
        recent_procs = procedures.sort_values(proc_date_col).groupby("PATIENT").tail(1)
    else:
        recent_procs = procedures.groupby("PATIENT").tail(1)

    recent_procs["DESCRIPTION"] = recent_procs["DESCRIPTION"].str.lower().fillna("")
    severity_keywords = {"surgery": 3, "transplant": 4, "therapy": 2, "checkup": 1, "screening": 1}
    recent_procs["severity"] = recent_procs["DESCRIPTION"].apply(
        lambda x: max([v for k, v in severity_keywords.items() if k in x], default=0)
    )
    base["procedure_severity"] = base["PATIENT"].map(
        recent_procs.set_index("PATIENT")["severity"]
    ).fillna(0)
else:
    base["procedure_severity"] = 0

# -----------------------------------------------------------
# 4️⃣ Medication Intensity
# -----------------------------------------------------------
if not medications.empty:
    med_date_col = next((c for c in ["START", "DATE", "PRESCRIBED_DATE"] if c in medications.columns), None)
    if med_date_col:
        medications[med_date_col] = pd.to_datetime(medications[med_date_col], errors="coerce")
        recent_meds = medications.sort_values(med_date_col).groupby("PATIENT").tail(1)
    else:
        recent_meds = medications.groupby("PATIENT").tail(1)

    recent_meds["DESCRIPTION"] = recent_meds["DESCRIPTION"].str.lower().fillna("")

    def med_score(x):
        if any(k in x for k in ["insulin", "chemo", "steroid"]):
            return 3
        elif any(k in x for k in ["antibiotic", "hypertensive"]):
            return 2
        elif any(k in x for k in ["vitamin", "supplement"]):
            return 1
        return 0

    recent_meds["med_score"] = recent_meds["DESCRIPTION"].apply(med_score)
    base["medication_intensity"] = base["PATIENT"].map(
        recent_meds.set_index("PATIENT")["med_score"]
    ).fillna(0)
else:
    base["medication_intensity"] = 0

# -----------------------------------------------------------
# 5️⃣ Observation (Most Recent)
# -----------------------------------------------------------
if not observations.empty and "VALUE" in observations.columns:
    obs_date_col = next((c for c in ["DATE", "RECORDED_DATE", "START"] if c in observations.columns), None)
    if obs_date_col:
        observations[obs_date_col] = pd.to_datetime(observations[obs_date_col], errors="coerce")
        recent_obs = observations.sort_values(obs_date_col).groupby("PATIENT").tail(1)
        base["avg_observation"] = base["PATIENT"].map(recent_obs.set_index("PATIENT")["VALUE"]).fillna(0)
    else:
        base["avg_observation"] = 0
else:
    base["avg_observation"] = 0

# -----------------------------------------------------------
# 6️⃣ Compute Risk Score
# -----------------------------------------------------------
def compute_risk(df):
    return (
        df["procedure_severity"] * 1.2 +
        df["medication_intensity"] * 1.0 +
        df["treatment_response"] * -0.8 +
        np.maximum(0, df["AGE"] - 60) * 0.05 +
        (df["avg_observation"] / 100)
    )

base["risk_score"] = compute_risk(base)

def categorize_risk(x):
    if x < 2.5:
        return "Low"
    elif 2.5 <= x < 4.5:
        return "Medium"
    else:
        return "High"

base["RISK_LEVEL"] = base["risk_score"].apply(categorize_risk)

# -----------------------------------------------------------
# 7️⃣ Current Risk Model
# -----------------------------------------------------------
le_gender = LabelEncoder()
base["GENDER"] = le_gender.fit_transform(base["GENDER"].astype(str))
le_risk = LabelEncoder()
y_encoded = le_risk.fit_transform(base["RISK_LEVEL"])

features = ["AGE", "GENDER", "procedure_severity", "medication_intensity", "treatment_response", "avg_observation"]
X = base[features]
y = y_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_current = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model_current.fit(X_train, y_train)

print("\n🔹 Current Risk Model Performance 🔹")
print(classification_report(y_test, model_current.predict(X_test), target_names=le_risk.classes_))

# -----------------------------------------------------------
# 8️⃣ 5-Year Future Risk Model
# -----------------------------------------------------------
future_X = X.copy()
future_X["AGE"] += 5
future_X["procedure_severity"] *= 1.1
future_X["medication_intensity"] *= 1.1
future_X["treatment_response"] *= 0.9

future_risk_score = compute_risk(future_X)
future_labels = pd.Series(future_risk_score).apply(categorize_risk)
y_future_encoded = le_risk.transform(future_labels)

model_future = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model_future.fit(X_train, y_future_encoded)
print("\n🔹 Future 5-Year Risk Model Trained 🔹")

# -----------------------------------------------------------
# 9️⃣ Treatment Recommendation Model
# -----------------------------------------------------------
if not conditions.empty and not careplans.empty:
    print("\n🔹 Training Treatment Recommendation Model 🔹")
    treat_df = conditions.merge(careplans, on="PATIENT", suffixes=("_cond", "_care"), how="inner")
    treat_df = treat_df[["PATIENT", "DESCRIPTION_cond", "DESCRIPTION_care"]].dropna()

    le_cond = LabelEncoder()
    le_treat = LabelEncoder()
    treat_df["condition_code"] = le_cond.fit_transform(treat_df["DESCRIPTION_cond"].astype(str).str.lower())
    treat_df["treatment_code"] = le_treat.fit_transform(treat_df["DESCRIPTION_care"].astype(str).str.lower())

    X_treat = treat_df[["condition_code"]]
    y_treat = treat_df["treatment_code"]

    treatment_model = RandomForestClassifier(n_estimators=200, random_state=42)
    treatment_model.fit(X_treat, y_treat)

    with open(os.path.join(OUTPUT_DIR, "treatment_model.pkl"), "wb") as f:
        pickle.dump({
            "model": treatment_model,
            "condition_encoder": le_cond,
            "treatment_encoder": le_treat
        }, f)
    print("✅ Treatment Recommendation Model Trained Successfully")
else:
    print("⚠️ Skipping treatment model (missing conditions/careplans)")

# -----------------------------------------------------------
# 🔟 Save Models & Processed Data
# -----------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "risk_model_current.pkl"), "wb") as f:
    pickle.dump({"model": model_current, "gender_encoder": le_gender, "risk_encoder": le_risk}, f)

with open(os.path.join(OUTPUT_DIR, "risk_model_future.pkl"), "wb") as f:
    pickle.dump({"model": model_future, "gender_encoder": le_gender, "risk_encoder": le_risk}, f)

base.to_csv(os.path.join(OUTPUT_DIR, "processed_patient_risk.csv"), index=False)
print(f"\n✅ All models and processed data saved successfully in: {OUTPUT_DIR}")

# train_model.py
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
DATA_DIR = os.path.join(os.getcwd(), "merged_dataset")
OUTPUT_DIR = os.path.join(os.getcwd(), "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
procedures = pd.read_csv(os.path.join(DATA_DIR, "procedures.csv"))
careplans = pd.read_csv(os.path.join(DATA_DIR, "careplans.csv"))
observations = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"))

# -----------------------------------------------------------
# BASIC PATIENT INFO
# -----------------------------------------------------------
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365
base = patients[["Id", "FIRST", "LAST", "GENDER", "AGE"]].rename(columns={"Id": "PATIENT"})

# -----------------------------------------------------------
# RECENT CONDITION
# -----------------------------------------------------------
conditions["START"] = pd.to_datetime(conditions["START"], errors="coerce")
recent_conditions = conditions.sort_values("START").groupby("PATIENT").tail(1)
base = base.merge(recent_conditions[["PATIENT", "DESCRIPTION"]], on="PATIENT", how="left")
base.rename(columns={"DESCRIPTION": "recent_condition"}, inplace=True)

# -----------------------------------------------------------
# TREATMENT RESPONSE (careplan)
# -----------------------------------------------------------
careplans["DESCRIPTION"] = careplans["DESCRIPTION"].str.lower().fillna("")
response_map = {"improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worsen": -1}
careplans["response_score"] = careplans["DESCRIPTION"].apply(
    lambda x: sum([v for k, v in response_map.items() if k in x])
)
response_score = careplans.groupby("PATIENT")["response_score"].mean()
base["treatment_response"] = base["PATIENT"].map(response_score).fillna(0)

# -----------------------------------------------------------
# RECENT PROCEDURE SEVERITY
# -----------------------------------------------------------
procedure_date_col = next((c for c in ["DATE", "START", "PERFORMED"] if c in procedures.columns), None)
if procedure_date_col:
    procedures[procedure_date_col] = pd.to_datetime(procedures[procedure_date_col], errors="coerce")
else:
    procedures["temp"] = range(len(procedures))
    procedure_date_col = "temp"

recent_proc = procedures.sort_values(procedure_date_col).groupby("PATIENT").tail(1)
recent_proc["DESCRIPTION"] = recent_proc["DESCRIPTION"].str.lower().fillna("")

severity_keywords = {
    "surgery": 3, "transplant": 4, "therapy": 2, "checkup": 1, "screening": 1
}
recent_proc["severity"] = recent_proc["DESCRIPTION"].apply(
    lambda x: max([v for k, v in severity_keywords.items() if k in x], default=0)
)

base["procedure_severity"] = base["PATIENT"].map(
    recent_proc.set_index("PATIENT")["severity"]
).fillna(0)

# -----------------------------------------------------------
# RECENT MEDICATION INTENSITY
# -----------------------------------------------------------
medications["START"] = pd.to_datetime(medications["START"], errors="coerce")
recent_meds = medications.sort_values("START").groupby("PATIENT").tail(1)
recent_meds["DESCRIPTION"] = recent_meds["DESCRIPTION"].str.lower().fillna("")

def med_score(desc):
    if any(k in desc for k in ["insulin", "chemo", "steroid"]):
        return 3
    elif any(k in desc for k in ["antibiotic", "hypertensive"]):
        return 2
    elif any(k in desc for k in ["vitamin", "supplement"]):
        return 1
    return 0

recent_meds["med_score"] = recent_meds["DESCRIPTION"].apply(med_score)
base["medication_intensity"] = base["PATIENT"].map(
    recent_meds.set_index("PATIENT")["med_score"]
).fillna(0)

# -----------------------------------------------------------
# OBSERVATION VALUE (Latest) - FIXED
# -----------------------------------------------------------
if "DATE" in observations.columns:
    observations["DATE"] = pd.to_datetime(observations["DATE"], errors="coerce")
    recent_obs = observations.sort_values("DATE").groupby("PATIENT").tail(1)

    if "VALUE" in observations.columns:
        # Clean VALUE column (extract numeric part only)
        def extract_numeric(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return val
            # Remove non-numeric characters (like %, mmHg, etc.)
            val = ''.join(ch for ch in str(val) if (ch.isdigit() or ch == '.'))
            try:
                return float(val)
            except:
                return np.nan

        recent_obs["VALUE"] = recent_obs["VALUE"].apply(extract_numeric)
        base["avg_observation"] = base["PATIENT"].map(
            recent_obs.set_index("PATIENT")["VALUE"]
        ).fillna(0)
    else:
        base["avg_observation"] = 0
else:
    base["avg_observation"] = 0

# -----------------------------------------------------------
# COMPUTE RISK SCORE
# -----------------------------------------------------------
def compute_risk(df):
    return (
        df["procedure_severity"] * 1.2 +
        df["medication_intensity"] * 1.0 +
        df["treatment_response"] * -0.8 +
        np.maximum(0, df["AGE"] - 60) * 0.05 +
        (df["avg_observation"] / 100)
    )

base["risk_score"] = compute_risk(base)

def risk_level(x):
    if x < 2.5: return "Low"
    elif x < 4.5: return "Medium"
    return "High"

base["RISK_LEVEL"] = base["risk_score"].apply(risk_level)

# -----------------------------------------------------------
# RISK MODEL TRAINING (CURRENT)
# -----------------------------------------------------------
le_gender = LabelEncoder()
base["GENDER"] = le_gender.fit_transform(base["GENDER"].astype(str))

X = base[["AGE", "GENDER", "procedure_severity", "medication_intensity", "treatment_response", "avg_observation"]]
le_risk = LabelEncoder()
y = le_risk.fit_transform(base["RISK_LEVEL"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=300, random_state=42)
risk_model.fit(X_train, y_train)

print("\n✅ Current Risk Model Report:")
print(classification_report(y_test, risk_model.predict(X_test), target_names=le_risk.classes_))


# -----------------------------------------------------------
# FUTURE (5-YEAR) RISK PREDICTION MODEL (ML-BASED)
# -----------------------------------------------------------

# Step 1: Use existing base data
future_data = base.copy()

# Step 2: Create a synthetic future risk label
# This simulates gradual change based on condition & severity
np.random.seed(42)
future_data["future_risk"] = (
    base["risk_score"]
    + (base["procedure_severity"] * np.random.uniform(0.05, 0.15, len(base)))
    + (base["medication_intensity"] * np.random.uniform(0.03, 0.10, len(base)))
    - (base["treatment_response"] * np.random.uniform(0.02, 0.08, len(base)))
)
future_data["future_risk"] = np.clip(future_data["future_risk"], 0, 1)  # keep between 0-1

# Step 3: Select features for model training
X_future = base[["AGE", "procedure_severity", "medication_intensity", "treatment_response"]]
y_future = future_data["future_risk"]

# Step 4: Train regression model
future_risk_model = RandomForestRegressor(n_estimators=200, random_state=42)
future_risk_model.fit(X_future, y_future)

# Step 5: Evaluate
y_pred = future_risk_model.predict(X_future)
mse = mean_squared_error(y_future, y_pred)
print(f"\n✅ Future (5-Year) Risk Model Trained — MSE: {mse:.4f}")

# Step 6: Predict future risks
base["predicted_5yr_risk"] = future_risk_model.predict(X_future)
base["risk_trend_change"] = base["predicted_5yr_risk"] - base["risk_score"]
base["RISK_LEVEL_5YR"] = base["predicted_5yr_risk"].apply(risk_level)

# -----------------------------------------------------------
# SAVE THE MODEL
# -----------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "future_risk_model.pkl"), "wb") as f:
    pickle.dump(future_risk_model, f)

# Save risk trend for visualization
trend_json = base[[
    "PATIENT", "recent_condition", "risk_score", "predicted_5yr_risk", 
    "risk_trend_change", "RISK_LEVEL", "RISK_LEVEL_5YR"
]]
trend_json.to_json(os.path.join(OUTPUT_DIR, "risk_trend.json"), orient="records")

print("\n📊 ML-Based 5-Year Risk Trend Model and Data Saved Successfully.")
# -----------------------------------------------------------
# TREATMENT RECOMMENDATION MODEL
# -----------------------------------------------------------
treat_df = conditions.merge(careplans, on="PATIENT", suffixes=("_cond", "_care"), how="inner")
treat_df = treat_df[["DESCRIPTION_cond", "DESCRIPTION_care"]].dropna()
le_cond = LabelEncoder()
le_treat = LabelEncoder()
treat_df["cond_code"] = le_cond.fit_transform(treat_df["DESCRIPTION_cond"].astype(str).str.lower())
treat_df["treat_code"] = le_treat.fit_transform(treat_df["DESCRIPTION_care"].astype(str).str.lower())

treat_model = RandomForestClassifier(
    n_estimators=20,   # fewer trees = faster
    max_depth=10,      # limit tree depth
    n_jobs=-1,         # use all CPU cores
    random_state=42
)
treat_model.fit(treat_df[["cond_code"]], treat_df["treat_code"])


print("\n✅ Treatment Recommendation Model Trained Successfully.")

# -----------------------------------------------------------
# SAVE MODELS & DATA
# -----------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "risk_model.pkl"), "wb") as f:
    pickle.dump({
        "model": risk_model,
        "gender_encoder": le_gender,
        "risk_encoder": le_risk
    }, f)

with open(os.path.join(OUTPUT_DIR, "treatment_model.pkl"), "wb") as f:
    pickle.dump({
        "model": treat_model,
        "cond_encoder": le_cond,
        "treat_encoder": le_treat
    }, f)

# Export risk trend for visualization
risk_graph = base[["PATIENT", "recent_condition", "risk_score", "risk_5yr", "risk_trend_change", "RISK_LEVEL", "RISK_LEVEL_5YR"]]
risk_graph.to_json(os.path.join(OUTPUT_DIR, "risk_trend.json"), orient="records")

print("\n📊 Risk Trend and Treatment Models Ready!")
print(f"Saved models and risk trend to: {OUTPUT_DIR}")'''

# train_model.py
"""
Full training pipeline for:
- Current risk classification (RandomForestClassifier)
- 5-year future risk prediction (RandomForestRegressor), trained only on real 5-year-forward data when available
- Treatment recommendation (condition -> careplan) (RandomForestClassifier)
- Risk factor contributions (SHAP if available)
- Patient clustering + simple lifestyle suggestions per cluster
Outputs saved under model_outputs/
"""

#!/usr/bin/env python3
"""
Robust patient risk pipeline:
- loads CSVs from merged_dataset/
- builds per-patient "current" features
- trains a RandomForest classifier for current risk labels (Low/Medium/High)
- attempts to train a 5-year regressor (or a 1-year fallback)
- trains a simple treatment recommender (condition -> careplan)
- computes clustering templates and exports frontend JSON artifacts
"""
# train_model.py
# train_model.py
''''
import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

# Optional libraries with fallbacks
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# -------------------------------------------------
# Configuration
# -------------------------------------------------
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "merged_dataset")
OUTPUT_DIR = os.path.join(ROOT, "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def safe_read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {name} — continuing with empty DataFrame.")
        return pd.DataFrame()
    print(f"📂 Loading {name} ...")
    return pd.read_csv(path)

def extract_numeric(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    text = str(v)
    filtered = ''.join(ch for ch in text if (ch.isdigit() or ch == '.' or ch == '-'))
    try:
        return float(filtered) if filtered not in ("", ".", "-") else np.nan
    except:
        return np.nan

def safe_dt_parse(s):
    try:
        return pd.to_datetime(s)
    except:
        return pd.NaT

# Normalizer for output messages
def info(msg):
    print(f"\nℹ️  {msg}")

# -------------------------------------------------
# Load raw data
# -------------------------------------------------
patients = safe_read_csv("patients.csv")
conditions = safe_read_csv("conditions.csv")
procedures = safe_read_csv("procedures.csv")
medications = safe_read_csv("medications.csv")
careplans = safe_read_csv("careplans.csv")
observations = safe_read_csv("observations.csv")

if patients.empty:
    raise RuntimeError("patients.csv is required. Place it in merged_dataset/ and try again.")

# normalize patient id column
pid_col = next((c for c in ["Id", "ID", "PATIENT", "Patient", "patient"] if c in patients.columns), None)
if pid_col is None:
    raise RuntimeError("No patient id column found in patients.csv")
patients = patients.rename(columns={pid_col: "PATIENT"})
patients["PATIENT"] = patients["PATIENT"].astype(str)

# AGE calculation (safely)
patients["BIRTHDATE"] = pd.to_datetime(patients.get("BIRTHDATE"), errors="coerce")
patients["AGE"] = ((pd.Timestamp.now() - patients["BIRTHDATE"]).dt.days // 365).fillna(0).astype(int)

# -------------------------------------------------
# Build base feature table (one row per patient)
# -------------------------------------------------
info("Building patient-level feature table ...")
base = patients[["PATIENT", "AGE"]].copy()
base["PATIENT"] = base["PATIENT"].astype(str)

def feature_score(desc, mapping):
    if not isinstance(desc, str): return 0
    ds = desc.lower()
    for k, v in mapping.items():
        if k in ds: return v
    return 0

# Procedure severity (last procedure description -> mapped score)
if not procedures.empty and "DESCRIPTION" in procedures.columns:
    procedures["PATIENT"] = procedures["PATIENT"].astype(str)
    last_proc = procedures.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in procedures.columns] if any(c in procedures.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index()
    base = base.merge(last_proc, on="PATIENT", how="left")
    base["procedure_severity"] = base["DESCRIPTION"].apply(lambda x: feature_score(x, {"transplant": 4, "surgery": 3, "therapy": 2, "screening": 1}))
else:
    base["procedure_severity"] = 0

# Medication intensity (last medication description -> mapped score)
if not medications.empty and "DESCRIPTION" in medications.columns:
    medications["PATIENT"] = medications["PATIENT"].astype(str)
    last_med = medications.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in medications.columns] if any(c in medications.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"DESCRIPTION_med"})
    base = base.merge(last_med, on="PATIENT", how="left")
    base["medication_intensity"] = base["DESCRIPTION_med"].apply(lambda x: feature_score(x, {
        "insulin": 3, "chemo": 3, "steroid": 3,
        "antibiotic": 2, "antihypertensive": 2,
        "vitamin": 1, "supplement": 1
    }))
else:
    base["medication_intensity"] = 0

# Treatment response (last careplan description -> mapped score)
if not careplans.empty and "DESCRIPTION" in careplans.columns:
    careplans["PATIENT"] = careplans["PATIENT"].astype(str)
    last_care = careplans.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in careplans.columns] if any(c in careplans.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"DESCRIPTION_care"})
    base = base.merge(last_care, on="PATIENT", how="left")
    base["treatment_response"] = base["DESCRIPTION_care"].apply(lambda x: feature_score(x, {
        "improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worse": -1
    }))
else:
    base["treatment_response"] = 0

# observation_value - try to parse most recent numeric observation value
if not observations.empty and "VALUE" in observations.columns:
    observations["PATIENT"] = observations["PATIENT"].astype(str)
    # Try to use date to get recent observation
    date_col = next((c for c in ("DATE", "date", "OBSERVATION_DATE", "timestamp", "time", "EVENT_TIME") if c in observations.columns), None)
    if date_col:
        observations[date_col] = pd.to_datetime(observations[date_col], errors="coerce")
        recent_obs = observations.sort_values(by=date_col).groupby("PATIENT").last().reset_index()
    else:
        recent_obs = observations.groupby("PATIENT").last().reset_index()
    recent_obs = recent_obs[["PATIENT", "VALUE"]]
    base = base.merge(recent_obs, on="PATIENT", how="left")
    base["observation_value"] = base["VALUE"].apply(extract_numeric).fillna(0)
else:
    base["observation_value"] = 0

# Ensure numeric columns exist
for col in ["procedure_severity", "medication_intensity", "treatment_response", "observation_value"]:
    if col not in base.columns:
        base[col] = 0

# -------------------------------------------------
# Compute current risk score and risk level (labels)
# -------------------------------------------------
info("Computing current risk scores ...")
# Use a slightly more cautious AGE effect (only age above 60 contributes)
base["age_over_60"] = np.maximum(base["AGE"] - 60, 0)

base["risk_score"] = (
    base["procedure_severity"] * 1.2 +
    base["medication_intensity"] * 1.0 -
    0.8 * base["treatment_response"] +
    base["age_over_60"] * 0.03 +
    base["observation_value"] / 100.0
)

# Create categorical risk level based on tertiles (data-driven)
q_low, q_high = base["risk_score"].quantile([0.33, 0.66])
base["RISK_LEVEL"] = base["risk_score"].apply(lambda s: "Low" if s <= q_low else ("Medium" if s <= q_high else "High"))

# Save base snapshot for reference
base.to_csv(os.path.join(OUTPUT_DIR, "patient_base_features.csv"), index=False)

# -------------------------------------------------
# 🔹 Model 1: Current Risk Classifier (improved)
# -------------------------------------------------
info("Training current risk classifier ...")

features = ["AGE", "procedure_severity", "medication_intensity", "treatment_response", "observation_value", "age_over_60"]
X = base[features].fillna(0)
y = base["RISK_LEVEL"].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature columns
with open(os.path.join(OUTPUT_DIR, "risk_feature_info.pkl"), "wb") as f:
    pickle.dump({"feature_cols": features, "scaler": scaler, "label_encoder": le}, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, stratify=y_encoded, test_size=0.25, random_state=RANDOM_STATE)

# Choose classifier (prefer XGBoost if available)
if HAS_XGBOOST:
    clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)
else:
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✅ Risk Model Accuracy: {acc:.4f}")
print(f"✅ Risk Model F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save risk model
with open(os.path.join(OUTPUT_DIR, "current_risk_classifier.pkl"), "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le, "feature_cols": features, "scaler": scaler}, f)

# -------------------------------------------------
# 🔹 Future 5-Year Risk Regressor
#    - Prefer real longitudinal slope (if available).
#    - If not available, simulate a plausible distribution using residuals from a regressor.
# -------------------------------------------------
info("Preparing target for 5-year future risk projection ...")

# Attempt 1: If observations has time series per patient, compute numeric slope of observations and derive a percent yearly slope
has_longitudinal_obs = False
slope_df = pd.DataFrame({"PATIENT": base["PATIENT"].astype(str), "obs_slope_per_year": 0.0})

if not observations.empty and "VALUE" in observations.columns:
    # Ensure date column present
    date_col = next((c for c in ("DATE", "date", "OBSERVATION_DATE", "timestamp", "time", "EVENT_TIME") if c in observations.columns), None)
    if date_col:
        observations[date_col] = pd.to_datetime(observations[date_col], errors="coerce")
        # Keep only numeric values
        observations["VALUE_NUM"] = observations["VALUE"].apply(extract_numeric)
        # For patients with at least 2 numeric observations with dates, estimate slope (value change per year)
        slopes = []
        for pid, grp in observations.groupby("PATIENT"):
            grp2 = grp.dropna(subset=["VALUE_NUM", date_col]).sort_values(by=date_col)
            if grp2.shape[0] >= 2:
                # convert dates to ordinal for linear regression
                x = grp2[date_col].map(datetime.toordinal).values.reshape(-1, 1)
                yv = grp2["VALUE_NUM"].values
                # simple linear regression slope per day -> per year
                lr = LinearRegression()
                try:
                    lr.fit(x, yv)
                    slope_per_day = float(lr.coef_[0])
                    slope_per_year = slope_per_day * 365.25
                    slopes.append((str(pid), slope_per_year))
                except Exception:
                    # in case of numerical issues, skip
                    continue
        if slopes:
            slopes_df = pd.DataFrame(slopes, columns=["PATIENT", "obs_slope_per_year"])
            slope_df = slope_df.merge(slopes_df, on="PATIENT", how="left").fillna({"obs_slope_per_year": 0.0})
            has_longitudinal_obs = True
            info(f"Found longitudinal observation trends for {len(slopes)} patients — using these slopes for projections.")
    else:
        info("Observations exist but no date column found; cannot compute longitudinal slopes.")
else:
    info("No observations with numeric VALUE found for slope computation.")

# Merge slopes into base
base = base.merge(slope_df, on="PATIENT", how="left")

# -----------------------------
# SAFETY PATCH: ensure obs_slope_per_year column exists
# -----------------------------
if "obs_slope_per_year" not in base.columns:
    print("⚠️  No longitudinal observation trend data found — setting slope = 0 for all patients.")
    base["obs_slope_per_year"] = 0.0

base["obs_slope_per_year"] = base["obs_slope_per_year"].fillna(0.0)

# If we have slopes, create future target deterministically:
if has_longitudinal_obs:
    # Convert observation slope into expected change percent of risk per year.
    # We'll map the observation slope magnitude into a percentage change range using a robust scaler-like transform.
    # This is heuristic but grounded in observed per-patient trend.
    info("Constructing future_risk_score using observation slopes ...")
    # Normalize slopes to median absolute dev to avoid outliers dominating
    mad = np.median(np.abs(base["obs_slope_per_year"] - np.median(base["obs_slope_per_year"]))) or 1.0
    # percent change per year = slope_normalized * scaling_factor
    slope_norm = base["obs_slope_per_year"] / mad
    # scaling_factor maps slope_norm to a plausible percent change (this is heuristic)
    scaling_factor = 0.02  # 2% per normalized slope unit per year (conservative)
    percent_change_per_year = slope_norm * scaling_factor
    # limit extreme values to reasonable bounds
    percent_change_per_year = np.clip(percent_change_per_year, -0.25, 0.25)
    # 5-year multiplier
    multiplier_5yr = (1 + percent_change_per_year) ** 5
    base["future_risk_score"] = base["risk_score"] * multiplier_5yr
    info("Future risk target constructed from slopes. Note: this uses observation trends and is only as good as the input time-series.")
else:
    # Fallback: no longitudinal labels — build a regressor to estimate risk_score and then simulate 5-year scenarios
    info("No longitudinal observation slopes available — building a data-driven simulation for 5-year projection (NOT a true observed future label).")
    # Train a regressor to predict current risk from features (this learns mapping X -> risk_score)
    Xf = base[features].fillna(0)
    yf = base["risk_score"].fillna(base["risk_score"].mean())
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, random_state=RANDOM_STATE)

    # scale
    scaler_f = StandardScaler()
    Xf_train_s = scaler_f.fit_transform(Xf_train)
    Xf_test_s = scaler_f.transform(Xf_test)

    reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)
    try:
        reg.fit(Xf_train_s, yf_train)
    except Exception as e:
        warnings.warn(f"Regressor training failed with: {e}. Falling back to simple identity predictor.")
        reg = None

    if reg is not None:
        pred_test = reg.predict(Xf_test_s)
        print(f"Regressor R² on holdout: {r2_score(yf_test, pred_test):.4f}, RMSE: {np.sqrt(mean_squared_error(yf_test, pred_test)):.4f}")
        # residuals capture unexplained variation
        residuals = (yf_test - pred_test).values if hasattr(yf_test, "values") else np.array(yf_test - pred_test)
        resid_std = residuals.std() if residuals.size > 0 else 0.0
        # For each patient, simulate a 5-year path using year-on-year percent change equal to reg-pred residual ratio:
        X_all_s = scaler_f.transform(Xf.fillna(0))
        base_pred = reg.predict(X_all_s)
        # Yearly percent change simulated from residuals: delta / predicted -> percents (bounded)
        eps = 1e-6
        percents = []
        rng = np.random.default_rng(RANDOM_STATE)
        for i, pid in enumerate(base["PATIENT"]):
            # sample 100 scenarios to get expected 5-year factor
            samples = []
            for _ in range(100):
                # sample a residual noise
                noise = rng.normal(loc=0.0, scale=resid_std if resid_std > 0 else 0.01)
                # simulated next-year predicted = base_pred + noise
                next_risk = base_pred[i] + noise
                # percent change estimated
                pct = (next_risk - base_pred[i]) / (abs(base_pred[i]) + eps)
                samples.append(pct)
            # expected annual pct = mean of samples (bounded)
            yearly_pct = np.clip(np.mean(samples), -0.2, 0.2)  # limit to +/-20% per year to avoid extreme drift
            percents.append(yearly_pct)
        percents = np.array(percents)
        # 5-year multiplier
        multiplier_5yr = (1 + percents) ** 5
        base["future_risk_score"] = base["risk_score"] * multiplier_5yr
        # Save reg/scaler for transparency
        with open(os.path.join(OUTPUT_DIR, "future_regressor_sim_info.pkl"), "wb") as f:
            pickle.dump({"regressor": reg, "scaler": scaler_f, "residual_std": float(resid_std)}, f)
        info("Constructed simulated future_risk_score using learned regressor + residual sampling. This produces both increases and decreases.")
    else:
        # last fallback: small symmetric noise multiplier (conservative)
        info("Regressor unavailable — applying small symmetric noise multipliers (conservative).")
        rng = np.random.default_rng(RANDOM_STATE)
        yearly_pct = rng.normal(0.0, 0.02, size=len(base))  # mean 0, sd 2% per year
        multiplier_5yr = (1 + yearly_pct) ** 5
        base["future_risk_score"] = base["risk_score"] * multiplier_5yr
        info("Warning: This fallback is only used when no model was trainable. Collect longitudinal labels for improved projections.")

# basic sanity: ensure no NaN
base["future_risk_score"] = base["future_risk_score"].fillna(base["risk_score"])

# Evaluate simple statistics (how many increased vs decreased)
increased = (base["future_risk_score"] > base["risk_score"]).sum()
decreased = (base["future_risk_score"] < base["risk_score"]).sum()
equal = (base["future_risk_score"] == base["risk_score"]).sum()
info(f"Projection summary: {increased} patients increased, {decreased} decreased, {equal} unchanged (out of {len(base)})")

# Save future snapshot for auditing
base[["PATIENT", "risk_score", "future_risk_score"]].to_csv(os.path.join(OUTPUT_DIR, "patient_future_projection.csv"), index=False)

# Train a regressor (future_model) to predict future_risk_score from current features (this helps downstream use)
info("Training a regressor to map current features -> future_risk_score (for serving).")

Xf_final = base[features].fillna(0)
yf_final = base["future_risk_score"].fillna(base["risk_score"].mean())

scaler_final = StandardScaler()
Xf_final_s = scaler_final.fit_transform(Xf_final)

if HAS_XGBOOST:
    future_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE)
else:
    future_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)

future_model.fit(Xf_final_s, yf_final)
yf_pred = future_model.predict(Xf_final_s)
print(f"Future regressor R² (train): {r2_score(yf_final, yf_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(yf_final, yf_pred)):.4f}")

# Save future model + scaler
with open(os.path.join(OUTPUT_DIR, "future_5yr_regressor.pkl"), "wb") as f:
    pickle.dump({"model": future_model, "scaler": scaler_final, "feature_cols": features}, f)

# -------------------------------------------------
# 🔹 Model 3: Treatment Recommender (Condition -> Careplan)
# Improved: uses text vectorization and class balancing
# -------------------------------------------------
info("Training treatment recommender ...")

if not conditions.empty and not careplans.empty and "DESCRIPTION" in conditions.columns and "DESCRIPTION" in careplans.columns:
    # Merge conditions and careplans at patient-level (we use last description for each patient)
    cond_last = conditions.sort_values(by=[c for c in ("DATE", "date", "START") if c in conditions.columns] if any(c in conditions.columns for c in ("DATE","date","START")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"cond_descr"})
    care_last = careplans.sort_values(by=[c for c in ("DATE", "date", "START") if c in careplans.columns] if any(c in careplans.columns for c in ("DATE","date","START")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"care_descr"})
    pairs = pd.merge(cond_last, care_last, on="PATIENT", how="inner").dropna()

    if pairs.empty:
        info("No matching patient-level condition->careplan pairs found. Skipping treatment recommender.")
    else:
        # Text vectorization: use TF-IDF with limited vocabulary (to keep models small)
        vect = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
        X_text = vect.fit_transform(pairs["cond_descr"].astype(str).str.lower())

        # Encode targets (care plan descriptions)
        le_treat = LabelEncoder()
        y_t = le_treat.fit_transform(pairs["care_descr"].astype(str).str.lower())

        # If classes are imbalanced, apply SMOTE on numeric representation (TF-IDF -> dense may be large)
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_text, y_t, test_size=0.2, random_state=RANDOM_STATE, stratify=y_t)

        # Convert sparse to dense for SMOTE if necessary (be cautious on memory). If many classes, SMOTE may fail; fallback to class_weight
        use_smote = HAS_SMOTE and (X_train_t.shape[0] < 20000) and (len(np.unique(y_t)) <= 10)
        if use_smote:
            try:
                X_train_dense = X_train_t.toarray()
                sm = SMOTE(random_state=RANDOM_STATE)
                X_res, y_res = sm.fit_resample(X_train_dense, y_train_t)
                X_train_final = X_res
                y_train_final = y_res
                info(f"Applied SMOTE: new training size {X_train_final.shape[0]}")
            except Exception as e:
                warnings.warn(f"SMOTE failed: {e}. Falling back to no-oversample.")
                X_train_final = X_train_t
                y_train_final = y_train_t
        else:
            X_train_final = X_train_t
            y_train_final = y_train_t

        # Choose classifier
        if HAS_XGBOOST:
            clf_t = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)
        else:
            clf_t = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

        # Fit (handle sparse/dense input differences)
        try:
            clf_t.fit(X_train_final, y_train_final)
        except Exception:
            # If classifier doesn't accept sparse or dense shapes, convert appropriately
            clf_t.fit(X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final, y_train_final)

        # Predict + evaluate
        try:
            ytp = clf_t.predict(X_test_t)
        except Exception:
            ytp = clf_t.predict(X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t)

        treat_acc = accuracy_score(y_test_t, ytp)
        print(f"✅ Treatment Model Accuracy: {treat_acc:.4f}")
        try:
            print("\nClassification Report (treatment recommender):\n", classification_report(y_test_t, ytp, target_names=le_treat.classes_[:len(np.unique(y_t))]))
        except Exception:
            # classification_report may error if labels mismatched, so skip gracefully
            pass

        # Save treatment model + vectorizer + label encoder
        treatment_model = {"model": clf_t, "vect": vect, "treat_encoder": le_treat}
        with open(os.path.join(OUTPUT_DIR, "treatment_recommender.pkl"), "wb") as f:
            pickle.dump(treatment_model, f)
else:
    info("Missing condition or careplan data — skipping treatment recommender.")

info("\n🎯 All models and artifacts saved to model_outputs/.")
info("Important notes:")
print("""
- If you have longitudinal (follow-up) labels for actual future risk, retrain the Future regressor with those observed future values — this will greatly improve accuracy.
- When no longitudinal labels exist, the script simulates plausible directions using observation slopes (if present) or residual-based sampling. Simulations are conservative (clamped yearly changes).
- Collect and feed real follow-up measurements for best results.
- You can tune hyperparameters (GridSearchCV) further if you have sufficient data.
""")





# -------------------------
# Feature importances & (optional) SHAP
# -------------------------
feature_importances = dict(zip(feature_cols, clf.feature_importances_.round(4).tolist()))
fi_path = os.path.join(OUTPUT_DIR, "feature_importances.json")
with open(fi_path, "w") as f:
    json.dump(feature_importances, f, indent=2)
print(f"Saved feature importances -> {fi_path}")

if SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(clf)
        X_all = X.astype(float)
        shap_vals = explainer.shap_values(X_all)
        if isinstance(shap_vals, list):
            shap_summary = {col: float(np.mean(np.abs(shap_vals[0])[:, i])) for i, col in enumerate(feature_cols)}
        else:
            shap_summary = {col: float(np.mean(np.abs(shap_vals[:, i]))) for i, col in enumerate(feature_cols)}
        with open(os.path.join(OUTPUT_DIR, "shap_explanations.json"), "w") as f:
            json.dump({"shap_summary": shap_summary}, f, indent=2)
        print("Saved SHAP summary.")
    except Exception as e:
        print("⚠️ SHAP explanation failed:", str(e))
else:
    print("⚠️ shap not installed — skipping SHAP. To enable: pip install shap")

# -------------------------
# Clustering
# -------------------------
cluster_output = os.path.join(OUTPUT_DIR, "patient_clusters.json")
try:
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(base[feature_cols].fillna(0).astype(float))
    n_patients = max(1, len(base))
    k = min(6, max(2, int(np.sqrt(n_patients))))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = km.fit_predict(X_feat)
    base["cluster"] = clusters
    cluster_templates = {}
    for c in range(k):
        avg = base.loc[base["cluster"] == c, feature_cols].mean()
        tips = []
        if avg["AGE"] > 60:
            tips.append("Focus on fall prevention, medication review, bone health.")
        if avg["medication_intensity"] >= 2:
            tips.append("Schedule medication adherence review and check for polypharmacy risks.")
        if avg["procedure_severity"] >= 3:
            tips.append("Ensure post-op rehab & close specialist follow-up.")
        if avg["observation_value"] > 140:
            tips.append("Monitor relevant vitals (e.g. blood pressure) and consult primary care.")
        if not tips:
            tips = ["Maintain healthy diet, regular exercise, annual checkups."]
        cluster_templates[c] = {"avg_features": avg.to_dict(), "recommendations": tips}
    with open(cluster_output, "w") as f:
        json.dump({"cluster_templates": cluster_templates}, f, indent=2, default=float)
    print(f"Patient clustering done (k={k}). Templates -> {cluster_output}")
except Exception as ex:
    print("⚠️ Clustering failed:", str(ex))

# -------------------------
# Frontend artifacts
# -------------------------
print("Preparing frontend JSON artifacts...")
frontend_rows = []
for _, row in base.iterrows():
    pid = str(row["PATIENT"])
    feat = {c: float(row[c]) for c in feature_cols}
    x_vec = np.array([feat[c] for c in feature_cols]).reshape(1, -1)
    cur_label = le_risk.inverse_transform(clf.predict(x_vec))[0]
    cur_prob = float(clf.predict_proba(x_vec).max()) if hasattr(clf, "predict_proba") else None

    predicted_5yr = None
    if future_model is not None:
        try:
            predicted_5yr = float(future_model.predict(np.array([feat[c] for c in feature_cols]).reshape(1, -1))[0])
        except Exception:
            predicted_5yr = None

    treatment_suggestion = None
    if treatment_model is not None and row.get("recent_condition"):
        cond_text = str(row["recent_condition"]).strip().lower()
        try:
            cond_code = treatment_model["cond_encoder"].transform([cond_text])[0]
            pred_tcode = treatment_model["model"].predict(np.array([[cond_code]]))[0]
            treatment_suggestion = treatment_model["treat_encoder"].inverse_transform([pred_tcode])[0]
        except Exception:
            treatment_suggestion = None

    contributions = {k: float(v) for k, v in feature_importances.items()}

    frontend_rows.append({
        "PATIENT": pid,
        "FIRST": row.get("FIRST"),
        "LAST": row.get("LAST"),
        "GENDER": row.get("GENDER"),
        "AGE": int(row.get("AGE") or 0),
        "recent_condition": row.get("recent_condition"),
        "recent_observation_desc": row.get("recent_observation_desc"),
        "recent_observation_value": float(row.get("observation_value") or 0),
        "current_risk_label": str(cur_label),
        "current_risk_score": float(row.get("risk_score")),
        "current_risk_confidence": cur_prob,
        "predicted_5yr_risk_score": predicted_5yr,
        "predicted_5yr_risk_level": (risk_label_from_score(predicted_5yr) if predicted_5yr is not None else None),
        "treatment_recommendation": treatment_suggestion,
        "risk_factor_contribution": contributions,
        "cluster": int(row.get("cluster") if "cluster" in row.index else -1)
    })

with open(os.path.join(OUTPUT_DIR, "patient_prediction_summary.json"), "w") as f:
    json.dump(frontend_rows, f, indent=2, default=float)

base_csv = os.path.join(OUTPUT_DIR, "processed_patient_features.csv")
base.to_csv(base_csv, index=False)
print(f"Saved processed patient features -> {base_csv}")
print(f"Saved frontend summary -> {os.path.join(OUTPUT_DIR, 'patient_prediction_summary.json')}")

with open(os.path.join(OUTPUT_DIR, "current_risk_model_bundle.pkl"), "wb") as f:
    pickle.dump({"clf": clf, "risk_label_encoder": le_risk, "feature_cols": feature_cols}, f)

if treatment_model is not None:
    with open(os.path.join(OUTPUT_DIR, "treatment_model_bundle.pkl"), "wb") as f:
        pickle.dump(treatment_model, f)

print("\n✅ All done. Outputs saved to:", OUTPUT_DIR)
print("Files of interest for frontend:")
for fn in ["patient_prediction_summary.json", "feature_importances.json", "patient_clusters.json", "current_risk_model_bundle.pkl"]:
    print(" -", os.path.join(OUTPUT_DIR, fn))
if future_model is not None:
    print(" - future_5yr_regressor.pkl (or fallback 1yr regressor)")
'''

# train_model.py
import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

# Optional libraries with fallbacks
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# -------------------------------------------------
# Configuration
# -------------------------------------------------
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "merged_dataset")
OUTPUT_DIR = os.path.join(ROOT, "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def safe_read_csv(name, usecols=None, chunksize=500000):
    """
    Safely reads large CSV files from DATA_DIR.
    Uses chunked loading to avoid MemoryError.
    Automatically concatenates chunks into one DataFrame.
    """
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {name} — continuing with empty DataFrame.")
        return pd.DataFrame()

    print(f"📂 Loading {name} ...")

    try:
        # Try normal read (fast path for small files)
        return pd.read_csv(path, usecols=usecols, low_memory=False)

    except MemoryError:
        # Handle large file by reading in chunks
        print(f"⚠️ MemoryError while reading {name}, using chunked loading...")
        chunks = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        print(f"✅ Loaded {name} in {len(chunks)} chunks ({len(df)} rows total)")
        return df

    except Exception as e:
        print(f"❌ Failed to read {name}: {e}")
        return pd.DataFrame()


def extract_numeric(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    text = str(v)
    filtered = ''.join(ch for ch in text if (ch.isdigit() or ch == '.' or ch == '-'))
    try:
        return float(filtered) if filtered not in ("", ".", "-") else np.nan
    except:
        return np.nan


def safe_dt_parse(s):
    try:
        return pd.to_datetime(s)
    except:
        return pd.NaT


def info(msg):
    """Clean logging for status messages"""
    print(f"\nℹ️  {msg}")


# -------------------------------------------------
# Load raw data safely
# -------------------------------------------------
patients = safe_read_csv("patients.csv")
conditions = safe_read_csv("conditions.csv")

# ✅ Handle large observations.csv safely
def load_large_csv(name, usecols=None, chunksize=100000):
    """Read a large CSV from DATA_DIR in chunks safely."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {name}")
        return pd.DataFrame()
    dfs = []
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            dfs.append(chunk)
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Successfully loaded {name} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"❌ Failed to read {name}: {e}")
        return pd.DataFrame()

# ✅ Use the chunk loader for heavy files
observations = load_large_csv("observations.csv", usecols=["PATIENT", "CODE", "DESCRIPTION", "VALUE"])
procedures = safe_read_csv("procedures.csv")
medications = safe_read_csv("medications.csv")
careplans = safe_read_csv("careplans.csv")


if patients.empty:
    raise RuntimeError("patients.csv is required. Place it in merged_dataset/ and try again.")

# normalize patient id column
pid_col = next((c for c in ["Id", "ID", "PATIENT", "Patient", "patient"] if c in patients.columns), None)
if pid_col is None:
    raise RuntimeError("No patient id column found in patients.csv")
patients = patients.rename(columns={pid_col: "PATIENT"})
patients["PATIENT"] = patients["PATIENT"].astype(str)

# AGE calculation (safely)
patients["BIRTHDATE"] = pd.to_datetime(patients.get("BIRTHDATE"), errors="coerce")
patients["AGE"] = ((pd.Timestamp.now() - patients["BIRTHDATE"]).dt.days // 365).fillna(0).astype(int)

# -------------------------------------------------
# Build base feature table (one row per patient)
# -------------------------------------------------
info("Building patient-level feature table ...")
base = patients[["PATIENT", "AGE"]].copy()
base["PATIENT"] = base["PATIENT"].astype(str)

def feature_score(desc, mapping):
    if not isinstance(desc, str): return 0
    ds = desc.lower()
    for k, v in mapping.items():
        if k in ds: return v
    return 0

# Procedure severity (last procedure description -> mapped score)
if not procedures.empty and "DESCRIPTION" in procedures.columns:
    procedures["PATIENT"] = procedures["PATIENT"].astype(str)
    last_proc = procedures.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in procedures.columns] if any(c in procedures.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index()
    base = base.merge(last_proc, on="PATIENT", how="left")
    base["procedure_severity"] = base["DESCRIPTION"].apply(lambda x: feature_score(x, {"transplant": 4, "surgery": 3, "therapy": 2, "screening": 1}))
else:
    base["procedure_severity"] = 0

# Medication intensity (last medication description -> mapped score)
if not medications.empty and "DESCRIPTION" in medications.columns:
    medications["PATIENT"] = medications["PATIENT"].astype(str)
    last_med = medications.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in medications.columns] if any(c in medications.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"DESCRIPTION_med"})
    base = base.merge(last_med, on="PATIENT", how="left")
    base["medication_intensity"] = base["DESCRIPTION_med"].apply(lambda x: feature_score(x, {
        "insulin": 3, "chemo": 3, "steroid": 3,
        "antibiotic": 2, "antihypertensive": 2,
        "vitamin": 1, "supplement": 1
    }))
else:
    base["medication_intensity"] = 0

# Treatment response (last careplan description -> mapped score)
if not careplans.empty and "DESCRIPTION" in careplans.columns:
    careplans["PATIENT"] = careplans["PATIENT"].astype(str)
    last_care = careplans.sort_values(by=[c for c in ("DATE", "date", "START", "start") if c in careplans.columns] if any(c in careplans.columns for c in ("DATE","date","START","start")) else []).groupby("PATIENT")["DESCRIPTION"].last().reset_index().rename(columns={"DESCRIPTION":"DESCRIPTION_care"})
    base = base.merge(last_care, on="PATIENT", how="left")
    base["treatment_response"] = base["DESCRIPTION_care"].apply(lambda x: feature_score(x, {
        "improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worse": -1
    }))
else:
    base["treatment_response"] = 0

# observation_value - try to parse most recent numeric observation value
if not observations.empty and "VALUE" in observations.columns:
    observations["PATIENT"] = observations["PATIENT"].astype(str)
    # Try to use date to get recent observation
    date_col = next((c for c in ("DATE", "date", "OBSERVATION_DATE", "timestamp", "time", "EVENT_TIME") if c in observations.columns), None)
    if date_col:
        observations[date_col] = pd.to_datetime(observations[date_col], errors="coerce")
        recent_obs = observations.sort_values(by=date_col).groupby("PATIENT").last().reset_index()
    else:
        recent_obs = observations.groupby("PATIENT").last().reset_index()
    recent_obs = recent_obs[["PATIENT", "VALUE"]]
    base = base.merge(recent_obs, on="PATIENT", how="left")
    base["observation_value"] = base["VALUE"].apply(extract_numeric).fillna(0)
else:
    base["observation_value"] = 0

# Ensure numeric columns exist
for col in ["procedure_severity", "medication_intensity", "treatment_response", "observation_value"]:
    if col not in base.columns:
        base[col] = 0

# -------------------------------------------------
# Compute current risk score and risk level (labels)
# -------------------------------------------------
info("Computing current risk scores ...")
# Use a slightly more cautious AGE effect (only age above 60 contributes)
base["age_over_60"] = np.maximum(base["AGE"] - 60, 0)

base["risk_score"] = (
    base["procedure_severity"] * 1.2 +
    base["medication_intensity"] * 1.0 -
    0.8 * base["treatment_response"] +
    base["age_over_60"] * 0.03 +
    base["observation_value"] / 100.0
)

# Create categorical risk level based on tertiles (data-driven)
q_low, q_high = base["risk_score"].quantile([0.33, 0.66])
base["RISK_LEVEL"] = base["risk_score"].apply(lambda s: "Low" if s <= q_low else ("Medium" if s <= q_high else "High"))

# Save base snapshot for reference
base.to_csv(os.path.join(OUTPUT_DIR, "patient_base_features.csv"), index=False)

# --------------------------------------------
# 🔹 Model 1: Current Risk Classifier (Improved Realistic Version)
# --------------------------------------------
info("Training current risk classifier (improved realistic version) ...")

# ✅ Additional meaningful features
base["procedure_count"] = procedures.groupby("PATIENT")["CODE"].transform("count")
base["medication_count"] = medications.groupby("PATIENT")["CODE"].transform("count")
base["careplan_count"] = careplans.groupby("PATIENT")["CODE"].transform("count")
# --------------------------------------------
# Clean and convert observation VALUE column
# --------------------------------------------
def extract_numeric_value(val):
    """Extract numeric part safely from mixed text (e.g., '140 mg/dL' → 140.0)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    text = str(val)
    # Keep only numbers, decimal points, and minus signs
    filtered = ''.join(ch for ch in text if (ch.isdigit() or ch in ".-"))
    try:
        return float(filtered) if filtered not in ("", ".", "-") else np.nan
    except:
        return np.nan

# Apply numeric extraction to VALUE column
if "VALUE" in observations.columns:
    
    observations["NUMERIC_VALUE"] = observations["VALUE"].apply(extract_numeric_value)
else:
    observations["NUMERIC_VALUE"] = np.nan

# Compute average numeric observation per patient
base["avg_observation_value"] = (
    observations.groupby("PATIENT")["NUMERIC_VALUE"].transform("mean")
)
base["avg_observation_value"] = base["avg_observation_value"].fillna(0)


# Replace NaN values with 0
for col in ["procedure_count", "medication_count", "careplan_count", "avg_observation_value"]:
    base[col] = base[col].fillna(0)

# ✅ Add engineered feature: age_over_60 (already present)
base["age_over_60"] = (base["AGE"] > 60).astype(int)

# 🔹 Final feature set
features = [
    "AGE",
    "procedure_severity",
    "medication_intensity",
    "treatment_response",
    "observation_value",
    "age_over_60",
    "procedure_count",
    "medication_count",
    "careplan_count",
    "avg_observation_value"
]

# Remove any infinite values
base = base.replace([np.inf, -np.inf], np.nan).fillna(0)

X = base[features].fillna(0)
y = base["RISK_LEVEL"].astype(str)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save feature info
with open(os.path.join(OUTPUT_DIR, "risk_feature_info.pkl"), "wb") as f:
    pickle.dump({"feature_cols": features, "scaler": scaler, "label_encoder": le}, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, stratify=y_encoded, test_size=0.25, random_state=RANDOM_STATE
)

# Handle imbalance with SMOTE if available
if HAS_SMOTE:
    print("✅ Applying SMOTE balancing ...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)
else:
    print("⚠️ SMOTE not available, continuing with original data.")

print("\n📊 Risk Level Distribution (Train set):")
print(pd.Series(y_train).value_counts())

# --------------------------------------------
# ✅ Improved XGBoost Classifier (less overfitting)
# --------------------------------------------
if HAS_XGBOOST:
    print("✅ Using tuned XGBoost Classifier ...")
    clf = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    print("✅ Using tuned Random Forest Classifier ...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# Train model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate model
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✅ Risk Model Accuracy: {acc:.4f}")
print(f"✅ Risk Model F1 Score: {f1:.4f}")

print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save risk model
with open(os.path.join(OUTPUT_DIR, "current_risk_classifier.pkl"), "wb") as f:
    pickle.dump({
        "model": clf,
        "label_encoder": le,
        "feature_cols": features,
        "scaler": scaler
    }, f)

print("💾 Improved Risk Classifier saved successfully!")

# -------------------------------------------------
# 🔹 Future 5-Year Risk Regressor
# -------------------------------------------------
info("Preparing target for 5-year future risk projection ...")

# Ensure features are defined and available
default_features = [
    "AGE", "procedure_severity", "medication_intensity", "treatment_response",
    "observation_value", "age_over_60", "procedure_count",
    "medication_count", "careplan_count", "avg_observation_value"
]
features = [f for f in default_features if f in base.columns]

# ✅ Ensure all required columns exist
for col in default_features:
    if col not in base.columns:
        base[col] = 0.0

# -------------------------------
# Compute observation slopes if available
# -------------------------------
has_longitudinal_obs = False
slope_df = pd.DataFrame({"PATIENT": base["PATIENT"].astype(str), "obs_slope_per_year": 0.0})

if not observations.empty and "VALUE" in observations.columns:
    date_col = next((c for c in ("DATE", "date", "OBSERVATION_DATE", "timestamp", "time", "EVENT_TIME") if c in observations.columns), None)
    if date_col:
        observations[date_col] = pd.to_datetime(observations[date_col], errors="coerce")
        observations["VALUE_NUM"] = observations["VALUE"].apply(extract_numeric)

        slopes = []
        for pid, grp in observations.groupby("PATIENT"):
            grp2 = grp.dropna(subset=["VALUE_NUM", date_col]).sort_values(by=date_col)
            if grp2.shape[0] >= 2:
                x = grp2[date_col].map(datetime.toordinal).values.reshape(-1, 1)
                yv = grp2["VALUE_NUM"].values
                lr = LinearRegression()
                try:
                    lr.fit(x, yv)
                    slope_per_year = float(lr.coef_[0]) * 365.25
                    slopes.append((str(pid), slope_per_year))
                except Exception:
                    continue
        if slopes:
            slopes_df = pd.DataFrame(slopes, columns=["PATIENT", "obs_slope_per_year"])
            slope_df = slope_df.merge(slopes_df, on="PATIENT", how="left").fillna({"obs_slope_per_year": 0.0})
            has_longitudinal_obs = True
            info(f"✅ Found longitudinal observation trends for {len(slopes)} patients.")
    else:
        info("⚠️ Observations exist but no date column found; cannot compute slopes.")
else:
    info("ℹ️ No observations with numeric VALUE found for slope computation.")

# Merge slopes into base
base = base.merge(slope_df, on="PATIENT", how="left")
base["obs_slope_per_year"] = base["obs_slope_per_year"].fillna(0.0)

# -------------------------------
# Construct target (future_risk_score)
# -------------------------------
if has_longitudinal_obs:
    info("Constructing future_risk_score using observation slopes ...")
    mad = np.median(np.abs(base["obs_slope_per_year"] - np.median(base["obs_slope_per_year"]))) or 1.0
    slope_norm = base["obs_slope_per_year"] / mad
    percent_change_per_year = np.clip(slope_norm * 0.02, -0.25, 0.25)
    multiplier_5yr = (1 + percent_change_per_year) ** 5
    base["future_risk_score"] = base["risk_score"] * multiplier_5yr
else:
    
    Xf = base[features].fillna(0)
    yf = base["risk_score"].fillna(base["risk_score"].mean())
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, random_state=RANDOM_STATE)

    scaler_f = StandardScaler()
    Xf_train_s = scaler_f.fit_transform(Xf_train)
    Xf_test_s = scaler_f.transform(Xf_test)

    reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)
    try:
        reg.fit(Xf_train_s, yf_train)
        pred_test = reg.predict(Xf_test_s)
        print(f"Regressor R² on holdout: {r2_score(yf_test, pred_test):.4f}, RMSE: {np.sqrt(mean_squared_error(yf_test, pred_test)):.4f}")
        residuals = (yf_test - pred_test).values
        resid_std = residuals.std() if residuals.size > 0 else 0.01

        X_all_s = scaler_f.transform(Xf.fillna(0))
        base_pred = reg.predict(X_all_s)
        eps = 1e-6
        rng = np.random.default_rng(RANDOM_STATE)

        percents = []
        for i in range(len(base_pred)):
            samples = []
            for _ in range(50):
                noise = rng.normal(0, resid_std)
                next_risk = base_pred[i] + noise
                pct = (next_risk - base_pred[i]) / (abs(base_pred[i]) + eps)
                samples.append(pct)
            yearly_pct = np.clip(np.mean(samples), -0.2, 0.2)
            percents.append(yearly_pct)

        percents = np.array(percents)
        multiplier_5yr = (1 + percents) ** 5
        base["future_risk_score"] = base["risk_score"] * multiplier_5yr

        with open(os.path.join(OUTPUT_DIR, "future_regressor_sim_info.pkl"), "wb") as f:
            pickle.dump({"regressor": reg, "scaler": scaler_f, "residual_std": float(resid_std)}, f)
    except Exception as e:
        info(f"❌ Regression fallback failed: {e}. Applying noise-based projection.")
        rng = np.random.default_rng(RANDOM_STATE)
        yearly_pct = rng.normal(0.0, 0.02, size=len(base))
        multiplier_5yr = (1 + yearly_pct) ** 5
        base["future_risk_score"] = base["risk_score"] * multiplier_5yr

# Clean target column
base["future_risk_score"] = base["future_risk_score"].fillna(base["risk_score"])

# Summary
increased = (base["future_risk_score"] > base["risk_score"]).sum()
decreased = (base["future_risk_score"] < base["risk_score"]).sum()
equal = (base["future_risk_score"] == base["risk_score"]).sum()
info(f"📊 Projection summary: {increased} ↑ | {decreased} ↓ | {equal} = (out of {len(base)})")

# Save projection audit
base[["PATIENT", "risk_score", "future_risk_score"]].to_csv(
    os.path.join(OUTPUT_DIR, "patient_future_projection.csv"), index=False
)

# -------------------------------
# Train Future Risk Regressor (for serving)
# -------------------------------
info("Training a serving regressor: current features → future_risk_score")

Xf_final = base[features].fillna(0)
yf_final = base["future_risk_score"].fillna(base["risk_score"].mean())

scaler_final = StandardScaler()
Xf_final_s = scaler_final.fit_transform(Xf_final)

if HAS_XGBOOST:
    future_model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE
    )
else:
    future_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
    )

future_model.fit(Xf_final_s, yf_final)
yf_pred = future_model.predict(Xf_final_s)
print(f"✅ Future regressor R² (train): {r2_score(yf_final, yf_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(yf_final, yf_pred)):.4f}")

# Save model for Flask use
with open(os.path.join(OUTPUT_DIR, "future_5yr_regressor.pkl"), "wb") as f:
    pickle.dump({"model": future_model, "scaler": scaler_final, "feature_cols": features}, f)

info("💾 future_5yr_regressor.pkl saved successfully.")

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import Counter

# -------------------------------
# Step 2: Validate Required Columns
# -------------------------------
required_cols = ["PATIENT", "DESCRIPTION", "START", "STOP"]
for col in required_cols:
    if col not in careplans.columns:
        raise ValueError(f"❌ Missing column: {col} in careplans.csv")

# -------------------------------
# Step 3: Create Extra Features
# -------------------------------
careplans["START"] = pd.to_datetime(careplans["START"], errors="coerce")
careplans["STOP"] = pd.to_datetime(careplans["STOP"], errors="coerce")
careplans["DURATION_DAYS"] = (careplans["STOP"] - careplans["START"]).dt.days.fillna(0)

# Aggregate per patient
agg_features = (
    careplans.groupby("PATIENT")
    .agg({
        "DURATION_DAYS": "mean",
        "DESCRIPTION": lambda x: " ".join(str(v) for v in x)
    })
    .reset_index()
)

# -------------------------------
# Step 4: Create Treatment Labels
# -------------------------------
if "TREATMENT" not in careplans.columns:
    
    careplans["TREATMENT"] = careplans["DESCRIPTION"].fillna("Unknown")

treatments = (
    careplans.groupby("PATIENT")["TREATMENT"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)

# Merge data
data = pd.merge(agg_features, treatments, on="PATIENT", how="inner")

# -------------------------------
# Step 5: Prepare Data for Model
# -------------------------------
X_text = data["DESCRIPTION"]
X_numeric = data[["DURATION_DAYS"]]
y = data["TREATMENT"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Filter classes with <2 samples
label_counts = Counter(y_encoded)
valid_idx = [i for i, val in enumerate(y_encoded) if label_counts[val] > 1]

X_text = X_text.iloc[valid_idx]
X_numeric = X_numeric.iloc[valid_idx]
y_encoded = y_encoded[valid_idx]

# -------------------------------
# Step 6: TF-IDF + Scaled Features
# -------------------------------
tfidf = TfidfVectorizer(max_features=800, stop_words="english")  # reduced feature size
X_text_tfidf = tfidf.fit_transform(X_text)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Combine numeric + text
from scipy.sparse import hstack
X_final = hstack([X_text_tfidf, X_scaled])

# -------------------------------
# Step 7: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# Step 8: Model Training (Random Forest + GridSearch)
# -------------------------------
print("\n🌲 Training Random Forest Recommender ...")

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,   # avoids memory/disk errors
    verbose=2
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# -------------------------------
# Step 9: Evaluate Accuracy
# -------------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
print(f"🏆 Best Parameters: {grid.best_params_}")


# -------------------------------
# Step 10: Save Model Files
# -------------------------------
os.makedirs("models", exist_ok=True)
with open("models/treatment_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ All models saved successfully in /models folder.")
