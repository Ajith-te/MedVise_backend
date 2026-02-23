import os
import random
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone


from database import db


ALLOWED_EXTENSIONS = {".csv"}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_collection(email, dataset_name):
    collection_name = email.replace("@", "_").replace(".", "_")
    collection = db[collection_name]

    data = list(collection.find({"dataset_name": dataset_name}, {"_id": 0}))

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def convert_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].apply(
                lambda x: x.tz_localize(None)
                if hasattr(x, "tzinfo") and x is not pd.NaT and x.tzinfo
                else x
            )
    return df

# Patient Preprocessing
def preprocess_patient_details(email):
    patients = load_collection(email, "patients.csv")
    if patients.empty:
        return []

    patients = convert_dates(patients, ["BIRTHDATE"])
    datasets = {
        "conditions": load_collection(email, "conditions.csv"),
        "medications": load_collection(email, "medications.csv"),
        "procedures": load_collection(email, "procedures.csv"),
        "observations": load_collection(email, "observations.csv"),
    }

    for name, df in datasets.items():
        if name == "observations":
            datasets[name] = convert_dates(df, ["DATE"])
        else:
            datasets[name] = convert_dates(df, ["START", "STOP"])
        datasets[name] = datasets[name].drop_duplicates()


    all_patients = []

    for _, row in patients.iterrows():
        pid = str(row.get("Id", "")).strip()
        age = int((pd.Timestamp.now() - row["BIRTHDATE"]).days / 365) if pd.notna(row.get("BIRTHDATE")) else None
        patient_conditions = datasets["conditions"][datasets["conditions"]["PATIENT"] == pid]
        patient_conditions = patient_conditions[patient_conditions["DESCRIPTION"].apply(is_real_health_condition)]
        last_condition = patient_conditions.sort_values("START", ascending=False).iloc[0] if not patient_conditions.empty else None

        def filter_after_date(df, date_col):
            df_patient = df[df["PATIENT"] == pid].copy()
            return df_patient.to_dict(orient="records")

        patient_detail = {
            "id": pid,
            "name": f"{row.get('FIRST','')} {row.get('LAST','')}".strip() or "Unnamed Patient",
            "birthdate": row.get("BIRTHDATE"),
            "age": age,
            "gender": row.get("GENDER", "N/A"),
            "city": row.get("ADDRESS_CITY", "N/A"),
            "state": row.get("ADDRESS_STATE", "N/A"),
            "country": row.get("ADDRESS_COUNTRY", "N/A"),
            "current_condition": last_condition.to_dict() if last_condition is not None else {},
            "conditions": filter_after_date(datasets["conditions"], "START"),
            "medications": filter_after_date(datasets["medications"], "START"),
            "observations": filter_after_date(datasets["observations"], "DATE"),
            "procedures": filter_after_date(datasets["procedures"], "START"),
        }
        all_patients.append(patient_detail)
    return all_patients



# -------------------------------------------------
# Utility Functions
# -------------------------------------------------






def safe_datetime(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    return val

def sanitize_patient(patient):
    if not isinstance(patient, dict):
        return patient
    sanitized = {}
    for k, v in patient.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_patient(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_patient(i) if isinstance(i, dict) else safe_datetime(i)
                for i in v
            ]
        else:
            sanitized[k] = safe_datetime(v)
    return sanitized

def is_real_health_condition(desc):
    if pd.isna(desc):
        return False
    desc_lower = desc.lower()
    keywords = [
        "disorder", "disease", "syndrome", "cancer", "diabetes", "asthma", "pain",
        "chronic", "hypertension", "infection", "injury", "stroke", "fracture",
        "renal", "liver", "hepatitis", "arthritis", "migraine", "depression", "anxiety"
    ]
    return any(k in desc_lower for k in keywords)




# -------------------------------------------------
# Model Loading
# -------------------------------------------------
MODEL_DIR = "model_outputs"
risk_model_path = os.path.join(MODEL_DIR, "current_risk_classifier.pkl")
future_model_path = os.path.join(MODEL_DIR, "future_5yr_regressor.pkl")
treat_model_path = os.path.join(MODEL_DIR, "treatment_recommender.pkl")

risk_model = future_model = treatment_model = None

if os.path.exists(risk_model_path):
    with open(risk_model_path, "rb") as f:
        clf_bundle = pickle.load(f)
        risk_model = clf_bundle["model"]
        risk_label_encoder = clf_bundle.get("label_encoder") or clf_bundle.get("risk_label_encoder")
        feature_cols = clf_bundle["feature_cols"]
        print("✅ Risk model loaded successfully.")
if os.path.exists(future_model_path):
    with open(future_model_path, "rb") as f:
        bundle = pickle.load(f)
        if isinstance(bundle, dict) and "model" in bundle:
            future_model = bundle["model"]
            future_feature_cols = bundle.get("feature_cols", [])
        else:
            future_model = bundle
            future_feature_cols = []
        print("✅ 5-Year Future Risk Regressor loaded.")

if os.path.exists(treat_model_path):
    with open(treat_model_path, "rb") as f:
        treatment_model = pickle.load(f)
        print("✅ Treatment model loaded.")
import pickle

with open("model_outputs/future_5yr_regressor.pkl", "rb") as f:
    model_data = pickle.load(f)

print("Model expects:", model_data["feature_cols"])


def compute_features(patient_data):
    age = int(patient_data.get("age", 0) or 0)

    # Default values
    procedure_desc = ""
    medication_desc = ""
    careplan_desc = ""
    obs_value = 0

    if isinstance(patient_data.get("procedures"), list) and len(patient_data["procedures"]) > 0:
        df = pd.DataFrame(patient_data["procedures"])
        if "START" in df.columns:
            df = df.sort_values("START", ascending=False)
        procedure_desc = str(df.iloc[0].get("DESCRIPTION", "")).lower()

    if isinstance(patient_data.get("medications"), list) and len(patient_data["medications"]) > 0:
        df = pd.DataFrame(patient_data["medications"])
        if "START" in df.columns:
            df = df.sort_values("START", ascending=False)
        medication_desc = str(df.iloc[0].get("DESCRIPTION", "")).lower()

    if isinstance(patient_data.get("conditions"), list) and len(patient_data["conditions"]) > 0:
        df = pd.DataFrame(patient_data["conditions"])
        if "START" in df.columns:
            df = df.sort_values("START", ascending=False)
        careplan_desc = str(df.iloc[0].get("DESCRIPTION", "")).lower()

    if isinstance(patient_data.get("observations"), list) and len(patient_data["observations"]) > 0:
        df = pd.DataFrame(patient_data["observations"])
        if "DATE" in df.columns:
            df = df.sort_values("DATE", ascending=False)
        obs_value = df.iloc[0].get("VALUE", 0)

    def feature_score(desc, mapping):
        for k, v in mapping.items():
            if k in desc:
                return v
        return 0

    procedure_severity = feature_score(procedure_desc, {
        "transplant": 4, "surgery": 3, "therapy": 2, "screening": 1
    })
    medication_intensity = feature_score(medication_desc, {
        "insulin": 3, "chemo": 3, "steroid": 3,
        "antibiotic": 2, "antihypertensive": 2,
        "vitamin": 1, "supplement": 1
    })
    treatment_response = feature_score(careplan_desc, {
        "improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worse": -1
    })

    try:
        obs_numeric = float(obs_value)
    except:
        obs_numeric = 0

    # 🔟 Now add extra derived features to match model expectation
    age_over_60 = 1 if age > 60 else 0
    procedure_count = len(patient_data.get("procedures", []))
    medication_count = len(patient_data.get("medications", []))
    careplan_count = len(patient_data.get("conditions", []))
    avg_observation_value = obs_numeric  # simple average substitute

    features = {
        "AGE": age,
        "procedure_severity": procedure_severity,
        "medication_intensity": medication_intensity,
        "treatment_response": treatment_response,
        "observation_value": obs_numeric,
        "age_over_60": age_over_60,
        "procedure_count": procedure_count,
        "medication_count": medication_count,
        "careplan_count": careplan_count,
        "avg_observation_value": avg_observation_value
    }

    print(f"🧩 Computed features for {patient_data.get('name','Unknown')}: {features}")
    return pd.DataFrame([features])



# -------------------------------------------------
# Risk Level Helper
# -------------------------------------------------
def risk_label_from_score(score):
    if score >= 0.5:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"

def risk_numeric_from_label(label):
    """Convert label to numeric score"""
    if not label:
        return 0
    l = str(label).lower()
    if l == "high":
        return 3
    elif l == "medium":
        return 2
    elif l == "low":
        return 1
    else:
        return 0

  
# ----------------------------------------
def risk_label_from_score(score):
    if score >= 0.5:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"


def calculate_age(birthdate_str):
    try:
        if pd.isna(birthdate_str) or birthdate_str == '':
            return None
        birthdate = datetime.strptime(str(birthdate_str), "%Y-%m-%d")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except Exception:
        return None
    





# Base directory where your dataset CSV files are located
DATA_DIR = "uploads/synthea_data_csv"

def compute_patient_features(patient_id):
    """
    Compute the 10 model features for the given patient_id
    using data from patients.csv, procedures.csv, medications.csv,
    careplans.csv, and observations.csv.
    """
    try:
        # ---------------------------------------
        # 1️⃣ Load all necessary CSV files
        # ---------------------------------------
        patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
        procedures = pd.read_csv(os.path.join(DATA_DIR, "procedures.csv"))
        medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
        careplans = pd.read_csv(os.path.join(DATA_DIR, "careplans.csv"))
        observations = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"))

        # ---------------------------------------
        # 2️⃣ AGE calculation from patients.csv
        # ---------------------------------------
        patient_row = patients[patients["Id"] == patient_id]
        if not patient_row.empty:
            birthdate = pd.to_datetime(patient_row["BIRTHDATE"].iloc[0], errors="coerce")
            if pd.notnull(birthdate):
                age = (pd.Timestamp("now") - birthdate).days // 365
            else:
                age = 0
        else:
            age = 0

        # ---------------------------------------
        # 3️⃣ Procedure-related features
        # ---------------------------------------
        patient_procedures = procedures[procedures["PATIENT"] == patient_id]
        procedure_count = len(patient_procedures)
        procedure_severity = (
            patient_procedures["CODE"].nunique() % 3 if not patient_procedures.empty else 0
        )

        # ---------------------------------------
        # 4️⃣ Medication-related features
        # ---------------------------------------
        patient_medications = medications[medications["PATIENT"] == patient_id]
        medication_count = len(patient_medications)
        medication_intensity = (
            patient_medications["CODE"].nunique() % 3 if not patient_medications.empty else 0
        )

        # ---------------------------------------
        # 5️⃣ Careplan-related features
        # ---------------------------------------
        patient_careplans = careplans[careplans["PATIENT"] == patient_id]
        careplan_count = len(patient_careplans)

        # ---------------------------------------
        # 6️⃣ Observation-related features
        # ---------------------------------------
        patient_observations = observations[observations["PATIENT"] == patient_id]
        if not patient_observations.empty and "VALUE" in patient_observations.columns:
            obs_vals = pd.to_numeric(patient_observations["VALUE"], errors="coerce").dropna()
            observation_value = obs_vals.mean() if not obs_vals.empty else 0
        else:
            observation_value = 0
        avg_observation_value = observation_value

        # ---------------------------------------
        # 7️⃣ Treatment response (simulated logic)
        # ---------------------------------------
        treatment_response = random.choice([0, 1]) if patient_medications.empty else len(patient_medications) % 2

        # ---------------------------------------
        # 8️⃣ Derived features
        # ---------------------------------------
        age_over_60 = 1 if age > 60 else 0

        # ---------------------------------------
        # ✅ Return all 10 model features
        # ---------------------------------------
        features = {
            "AGE": int(age),
            "procedure_severity": int(procedure_severity),
            "medication_intensity": int(medication_intensity),
            "treatment_response": int(treatment_response),
            "observation_value": float(observation_value),
            "age_over_60": int(age_over_60),
            "procedure_count": int(procedure_count),
            "medication_count": int(medication_count),
            "careplan_count": int(careplan_count),
            "avg_observation_value": float(avg_observation_value),
        }

        print(f"🧩 Computed patient features for {patient_id}: {features}")
        return features

    except Exception as e:
        print(f"❌ Error computing features for {patient_id}: {e}")
        return {
            "AGE": 0, "procedure_severity": 0, "medication_intensity": 0,
            "treatment_response": 0, "observation_value": 0, "age_over_60": 0,
            "procedure_count": 0, "medication_count": 0,
            "careplan_count": 0, "avg_observation_value": 0
        }


