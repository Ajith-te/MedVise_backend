import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

# -------------------------------------------------
# Flask Configuration
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads/synthea_data_csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {".csv"}

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_csv(file_name, folder):
    path = os.path.join(UPLOAD_FOLDER, folder, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

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
# Patient Preprocessing
# -------------------------------------------------
def preprocess_patient_details(email):
    folder = email
    patients = load_csv("patients.csv", folder)
    if patients.empty:
        return []

    patients = convert_dates(patients, ["BIRTHDATE"])
    datasets = {
        "conditions": load_csv("conditions.csv", folder),
        "medications": load_csv("medications.csv", folder),
        "procedures": load_csv("procedures.csv", folder),
        "observations": load_csv("observations.csv", folder),
    }

    for name, df in datasets.items():
        if name == "observations":
            datasets[name] = convert_dates(df, ["DATE"])
        else:
            datasets[name] = convert_dates(df, ["START", "STOP"])
        datasets[name] = df.drop_duplicates()

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

# ----------------------------------------
# API 1: Predict Current Risk (probability + label)
# ----------------------------------------
@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        data = request.get_json()
        print("📥 Received risk prediction data:", data)

        # ✅ Load saved model and preprocessing objects
        model_path = os.path.join("model_outputs", "current_risk_classifier.pkl")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found. Please train the model first."}), 500

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        le = model_data["label_encoder"]
        feature_cols = model_data["feature_cols"]

        # ✅ Get all features directly from frontend (selected patient)
        age = int(data.get("AGE", 0))
        observation_val = float(data.get("observation_value", 0))
        procedure_count = int(data.get("procedure_count", 0))
        medication_count = int(data.get("medication_count", 0))
        careplan_count = int(data.get("careplan_count", 0))
        avg_observation_value = float(data.get("avg_observation_value", observation_val))

        # ✅ Compute engineered features
        age_over_60 = 1 if age > 60 else 0

        def feature_score(desc, mapping):
            if not isinstance(desc, str):
                return 0
            desc = desc.lower()
            for k, v in mapping.items():
                if k in desc:
                    return v
            return 0

        procedure_severity = feature_score(data.get("procedure_desc", ""), {
            "transplant": 4, "surgery": 3, "therapy": 2, "screening": 1
        })
        medication_intensity = feature_score(data.get("medication_desc", ""), {
            "insulin": 3, "chemo": 3, "steroid": 3,
            "antibiotic": 2, "antihypertensive": 2,
            "vitamin": 1, "supplement": 1
        })
        treatment_response = feature_score(data.get("careplan_desc", ""), {
            "improve": 2, "recovery": 2, "monitor": 1, "no change": 0, "worse": -1
        })

        # ✅ Combine all features
        input_data = pd.DataFrame([{
            "AGE": age,
            "procedure_severity": procedure_severity,
            "medication_intensity": medication_intensity,
            "treatment_response": treatment_response,
            "observation_value": observation_val,
            "age_over_60": age_over_60,
            "procedure_count": procedure_count,
            "medication_count": medication_count,
            "careplan_count": careplan_count,
            "avg_observation_value": avg_observation_value
        }])

        # Ensure correct order
        input_data = input_data[feature_cols]

        # ✅ Scale & Predict
        X_scaled = scaler.transform(input_data)
        pred_class = model.predict(X_scaled)[0]
        pred_label = le.inverse_transform([pred_class])[0]
        pred_proba = model.predict_proba(X_scaled).max()

        print(f"✅ Current Risk Predicted: {pred_label} | Confidence: {pred_proba:.3f}")

        return jsonify({
            "predicted_risk": pred_label,
            "confidence_score": round(float(pred_proba), 3),
            "features_used": input_data.to_dict(orient="records")[0]
        })

    except Exception as e:
        print(f"❌ Error in /predict_risk: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_future_risk", methods=["POST"])
def predict_future_risk():
    try:
        data = request.get_json()
        print("📥 Received future risk data:", data)

        # ✅ Load model
        model_path = os.path.join("model_outputs", "future_5yr_regressor.pkl")
        if not os.path.exists(model_path):
            return jsonify({"error": "Future risk model not found"}), 500

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_cols = model_data["feature_cols"]

        # ✅ Get features from frontend (same patient)
        X = pd.DataFrame([[float(data.get(col, 0)) for col in feature_cols]], columns=feature_cols)
        X_scaled = scaler.transform(X)

        predicted_future_risk = float(model.predict(X_scaled)[0])
        print(f"✅ Predicted Future Risk Score: {predicted_future_risk:.3f}")

        return jsonify({
            "predicted_future_risk": predicted_future_risk,
            "features_used": X.to_dict(orient="records")[0]
        })

    except Exception as e:
        print(f"❌ Error in /predict_future_risk: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# API 3: Recommend Treatment
# ----------------------------------------
# Load treatment recommendation models
with open(os.path.join(MODEL_DIR, "treatment_recommender.pkl"), "rb") as f:
    treatment_recommender = pickle.load(f)

with open(os.path.join(MODEL_DIR, "treatment_vectorizer.pkl"), "rb") as f:
    treatment_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "treatment_label_encoder.pkl"), "rb") as f:
    treatment_label_encoder = pickle.load(f)

print("✅ Treatment model loaded successfully (with actual label names)")
# ----------------------------------------
# API 3: High Risk Patients (Fetch and Predict)
# ----------------------------------------
@app.route("/high_risk_patients", methods=["GET"])
def get_high_risk_patients():
    if not risk_model:
        return jsonify({"error": "Risk model not loaded"}), 500

    try:
        doctor_email = request.args.get("email")
        if not doctor_email:
            return jsonify({"error": "Doctor email required"}), 400

        upload_path = os.path.join(UPLOAD_FOLDER, doctor_email, "patients.csv")

        if not os.path.exists(upload_path):
            return jsonify({"error": f"No uploaded CSV found for {doctor_email}"}), 404

        df = pd.read_csv(upload_path)

        # Ensure all feature columns exist (fill missing ones with 0)
        missing_cols = [col for col in feature_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0

        X = df[feature_cols].values

        # Predict risk using the current model
        if hasattr(risk_model, "predict_proba"):
            probs = risk_model.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            labels = risk_label_encoder.inverse_transform(preds)
            df["predicted_risk"] = labels
            df["actual_score"] = [round(p[np.argmax(p)], 3) for p in probs]
        else:
            y_pred = risk_model.predict(X)
            labels = risk_label_encoder.inverse_transform(y_pred)
            df["predicted_risk"] = labels
            df["actual_score"] = y_pred

        # Keep only High-risk patients
        high_risk = df[df["predicted_risk"] == "High"]

        print(f"✅ {len(high_risk)} high-risk patients identified for {doctor_email}")
        return jsonify(high_risk.to_dict(orient="records"))

    except Exception as e:
        print(f"❌ Error in /high_risk_patients: {e}")
        return jsonify({"error": str(e)}), 500


from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

def calculate_age(birthdate_str):
    try:
        if pd.isna(birthdate_str) or birthdate_str == '':
            return None
        birthdate = datetime.strptime(str(birthdate_str), "%Y-%m-%d")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except Exception:
        return None
import pandas as pd
import numpy as np
import random
import os

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



# -------------------------------------------------
# API 4: Upload CSV Files
# -------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    from database import insert_many
    import pandas as pd
    import datetime

    email = request.form.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    # 🔹 Sanitize email for MongoDB collection name
    collection_name = email.replace("@", "_").replace(".", "_")

    user_folder = os.path.join(UPLOAD_FOLDER, collection_name)
    os.makedirs(user_folder, exist_ok=True)

    files = request.files.getlist("files")
    saved_files = []

    for file in files:
        filename = os.path.basename(file.filename)

        if allowed_file(filename):
            filepath = os.path.join(user_folder, filename)
            file.save(filepath)
            saved_files.append(filename)

            # ✅ Read CSV
            df = pd.read_csv(filepath)

            # Optional metadata
            df["uploaded_by"] = email
            df["dataset_name"] = filename
            df["upload_date"] = datetime.datetime.now()

            records = df.to_dict(orient="records")

            # 🔹 Insert into MongoDB collection = sanitized email
            insert_many(collection_name, records)

    if not saved_files:
        return jsonify({"error": "No valid CSV files uploaded"}), 400

    return jsonify({
        "message": "Files stored successfully",
        "collection": collection_name,
        "files": saved_files
    })

# -------------------------------------------------
# API 5: Get Patients Summary
# -------------------------------------------------
@app.route("/patients", methods=["GET"])
def get_patients():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    all_patients = preprocess_patient_details(email)
    summary = [
        {
            "id": p["id"],
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "condition": p["current_condition"].get("DESCRIPTION", "N/A"),
        }
        for p in all_patients
    ]
    return jsonify({"patients": summary})
@app.route("/recommend_treatment", methods=["POST"])
def recommend_treatment():
    try:
        data = request.get_json(force=True)
        print("📥 Received treatment recommendation request:", data)

        # -------------------------------
        # Load all trained components
        # -------------------------------
        with open("models/treatment_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # -------------------------------
        # Validate and process input
        # -------------------------------
        condition_desc = data.get("condition_desc", "").strip()
        duration_days = data.get("duration_days", 0)  # numeric field

        if not condition_desc:
            return jsonify({"error": "❌ 'condition_desc' is required"}), 400

        # TF-IDF transformation (text)
        X_text = tfidf.transform([condition_desc])

        # Scale numeric input
        X_num = scaler.transform([[float(duration_days)]])

        # Combine both into final input
        from scipy.sparse import hstack
        X_input = hstack([X_text, X_num])

        # -------------------------------
        # Make prediction
        # -------------------------------
        pred_code = model.predict(X_input)[0]
        predicted_treatment = label_encoder.inverse_transform([pred_code])[0]

        print(f"🧠 Predicted Treatment: {predicted_treatment}")

        # -------------------------------
        # Return API response
        # -------------------------------
        return jsonify({
            "recommended_treatment": predicted_treatment,
            "model_type": "Random Forest (TF-IDF + Duration)",
            "input_used": {
                "condition_desc": condition_desc,
                "duration_days": duration_days
            }
        })

    except Exception as e:
        print(f"❌ Error in /recommend_treatment: {e}")
        return jsonify({"error": str(e)}), 500


def preprocess_patient(email):
    folder = email
    patients = load_csv("patients.csv", folder)
    if patients.empty:
        return []

    patients = convert_dates(patients, ["BIRTHDATE"])
    datasets = {
        "conditions": load_csv("conditions.csv", folder),
        "medications": load_csv("medications.csv", folder),
        "procedures": load_csv("procedures.csv", folder),
        "observations": load_csv("observations.csv", folder),
    }

    for name, df in datasets.items():
        if name == "observations":
            datasets[name] = convert_dates(df, ["DATE"])
        else:
            datasets[name] = convert_dates(df, ["START", "STOP"])
        datasets[name] = df.drop_duplicates()

    exclude_procedures = ["dental", "oral health", "fluoride", "plaque", "calculus", "gingivae"]
    exclude_observations = ["qaly", "daly", "qols"]
    exclude_medications = ["sodium fluoride"]

    if not datasets["procedures"].empty:
        datasets["procedures"] = datasets["procedures"][
            ~datasets["procedures"]["DESCRIPTION"].str.lower().str.contains("|".join(exclude_procedures), na=False)
        ]

    if not datasets["observations"].empty:
        datasets["observations"] = datasets["observations"][
            ~datasets["observations"]["DESCRIPTION"].str.lower().str.contains("|".join(exclude_observations), na=False)
        ]

    if not datasets["medications"].empty:
        datasets["medications"] = datasets["medications"][
            ~datasets["medications"]["DESCRIPTION"].str.lower().str.contains("|".join(exclude_medications), na=False)
        ]

    all_patients = []

    for _, row in patients.iterrows():
        pid = str(row.get("Id", "")).strip()
        age = int((pd.Timestamp.now() - row["BIRTHDATE"]).days / 365) if pd.notna(row.get("BIRTHDATE")) else None

        patient_conditions = datasets["conditions"][datasets["conditions"]["PATIENT"] == pid]
        patient_conditions = patient_conditions[patient_conditions["DESCRIPTION"].apply(is_real_health_condition)]

        last_condition = patient_conditions.sort_values("START", ascending=False).iloc[0] if not patient_conditions.empty else None
        last_date = last_condition["START"] if last_condition is not None else None

        def filter_after_date(df, date_col):
            df_patient = df[df["PATIENT"] == pid].copy()
            if last_date is not None and date_col in df_patient.columns:
                df_patient.loc[:, date_col] = pd.to_datetime(df_patient[date_col], errors="coerce").dt.tz_localize(None)
                last_local = pd.to_datetime(last_date, errors="coerce").tz_localize(None)
                df_patient = df_patient[df_patient[date_col] >= last_local]
            return df_patient.to_dict(orient="records")

        patient_detail = {
            "id": pid,
            "name": f"{row.get('FIRST','')} {row.get('LAST','')}".strip() or "Unnamed Patient",
            "birthdate": row.get("BIRTHDATE"),
            "age": age,
            "gender": row.get("GENDER", "N/A") if pd.notna(row.get("GENDER")) else "N/A",
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

@app.route("/patient/<patient_id>", methods=["GET"])
def get_patient_details(patient_id):
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    all_patients = preprocess_patient(email)
    patient = next((p for p in all_patients if str(p["id"]) == str(patient_id)), None)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(sanitize_patient(patient))

# -------------------------------------------------
# API 6: Predictive Analysis (Summary per Patient)
# -------------------------------------------------
@app.route("/predictive_analysis", methods=["GET"])
def predictive_analysis():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    try:
        patients = preprocess_patient_details(email)
        if not patients or len(patients) == 0:
            return jsonify({"error": "No patient data found"}), 404

        summary_data = []

        for p in patients:
            cond_desc = str(p.get("current_condition", {}).get("DESCRIPTION", "Unknown")).lower()

            # 🧠 Generate patient-specific features
            feature_df = compute_features(p)

            # ✅ Predict risk
            predicted_risk = "N/A"
            if risk_model is not None:
                try:
                    X = feature_df[feature_cols] if set(feature_cols).issubset(feature_df.columns) else feature_df
                    if hasattr(risk_model, "predict_proba"):
                        probs = risk_model.predict_proba(X)[0]
                        pred = np.argmax(probs)
                        predicted_risk = risk_label_encoder.inverse_transform([pred])[0] if risk_label_encoder else str(pred)
                    else:
                        y_pred = risk_model.predict(X)[0]
                        predicted_risk = risk_label_encoder.inverse_transform([int(y_pred)])[0] if risk_label_encoder else str(y_pred)
                except Exception as e:
                    print(f"⚠️ Risk prediction failed for {p.get('id')}: {e}")

            # ✅ Predict treatment
            recommended_treatment = "N/A"
            if treatment_model is not None:
                try:
                    prediction_encoded = treatment_model.predict(feature_df)[0]
                    recommended_treatment = treatment_label_encoder.inverse_transform([prediction_encoded])[0]
                except Exception as e:
                    print(f"⚠️ Treatment prediction failed for {p.get('id')}: {e}")

            summary_data.append({
                "id": p.get("id"),
                "name": p.get("name", "Unknown"),
                "age": p.get("age", "Unknown"),
                "gender": p.get("gender", "Unknown"),
                "condition": cond_desc,
                "predicted_risk": predicted_risk,
                "recommended_treatment": recommended_treatment
            })
            print(f"🔹 Features for {p.get('name')}: {feature_df.to_dict(orient='records')[0]}")


        return jsonify(summary_data), 200

    except Exception as e:
        print(f"❌ Error in /predictive_analysis: {e}")
        return jsonify({"error": str(e)}), 500



# -------------------------------------------------
# Run Flask App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
