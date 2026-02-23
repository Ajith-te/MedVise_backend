import os
import pickle
import numpy as np
import pandas as pd

from flask import Blueprint, request, jsonify

from utils import preprocess_patient_details, compute_features

predict_bp = Blueprint("predict", __name__)

# --------------------------------------------------
# Load Models ONCE (important for performance)
# --------------------------------------------------

RISK_MODEL = None
RISK_SCALER = None
RISK_LE = None
RISK_FEATURES = None

FUTURE_MODEL = None
FUTURE_SCALER = None
FUTURE_FEATURES = None


def load_models():
    global RISK_MODEL, RISK_SCALER, RISK_LE, RISK_FEATURES
    global FUTURE_MODEL, FUTURE_SCALER, FUTURE_FEATURES

    try:
        # Load current risk model
        risk_path = os.path.join("model_outputs", "current_risk_classifier.pkl")
        if os.path.exists(risk_path):
            with open(risk_path, "rb") as f:
                data = pickle.load(f)
                RISK_MODEL = data["model"]
                RISK_SCALER = data["scaler"]
                RISK_LE = data["label_encoder"]
                RISK_FEATURES = data["feature_cols"]

        # Load future risk model
        future_path = os.path.join("model_outputs", "future_5yr_regressor.pkl")
        if os.path.exists(future_path):
            with open(future_path, "rb") as f:
                data = pickle.load(f)
                FUTURE_MODEL = data["model"]
                FUTURE_SCALER = data["scaler"]
                FUTURE_FEATURES = data["feature_cols"]

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")


# Call once at import
load_models()


# --------------------------------------------------
# API 1: Predict Current Risk
# --------------------------------------------------
@predict_bp.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        if RISK_MODEL is None:
            return jsonify({"error": "Risk model not loaded"}), 500

        data = request.get_json()

        age = int(data.get("AGE", 0))
        observation_val = float(data.get("observation_value", 0))
        procedure_count = int(data.get("procedure_count", 0))
        medication_count = int(data.get("medication_count", 0))
        careplan_count = int(data.get("careplan_count", 0))
        avg_observation_value = float(data.get("avg_observation_value", observation_val))

        age_over_60 = 1 if age > 60 else 0

        input_df = pd.DataFrame([{
            "AGE": age,
            "observation_value": observation_val,
            "age_over_60": age_over_60,
            "procedure_count": procedure_count,
            "medication_count": medication_count,
            "careplan_count": careplan_count,
            "avg_observation_value": avg_observation_value
        }])

        input_df = input_df[RISK_FEATURES]

        X_scaled = RISK_SCALER.transform(input_df)

        pred_class = RISK_MODEL.predict(X_scaled)[0]
        pred_label = RISK_LE.inverse_transform([pred_class])[0]
        pred_proba = float(RISK_MODEL.predict_proba(X_scaled).max())

        return jsonify({
            "predicted_risk": pred_label,
            "confidence_score": round(pred_proba, 3),
            "features_used": input_df.to_dict(orient="records")[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# API 2: Predict Future 5-Year Risk
# --------------------------------------------------
@predict_bp.route("/predict_future_risk", methods=["POST"])
def predict_future_risk():
    try:
        if FUTURE_MODEL is None:
            return jsonify({"error": "Future risk model not loaded"}), 500

        data = request.get_json()

        X = pd.DataFrame(
            [[float(data.get(col, 0)) for col in FUTURE_FEATURES]],
            columns=FUTURE_FEATURES
        )

        X_scaled = FUTURE_SCALER.transform(X)
        prediction = float(FUTURE_MODEL.predict(X_scaled)[0])

        return jsonify({
            "predicted_future_risk": prediction,
            "features_used": X.to_dict(orient="records")[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# API 3: Predictive Analysis (All Patients)
# --------------------------------------------------
@predict_bp.route("/predictive_analysis", methods=["GET"])
def predictive_analysis():

    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    try:
        patients = preprocess_patient_details(email)
        if not patients:
            return jsonify({"error": "No patient data found"}), 404

        results = []

        for p in patients:
            feature_df = compute_features(p)

            predicted_risk = "N/A"

            if RISK_MODEL is not None:
                try:
                    X = feature_df[RISK_FEATURES]
                    X_scaled = RISK_SCALER.transform(X)

                    pred = RISK_MODEL.predict(X_scaled)[0]
                    predicted_risk = RISK_LE.inverse_transform([pred])[0]
                except:
                    pass

            results.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "age": p.get("age"),
                "gender": p.get("gender"),
                "predicted_risk": predicted_risk
            })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500