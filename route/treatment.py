import os
import pickle
import numpy as np
import pandas as pd

from flask import Blueprint, request, jsonify
from scipy.sparse import hstack

from utils import preprocess_patient_details

treatment_bp = Blueprint("treatment", __name__)

MODEL_DIR = "model_outputs"

# --------------------------------------------------
# Load Models ONCE
# --------------------------------------------------

TREATMENT_MODEL = None
TFIDF = None
LABEL_ENCODER = None
SCALER = None


def load_treatment_models():
    global TREATMENT_MODEL, TFIDF, LABEL_ENCODER, SCALER

    try:
        with open(os.path.join(MODEL_DIR, "treatment_model.pkl"), "rb") as f:
            TREATMENT_MODEL = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            TFIDF = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            LABEL_ENCODER = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            SCALER = pickle.load(f)

        print("✅ Treatment models loaded successfully")

    except Exception as e:
        print(f"❌ Treatment model loading failed: {e}")


load_treatment_models()


# --------------------------------------------------
# API 1: Recommend Treatment
# --------------------------------------------------
@treatment_bp.route("/recommend_treatment", methods=["POST"])
def recommend_treatment():

    try:
        if TREATMENT_MODEL is None:
            return jsonify({"error": "Treatment model not loaded"}), 500

        data = request.get_json()

        condition_desc = data.get("condition_desc", "").strip()
        duration_days = float(data.get("duration_days", 0))

        if not condition_desc:
            return jsonify({"error": "condition_desc is required"}), 400

        # Transform text
        X_text = TFIDF.transform([condition_desc])

        # Scale numeric
        X_num = SCALER.transform([[duration_days]])

        # Combine features
        X_input = hstack([X_text, X_num])

        pred_code = TREATMENT_MODEL.predict(X_input)[0]
        predicted_treatment = LABEL_ENCODER.inverse_transform([pred_code])[0]

        return jsonify({
            "recommended_treatment": predicted_treatment,
            "input_used": {
                "condition_desc": condition_desc,
                "duration_days": duration_days
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# API 2: High Risk Patients (MongoDB Based)
# --------------------------------------------------
@treatment_bp.route("/high_risk_patients", methods=["GET"])
def get_high_risk_patients():

    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    try:
        # Use your preprocess function (already DB based)
        patients = preprocess_patient_details(email)

        if not patients:
            return jsonify({"message": "No patients found"}), 404

        high_risk_list = []

        for p in patients:
            risk = p.get("predicted_risk")

            if risk and risk.lower() == "high":
                high_risk_list.append({
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "age": p.get("age"),
                    "gender": p.get("gender"),
                    "condition": p.get("current_condition", {}).get("DESCRIPTION")
                })

        return jsonify({
            "total_high_risk": len(high_risk_list),
            "patients": high_risk_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500