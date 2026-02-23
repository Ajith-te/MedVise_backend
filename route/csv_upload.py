import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

from database import get_patient_by_id, insert_many
from utils import allowed_file, preprocess_patient_details, sanitize_patient
from route.predict import predict_bp
from route.treatment import treatment_bp


app = Flask(__name__)
CORS(app)

app.register_blueprint(predict_bp)
app.register_blueprint(treatment_bp)


CHUNK_SIZE = 5000


def get_collection_name(email):
    return email.replace("@", "_").replace(".", "_")


# Upload CSV Files
@app.route("/upload", methods=["POST"])
def upload_files():

    email = request.form.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    collection_name = get_collection_name(email)

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved_files = []
    total_inserted = 0

    for file in files:
        filename = file.filename

        if not filename or not allowed_file(filename):
            continue

        try:
            file.stream.seek(0)  # Reset pointer

            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

                chunk["uploaded_by"] = email
                chunk["dataset_file_name"] = filename
                chunk["uploaded_at"] = datetime.now(timezone.utc)

                records = chunk.to_dict(orient="records")

                if records:
                    insert_many(collection_name, records)
                    total_inserted += len(records)

            saved_files.append(filename)

        except Exception as e:
            return jsonify({
                "error": f"Error processing file {filename}",
                "details": str(e)
            }), 500

    if not saved_files:
        return jsonify({"error": "No valid CSV files uploaded"}), 400

    return jsonify({
        "message": "Files processed successfully",
        "collection": collection_name,
        "files": saved_files,
        "total_records_inserted": total_inserted
    }), 200


# Get Patients Summary
@app.route("/patients", methods=["GET"])
def get_patients():

    try:
        email = request.args.get("email")
        if not email:
            return jsonify({"error": "Email required"}), 400

        collection_name = get_collection_name(email)

        all_patients = preprocess_patient_details(collection_name)

        if not all_patients:
            return jsonify({
                "patients": [],
                "total": 0,
                "message": "No patients found"
            }), 200

        summary = [
            {
                "id": p.get("id"),
                "name": p.get("name", "Unknown"),
                "age": p.get("age"),
                "gender": p.get("gender", "N/A"),
                "condition": p.get("current_condition", {}).get("DESCRIPTION", "N/A"),
            }
            for p in all_patients
        ]

        return jsonify({
            "patients": summary,
            "total": len(summary)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500


# Get Single Patient Details
@app.route("/patient/<patient_id>", methods=["GET"])
def get_patient_details(patient_id):

    try:
        email = request.args.get("email")
        if not email:
            return jsonify({"error": "Email required"}), 400

        collection_name = get_collection_name(email)

        patient = get_patient_by_id(collection_name, patient_id)

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        return jsonify({
            "patient": sanitize_patient(patient)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500