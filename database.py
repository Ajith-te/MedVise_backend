import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["MedVise"]

def insert_many(collection_name, records):
    collection = db[collection_name]
    collection.insert_many(records)


def get_patient_by_id(collection_name, patient_id):
    collection = db[collection_name]
    patient =collection.find_one({"PATIENT": patient_id})

    if patient:
        patient["_id"] = str(patient["_id"])  # convert here
    
    return patient

