# backend/utils/database.py

from pymongo import MongoClient

# -------------------------
# 1️⃣ MongoDB Configuration
# -------------------------
MONGO_URI = "mongodb://localhost:27017/"  # your local MongoDB

# -------------------------
# 2️⃣ Connect to MongoDB
# -------------------------
client = MongoClient(MONGO_URI)
db = client["MedVise"]  # Database name
FILE_DATA = db['file_data']
# -------------------------
# 3️⃣ Helper Functions
# -------------------------

def insert_many(collection_name, records):
    collection = db[collection_name]   # 🔥 THIS LINE IS IMPORTANT
    if records:
        collection.insert_many(records)


def insert_one(collection_name, record):
    """
    Insert a single record (dict) into a MongoDB collection
    """
    collection = db[collection_name]
    collection.insert_one(record)

def find_all(collection_name, query={}):
    """
    Fetch all documents from a collection (query is optional)
    Returns a list of dicts without _id
    """
    collection = db[collection_name]
    return list(collection.find(query, {"_id": 0}))

def find_one(collection_name, query={}):
    """
    Fetch a single document matching query
    Returns a dict without _id
    """
    collection = db[collection_name]
    return collection.find_one(query, {"_id": 0})