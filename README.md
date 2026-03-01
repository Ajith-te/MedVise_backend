# 🏥 MedVise Backend

MedVise Backend is a Flask-based REST API for healthcare data processing and disease prediction using Machine Learning.

---

## 🚀 Features

- CSV file upload
- MongoDB integration
- Patient data storage
- ML prediction using XGBoost / Scikit-learn
- REST API endpoints
- CORS enabled

---

## 🛠 Tech Stack

- Python
- Flask
- MongoDB
- Pandas
- Scikit-learn
- XGBoost

---

## 📂 Project Structure

```
MedVise_backend/
│── app.py
│── database.py
│── route/
│── model/
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ajith-te/MedVise_backend.git
cd MedVise_backend
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

Server will start at:

```
http://127.0.0.1:5000/
```

---

## 📡 API Endpoints

### Upload CSV
```
POST /upload
```

### Predict
```
POST /predict
```

---

## 🔐 Environment Variables

Create `.env` file:

```
MONGO_URI=your_mongodb_connection_string
```

---

## 📌 Future Improvements

- JWT Authentication
- Docker Deployment
- Model Versioning
- Cloud Deployment (AWS)

---

## 👨‍💻 Author
