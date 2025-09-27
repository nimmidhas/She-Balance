from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import jwt
from datetime import datetime, timedelta
from jwt import PyJWTError

# JWT configuration
SECRET_KEY = "she-balance-secret-key-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

security = HTTPBearer()

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize database and app
init_db()
app = FastAPI(title="She Balance AI", version="1.0.0")

# Fix CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI models
try:
    model = joblib.load('model/siddha_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    df = joblib.load('model/dataset.pkl')
    print("✅ AI models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

class SymptomRequest(BaseModel):
    symptoms: str

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/api/register")
async def register(user_data: UserRegister):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(user_data.password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (user_data.username, user_data.email, password_hash)
        )
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()

@app.post("/api/login")
async def login(user_data: UserLogin):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (user_data.username,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user and user[2] == hash_password(user_data.password):
        access_token = create_access_token(data={"sub": user[1], "user_id": user[0]})
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "username": user[1],
            "message": "Login successful"
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

def predict_remedy(symptoms_text):
    symptoms_text = symptoms_text.lower()
    symptom_vector = vectorizer.transform([symptoms_text])
    probabilities = model.predict_proba(symptom_vector)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    
    results = []
    for idx in top_3_indices:
        if probabilities[idx] > 0.01:
            remedy_term = label_encoder.inverse_transform([idx])[0]
            remedy_data = df[df['Remedy (Siddha Term)'] == remedy_term]
            if not remedy_data.empty:
                remedy_row = remedy_data.iloc[0]
                results.append({
                    "id": int(remedy_row['ID']),
                    "remedy_name": remedy_row['Remedy (English)'],
                    "siddha_term": remedy_row['Remedy (Siddha Term)'],
                    "symptoms": remedy_row['Symptom'],
                    "ingredients": remedy_row['Primary Ingredients'],
                    "preparation": remedy_row['Preparation Method'],
                    "dosage": remedy_row['Dosage'],
                    "target_dosha": remedy_row['Target Dosha'],
                    "confidence": float(probabilities[idx])
                })
    return results

@app.get("/")
async def root():
    return {"message": "🌿 She Balance AI API is running!", "version": "1.0.0"}

@app.get("/api/predict")
async def predict_from_symptoms(symptoms: str, token: dict = Depends(verify_token)):
    """Protected endpoint - requires authentication"""
    if not symptoms or len(symptoms.strip()) < 3:
        raise HTTPException(status_code=400, detail="Please provide symptoms (min 3 characters)")
    
    try:
        remedies = predict_remedy(symptoms)
        if not remedies:
            raise HTTPException(status_code=404, detail="No remedies found")
        
        return {
            "input_symptoms": symptoms,
            "remedies_found": len(remedies),
            "remedies": remedies,
            "user": token.get("sub", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/public/predict")
async def public_predict_from_symptoms(symptoms: str):
    """Public endpoint - no authentication required"""
    if not symptoms or len(symptoms.strip()) < 3:
        raise HTTPException(status_code=400, detail="Please provide symptoms (min 3 characters)")
    
    try:
        remedies = predict_remedy(symptoms)
        if not remedies:
            raise HTTPException(status_code=404, detail="No remedies found")
        
        return {
            "input_symptoms": symptoms,
            "remedies_found": len(remedies),
            "remedies": remedies,
            "note": "This is a public demo endpoint"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/remedies")
async def get_all_remedies(token: dict = Depends(verify_token)):
    """Protected endpoint - requires authentication"""
    remedies = []
    for _, row in df.iterrows():
        remedies.append({
            "id": int(row['ID']),
            "symptom": row['Symptom'],
            "remedy_name": row['Remedy (English)'],
            "siddha_term": row['Remedy (Siddha Term)'],
            "target_dosha": row['Target Dosha']
        })
    return remedies

@app.get("/api/public/remedies")
async def get_public_remedies():
    """Public endpoint - no authentication required"""
    remedies = []
    for _, row in df.iterrows():
        remedies.append({
            "id": int(row['ID']),
            "symptom": row['Symptom'],
            "remedy_name": row['Remedy (English)'],
            "siddha_term": row['Remedy (Siddha Term)'],
            "target_dosha": row['Target Dosha']
        })
    return remedies[:10]  # Return only first 10 for demo

@app.get("/api/user/profile")
async def get_user_profile(token: dict = Depends(verify_token)):
    """Get user profile - requires authentication"""
    return {
        "username": token.get("sub"),
        "user_id": token.get("user_id"),
        "message": "Profile retrieved successfully"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting She Balance AI Server with Authentication...")
    print("🔐 Authentication: Enabled")
    print("🌐 Frontend: http://localhost:3000")
    print("🔧 API: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)