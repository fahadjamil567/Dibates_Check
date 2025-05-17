from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import os

app = FastAPI()

# Load the model and scaler
try:
    model = joblib.load('disease_model.joblib')
    scaler = joblib.load('disease_scaler.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array and reshape
        features = np.array(input_data.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 