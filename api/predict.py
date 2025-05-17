from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
import json
from .model import SimpleModel

app = FastAPI()

# Load the model from JSON
try:
    with open('api/model_data.json', 'r') as f:
        model = SimpleModel.from_json(f.read())
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = float(model.predict_proba(features)[0])
        
        return {
            "prediction": int(prediction[0]),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 