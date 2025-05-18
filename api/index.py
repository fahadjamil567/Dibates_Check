from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Disease Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Basic processing
    image = image.resize((64, 64))
    img_array = np.array(image)
    
    # Simple prediction (mock)
    avg_color = np.mean(img_array, axis=(0, 1))
    prediction = 1 if np.mean(avg_color) > 127 else 0
    
    return {
        "prediction": prediction,
        "healthy_probability": 0.8 if prediction == 0 else 0.2,
        "diseased_probability": 0.2 if prediction == 0 else 0.8
    } 