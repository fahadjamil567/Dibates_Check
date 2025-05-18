from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((64, 64))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Simple feature extraction - average color values
    features = np.mean(img_array, axis=(0, 1))
    
    return features

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    features = preprocess_image(image)
    
    # Simple rule-based prediction using color values
    # If the average red channel is higher than others, classify as diseased
    prediction = 1 if features[0] > np.mean(features[1:]) else 0
    
    # Calculate simple probabilities based on color channel differences
    diff = np.abs(features[0] - np.mean(features[1:]))
    prob = np.clip(0.5 + diff, 0.1, 0.9)
    
    if prediction == 1:
        probabilities = [1 - prob, prob]
    else:
        probabilities = [prob, 1 - prob]
    
    return {
        "prediction": prediction,
        "healthy_probability": float(probabilities[0]),
        "diseased_probability": float(probabilities[1])
    } 