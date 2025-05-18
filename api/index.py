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
    
    return img_array

@app.get("/")
async def root():
    return {"message": "Disease Detection API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    features = preprocess_image(image)
    
    # Simple mock prediction based on average pixel values
    avg_color = np.mean(features, axis=(0, 1))
    prediction = 1 if avg_color[0] > np.mean(avg_color[1:]) else 0
    
    # Calculate probabilities based on color differences
    diff = float(np.abs(avg_color[0] - np.mean(avg_color[1:])))
    prob = min(max(0.5 + diff, 0.1), 0.9)
    
    probabilities = [1 - prob, prob] if prediction == 1 else [prob, 1 - prob]
    
    return {
        "prediction": prediction,
        "healthy_probability": float(probabilities[0]),
        "diseased_probability": float(probabilities[1])
    } 