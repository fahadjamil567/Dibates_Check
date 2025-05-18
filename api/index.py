from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler

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
    
    # Flatten the image
    features = img_array.reshape(1, -1)
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
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
    
    # For demo purposes, return a mock prediction
    # In production, you would load and use your trained model here
    prediction = int(np.random.choice([0, 1], p=[0.7, 0.3]))
    probabilities = np.random.dirichlet([5, 2]) if prediction == 0 else np.random.dirichlet([2, 5])
    
    return {
        "prediction": prediction,
        "healthy_probability": float(probabilities[0]),
        "diseased_probability": float(probabilities[1])
    } 