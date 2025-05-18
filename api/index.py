from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
from PIL import Image
import onnxruntime as ort

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model
session = ort.InferenceSession('disease_model.onnx')

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((64, 64))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW format
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    
    # Get probabilities using softmax
    scores = outputs[0][0]
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()
    
    prediction = int(np.argmax(probabilities))
    
    return {
        "prediction": prediction,
        "healthy_probability": float(probabilities[0]),
        "diseased_probability": float(probabilities[1])
    } 