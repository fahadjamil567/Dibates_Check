from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "healthy"})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and process image
        image = Image.open(file.stream)
        image = image.resize((64, 64))
        img_array = np.array(image)
        
        # Simple mock prediction
        avg_color = np.mean(img_array, axis=(0, 1))
        prediction = 1 if np.mean(avg_color) > 127 else 0
        
        return jsonify({
            'prediction': prediction,
            'healthy_probability': 0.8 if prediction == 0 else 0.2,
            'diseased_probability': 0.2 if prediction == 0 else 0.8
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 