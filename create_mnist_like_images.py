import numpy as np
from sklearn.datasets import load_digits
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load MNIST-like dataset from scikit-learn
digits = load_digits()

# Create test_images directory if it doesn't exist
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# Function to save digit image
def save_digit_image(data, digit, index, size=(8, 8)):
    # Reshape the data to 8x8
    image_data = data.reshape(size)
    
    # Scale to 0-255 range
    image_data = ((image_data - image_data.min()) * (255.0 / (image_data.max() - image_data.min()))).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(image_data)
    
    # Save 8x8 version
    img.save(f'test_images/mnist_{digit}_{index}_8x8.png')
    
    # Save larger version for display
    img_large = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_large.save(f'test_images/mnist_{digit}_{index}.png')

# Save some example digits
for digit in range(10):
    # Get indices for this digit
    indices = np.where(digits.target == digit)[0]
    
    # Save first two examples of each digit
    for i in range(min(2, len(indices))):
        save_digit_image(digits.data[indices[i]], digit, i)

print("Created MNIST-like test images in the test_images directory.")
print("You can use these images to test the model.") 