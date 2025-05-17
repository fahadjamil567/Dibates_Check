import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import os

# Create test_images directory if it doesn't exist
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Function to save digit image
def save_digit_image(data, label, index, folder='test_images'):
    # Reshape the data to 8x8 image
    image = data.reshape(8, 8)
    
    # Create figure with white background
    plt.figure(figsize=(3, 3), facecolor='white')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Save with high DPI for better quality
    filename = f'{folder}/digit_{label}_sample_{index}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=300)
    plt.close()

# Generate multiple samples for each digit
samples_per_digit = 3
print("Generating test images...")

for digit in range(10):
    # Get indices for this digit
    digit_indices = np.where(y == digit)[0]
    
    # Select random samples
    selected_indices = np.random.choice(digit_indices, samples_per_digit, replace=False)
    
    for idx, sample_idx in enumerate(selected_indices):
        save_digit_image(X[sample_idx], digit, idx)

print(f"Generated {samples_per_digit} samples for each digit (0-9)")
print(f"Images saved in the 'test_images' directory") 