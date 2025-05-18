import numpy as np
import cv2
import os

def create_synthetic_image(size=(64, 64), is_diseased=False):
    """Create a synthetic medical image"""
    # Create base image
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add tissue-like texture
    noise = np.random.normal(0, 20, size + (3,)).astype(np.uint8)
    image = cv2.add(image, noise)
    
    if is_diseased:
        # Add disease markers (dark spots)
        num_spots = np.random.randint(3, 7)
        for _ in range(num_spots):
            x = np.random.randint(5, size[0]-15)
            y = np.random.randint(5, size[1]-15)
            radius = np.random.randint(3, 8)
            color = (100, 50, 50)  # Dark reddish spots
            cv2.circle(image, (x, y), radius, color, -1)
            # Add some texture to the spots
            cv2.circle(image, (x, y), radius+2, (150, 100, 100), 1)
    else:
        # Add healthy tissue patterns
        num_patterns = np.random.randint(4, 8)
        for _ in range(num_patterns):
            x = np.random.randint(5, size[0]-15)
            y = np.random.randint(5, size[1]-15)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            color = (180, 180, 180)  # Light gray patterns
            cv2.ellipse(image, (x, y), (w, h), 
                       np.random.randint(0, 180), 0, 360, color, 1)
    
    return image

def generate_dataset(num_images=100):
    """Generate training dataset"""
    # Create directories if they don't exist
    os.makedirs('train_images/healthy', exist_ok=True)
    os.makedirs('train_images/diseased', exist_ok=True)
    
    print("Generating healthy images...")
    for i in range(num_images):
        img = create_synthetic_image(is_diseased=False)
        cv2.imwrite(f'train_images/healthy/healthy_{i}.png', img)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} healthy images")
    
    print("\nGenerating diseased images...")
    for i in range(num_images):
        img = create_synthetic_image(is_diseased=True)
        cv2.imwrite(f'train_images/diseased/diseased_{i}.png', img)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} diseased images")

if __name__ == "__main__":
    print("Starting synthetic image generation...")
    generate_dataset(100)  # Generate 100 images per class
    print("\nImage generation complete!")
    print("- Check train_images/healthy for healthy images")
    print("- Check train_images/diseased for diseased images") 