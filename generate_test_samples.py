import numpy as np
import cv2
import os

def create_test_image(size=(64, 64), image_type='normal'):
    """Create a test image with specific characteristics"""
    # Create base image
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add base texture
    noise = np.random.normal(0, 15, size + (3,)).astype(np.uint8)
    image = cv2.add(image, noise)
    
    if image_type == 'normal':
        # Add healthy tissue patterns
        for _ in range(5):
            x = np.random.randint(10, size[0]-20)
            y = np.random.randint(10, size[1]-20)
            w = np.random.randint(8, 12)
            h = np.random.randint(8, 12)
            color = (180, 180, 180)
            cv2.ellipse(image, (x, y), (w, h), 
                       np.random.randint(0, 180), 0, 360, color, 1)
    
    elif image_type == 'early_stage':
        # Add healthy patterns and one small spot
        for _ in range(4):
            x = np.random.randint(10, size[0]-20)
            y = np.random.randint(10, size[1]-20)
            w = np.random.randint(8, 12)
            h = np.random.randint(8, 12)
            color = (180, 180, 180)
            cv2.ellipse(image, (x, y), (w, h), 
                       np.random.randint(0, 180), 0, 360, color, 1)
        
        # Add one small disease marker
        x = np.random.randint(20, size[0]-20)
        y = np.random.randint(20, size[1]-20)
        cv2.circle(image, (x, y), 4, (100, 50, 50), -1)
        cv2.circle(image, (x, y), 5, (150, 100, 100), 1)
    
    elif image_type == 'advanced':
        # Add multiple disease markers
        for _ in range(4):
            x = np.random.randint(15, size[0]-15)
            y = np.random.randint(15, size[1]-15)
            radius = np.random.randint(5, 8)
            color = (100, 50, 50)
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius+1, (150, 100, 100), 1)
    
    # Resize to a larger size for better visibility
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    return image

def generate_test_samples():
    """Generate test samples with different characteristics"""
    # Create test_samples directory if it doesn't exist
    os.makedirs('test_samples', exist_ok=True)
    
    # Generate multiple samples of each type
    types = {
        'normal': 3,
        'early_stage': 3,
        'advanced': 3
    }
    
    print("Generating test samples...")
    for img_type, count in types.items():
        for i in range(count):
            img = create_test_image(image_type=img_type)
            filename = f'test_samples/{img_type}_{i+1}.png'
            cv2.imwrite(filename, img)
            print(f"Created {filename}")

if __name__ == "__main__":
    print("Starting test sample generation...")
    generate_test_samples()
    print("\nTest samples have been generated in the 'test_samples' directory.")
    print("You can use these images to test the disease detection system:") 