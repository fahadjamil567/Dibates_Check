import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the existing model
try:
    existing_model = joblib.load('digit_model.joblib')
    print("Testing existing model...")
    
    # Test the existing model
    existing_predictions = existing_model.predict(X_test)
    existing_accuracy = accuracy_score(y_test, existing_predictions)
    print(f"\nExisting Model Accuracy: {existing_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, existing_predictions))
    
    # Plot confusion matrix for existing model
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, existing_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Existing Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('existing_model_confusion_matrix.png')
    plt.close()
    
except:
    print("No existing model found or error loading model.")
    existing_accuracy = 0

# Train a new model with optimized parameters
print("\nTraining new model with optimized parameters...")
new_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Perform cross-validation
cv_scores = cross_val_score(new_model, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the new model
new_model.fit(X_train, y_train)

# Test the new model
new_predictions = new_model.predict(X_test)
new_accuracy = accuracy_score(y_test, new_predictions)
print(f"\nNew Model Accuracy: {new_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, new_predictions))

# Plot confusion matrix for new model
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, new_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - New Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('new_model_confusion_matrix.png')
plt.close()

# Save the better model
if new_accuracy > existing_accuracy:
    print("\nNew model performs better. Saving new model...")
    joblib.dump(new_model, 'digit_model.joblib')
    print("New model saved as 'digit_model.joblib'")
else:
    print("\nExisting model performs better. Keeping existing model.")

# Test with our generated test images
print("\nTesting with generated test images...")
from PIL import Image
import os

def load_and_preprocess_test_image(file_path):
    img = Image.open(file_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array.flatten()

test_dir = 'test_images'
for file in os.listdir(test_dir):
    if file.endswith('_8x8.png'):
        img_path = os.path.join(test_dir, file)
        img_vector = load_and_preprocess_test_image(img_path)
        
        # Get predictions from both models
        if existing_accuracy > 0:
            existing_pred = existing_model.predict([img_vector])[0]
            existing_conf = existing_model.predict_proba([img_vector])[0].max() * 100
        
        new_pred = new_model.predict([img_vector])[0]
        new_conf = new_model.predict_proba([img_vector])[0].max() * 100
        
        print(f"\nTest image: {file}")
        if existing_accuracy > 0:
            print(f"Existing model prediction: {existing_pred} (confidence: {existing_conf:.2f}%)")
        print(f"New model prediction: {new_pred} (confidence: {new_conf:.2f}%)") 