import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances ({title})")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{title.lower().replace(" ", "_")}.png')
    plt.close()

# Load and prepare the dataset
print("Loading Diabetes dataset...")
# Pima Indians Diabetes Dataset
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', 
                  names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

# Prepare features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']
feature_names = X.columns

# Split the data with stratification
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline with preprocessing and model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'class_weight': ['balanced'],
    'criterion': ['gini']
}

# Create and train the model
print("Training model with GridSearchCV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Print training results
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest set accuracy:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, "Diabetes Prediction Model")

# Plot feature importance
plot_feature_importance(best_model, feature_names, "Diabetes Prediction")

# Save the model and scaler
print("\nSaving model and scaler...")
joblib.dump(best_model, 'disease_model.joblib')
joblib.dump(scaler, 'disease_scaler.joblib')

print("\nModel Training Summary:")
print("-----------------------")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nBest Model Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# Create sample test cases
print("\nCreating sample test cases...")
sample_cases = pd.DataFrame([
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],    # Healthy case
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],   # Diabetic case
    [3, 111, 58, 31, 44, 29.5, 0.430, 22],  # Borderline case
], columns=feature_names)

# Save sample cases
sample_cases.to_csv('sample_cases.csv', index=False)
print("\nSample test cases saved to 'sample_cases.csv'")

print("\nTraining complete. Model and scaler saved as 'disease_model.joblib' and 'disease_scaler.joblib'")
print("You can now use the model with the Streamlit app (app.py)")
