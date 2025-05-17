import joblib
from api.model import SimpleModel

# Load the original model and scaler
model = joblib.load('disease_model.joblib')
scaler = joblib.load('disease_scaler.joblib')

# Convert to simple model
simple_model = SimpleModel.from_sklearn(model, scaler)

# Save as JSON
with open('api/model_data.json', 'w') as f:
    f.write(simple_model.to_json()) 