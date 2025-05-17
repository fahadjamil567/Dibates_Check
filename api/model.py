import numpy as np
import json

class SimpleModel:
    def __init__(self, weights, bias, scaler_mean, scaler_scale):
        self.weights = np.array(weights)
        self.bias = bias
        self.scaler_mean = np.array(scaler_mean)
        self.scaler_scale = np.array(scaler_scale)
    
    def predict_proba(self, X):
        # Scale the input
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        # Calculate logits
        logits = np.dot(X_scaled, self.weights) + self.bias
        # Convert to probability using sigmoid
        prob = 1 / (1 + np.exp(-logits))
        return prob

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    @classmethod
    def from_sklearn(cls, sklearn_model, scaler):
        """Convert sklearn model to simple model"""
        # Get coefficients and intercept from the sklearn model
        weights = sklearn_model.coef_[0] if hasattr(sklearn_model, 'coef_') else sklearn_model.feature_importances_
        bias = sklearn_model.intercept_[0] if hasattr(sklearn_model, 'intercept_') else 0
        
        # Get scaler parameters
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        
        return cls(weights.tolist(), float(bias), scaler_mean.tolist(), scaler_scale.tolist())

    def to_json(self):
        """Serialize model to JSON"""
        return json.dumps({
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'scaler_mean': self.scaler_mean.tolist(),
            'scaler_scale': self.scaler_scale.tolist()
        })

    @classmethod
    def from_json(cls, json_str):
        """Create model from JSON"""
        data = json.loads(json_str)
        return cls(
            data['weights'],
            data['bias'],
            data['scaler_mean'],
            data['scaler_scale']
        ) 