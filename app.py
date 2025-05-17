import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Instructions")
    st.write("""
    1. Enter your health parameters in the form
    2. Click 'Predict' to see your diabetes risk assessment
    3. View the prediction and confidence score
    """)
    
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

# Main content
st.title("🏥 Diabetes Risk Predictor")

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('disease_model.joblib')
    scaler = joblib.load('disease_scaler.joblib')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()

    # Create input form
    st.subheader("Enter Your Health Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=85)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=66)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=29)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.6)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.351)
        age = st.number_input("Age", min_value=0, max_value=120, value=31)

    # Create feature array
    features = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]])

    # Add predict button
    if st.button("Predict"):
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction[0] == 1:
                st.error("⚠️ High Risk of Diabetes")
            else:
                st.success("✅ Low Risk of Diabetes")
                
            st.write("---")
            st.write("Confidence Scores:")
            st.write(f"Low Risk: {prediction_proba[0][0]:.2%}")
            st.write(f"High Risk: {prediction_proba[0][1]:.2%}")
        
        with result_col2:
            # Create probability distribution plot
            fig, ax = plt.subplots(figsize=(8, 4))
            probas = prediction_proba[0]
            ax.bar(['Low Risk', 'High Risk'], probas, color=['green', 'red'])
            ax.set_ylim(0, 1)
            ax.set_title('Probability Distribution')
            for i, v in enumerate(probas):
                ax.text(i, v + 0.01, f'{v:.2%}', ha='center')
            st.pyplot(fig)
            plt.close()

        # Show feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                           'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            ax.set_title('Feature Importance for Prediction')
            st.pyplot(fig)
            plt.close()

    # Load and display sample cases
    st.subheader("Sample Test Cases")
    try:
        sample_cases = pd.read_csv('sample_cases.csv')
        st.write("You can try these sample cases:")
        st.dataframe(sample_cases)
    except:
        st.write("Sample cases file not found.")

except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.write("Please make sure the model files (disease_model.joblib and disease_scaler.joblib) exist in the current directory.") 