# Diabetes Risk Predictor

A web application that predicts diabetes risk based on various health parameters using machine learning.

## Features

- Input various health parameters
- Get instant risk assessment
- View confidence scores and probability distribution
- Analyze feature importance
- Sample test cases included

## Technologies Used

- Python 3.9+
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment on Vercel

1. Fork this repository
2. Sign up for Vercel (if you haven't already)
3. Create a new project and import your forked repository
4. Deploy with the following settings:
   - Framework Preset: Other
   - Build Command: None
   - Output Directory: None
   - Install Command: `pip install -r requirements.txt`

## Model Information

The application uses a Random Forest Classifier trained on the Pima Indians Diabetes Dataset. The model considers the following features:
- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age

## License

MIT License 