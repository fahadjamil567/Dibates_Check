import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Disease Detection App",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the model architecture here instead of importing
class DiseaseNet(nn.Module):
    def __init__(self):
        super(DiseaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = DiseaseNet()
    model.load_state_dict(torch.load('disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def main():
    st.title("Disease Detection System")
    st.write("Upload an image for disease detection")

    try:
        model = load_model()
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            input_tensor = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            
            # Display results
            with col2:
                st.header("Prediction Results")
                
                if prediction.item() == 1:
                    st.error("⚠️ Disease Detected")
                else:
                    st.success("✅ No Disease Detected")
                
                # Display probabilities
                st.write("Confidence Scores:")
                st.write(f"- Healthy: {probabilities[0][0].item():.2%}")
                st.write(f"- Diseased: {probabilities[0][1].item():.2%}")
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Healthy', 'Diseased']
                probs = [probabilities[0][0].item(), probabilities[0][1].item()]
                ax.bar(labels, probs, color=['green', 'red'])
                ax.set_ylim(0, 1)
                plt.title('Prediction Probabilities')
                
                for i, v in enumerate(probs):
                    ax.text(i, v + 0.01, f'{v:.2%}', ha='center')
                
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the model file (disease_model.pth) is present in the current directory.")

if __name__ == "__main__":
    main() 