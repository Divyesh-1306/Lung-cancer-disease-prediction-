import os
import requests
os.system("chmod +x setup.sh")
import tensorflow as tf 
import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import io

# Define the model path and Google Drive link
model_path = os.path.join(os.getcwd(), 'lung_cancer_model.h5')
model_url = "https://drive.google.com/uc?export=download&id=120kjABeUv9lzs6U9S9xa14KCXY2cMtwb"  # Replace with your direct link

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    st.write("Downloading model...")
    try:
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

# Load the trained model
try:
    model = keras.models.load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Define class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Function to preprocess the image
def preprocess_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app
st.title('Lung Cancer Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Make a prediction
    processed_image = preprocess_image(uploaded_file)

    if processed_image is not None:
        try:
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]

            st.write(f'## Prediction: {predicted_label}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")
