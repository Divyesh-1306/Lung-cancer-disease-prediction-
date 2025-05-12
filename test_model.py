import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model('lung_cancer_model.h5')

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Example usage:
image_path = 'The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Normal cases/Normal case (11).jpg'  # Replace with the path to your image
processed_image = load_and_preprocess_image(image_path)

if processed_image is not None:
    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # Define class labels
    class_labels = ['Benign', 'Malignant', 'Normal']
    predicted_label = class_labels[predicted_class]

    print(f"The predicted class is: {predicted_label}")
