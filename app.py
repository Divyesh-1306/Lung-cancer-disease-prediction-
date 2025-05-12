from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('lung_cancer_model.h5')

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Define class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Function to preprocess the image
def preprocess_image(image_data):
    try:
        image = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image)).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.get_json()['image']
        image_data = base64.b64decode(image_data)

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        if processed_image is None:
            return jsonify({'error': 'Could not preprocess image'}), 400

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        # Return the prediction as JSON
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    pass
