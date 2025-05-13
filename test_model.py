import tensorflow as tf
from tensorflow import keras

model_path = 'lung_cancer_model.h5'

try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()  # Print model summary to verify the architecture
except Exception as e:
    print(f"Error loading model: {e}")
