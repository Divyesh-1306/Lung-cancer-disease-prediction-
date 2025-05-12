import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers

# Define data directories
DATA_DIR = 'The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
BENIGN_DIR = os.path.join(DATA_DIR, 'Bengin cases')
MALIGNANT_DIR = os.path.join(DATA_DIR, 'Malignant cases')
NORMAL_DIR = os.path.join(DATA_DIR, 'Normal cases')

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Function to load and preprocess images
def load_images(directory, label):
    images = []
    labels = []
    print(f"Loading images from directory: {directory}")
    for filename in os.listdir(directory):
        print(f"Processing file: {filename}")
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
                img_array = np.array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images, labels

# Load images and labels
benign_images, benign_labels = load_images(BENIGN_DIR, 0)  # 0 for Benign
malignant_images, malignant_labels = load_images(MALIGNANT_DIR, 1)  # 1 for Malignant
normal_images, normal_labels = load_images(NORMAL_DIR, 2)  # 2 for Normal

print(f"Number of benign images: {len(benign_images)}")
print(f"Number of malignant images: {len(malignant_images)}")
print(f"Number of normal images: {len(normal_images)}")

# Combine data
all_images = benign_images + malignant_images + normal_images
all_labels = benign_labels + malignant_labels + normal_labels

# Convert to numpy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

print("Data loading and preprocessing complete.")

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes: Benign, Malignant, Normal
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model definition complete.")

# Train the model
epochs = 10
batch_size = 32

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test))

print("Model training complete.")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save('lung_cancer_model.h5')
print("Model saved to lung_cancer_model.h5")
