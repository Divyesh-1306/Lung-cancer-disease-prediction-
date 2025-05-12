# Lung Cancer Detection

This project is a deep learning application for detecting lung cancer from medical images.

## Overview

The application uses a convolutional neural network (CNN) model trained on the IQ-OTHNCCD lung cancer dataset to classify images as either cancerous or non-cancerous.

## Files

- `app.py`: Main application file.
- `streamlit_app.py`: Streamlit application file.
- `train.py`: Script for training the CNN model.
- `test_model.py`: Script for testing the trained model.
- `lung_cancer_model.h5`: Trained CNN model.
- `Test cases/`: Directory containing sample test images.
- `tensorflow/`: Directory containing tensorflow wheel file.
- `The IQ-OTHNCCD lung cancer dataset/`: Dataset used for training and testing.

## Prerequisites

- Python 3.9
- TensorFlow
- Streamlit
- Other dependencies (install using `pip install -r requirements.txt`)

## Installation

1.  Install the required dependencies:

    ```bash
    pip install tensorflow-2.19.0-cp39-cp39-win_amd64.whl
    pip install -r requirements.txt
    ```
2.  Download the IQ-OTHNCCD lung cancer dataset and place it in the project directory.

## Usage

1.  Train the model:

    ```bash
    python train.py
    ```
2.  Test the model:

    ```bash
    python test_model.py
    ```
3.  Run the Streamlit application:

    ```bash
    streamlit run streamlit_app.py
    ```

## Dataset

The IQ-OTHNCCD lung cancer dataset was used to train and test the model.

## Test Cases

The `Test cases/` directory contains sample images for testing the application.

## Model

The trained CNN model is saved as `lung_cancer_model.h5`.

## Contributing

Contributions are welcome! Please submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
