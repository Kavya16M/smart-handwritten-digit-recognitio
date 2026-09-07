# Smart Handwritten Digit Recognition System

An end-to-end Machine Learning and Deep Learning project for recognizing handwritten digits from 0 to 9.

The project compares multiple machine learning and deep learning models and integrates the best-performing CNN model into an interactive Flask web application.

## Features

- Draw a handwritten digit directly on the website
- Upload an image of a handwritten digit
- Predict digits from 0 to 9
- Display prediction confidence
- Image preprocessing before prediction
- Interactive web interface
- Comparison of five ML and Deep Learning models

## Dataset

The project uses the MNIST handwritten digit dataset.

- 60,000 training images
- 10,000 testing images
- 10 digit classes: 0–9
- Image size: 28 × 28 pixels
- Grayscale images

## Models Implemented

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Neural Network
5. Convolutional Neural Network (CNN)

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 92.57% | 92.56% | 92.57% | 92.56% |
| SVM | 98.37% | 98.37% | 98.37% | 98.37% |
| Random Forest | 97.04% | 97.04% | 97.04% | 97.04% |
| Neural Network | 97.69% | 97.70% | 97.69% | 97.69% |
| **CNN** | **99.06%** | **99.06%** | **99.06%** | **99.06%** |

CNN achieved the highest test accuracy and was selected for the final web application.

## CNN Architecture

The final CNN consists of:

- Convolutional layer with 32 filters
- Max Pooling layer
- Convolutional layer with 64 filters
- Max Pooling layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer
- Output layer with 10 classes

## How the Web Application Works

1. User draws a digit or uploads an image.
2. The image is converted to grayscale.
3. The digit region is detected and cropped.
4. The image is resized and centered into 28 × 28 format.
5. Pixel values are normalized.
6. The processed image is passed to the trained CNN.
7. The predicted digit and confidence score are displayed.

## Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Matplotlib
- Flask
- HTML
- CSS
- JavaScript

## Project Structure

```text
AI-Digit-Recognition/
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── experiments/
│   ├── 01_logistic_regression.py
│   ├── 02_svm.py
│   ├── 03_random_forest.py
│   ├── 04_neural_network.py
│   ├── 05_cnn.py
│   └── 06_digit_validator.py
│
├── models/
│   ├── digit_cnn.keras
│   └── digit_validator.keras
│
├── requirements.txt
├── .gitignore
└── README.md
