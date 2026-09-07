import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# PROJECT PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# =========================
# LOAD DATASET
# =========================

print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("MNIST dataset loaded.")


# =========================
# PREPROCESSING
# =========================

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(
    -1,
    28,
    28,
    1
)

X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# =========================
# BUILD CNN MODEL
# =========================

model = models.Sequential([

    layers.Input(
        shape=(28, 28, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# =========================
# MODEL SUMMARY
# =========================

print("\n===== CNN MODEL SUMMARY =====")

model.summary()


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# EARLY STOPPING
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# =========================
# TRAIN MODEL
# =========================

print("\n===== TRAINING CNN =====")

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping]
)


# =========================
# SAVE TRAINED MODEL
# =========================

model_path = os.path.join(
    MODELS_DIR,
    "digit_cnn.keras"
)

model.save(
    model_path
)

print("\n===== MODEL SAVED =====")

print("CNN model saved successfully!")

print(
    "Saved at:",
    model_path
)


# =========================
# PREDICTION
# =========================

print("\nMaking predictions...")

y_pred_probs = model.predict(
    X_test
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

print("Prediction completed.")


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n===== CNN METRICS =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


# =========================
# CONFUSION MATRIX VISUAL
# =========================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - CNN"
)

plt.colorbar()

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    range(10)
)

plt.yticks(
    range(10)
)

for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()


# =========================
# SAVE CONFUSION MATRIX
# =========================

confusion_path = os.path.join(
    RESULTS_DIR,
    "cnn_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nConfusion matrix saved at:",
    confusion_path
)

plt.show()


# =========================
# ACCURACY GRAPH
# =========================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Accuracy vs Epochs - CNN"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# =========================
# SAVE ACCURACY GRAPH
# =========================

accuracy_path = os.path.join(
    RESULTS_DIR,
    "cnn_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Accuracy graph saved at:",
    accuracy_path
)

plt.show()


# =========================
# LOSS GRAPH
# =========================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Loss vs Epochs - CNN"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# =========================
# SAVE LOSS GRAPH
# =========================

loss_path = os.path.join(
    RESULTS_DIR,
    "cnn_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Loss graph saved at:",
    loss_path
)

plt.show()


# =========================
# SAMPLE PREDICTION
# =========================

index = 0

sample_image = X_test[index]

sample_prediction = model.predict(
    sample_image.reshape(
        1,
        28,
        28,
        1
    ),
    verbose=0
)

predicted_digit = np.argmax(
    sample_prediction
)

confidence = np.max(
    sample_prediction
)


print("\n===== SAMPLE PREDICTION =====")

print(
    "Actual Digit:",
    y_test[index]
)

print(
    "Predicted Digit:",
    predicted_digit
)

print(
    "Confidence:",
    confidence * 100,
    "%"
)


# =========================
# SAMPLE IMAGE
# =========================

plt.figure(
    figsize=(4, 4)
)

plt.imshow(
    sample_image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"Actual: {y_test[index]} | "
    f"Predicted: {predicted_digit}"
)

plt.axis(
    "off"
)

plt.tight_layout()


# =========================
# SAVE SAMPLE PREDICTION
# =========================

sample_path = os.path.join(
    RESULTS_DIR,
    "cnn_sample_prediction.png"
)

plt.savefig(
    sample_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Sample prediction saved at:",
    sample_path
)

plt.show()


# =========================
# COMPLETED
# =========================

print(
    "\nCNN experiment completed successfully."
)