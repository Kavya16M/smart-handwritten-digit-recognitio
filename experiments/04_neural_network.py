import numpy as np
import matplotlib.pyplot as plt
import os

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
# RESULTS PATH
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
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

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# =========================
# BUILD MODEL
# =========================

model = models.Sequential([

    layers.Input(shape=(784,)),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# MODEL SUMMARY
# =========================

print("\n===== MODEL SUMMARY =====")

model.summary()


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

print("\n===== TRAINING NEURAL NETWORK =====")

history = model.fit(

    X_train,
    y_train,

    epochs=15,

    batch_size=32,

    validation_split=0.1,

    callbacks=[early_stopping]
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


print("\n===== NEURAL NETWORK METRICS =====")

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

plt.figure(figsize=(8, 6))

plt.imshow(
    cm,
    cmap="Greens"
)

plt.title(
    "Confusion Matrix - Neural Network"
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
    "neural_network_confusion_matrix.png"
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

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Accuracy vs Epochs - Neural Network"
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
    "neural_network_accuracy.png"
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

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Loss vs Epochs - Neural Network"
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
    "neural_network_loss.png"
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
# COMPLETED
# =========================

print(
    "\nNeural Network experiment completed successfully."
)
