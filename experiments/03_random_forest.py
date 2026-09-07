import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

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

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(np.int64)

X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

print("MNIST dataset loaded.")


# =========================
# PREPROCESSING
# =========================

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# =========================
# BUILD MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# =========================
# TRAIN MODEL
# =========================

print("\nTraining Random Forest...")
print("This may take some time...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================
# PREDICTION
# =========================

print("\nMaking predictions...")

y_pred = model.predict(
    X_test
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


print("\n===== RANDOM FOREST METRICS =====")

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
    cmap="Oranges"
)

plt.title(
    "Confusion Matrix - Random Forest"
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
    "random_forest_confusion_matrix.png"
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
# PERFORMANCE GRAPH
# =========================

metrics = [
    accuracy,
    precision,
    recall,
    f1
]

names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    names,
    metrics
)

plt.ylim(
    0,
    1
)

plt.title(
    "Random Forest Performance"
)

plt.ylabel(
    "Score"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# =========================
# SAVE PERFORMANCE GRAPH
# =========================

metrics_path = os.path.join(
    RESULTS_DIR,
    "random_forest_metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Metrics graph saved at:",
    metrics_path
)

plt.show()


# =========================
# COMPLETED
# =========================

print(
    "\nRandom Forest experiment completed successfully."
)