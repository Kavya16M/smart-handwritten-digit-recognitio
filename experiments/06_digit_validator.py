import os
import random

import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# RANDOM SEEDS
# =========================

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# =========================
# LOAD MNIST
# =========================

print("Loading MNIST digits...")

(X_train, _), (X_test, _) = mnist.load_data()

print("MNIST loaded successfully.")


# =========================
# CREATE INVALID IMAGE
# =========================

def create_invalid_image():

    image = Image.new(
        "L",
        (28, 28),
        0
    )

    draw = ImageDraw.Draw(image)

    invalid_type = random.choice([
        "scribble",
        "lines",
        "circle",
        "rectangle",
        "noise",
        "blank"
    ])


    # Random scribble
    if invalid_type == "scribble":

        points = []

        for _ in range(
            random.randint(5, 15)
        ):

            points.append(
                (
                    random.randint(1, 26),
                    random.randint(1, 26)
                )
            )

        draw.line(
            points,
            fill=255,
            width=random.randint(2, 5)
        )


    # Random lines
    elif invalid_type == "lines":

        for _ in range(
            random.randint(2, 6)
        ):

            draw.line(
                (
                    random.randint(1, 26),
                    random.randint(1, 26),
                    random.randint(1, 26),
                    random.randint(1, 26)
                ),
                fill=255,
                width=random.randint(1, 4)
            )


    # Circle
    elif invalid_type == "circle":

        x1 = random.randint(2, 8)
        y1 = random.randint(2, 8)

        x2 = random.randint(19, 26)
        y2 = random.randint(19, 26)

        draw.ellipse(
            (
                x1,
                y1,
                x2,
                y2
            ),
            outline=255,
            width=random.randint(2, 4)
        )


    # Rectangle
    elif invalid_type == "rectangle":

        x1 = random.randint(2, 8)
        y1 = random.randint(2, 8)

        x2 = random.randint(19, 26)
        y2 = random.randint(19, 26)

        draw.rectangle(
            (
                x1,
                y1,
                x2,
                y2
            ),
            outline=255,
            width=random.randint(2, 4)
        )


    # Random noise
    elif invalid_type == "noise":

        noise = np.random.randint(
            0,
            256,
            (28, 28),
            dtype=np.uint8
        )

        return noise


    # Blank / almost blank
    elif invalid_type == "blank":

        if random.random() > 0.5:

            x = random.randint(2, 25)
            y = random.randint(2, 25)

            draw.ellipse(
                (
                    x,
                    y,
                    min(x + 2, 27),
                    min(y + 2, 27)
                ),
                fill=255
            )


    return np.array(image)


# =========================
# PREPARE VALID DIGITS
# =========================

print("Preparing valid digit images...")

valid_train = X_train.copy()

valid_test = X_test.copy()


# =========================
# CREATE INVALID DATA
# =========================

print("Generating invalid training images...")

invalid_train = np.array([
    create_invalid_image()
    for _ in range(len(valid_train))
])


print("Generating invalid testing images...")

invalid_test = np.array([
    create_invalid_image()
    for _ in range(len(valid_test))
])


# =========================
# CREATE LABELS
# =========================

# 1 = valid digit
# 0 = not a digit

valid_train_labels = np.ones(
    len(valid_train),
    dtype=np.float32
)

invalid_train_labels = np.zeros(
    len(invalid_train),
    dtype=np.float32
)


valid_test_labels = np.ones(
    len(valid_test),
    dtype=np.float32
)

invalid_test_labels = np.zeros(
    len(invalid_test),
    dtype=np.float32
)


# =========================
# COMBINE DATA
# =========================

X_validator_train = np.concatenate(
    [
        valid_train,
        invalid_train
    ]
)

y_validator_train = np.concatenate(
    [
        valid_train_labels,
        invalid_train_labels
    ]
)


X_validator_test = np.concatenate(
    [
        valid_test,
        invalid_test
    ]
)

y_validator_test = np.concatenate(
    [
        valid_test_labels,
        invalid_test_labels
    ]
)


# =========================
# SHUFFLE DATA
# =========================

train_indices = np.random.permutation(
    len(X_validator_train)
)

X_validator_train = X_validator_train[
    train_indices
]

y_validator_train = y_validator_train[
    train_indices
]


test_indices = np.random.permutation(
    len(X_validator_test)
)

X_validator_test = X_validator_test[
    test_indices
]

y_validator_test = y_validator_test[
    test_indices
]


# =========================
# NORMALIZE
# =========================

X_validator_train = (
    X_validator_train.astype("float32")
    / 255.0
)

X_validator_test = (
    X_validator_test.astype("float32")
    / 255.0
)


# =========================
# RESHAPE
# =========================

X_validator_train = X_validator_train.reshape(
    -1,
    28,
    28,
    1
)

X_validator_test = X_validator_test.reshape(
    -1,
    28,
    28,
    1
)


print(
    "Training samples:",
    X_validator_train.shape
)

print(
    "Testing samples:",
    X_validator_test.shape
)


# =========================
# BUILD VALIDATOR CNN
# =========================

validator = models.Sequential([

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
        64,
        activation="relu"
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# =========================
# MODEL SUMMARY
# =========================

print(
    "\n===== DIGIT VALIDATOR MODEL ====="
)

validator.summary()


# =========================
# COMPILE
# =========================

validator.compile(

    optimizer="adam",

    loss="binary_crossentropy",

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
# TRAIN
# =========================

print(
    "\n===== TRAINING DIGIT VALIDATOR ====="
)

history = validator.fit(

    X_validator_train,

    y_validator_train,

    epochs=12,

    batch_size=128,

    validation_split=0.1,

    callbacks=[
        early_stopping
    ]
)


# =========================
# TEST
# =========================

print(
    "\n===== TESTING DIGIT VALIDATOR ====="
)

test_loss, test_accuracy = validator.evaluate(

    X_validator_test,

    y_validator_test,

    verbose=0
)


print(
    "Validator Test Accuracy:",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


# =========================
# SAVE MODEL
# =========================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = os.path.abspath(
    "models/digit_validator.keras"
)

validator.save(
    model_path
)


print(
    "\nDigit validator saved successfully!"
)

print(
    "Saved at:",
    model_path
)
