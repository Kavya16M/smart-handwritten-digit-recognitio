import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist


print("Loading saved CNN model...")

model = tf.keras.models.load_model(
    "models/digit_cnn.keras"
)

print("Model loaded successfully!")


# Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()


# Normalize
X_test = X_test / 255.0


# Reshape for CNN
X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)


# Select one image
index = 0

sample_image = X_test[index]


# Predict
prediction = model.predict(
    sample_image.reshape(
        1,
        28,
        28,
        1
    )
)


predicted_digit = np.argmax(
    prediction
)

confidence = np.max(
    prediction
)


print("\n===== SAVED MODEL TEST =====")

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


# Display image
plt.imshow(
    sample_image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"Actual: {y_test[index]} | "
    f"Predicted: {predicted_digit}"
)

plt.axis("off")

plt.show()