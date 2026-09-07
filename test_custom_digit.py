import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageOps


# =========================
# LOAD SAVED MODEL
# =========================

print("Loading saved CNN model...")

model = tf.keras.models.load_model(
    "models/digit_cnn.keras"
)

print("Model loaded successfully!")


# =========================
# LOAD IMAGE
# =========================

image = Image.open(
    "my_digit.png"
).convert("L")


# =========================
# INVERT IMAGE IF NEEDED
# =========================

image_array = np.array(image)

# If background is mostly white,
# convert it to MNIST style:
# black background + white digit

if np.mean(image_array) > 127:
    image = ImageOps.invert(image)

image_array = np.array(image)


# =========================
# REMOVE BACKGROUND
# =========================

# Remove very dark background noise
image_array[image_array < 30] = 0


# =========================
# FIND DIGIT
# =========================

rows = np.any(
    image_array > 30,
    axis=1
)

cols = np.any(
    image_array > 30,
    axis=0
)

if not np.any(rows) or not np.any(cols):
    print("No digit detected in the image.")
    exit()


row_indices = np.where(rows)[0]
col_indices = np.where(cols)[0]

top = row_indices[0]
bottom = row_indices[-1]

left = col_indices[0]
right = col_indices[-1]


# =========================
# CROP DIGIT
# =========================

cropped = image_array[
    top:bottom + 1,
    left:right + 1
]


# =========================
# RESIZE DIGIT
# =========================

cropped_image = Image.fromarray(
    cropped.astype(np.uint8)
)

width, height = cropped_image.size

# MNIST digits do not completely fill 28x28.
# Resize digit to fit inside approximately 20x20.

if width > height:

    new_width = 20

    new_height = max(
        1,
        int(height * (20 / width))
    )

else:

    new_height = 20

    new_width = max(
        1,
        int(width * (20 / height))
    )


cropped_image = cropped_image.resize(
    (new_width, new_height)
)


# =========================
# CREATE 28 x 28 CANVAS
# =========================

final_image = Image.new(
    "L",
    (28, 28),
    0
)


# =========================
# CENTER DIGIT
# =========================

x_position = (
    28 - new_width
) // 2

y_position = (
    28 - new_height
) // 2


final_image.paste(
    cropped_image,
    (
        x_position,
        y_position
    )
)


# =========================
# NORMALIZE
# =========================

processed_image = np.array(
    final_image
).astype("float32")

processed_image = (
    processed_image / 255.0
)


# =========================
# RESHAPE FOR CNN
# =========================

model_input = processed_image.reshape(
    1,
    28,
    28,
    1
)


# =========================
# PREDICTION
# =========================

prediction = model.predict(
    model_input
)

predicted_digit = np.argmax(
    prediction
)

confidence = np.max(
    prediction
)


# =========================
# DISPLAY RESULT
# =========================

print("\n===== CUSTOM DIGIT TEST =====")

print(
    "Predicted Digit:",
    predicted_digit
)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)


# =========================
# DISPLAY PROCESSED IMAGE
# =========================

plt.imshow(
    processed_image,
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_digit} | "
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")

plt.show()