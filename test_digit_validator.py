import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load validator model
validator = tf.keras.models.load_model("models/digit_validator.keras")


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")

    # Invert if background is white
    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image = np.array(image).astype("float32") / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image


# Change this filename to the image you want to test
image_path = "scribble.png"

image = preprocess_image(image_path)

prediction = validator.predict(image, verbose=0)[0][0]

print("Validator output:", prediction)

if prediction >= 0.5:
    print("Result: VALID DIGIT")
else:
    print("Result: NOT A DIGIT")
