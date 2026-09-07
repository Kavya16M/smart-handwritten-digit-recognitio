import os
import io
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps


app = Flask(__name__)


# =========================
# MODEL PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DIGIT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "digit_cnn.keras"
)

VALIDATOR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "digit_validator.keras"
)


# =========================
# LOAD MODELS
# =========================

digit_model = tf.keras.models.load_model(
    DIGIT_MODEL_PATH
)

validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH
)

print("\n====================================")
print("Digit CNN loaded successfully.")
print("Digit validator loaded successfully.")
print("====================================\n")


# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(image):

    image = image.convert("L")

    image_array = np.array(image)

    # Invert if image has white background
    if np.mean(image_array) > 127:
        image = ImageOps.invert(image)

    image_array = np.array(image)

    # Remove very weak pixels
    image_array[image_array < 30] = 0

    # Find foreground
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)

    # Check blank image
    if not rows.any() or not cols.any():
        raise ValueError(
            "Please draw or upload a digit first."
        )

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    top = row_indices[0]
    bottom = row_indices[-1]

    left = col_indices[0]
    right = col_indices[-1]

    # Crop
    cropped = image_array[
        top:bottom + 1,
        left:right + 1
    ]

    cropped_image = Image.fromarray(cropped)

    # =========================
    # RESIZE
    # =========================

    width, height = cropped_image.size

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

    resized = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # =========================
    # CENTER ON 28x28
    # =========================

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    final_image.paste(
        resized,
        (x_offset, y_offset)
    )

    # =========================
    # NORMALIZE
    # =========================

    final_array = np.array(
        final_image
    ).astype("float32")

    final_array = final_array / 255.0

    final_array = final_array.reshape(
        1,
        28,
        28,
        1
    )

    return final_array


# =========================
# VALIDATE + PREDICT
# =========================

def validate_and_predict(processed_image):

    # -------------------------
    # VALIDATOR
    # -------------------------

    validator_result = validator_model.predict(
        processed_image,
        verbose=0
    )

    validator_score = float(
        validator_result[0][0]
    )

    print("\n---------- VALIDATION ----------")

    print(
        "Validator score:",
        validator_score
    )

    print(
        "Validator percentage:",
        round(validator_score * 100, 2),
        "%"
    )

    # Validator:
    # 1 = valid digit
    # 0 = not digit

    if validator_score < 0.5:

        print("Validator Result: NOT A DIGIT")
        print("--------------------------------\n")

        return {
            "valid": False,
            "message":
                "This does not appear to be a valid "
                "handwritten digit. Please draw or "
                "upload one digit from 0 to 9."
        }

    print("Validator Result: VALID DIGIT")

    # -------------------------
    # DIGIT CNN
    # -------------------------

    result = digit_model.predict(
        processed_image,
        verbose=0
    )

    predicted_digit = int(
        np.argmax(result)
    )

    confidence = round(
        float(np.max(result)) * 100,
        2
    )

    print(
        "Predicted Digit:",
        predicted_digit
    )

    print(
        "CNN Confidence:",
        confidence,
        "%"
    )

    print("--------------------------------\n")

    return {
        "valid": True,
        "prediction": predicted_digit,
        "confidence": confidence
    }


# =========================
# HOME / IMAGE UPLOAD
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        try:

            if "image" not in request.files:

                error = "Please select an image."

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error
                )

            file = request.files["image"]

            if file.filename == "":

                error = "Please select an image."

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error
                )

            image = Image.open(
                file.stream
            )

            processed_image = preprocess_image(
                image
            )

            output = validate_and_predict(
                processed_image
            )

            if not output["valid"]:

                error = output["message"]

            else:

                prediction = output[
                    "prediction"
                ]

                confidence = output[
                    "confidence"
                ]

        except ValueError as e:

            error = str(e)

        except Exception as e:

            print("Upload Error:", e)

            error = (
                "Unable to process the image. "
                "Please upload a clear handwritten digit."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# =========================
# CANVAS PREDICTION
# =========================

@app.route(
    "/predict-canvas",
    methods=["POST"]
)
def predict_canvas():

    try:

        data = request.get_json()

        if not data or "image" not in data:

            return jsonify({
                "error":
                    "Please draw a digit first."
            }), 400

        image_data = data["image"]

        # Remove base64 header
        if "," in image_data:
            image_data = image_data.split(
                ",",
                1
            )[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        processed_image = preprocess_image(
            image
        )

        output = validate_and_predict(
            processed_image
        )

        # Validator rejected image
        if not output["valid"]:

            return jsonify({
                "error": output["message"]
            }), 400

        # Valid digit
        return jsonify({
            "prediction":
                output["prediction"],

            "confidence":
                output["confidence"]
        })

    except ValueError as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:

        print("Canvas Error:", e)

        return jsonify({
            "error":
                "Unable to process the drawing."
        }), 500


# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
    