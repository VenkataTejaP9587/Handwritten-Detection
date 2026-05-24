import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("model/language_model.h5")

# Load dynamically saved class names
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

def predict_language(image):
    """
    Predict the language of a handwritten text image.

    Args:
        image: BGR image (numpy array) from OpenCV

    Returns:
        tuple: (language_name, confidence_score)
    """
    # Ensure image is BGR (3-channel) for MobileNetV2
    if len(image.shape) == 2:
        # Grayscale → BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # BGRA → BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Resize to model input size
    img = cv2.resize(image, (128, 128))

    # Convert BGR to RGB (MobileNetV2 expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply MobileNetV2 preprocessing
    img = img.astype("float32")
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[index], confidence