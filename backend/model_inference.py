from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(model_path, image_path):
    """
    Classifies a single image using the trained model.

    Parameters:
        model_path (str): Path to the trained model file.
        image_path (str): Path to the image to classify.

    Returns:
        str: Predicted class ("Cat" or "Dog").
    """
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"
