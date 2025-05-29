from MyImports import *

# Specifying the image size to resize all images
image_size = (256, 256)

import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_uploaded_image(uploaded_file, target_size=(224, 224), normalize=True):
    """
    Preprocesses an uploaded image file for model prediction.

    Parameters:
        uploaded_file: A file-like object (e.g., from st.file_uploader)
        target_size (tuple): Size to resize image to (width, height).
        normalize (bool): If True, scales pixel values to [0, 1].

    Returns:
        np.ndarray: Preprocessed image with shape (1, H, W, C)
    """
    # Load image using PIL
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)

    # Convert to numpy array
    img_array = tf.keras.utils.img_to_array(img)

    if normalize:
        img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
