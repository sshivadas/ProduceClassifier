from MyImports import *

# Specifying the image size to resize all images
image_size = (256, 256)

import tensorflow as tf

def prepare_image_for_model(uploaded_file, target_size):
    """
    Prepares an uploaded image file for model prediction.

    Parameters:
        uploaded_file: File-like object (e.g., from Streamlit's st.file_uploader)
        target_size (tuple): Desired image size (width, height)
        normalize (bool): Whether to normalize pixel values to [0, 1]

    Returns:
        tf.Tensor: Image tensor of shape (1, H, W, C), ready for model.predict()
    """
    # Load image
    img = tf.keras.utils.load_img(uploaded_file)

    # Convert to float32 NumPy array
    img_array = tf.keras.utils.img_to_array(img)

    # Optional: Normalize pixel values   
    img_array = img_array / 255.0

    # Resize image
    img_resized = tf.image.resize(img_array, target_size)

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img_batch = tf.expand_dims(img_resized, axis=0)

    return img_batch


# index           0        1        2         3
class_names = ['noise', 'onion', 'potato', 'tomato']

def predict_image_class(model, image_tensor):
    """
    Predict the class of an input image using a trained model.

    Parameters:
        model (tf.keras.Model): Trained Keras model.
        image_tensor (tf.Tensor): Image tensor of shape (1, H, W, C), preprocessed.
        class_names (list): List of class names (index must match model output indices).

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Predict probabilities
    predictions = model.predict(image_tensor)
    
    # Get index of highest probability
    pred_index = tf.argmax(predictions[0]).numpy()

    # Get class name and confidence
    predicted_class = class_names[pred_index]
    confidence = predictions[0][pred_index]

    return predicted_class, float(confidence)



