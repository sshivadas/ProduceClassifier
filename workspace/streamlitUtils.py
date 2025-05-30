from MyImports import *
from ModelUtils import predict_image_class
def process_streamlit_input_image(image_file):
    """
    Prepares an image file for model prediction.

    Parameters:
        image: image file .jpeg file(e.g., from Streamlit's st.file_uploaded)
        
    Returns:
        tf.Tensor: Image tensor of shape (1, H, W, C), ready for model.predict()
    """
    # Open image using PIL and ensure RGB format
    image = Image.open(image_file).convert('RGB')

    # Convert to float32 NumPy array
    img_array = tf.keras.utils.img_to_array(image)

    # Optional: Normalize pixel values   
    img_array = img_array / 255.0

    # Resize image
    img_resized = tf.image.resize(img_array, image_size)

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img_tensor = tf.expand_dims(img_resized, axis=0)

    return img_tensor