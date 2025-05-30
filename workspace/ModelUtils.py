from MyImports import *

def process_image(image_path):
    """
    Prepares an image for model prediction.

    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        tf.Tensor: Image tensor of shape (1, H, W, C), ready for model.predict()
    """
    #Load Image
    image= tf.keras.utils.load_img(image_path)
    
    # Convert to float32 NumPy array
    img_array = tf.keras.utils.img_to_array(image)

    # Resize image
    img_resized = tf.image.resize(img_array, image_size)

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img_tensor = tf.expand_dims(img_resized, axis=0)

    return img_tensor

def predict_image_class(model, image_tensor, class_names):
    """
    Predict the class of an input image using a trained model.

    Parameters:
        model (tf.keras.Model): Trained Keras model.
        image_tensor (tf.Tensor): Image tensor of shape (1, H, W, C), preprocessed.
        

    Returns:
        predicted_label (str): Predicted class label.
    """
    pred = model.predict(image_tensor, verbose=0)
    predicted_index = tf.argmax(pred, 1).numpy().item()
    predicted_label = class_names[predicted_index]
    return predicted_label

def classwise_accuracy(folder_path, model,class_names):
    """
    Compute and print per-class classification accuracy using a given model.

    Args:
        folder_path (Path): Path object pointing to the root folder containing class-wise subfolders.
        model: A model with a `.predict()` method that returns (pred_class, confidence).

    Prints:
        Accuracy percentage and number of images per class.
    """
    for class_folder in sorted(folder_path.iterdir()):
        if not class_folder.is_dir():
            continue

        cls_name = class_folder.name
        image_paths = list(class_folder.glob('*'))

        if not image_paths:
            print(f"⚠️  Skipping empty class: {cls_name}")
            continue

        correct_predictions = 0

        for img_path in image_paths:
            img_tensor = process_image(img_path)
            pred_class = predict_image_class(model, img_tensor,class_names)
            if pred_class == cls_name:
                correct_predictions += 1

        accuracy = (correct_predictions / len(image_paths)) * 100
        print(f"✅ Accuracy for class '{cls_name}': {accuracy:.2f}% ({len(image_paths)} images)")


