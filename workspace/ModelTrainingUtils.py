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



def plot_accuracy(model_history, n_epochs=None):
    """
    Plots training and validation accuracy over epochs.

    Parameters:
        model_history (keras.callbacks.History): History object returned by model.fit()
        n_epochs (int, optional): Number of epochs to plot. If None, inferred from history.
    """
    if not hasattr(model_history, 'history') or not isinstance(model_history.history, dict):
        raise ValueError("❗ Provided object is not a valid Keras History object.")

    history = model_history.history
    train_acc = history.get('accuracy')
    val_acc = history.get('val_accuracy')

    if train_acc is None or val_acc is None:
        raise KeyError("❗ 'accuracy' or 'val_accuracy' not found in training history.")

    if n_epochs is None:
        n_epochs = min(len(train_acc), len(val_acc))

    epochs = range(n_epochs)

    # Plot
    plt.figure(figsize=(7.5, 5))
    plt.plot(epochs, [acc * 100 for acc in train_acc[:n_epochs]], label='Train Accuracy', color='blue')
    plt.plot(epochs, [acc * 100 for acc in val_acc[:n_epochs]], label='Validation Accuracy', color='red')

    plt.title("Accuracy vs. Epoch", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def plot_loss(model_history, n_epochs=None):
    """
    Plots training and validation loss over epochs.

    Parameters:
        model_history (keras.callbacks.History): History object returned by model.fit()
        n_epochs (int, optional): Number of epochs to plot. If None, inferred from history.
    """
    if not hasattr(model_history, 'history') or not isinstance(model_history.history, dict):
        raise ValueError("❗ Provided object is not a valid Keras History object.")

    history = model_history.history
    train_loss = history.get('loss')
    val_loss = history.get('val_loss')

    if train_loss is None or val_loss is None:
        raise KeyError("❗ 'loss' or 'val_loss' not found in training history.")

    if n_epochs is None:
        n_epochs = min(len(train_loss), len(val_loss))

    epochs = range(n_epochs)

    # Plot
    plt.figure(figsize=(7.5, 5))
    plt.plot(epochs, train_loss[:n_epochs], label='Train Loss', color='blue')
    plt.plot(epochs, val_loss[:n_epochs], label='Validation Loss', color='red')

    plt.title("Loss vs. Epoch", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
