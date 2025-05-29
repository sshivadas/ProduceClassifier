from MyImports import *
import matplotlib.pyplot as plt

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
