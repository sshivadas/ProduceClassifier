from MyImports import *

def examine_dataset(folder_path):
    """
    Examines an image dataset organized by class folders.

    Parameters:
        folder_path (Path): Path object pointing to the train/test folder. 
                            Each subfolder is assumed to represent a class and contains images.

    Returns:
        image_dict (dict): A dictionary with class names as keys and one sample image (PIL.Image) per class as values.
        count_dict (dict): A dictionary with class names as keys and the number of images in that class as values.
    """

    image_dict = {}  # To store one sample image per class (for quick inspection or visualization)
    count_dict = {}  # To store the number of images per class

    # Iterate through all class folders in sorted order (helps with consistency in results)
    for class_dir in sorted(folder_path.iterdir()):
        if not class_dir.is_dir():
            continue  # Skip files or non-directory items
        
        image_files = list(class_dir.glob('*'))  # List all files (you could restrict to *.jpg, *.png if needed)

        if not image_files:
            print(f"⚠️  Skipping empty class: {class_dir.name}")
            continue

        count_dict[class_dir.name] = len(image_files)  # Count images in the class

        # Select one random image to inspect the class visually
        random_image_path = random.choice(image_files)

        try:
            image = tf.keras.utils.load_img(random_image_path)  # Load image using Keras utility (returns PIL.Image)
            image_dict[class_dir.name] = image
        except Exception as e:
            print(f"❌ Failed to load image {random_image_path.name} from class '{class_dir.name}': {e}")

        print(f"✅ Class '{class_dir.name}' has {len(image_files)} images.")

    return image_dict, count_dict

def visualize_sample_image(image_dict):
    """
    Displays one random sample image from each class using matplotlib.

    Parameters:
        image_dict (dict): A dictionary where keys are class names and 
                           values is one sample image per class.
    """
    num_classes = len(image_dict)
    if num_classes == 0:
        print("❗ No images to display. 'image_dict' is empty.")
        return

    plt.figure(figsize=(4 * num_classes, 5))  # Adjust figure size based on number of classes

    for i, (cls, img) in enumerate(image_dict.items()):
        ax = plt.subplot(1, num_classes, i + 1)  # Create subplot for each class
        plt.imshow(img)                         # Show the image
        plt.title(f'{cls}\n{img.size}')         # Title with class name and image dimensions
        plt.axis("off")                         # Hide axis ticks

    plt.tight_layout()
    plt.show()


def image_cnt_distribution_per_class(image_cnt_dict, title="Image Count per Class"):
    """
    Plots a bar chart showing the distribution of image counts across different classes.

    Parameters:
        image_cnt_dict (dict): Dictionary with class names as keys and image counts as values.
        title (str): Title of the bar plot (optional, default is "Image Count per Class").
    """
    if not image_cnt_dict:
        print("❗ The input dictionary is empty. Nothing to plot.")
        return

    # Convert dictionary to sorted DataFrame for plotting
    count_df = (
        pd.DataFrame(list(image_cnt_dict.items()), columns=["class", "count"])
        .sort_values("count", ascending=False)
    )

    # Create bar chart
    ax = count_df.plot.bar(
        x='class',
        y='count',
        color='green',
        figsize=(10, 6.5),
        legend=False,
        title=title
    )

    # Set axis labels and appearance
    ax.set_ylabel("Image Count", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of each bar
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f'{int(height)}',
            (patch.get_x() + patch.get_width() / 2, height),
            ha='center',
            va='bottom',
            fontsize=10,
            xytext=(0, 5),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.show()
