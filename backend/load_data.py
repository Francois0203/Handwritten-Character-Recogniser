import tensorflow_datasets as tfds
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_emnist():
    """
    Loads EMNIST ByClass dataset (digits + uppercase + lowercase letters)
    Returns: (x_train, y_train), (x_test, y_test) as numpy arrays
    """
    print("Loading EMNIST ByClass dataset...")

    ds_train = tfds.load('emnist/byclass', split='train', as_supervised=True)
    ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)

    # Convert to numpy arrays
    x_train = np.array([np.array(image) for image, label in tfds.as_numpy(ds_train)])
    y_train = np.array([label for image, label in tfds.as_numpy(ds_train)])
    x_test = np.array([np.array(image) for image, label in tfds.as_numpy(ds_test)])
    y_test = np.array([label for image, label in tfds.as_numpy(ds_test)])

    print(f"EMNIST loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")

    return (x_train, y_train), (x_test, y_test)

def load_custom_symbols(symbols_dir):
    """
    Loads custom symbol images from a folder
    Args:
        symbols_dir: path to folder containing subfolders per symbol
    Returns:
        x_data, y_data as numpy arrays
    """
    print("Loading custom symbols...")

    x_data = []
    y_data = []
    classes = sorted(os.listdir(symbols_dir))
    class_map = {cls:i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_folder = os.path.join(symbols_dir, cls)
        for img_file in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_file)
            img = load_img(img_path, color_mode='grayscale', target_size=(28,28))
            img_array = img_to_array(img)
            x_data.append(img_array)
            y_data.append(class_map[cls])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(f"Custom symbols loaded: {x_data.shape[0]} samples")

    return x_data, y_data

# Example usage (comment out when importing)
if __name__ == "__main__":
    # Load datasets
    (x_train, y_train), (x_test, y_test) = load_emnist()
    x_symbols, y_symbols = load_custom_symbols("../data/custom_symbols")  # optional

    import matplotlib.pyplot as plt
    import random

    # Function to visualize a few samples
    def visualize_samples(x, y, num_samples=9, title="Sample Images"):
        plt.figure(figsize=(6,6))
        indices = random.sample(range(len(x)), num_samples)
        for i, idx in enumerate(indices):
            plt.subplot(3,3,i+1)
            plt.imshow(x[idx].squeeze(), cmap='gray')  # .squeeze() in case shape is (28,28,1)
            plt.title(f"Label: {y[idx]}")
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    # Visualize 9 random EMNIST images
    visualize_samples(x_train, y_train, title="Random EMNIST Samples")

    # Optional: visualize custom symbols
    visualize_samples(x_symbols, y_symbols, title="Random Symbol Samples")