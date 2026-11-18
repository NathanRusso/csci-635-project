"""
Classifies a given image as one of the 3 trading cards using the CNN, ViT, and MobileViT.
"""
# Logging Imports
print("[INFO] Importing libraries...")
import logging
import os
import warnings

# Suppress warnings on run
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow startup messages
warnings.filterwarnings("ignore")           # Suppress Python/Keras warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)     # Suppress TensorFlow logging
original_stderr = os.dup(2)                 # Temporarily suppress C++ startup output
devnull = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(devnull, 2)
    import tensorflow as tf     # Note: Takes a few seconds
finally:
    os.dup2(original_stderr, 2)
    os.close(original_stderr)
    os.close(devnull)

# Other Imports
import argparse
import cv2 as cv
import keras
from keras_vision.MobileViT_v2 import build_MobileViT_v2
import numpy as np


# The size used for resizing
SIZE_FEATURE_WIDTH = 500
SIZE_FEATURE_HEIGHT = 700
SIZE_IMAGE_VIT = 224

# The paths to all 3 ML models
PATH_CNN = 'data/635-CNN.keras'
PATH_VIT = 'data/635-ViT.keras'
PATH_MOBILEVIT = 'data/635-MobileVit_v2.keras'

# Map of card class value to its name
card_map = {
    0: "Magic: The Gathering",
    1: "Pokemon",
    2: "Yu-Gi-Oh"
}


def cnn_classify(images):
    """
    Classifies each of the images using a CNN.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        Numpy array of softmax predictions
    """
    model = keras.models.load_model(PATH_CNN)
    return model.predict(images, verbose=0)


def vit_classify(images):
    """
    Classifies each of the images using a ViT.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        Numpy array of 3-value tuples for each prediction (softmax)
    """
    model = keras.models.load_model(PATH_VIT)
    images = tf.reshape(images, shape=(-1, SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH, 1))
    images = tf.image.resize(images, size=(SIZE_IMAGE_VIT, SIZE_IMAGE_VIT), method="area")
    images = images * 2 - 1
    images = tf.image.grayscale_to_rgb(images)
    return model.predict(images, verbose=0)


def mobilevit_classify(images):
    """
    Classifies each of the images using a MobileViT.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        Numpy array of softmax predictions
    """
    # Build the exact architecture saved in PATH_MOBILEVIT 
    base_model = build_MobileViT_v2(
        input_shape=(256, 192, 1),
        width_multiplier=0.5,
        num_classes=3,
        classifier_head_activation="softmax",
        pretrained=False,
    )
    model = keras.Sequential([
        keras.layers.Input(shape=(SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH, 1)),
        keras.layers.Resizing(256, 192),
        base_model,
    ])

    # Run a dummy forward pass so all internal sub-layers build their weight variables
    model(np.zeros((1, SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH, 1), dtype="float32"), training=False)

    # Load the trained weights directly from the .keras archive
    model.load_weights(PATH_MOBILEVIT)

    images = tf.reshape(images, shape=(-1, SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH, 1))
    return model.predict(images, verbose=0)


def classify(images):
    """
    Classifies each of the images for each model and returns their predictions.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        A 3-tuple containing lists of softmax predictions for each model
    """
    predictions_cnn = cnn_classify(images)
    predictions_vit = vit_classify(images)
    predictions_mobilevit = mobilevit_classify(images)

    return predictions_cnn, predictions_vit, predictions_mobilevit


def display_classify(images, images_paths):
    """
    Classifies each of the images and displays their predictions.

    Args:
        images: Numpy array of 700x500 images
        image_paths: A list of the string image paths
    """
    # Gets the softmax classification arrays from the models
    predictions_cnn, predictions_vit, predictions_mobilevit = classify(images)

    # Displays the output for each classification
    for i in range(len(images)):
        prediction_cnn = np.round(predictions_cnn[i] * 100, decimals=3)
        prediction_vit = np.round(predictions_vit[i] * 100, decimals=3)
        prediction_mobilevit = np.round(predictions_mobilevit[i] * 100, decimals=3)
        print(f"\nClassification Image #{i+1}: {images_paths[i]}")
        print(f"\tCNN:          {prediction_cnn}, {card_map[np.argmax(prediction_cnn)]}")
        print(f"\tViT:          {prediction_vit}, {card_map[np.argmax(prediction_vit)]}")
        print(f"\tMobileViT:    {prediction_mobilevit}, {card_map[np.argmax(prediction_mobilevit)]}")
    print()


def main():
    """
    Takes in a list of images and classifies them using the CNN, ViT, and MobileViT.
    """
    print(f"[INFO] Running application...")
    
    # Makes sure numpy shows all 3 decimal places minimum
    np.set_printoptions(formatter={'float_kind': '{:0.3f}'.format})

    # Get a list of cards to classify (image paths)
    parser = argparse.ArgumentParser(prog="ML Project", description="Trading Card Classifier")
    parser.add_argument("--images", nargs="+", required=True, help="List of image paths to classify")

    # Get the images for all of the paths
    args = parser.parse_args()
    image_paths = args.images
    images = []
    images_paths_selected = []
    for path in image_paths:
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not get image located at {path}!")
            continue
        images_paths_selected.append(path)
        if image.shape[2:] != (SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH):
            image = cv.resize(image, (SIZE_FEATURE_WIDTH, SIZE_FEATURE_HEIGHT), interpolation=cv.INTER_AREA)
        #card = card.ravel()
        image = cv.normalize(src=image, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        images.append(image)
    images = np.array(images)

    # Display the predictions for the cards
    display_classify(images, images_paths_selected)


if __name__ == "__main__":
    main()
