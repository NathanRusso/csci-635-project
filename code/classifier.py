"""
Classifies a given image as one of the 3 trading cards using the CNN, ViT, and MobileViT.
"""

import argparse
import cv2 as cv
import keras
import numpy as np
import tensorflow as tf


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


def cnn_classification(images):
    """
    Classifies each of the images using a CNN.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        ...
    """
    #model = keras.models.load_model(PATH_CNN)
    return [[0, 0, 0] for _ in range(len(images))]


def vit_classification(images):
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


def mobilevit_classification(images):
    """
    Classifies each of the images using a MobileViT.

    Args:
        images: Numpy array of 700x500 images

    Returns:
        ...
    """
    #model = keras.models.load_model(PATH_MOBILEVIT)
    return [[0, 0, 0] for _ in range(len(images))]


def classify(images):
    """
    Classifies each of the images and displays their predictions.

    Args:
        images: Numpy array of 700x500 images
    """
    # Gets the softmax classification arrays from the models
    predictions_cnn = cnn_classification(images)
    predictions_vit = vit_classification(images)
    predictions_mobilevit = mobilevit_classification(images)

    # Displays the output for each classification
    for i in range(len(images)):
        prediction_cnn = np.round(predictions_cnn[i], decimals=3)
        prediction_vit = np.round(predictions_vit[i] * 100, decimals=3)
        prediction_mobilevit = np.round(predictions_mobilevit[i], decimals=3)
        print(f"\nClassification Image #{i+1}:")
        print(f"\tCNN:          {prediction_cnn}, {card_map[np.argmax(prediction_cnn)]}")
        print(f"\tViT:          {prediction_vit}, {card_map[np.argmax(prediction_vit)]}")
        print(f"\tMobileViT:    {prediction_mobilevit}, {card_map[np.argmax(prediction_mobilevit)]}")
    print()


def main():
    """
    Takes in a list of images and classifies them using the CNN, ViT, and MobileViT.
    """
    # Makes sure numpy shows all 3 decimal places minimum
    np.set_printoptions(formatter={'float_kind': '{:0.3f}'.format})

    # Get a list of cards to classify (image paths)
    parser = argparse.ArgumentParser(prog="ML Project", description="Trading Card Classifier")
    parser.add_argument("--images", nargs="+", required=True, help="List of image paths to classify")

    # Get the images for all of the paths
    args = parser.parse_args()
    image_paths = args.images
    images = []
    for path in image_paths:
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not get image located at {path}!")
            continue
        if image.shape[2:] != (SIZE_FEATURE_HEIGHT, SIZE_FEATURE_WIDTH):
            image = cv.resize(image, (SIZE_FEATURE_WIDTH, SIZE_FEATURE_HEIGHT), interpolation=cv.INTER_AREA)
        #card = card.ravel()
        image = cv.normalize(src=image, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        images.append(image)
    images = np.array(images)

    # Display the predictions for the cards
    classify(images)


if __name__ == "__main__":
    main()
