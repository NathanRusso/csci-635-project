"""
File to create and train a Visual Transformer (ViT) model.
"""

import numpy as np
import math
import cv2 as cv
from enum import Enum
import keras
import keras_hub
import tensorflow as tf


# Data Model Path
PATH_MODEL = 'data/ViT.keras'

# Train Paths
PATH_TRAIN = 'data/train'
PATH_TRAIN_MTG = 'data/train/train_mtg.dat'
PATH_TRAIN_PKM = 'data/train/train_pkm.dat'
PATH_TRAIN_YGO = 'data/train/train_ygo.dat'

# Test Paths
PATH_TEST = 'data/test'
PATH_TEST_MTG = 'data/test/test_mtg.dat'
PATH_TEST_PKM = 'data/test/test_pkm.dat'
PATH_TEST_YGO = 'data/test/test_ygo.dat'

# Validation Paths
PATH_VALIDATION = 'data/validation'
PATH_VALIDATION_MTG = 'data/validation/validation_mtg.dat'
PATH_VALIDATION_PKM = 'data/validation/validation_pkm.dat'
PATH_VALIDATION_YGO = 'data/validation/validation_ygo.dat'

# Card Class Counts and Feature Sizes
COUNT_TRAIN = 9343
COUNT_TEST = 2919
COUNT_VALIDATION = 2335
SIZE_FEATURE = 350_000
SIZE_FEATURE_WIDTH = 500
SIZE_FEATURE_HEIGHT = 700
SIZE_BATCH = 32
SIZE_PATH = 25
EPOCHS = 1


# Train Memmaps
train_mtg = np.memmap(PATH_TRAIN_MTG, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))
train_pkm = np.memmap(PATH_TRAIN_PKM, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))
train_ygo = np.memmap(PATH_TRAIN_YGO, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))

# Test Memmaps
test_mtg = np.memmap(PATH_TEST_MTG, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))
test_pkm = np.memmap(PATH_TEST_PKM, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))
test_ygo = np.memmap(PATH_TEST_YGO, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))

# Validation Memmaps
validation_mtg = np.memmap(PATH_VALIDATION_MTG, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))
validation_pkm = np.memmap(PATH_VALIDATION_PKM, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))
validation_ygo = np.memmap(PATH_VALIDATION_YGO, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))


class CardType(Enum):
    """Card Type Enum for Magic, Pokemon, and Yu-Gi-Oh."""
    MTG = 0     # Magic: The Gathering
    PKM = 1     # Pokemon
    YGO = 2     # Yu-Gi-Oh


def train_generator():
    """
    Train data generator used for selection/iteration.
    """
    for i in range(COUNT_TRAIN):
        yield train_mtg[i], CardType.MTG.value
        yield train_pkm[i], CardType.PKM.value
        yield train_ygo[i], CardType.YGO.value


def validation_generator():
    """
    Validation data generator used for selection/iteration.
    """
    for i in range(COUNT_VALIDATION):
        yield validation_mtg[i], CardType.MTG.value
        yield validation_pkm[i], CardType.PKM.value
        yield validation_ygo[i], CardType.YGO.value


def test_generator():
    """
    test data generator used for evaluation.
    """
    for i in range(COUNT_TEST):
        yield test_mtg[i], CardType.MTG.value
        yield test_pkm[i], CardType.PKM.value
        yield test_ygo[i], CardType.YGO.value


def image_resizer(images, labels):
    """
    Resizes the images for the ViT.

    1. (350_000,) to (700, 500, 1)
    2. (700, 500, 1) to (384, 384)
    3. (384, 384, 1) [0, 1] to (384, 384, 1) [-1, 1]
    3. (384, 384, 1) to (384, 384, 3) where the 3 channels are the same.
    """
    images = tf.reshape(images, shape=(-1, 700, 500, 1))
    images = tf.image.resize(images, size=(384, 384), method="area")
    images = images * 2 - 1
    images = tf.image.grayscale_to_rgb(images)
    return images, labels


def create_train_vit():
    """
    Loads a pretrained ViT and trains it on our card data.
    Saves the model to the data file.
    https://www.kaggle.com/models/keras/vit/keras/vit_large_patch32_384_imagenet/3
    """

    # Downloads the 32x32 patch size, 384x384 image large ViT model
    #model.backbone.trainable = False
    model = keras_hub.models.ViTImageClassifier.from_preset(
        preset="vit_large_patch32_384_imagenet",
        num_classes=3,
        load_weights=True
    )

    # Sets the model optimizer, loss, and metric
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Output Generator Signature: [(350,000 features), (1 target class)]
    output_generator_signature = (
        tf.TensorSpec(shape=(SIZE_FEATURE,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    # Callback to stop the model if validation loss stops improving (prevent overfitting)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    # Set up the train dataset
    train_dataset = tf.data.Dataset.from_generator(train_generator, output_signature=output_generator_signature
    ).shuffle(buffer_size=SIZE_BATCH, reshuffle_each_iteration=True
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(COUNT_TRAIN * 3 / SIZE_BATCH)))

    # Set up the validation dataset
    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_signature=output_generator_signature
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(COUNT_VALIDATION * 3 / SIZE_BATCH)))

    # Set up the train dataset
    test_dataset = tf.data.Dataset.from_generator(test_generator, output_signature=output_generator_signature
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(COUNT_TEST * 3 / SIZE_BATCH)))

    # Trains the model on the training and validation sets
    model.fit(
        x=train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    # Checks the model's performance on the test data 
    model.evaluate(x=test_dataset)

    # Saves the model to be used later.
    model.save(PATH_MODEL)


if __name__ == "__main__":
    create_train_vit()
