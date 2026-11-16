"""
File to create and train a Visual Transformer (ViT) model.
"""

import numpy as np
import cv2 as cv

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


# Train Memmaps
train_mtg = np.memmap(PATH_TRAIN_MTG, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))
train_pkm = np.memmap(PATH_TRAIN_PKM, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))
train_ygo = np.memmap(PATH_TRAIN_YGO, dtype=np.float32, mode="r", shape=(COUNT_TRAIN, SIZE_FEATURE))

# Test Memmaps
train_mtg = np.memmap(PATH_TEST_MTG, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))
train_pkm = np.memmap(PATH_TEST_PKM, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))
train_ygo = np.memmap(PATH_TEST_YGO, dtype=np.float32, mode="r", shape=(COUNT_TEST, SIZE_FEATURE))

# Validation Memmaps
train_mtg = np.memmap(PATH_VALIDATION_MTG, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))
train_pkm = np.memmap(PATH_VALIDATION_PKM, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))
train_ygo = np.memmap(PATH_VALIDATION_YGO, dtype=np.float32, mode="r", shape=(COUNT_VALIDATION, SIZE_FEATURE))


def create_train_vit():
    """
    Creates a Visual Transformer (ViT) model and trains it.
    It saves the model to the data file.
    """

    #... = cv.resize(..., dsize=(SIZE_FEATURE_WIDTH, SIZE_FEATURE_HEIGHT))

    return

if __name__ == "__main__":
    create_train_vit()
