import os
import cv2 as cv
import math as m
import numpy as np
import pandas as pd
import time as t
#import pyarrow as pa
#import pyarrow.parquet as pq
#import dask.array as da
#import dask.dataframe as dd
#from dask_ml.model_selection import train_test_split
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Original Data Paths
PATH_MTG = 'data/mtg'
PATH_PKM = 'data/pokemon'
PATH_YGO = 'data/yugioh'

# Temporary Paths
TEMP_MTG = 'data/_temp/temp_mtg.dat'
TEMP_PKM = 'data/_temp/temp_pkm.dat'
TEMP_YGO = 'data/_temp/temp_ygo.dat'
TEMP_ALL = 'data/_temp/temp_all.dat'

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

# Card Counts and Feature Size
COUNT_MTG = 16_990
COUNT_PKM = 21_199
COUNT_YGO = 14_599
IMAGE_DATA_SIZE = 350_000
CHUNK_SIZE = 100


class CardType(Enum):
    """Card Type Enum for Magic, Pokemon, and Yu-Gi-Oh."""
    MTG = 0     # Magic: The Gathering
    PKM = 1     # Pokemon
    YGO = 2     # Yu-Gi-Oh


def convert_images_to_vectors(temporary_filename: str, path: str, count: int | None = None):
    """
    Converts the card JPGs in the path to Numpy normalized grayscale vectors.
    The vectors are normalize to [0, 1] using Min Max Scaling.

    Args:
        path (str): The data path for the card JPGs.
        count (int | None): The number of cards to process. All in path if None.
        temporary_filename (str): The temporary path/filename to store the card data on drive.
    
    Returns:
        A list of numpy 1D vectors for each image processed.
    """
    # Establish the number of cards to process
    start = t.perf_counter()
    filenames = os.listdir(path)
    if count is None: count = len(filenames)
    filenames = filenames[:count]
    card_vectors = np.memmap(temporary_filename, dtype=np.float32, mode='w+', shape=(count, IMAGE_DATA_SIZE))

    # Convert count number of cards
    for i, filename in enumerate(filenames):
        filepath = os.path.join(path, filename)
        card = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if card is not None:
            card = card.ravel()
            card = cv.normalize(src=card, dst=card_vectors[i], alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        else:
            print(f"[ERROR] Could not process card {filepath}.")
            card_vectors[i] = 0     # Fallback case

    # Flush the final bit of memory from RAM to the drive
    card_vectors.flush()

    # Returns all of the cards as Numpy grayscale vectors.
    end = t.perf_counter()
    print(f"\tTime Elapsed: {end - start}s.")
    return card_vectors


def train_test_validation_split(card_count: int, data_memmaps: list):
    """
    This splits all of the MTG, PKM, & YGO data from the temporary files into their own train, test, and validation data files.
    This is all done using memmaps.

    - Train: 64% (80% of 80%)
    - Test: 20%
    - Validation: 16% (20% of 80%)

    Args:
        card_count (int): The total number if cards to process for each set.
        data_memmaps (list): A list of the MTG, PKM, & YGO memmapst that link to each temporary file.
    """
    # Declare the amount of data for Train, Test, and Validation
    train_count = int(card_count * 0.64) 
    test_count = int(card_count * 0.20)
    validation_count = int(card_count * 0.16)
    split_counts = [train_count, test_count, validation_count]

    # Declare the random number generator to get random indices
    rng = np.random.default_rng()

    # Copies all of the data from the MTG, PKM, & YGO data memmaps into separate train, test, & validation memmaps
    for card_key, data_memmap in zip(['MTG', 'PKM', 'YGO'], data_memmaps):    # MTG, PKM, YGO
        print(f"[INFO] Splitting {card_key} data into Train, Test, and Validation sets...")
        time_start = t.perf_counter()

        # Generates a random sequence for how to pull data from the data memmap
        random_indices = rng.choice(card_count, size=card_count, replace=False)

        # Move key data into its own train, test, & validation memmaps
        buffer = 0
        for count, set in zip(split_counts, ["TRAIN", "TEST", "VALIDATION"]):
            # Creates the memmap for that card's set
            map = np.memmap(globals()[f"PATH_{set}_{card_key}"], dtype=np.float32, mode='w+', shape=(count, IMAGE_DATA_SIZE))

            # Copy data through chunks
            chunk_count, chunk_remainder = divmod(count, CHUNK_SIZE)
            for i in range(0, chunk_count):
                start, end = i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE
                r_start, r_end = start + buffer, end + buffer
                map[start:end] = data_memmap[random_indices[r_start:r_end]]

            # Copy the last bit of data without going over
            start = chunk_count * CHUNK_SIZE
            end = start + chunk_remainder
            r_start = start + buffer
            r_end = r_start + chunk_remainder
            map[start:end] = data_memmap[random_indices[r_start:r_end]]
            buffer = r_end  # Update buffer for next round

            # Flush the final bit of memmap memory from RAM
            map.flush()

        time_end = t.perf_counter()
        print(f"\tTime Elapsed: {time_end - time_start}s.")


def main():
    """
    Prepares the data used to analyze the Magic: The Gathering, Pokemon, and Yu-Gi-Oh card images.

    Preprocessing of images:
    1) Convert image to vectors
        - Load images from folders
        - Encode them as vectors
        - Normalize the vectors [0, 1]
    2) Splits the data into Train, Test, and Validation sets.
    """

    # Set the numbers of cards to process from each set
    card_count = min(COUNT_MTG, COUNT_PKM, COUNT_YGO)
    #card_count = 100

    # Gets the vectors for all cards JPGs per path
    print("[INFO] Converting 'Magic: The Gathering' card JPGs to vectors...")
    mtg_cards = convert_images_to_vectors(TEMP_MTG, PATH_MTG, card_count)
    print("[INFO] Converting 'Pokemon' card JPGs to vectors...")
    pkm_cards = convert_images_to_vectors(TEMP_PKM, PATH_PKM, card_count)
    print("[INFO] Converting 'Yu-Gi-Oh' card JPGs to vectors...")
    ygo_cards = convert_images_to_vectors(TEMP_YGO, PATH_YGO, card_count)

    # Copy and split the MTG, PKM, & YGO data from the temporary files into their own train, test, and validation data files
    print("[INFO] Splitting data into Train, Test, and Validation sets...")
    train_test_validation_split(card_count, [mtg_cards, pkm_cards, ygo_cards])

    # Release the OS file locks, deletes the Python references, and removes the temporary files from disk
    if hasattr(mtg_cards, '_mmap'): mtg_cards._mmap.close()
    if hasattr(pkm_cards, '_mmap'): pkm_cards._mmap.close()
    if hasattr(ygo_cards, '_mmap'): ygo_cards._mmap.close()
    del mtg_cards, pkm_cards, ygo_cards
    if os.path.exists(TEMP_MTG): os.remove(TEMP_MTG)
    if os.path.exists(TEMP_PKM): os.remove(TEMP_PKM)
    if os.path.exists(TEMP_YGO): os.remove(TEMP_YGO)


if __name__ == "__main__":
    main()
