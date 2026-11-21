import os
import cv2 as cv
import math as m
import numpy as np
import pandas as pd
import time as t
import config as cf


# Temporary Paths
PATH_TEMP_MTG = 'data/_temp/temp_mtg.dat'
PATH_TEMP_PKM = 'data/_temp/temp_pkm.dat'
PATH_TEMP_YGO = 'data/_temp/temp_ygo.dat'
PATH_TEMP_ALL = 'data/_temp/temp_all.dat'

# Processing Chunk Size
SIZE_CHUNK = 100

# Version 2 Variables
GREY_LOWER = 15     # Inclusive
GREY_UPPER = 240    # Inclusive
ANGLE_SHIFT = 20
BRIGHTNESS_SHIFT = 40


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
    card_vectors = np.memmap(temporary_filename, dtype=np.float32, mode='w+', shape=(count, cf.SIZE_FEATURE))

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


def convert_images_to_vectors_2(temporary_filename: str, path: str, count: int | None = None):
    """
    Converts the card JPGs in the path to Numpy normalized grayscale vectors.
    The vectors are normalize to [0, 1] using Min Max Scaling.
    Since this is version #2, it also pads the images with a grey blur background.
    It also warps the images vertically and horizontally, along with changing brightness.

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
    card_vectors = np.memmap(temporary_filename, dtype=np.float32, mode='w+', shape=(count, cf.SIZE_PADDED_FEATURE))

    # Convert count number of cards
    for i, filename in enumerate(filenames):
        # Pull the greys scale cards
        filepath = os.path.join(path, filename)
        card = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        # Continue if there was an issue
        if card is None:
            print(f"[ERROR] Could not process card {filepath}.")
            card_vectors[i] = 0     # Fallback case
            continue

        # Create a random blurred grey scale background to mimic "real life" backgrounds.
        background = np.random.randint(low=GREY_LOWER, high=GREY_UPPER+1, size=(cf.SIZE_PADDED_HEIGHT, cf.SIZE_PADDED_WIDTH), dtype=np.uint8)
        cv.GaussianBlur(src=background, dst=background, ksize=(11,11), sigmaX=0, borderType=cv.BORDER_REFLECT)

        # Add in random horizontal perspective tilts (only the card inner part)
        # Overlay the card on top of the "real life" background
        horizontal_degree_change = np.random.randint(low=-ANGLE_SHIFT, high=ANGLE_SHIFT+1)
        vertical_degree_change = np.random.randint(low=-ANGLE_SHIFT, high=ANGLE_SHIFT+1)

        # Vertical Perspective Changes
        theta_vertical = np.radians(abs(vertical_degree_change))
        new_height = cf.SIZE_FEATURE_HEIGHT * np.cos(theta_vertical)
        pinch_x = cf.SIZE_FEATURE_WIDTH * np.sin(theta_vertical) * 0.25
        offset_y = cf.SIZE_PADDING + (cf.SIZE_FEATURE_HEIGHT - new_height) / 2.0
        pinch_top = pinch_x if vertical_degree_change >= 0 else 0.0
        pinch_bottom = pinch_x if vertical_degree_change < 0 else 0.0

        # Horizontal Perspective Changes
        theta_horizontal = np.radians(abs(horizontal_degree_change))
        new_width = cf.SIZE_FEATURE_WIDTH * np.cos(theta_horizontal)
        pinch_y = cf.SIZE_FEATURE_HEIGHT * np.sin(theta_horizontal) * 0.25
        offset_x = cf.SIZE_PADDING + (cf.SIZE_FEATURE_WIDTH - new_width) / 2.0
        pinch_right = pinch_y if horizontal_degree_change >= 0 else 0.0
        pinch_left = pinch_y if horizontal_degree_change < 0 else 0.0

        src_corners = np.array(
            [[0, 0], [cf.SIZE_FEATURE_WIDTH, 0], [cf.SIZE_FEATURE_WIDTH, cf.SIZE_FEATURE_HEIGHT], [0, cf.SIZE_FEATURE_HEIGHT]],
            dtype=np.float32
        )

        dst_corners = np.array([
            [offset_x + pinch_top, offset_y + pinch_left],
            [offset_x + new_width - pinch_top, offset_y + pinch_right],
            [offset_x + new_width - pinch_bottom, offset_y + new_height - pinch_right],
            [offset_x + pinch_bottom, offset_y + new_height - pinch_left]
        ], dtype=np.float32)

        matrix = cv.getPerspectiveTransform(src=src_corners, dst=dst_corners)
        cv.warpPerspective(
            src=card,
            M=matrix,
            dsize=(cf.SIZE_PADDED_WIDTH, cf.SIZE_PADDED_HEIGHT),
            dst=background,
            borderMode=cv.BORDER_TRANSPARENT
        )
        card = background.astype(np.float32)

        # Randomly shift the brightness of the card: New = |α*Old+β|
        brightness_change = np.random.randint(-BRIGHTNESS_SHIFT, BRIGHTNESS_SHIFT)
        card = cv.add(src1=card, src2=brightness_change)

        # Flatten the card and normalize it to [0, 1]
        card = card.ravel()
        card = cv.normalize(src=card, dst=card_vectors[i], alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

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
            # map = np.memmap(getattr(cf, f"PATH_{set}_{card_key}"), dtype=np.float32, mode='w+', shape=(count, cf.SIZE_FEATURE))
            map = np.memmap(getattr(cf, f"PATH_{set}_{card_key}_2"), dtype=np.float32, mode='w+', shape=(count, cf.SIZE_PADDED_FEATURE))

            # Copy data through chunks
            chunk_count, chunk_remainder = divmod(count, SIZE_CHUNK)
            for i in range(0, chunk_count):
                start, end = i * SIZE_CHUNK, (i + 1) * SIZE_CHUNK
                r_start, r_end = start + buffer, end + buffer
                map[start:end] = data_memmap[random_indices[r_start:r_end]]

            # Copy the last bit of data without going over
            start = chunk_count * SIZE_CHUNK
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
    card_count = min(cf.COUNT_MTG, cf.COUNT_PKM, cf.COUNT_YGO)
    #card_count = 100

    # Gets the vectors for all cards JPGs per path
    print("[INFO] Converting 'Magic: The Gathering' card JPGs to vectors...")
    # mtg_cards = convert_images_to_vectors(PATH_TEMP_MTG, cf.PATH_MTG, card_count)
    mtg_cards = convert_images_to_vectors_2(PATH_TEMP_MTG, cf.PATH_MTG, card_count)
    print("[INFO] Converting 'Pokemon' card JPGs to vectors...")
    # pkm_cards = convert_images_to_vectors(PATH_TEMP_PKM, cf.PATH_PKM_2, card_count)
    pkm_cards = convert_images_to_vectors_2(PATH_TEMP_PKM, cf.PATH_PKM, card_count)
    print("[INFO] Converting 'Yu-Gi-Oh' card JPGs to vectors...")
    # ygo_cards = convert_images_to_vectors(PATH_TEMP_YGO, cf.PATH_YGO_2, card_count)
    ygo_cards = convert_images_to_vectors_2(PATH_TEMP_YGO, cf.PATH_YGO, card_count)

    # Copy and split the MTG, PKM, & YGO data from the temporary files into their own train, test, and validation data files
    print("[INFO] Splitting data into Train, Test, and Validation sets...")
    train_test_validation_split(card_count, [mtg_cards, pkm_cards, ygo_cards])

    # Release the OS file locks, deletes the Python references, and removes the temporary files from disk
    if hasattr(mtg_cards, '_mmap'): mtg_cards._mmap.close()
    if hasattr(pkm_cards, '_mmap'): pkm_cards._mmap.close()
    if hasattr(ygo_cards, '_mmap'): ygo_cards._mmap.close()
    del mtg_cards, pkm_cards, ygo_cards
    if os.path.exists(PATH_TEMP_MTG): os.remove(PATH_TEMP_MTG)
    if os.path.exists(PATH_TEMP_PKM): os.remove(PATH_TEMP_PKM)
    if os.path.exists(PATH_TEMP_YGO): os.remove(PATH_TEMP_YGO)


if __name__ == "__main__":
    main()
