import tensorflow as tf
import numpy as np
import config as cf

def make_labeled_dataset(mtg, pkm, ygo, batch_size=32,shuffle= False ):
    """
    This function takes in an train/test/validation split from each of the card games, a batch size, and bool to determine 
    if we need to shuffle the dataset.

    It then processes them in batches in order to apply a label for each class so we can train the model with TensorFlow

    If prompted it can shuffle the dataset so the model can get a variety of classes instead of each one sequentially.
    This can reduce potentially reduce bias in the model


    It returns a tensorflow dataset that is processed from the from_generator function
    """


    arrays= [mtg,pkm,ygo]
    sample_order = np.asarray(
        [
            (class_id, index)
            for class_id, data in enumerate(arrays)
            for index in range(len(data))
        ],
        dtype=np.int32,
    )

    def batches():
        order = sample_order.copy()
        if shuffle:
            np.random.shuffle(order)

        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            images = np.stack(
                [arrays[class_id][index] for class_id, index in batch]
            ).reshape(
                -1,
                cf.SIZE_PADDED_HEIGHT,
                cf.SIZE_PADDED_WIDTH,
                1,
            )
            labels = batch[:, 0]
            yield images, labels

    return tf.data.Dataset.from_generator(
        batches,
        output_signature=(
            tf.TensorSpec(
                (None, cf.SIZE_PADDED_HEIGHT, cf.SIZE_PADDED_WIDTH, 1),
                tf.float32,
            ),
            tf.TensorSpec((None,), tf.int32),
        ),
    ).prefetch(tf.data.AUTOTUNE)