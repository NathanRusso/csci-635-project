import tensorflow as tf
import numpy as np
import config as cf


def make_labeled_dataset_2(
    mtg,
    pkm,
    ygo,
    batch_size=32,
    shuffle=False,
    padding=0,
):
    """
    Create a labeled TensorFlow dataset from one or more data sources per class.

    Pass the original and processed datasets as lists:

        mtg=[train_mtg_original, train_mtg_2]
        pkm=[train_pkm_original, train_pkm_2]
        ygo=[train_ygo_original, train_ygo_2]

    padding should contain one value per source. Use 20 for original unpadded
    cards and 0 for cards already padded to 740x540:

        padding=(20, 0)
    """
    source_groups = []

    for arrays in (mtg, pkm, ygo):
        if isinstance(arrays, (list, tuple)):
            source_groups.append(list(arrays))
        else:
            source_groups.append([arrays])

    source_count = max(len(group) for group in source_groups)

    if isinstance(padding, (list, tuple)):
        padding_per_source = list(padding)
    else:
        padding_per_source = [padding]

    padding_per_source.extend([0] * (source_count - len(padding_per_source)))

    if len(padding_per_source) != source_count:
        raise ValueError("padding must contain one value per source array")

    sample_order = np.asarray(
        [
            (class_id, source_id, index)
            for class_id, source_group in enumerate(source_groups)
            for source_id, data in enumerate(source_group)
            for index in range(len(data))
        ],
        dtype=np.int32,
    )

    def batches():
        order = sample_order.copy()

        if shuffle:
            np.random.shuffle(order)

        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            images = []

            for class_id, source_id, index in batch:
                image = source_groups[class_id][source_id][index]
                source_padding = padding_per_source[source_id]

                if source_padding:
                    image = image.reshape(
                        cf.SIZE_FEATURE_HEIGHT,
                        cf.SIZE_FEATURE_WIDTH,
                    )
                    image = np.pad(
                        image,
                        (
                            (source_padding, source_padding),
                            (source_padding, source_padding),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    image = image.reshape(
                        cf.SIZE_PADDED_HEIGHT,
                        cf.SIZE_PADDED_WIDTH,
                    )

                images.append(image)

            images = np.asarray(images, dtype=np.float32).reshape(
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
                (
                    None,
                    cf.SIZE_PADDED_HEIGHT,
                    cf.SIZE_PADDED_WIDTH,
                    1,
                ),
                tf.float32,
            ),
            tf.TensorSpec((None,), tf.int32),
        ),
    ).prefetch(tf.data.AUTOTUNE)