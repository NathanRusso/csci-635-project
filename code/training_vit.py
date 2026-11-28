"""
File to create and train a Vision Transformer (ViT) model.
"""
import config as cf
import numpy as np
import keras
import keras_hub
import tensorflow as tf
import math


# Patch Count, Feature Sizes, and more
COUNT_PATCH = 196  # (224/16) x (224/16) = 14 x 14 = 196
SIZE_FEATURE_COMPRESSED = 224
SIZE_SHUFFLE = 32
SIZE_BATCH = 16
SIZE_PATCH = 16
EPOCHS_FROZEN = 3
EPOCHS_UNFROZEN = 7
EPOCHS_TEST = 1     # Replace EPOCHS_FROZEN and EPOCHS_UNFROZEN with this for testing

# Train Memmaps #1
train_mtg_1 = np.memmap(cf.PATH_TRAIN_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))
train_pkm_1 = np.memmap(cf.PATH_TRAIN_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))
train_ygo_1 = np.memmap(cf.PATH_TRAIN_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))

# Test Memmaps #1
test_mtg_1 = np.memmap(cf.PATH_TEST_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))
test_pkm_1 = np.memmap(cf.PATH_TEST_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))
test_ygo_1 = np.memmap(cf.PATH_TEST_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))

# Validation Memmaps #1
val_mtg_1 = np.memmap(cf.PATH_VALIDATION_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))
val_pkm_1 = np.memmap(cf.PATH_VALIDATION_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))
val_ygo_1 = np.memmap(cf.PATH_VALIDATION_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))

# Train Memmaps #2
train_mtg_2 = np.memmap(cf.PATH_TRAIN_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))
train_pkm_2 = np.memmap(cf.PATH_TRAIN_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))
train_ygo_2 = np.memmap(cf.PATH_TRAIN_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))

# Test Memmaps #2
test_mtg_2 = np.memmap(cf.PATH_TEST_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))
test_pkm_2 = np.memmap(cf.PATH_TEST_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))
test_ygo_2 = np.memmap(cf.PATH_TEST_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))

# Validation Memmaps #2
val_mtg_2 = np.memmap(cf.PATH_VALIDATION_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))
val_pkm_2 = np.memmap(cf.PATH_VALIDATION_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))
val_ygo_2 = np.memmap(cf.PATH_VALIDATION_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))


def train_generator():
    """
    Train data generator used for data selection/iteration.
    """
    for i in range(cf.COUNT_TRAIN):
        # Pad the original #1 datasets
        mtg_1 = np.pad(train_mtg_1[i], ((0, 49600)), mode="constant")
        pkm_1 = np.pad(train_pkm_1[i], ((0, 49600)), mode="constant")
        ygo_1 = np.pad(train_ygo_1[i], ((0, 49600)), mode="constant")        

        # Yield the original #1 datasets
        yield mtg_1, cf.CardType.MTG.value, 1
        yield pkm_1, cf.CardType.PKM.value, 1
        yield ygo_1, cf.CardType.YGO.value, 1

        # Yield the transformed #2 datasets
        yield train_mtg_2[i], cf.CardType.MTG.value, 2
        yield train_pkm_2[i], cf.CardType.PKM.value, 2
        yield train_ygo_2[i], cf.CardType.YGO.value, 2


def validation_generator():
    """
    Validation data generator used for data selection/iteration.
    """
    for i in range(cf.COUNT_VALIDATION):
        # Pad the original #1 datasets
        mtg_1 = np.pad(val_mtg_1[i], ((0, 49600)), mode="constant")
        pkm_1 = np.pad(val_pkm_1[i], ((0, 49600)), mode="constant")
        ygo_1 = np.pad(val_ygo_1[i], ((0, 49600)), mode="constant")

        # Yield the original #1 datasets
        yield mtg_1, cf.CardType.MTG.value, 1
        yield pkm_1, cf.CardType.PKM.value, 1
        yield ygo_1, cf.CardType.YGO.value, 1

        # Yield the transformed #2 datasets
        yield val_mtg_2[i], cf.CardType.MTG.value, 2
        yield val_pkm_2[i], cf.CardType.PKM.value, 2
        yield val_ygo_2[i], cf.CardType.YGO.value, 2


def test_generator():
    """
    Test data generator used for data selection/iteration.
    """
    for i in range(cf.COUNT_TEST):
        # Pad the original #1 datasets
        mtg_1 = np.pad(test_mtg_1[i], ((0, 49600)), mode="constant")
        pkm_1 = np.pad(test_pkm_1[i], ((0, 49600)), mode="constant")
        ygo_1 = np.pad(test_ygo_1[i], ((0, 49600)), mode="constant")

        # Yield the original #1 datasets
        yield mtg_1, cf.CardType.MTG.value, 1
        yield pkm_1, cf.CardType.PKM.value, 1
        yield ygo_1, cf.CardType.YGO.value, 1

        # Yield the transformed #2 datasets
        yield test_mtg_2[i], cf.CardType.MTG.value, 2
        yield test_pkm_2[i], cf.CardType.PKM.value, 2
        yield test_ygo_2[i], cf.CardType.YGO.value, 2


def image_resizer(images, labels, delimiters):
    """
    Resizes the images for the ViT.

    1. (740, 540, 1) or (700, 500, 1) to (224, 224)
    2. (224, 224, 1) [0, 1] to (224, 224, 1) [-1, 1]
    3. (224, 224, 1) to (224, 224, 3) where the 3 channels are the same.

    Args:
        images: TensorFlow Tensor containing a batch of images.
        labels: TensorFlow Tensor containing the corresponding batch of labels.

    Returns:
        A tuple containing the reshaped images Tensor and the corresponding labels Tensor.
    """

    def resize_image(args):
        image, delimiter = args
        image = tf.cond(
            tf.equal(delimiter, 1),
            true_fn=lambda: tf.reshape(image[:cf.SIZE_FEATURE], (700, 500, 1)),
            false_fn=lambda: tf.reshape(image, (740, 540, 1))
        )
        image = tf.image.resize(image, size=(SIZE_FEATURE_COMPRESSED, SIZE_FEATURE_COMPRESSED), method="area")

        return image

    images = tf.map_fn(
        resize_image,
        (images, delimiters),
        fn_output_signature=tf.TensorSpec(shape=(SIZE_FEATURE_COMPRESSED, SIZE_FEATURE_COMPRESSED, 1), dtype=tf.float32)
    )
    images = images * 2 - 1
    images = tf.image.grayscale_to_rgb(images)
    return images, labels


def create_datasets_vit():
    """
    Creates the Train, Test, and Validation sets needed for the ViT.
    They include both the features and targets.

    Returns:
        A 3-tuple containing the train, test, and validation datasets.
    """
    # Output Generator Signature: [(399_600 features), (1 target class), (1 delimiter class)]
    output_generator_signature = (
        tf.TensorSpec(shape=(cf.SIZE_PADDED_FEATURE,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    # Set up the train dataset
    train_dataset = tf.data.Dataset.from_generator(train_generator, output_signature=output_generator_signature
    ).shuffle(buffer_size=SIZE_SHUFFLE, reshuffle_each_iteration=True
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(cf.COUNT_TRAIN * 6 / SIZE_BATCH)))

    # Set up the validation dataset
    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_signature=output_generator_signature
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(cf.COUNT_VALIDATION * 6 / SIZE_BATCH)))

    # Set up the train dataset
    test_dataset = tf.data.Dataset.from_generator(test_generator, output_signature=output_generator_signature
    ).batch(SIZE_BATCH
    ).map(image_resizer, num_parallel_calls=1
    ).prefetch(1
    ).apply(tf.data.experimental.assert_cardinality(math.ceil(cf.COUNT_TEST * 6 / SIZE_BATCH)))

    return train_dataset, test_dataset, validation_dataset


def create_train_vit():
    """
    Loads a pretrained ViT and trains it on our card data.
    Saves the model to the data file.
    """

    # Callback to stop the model if validation loss stops improving (prevent overfitting)
    model_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    # Train, Test, and Validation datasets used in the model
    train_dataset, test_dataset, validation_dataset = create_datasets_vit()

    # Downloads the 32x32 patch size, 224x224 image large ViT model
    model = keras_hub.models.ViTImageClassifier.from_preset(
        preset="vit_base_patch16_224_imagenet",
        num_classes=3,
        load_weights=True,
        activation="softmax"
    )


    # Freezes weights and biases for the model besides the classification head
    model.backbone.trainable = False

    # Sets the model optimizer, loss, and metric
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),      # Default params
        loss=keras.losses.SparseCategoricalCrossentropy(),                              # Default params
        metrics=["accuracy"]
    )

    # Trains the model on the training and validation sets
    model.fit(
        x=train_dataset,
        epochs=EPOCHS_FROZEN,
        validation_data=validation_dataset,
        callbacks=model_callbacks,
    )


    # Unfreezes weights and biases so the entire model can change
    model.backbone.trainable = True

    # Sets the model optimizer, loss, and metric ~ Makes sure 'trainable' is updated
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.00001, weight_decay=0.004),    # Not default learning rate
        loss=keras.losses.SparseCategoricalCrossentropy(),                              # Default params
        metrics=["accuracy"]
    )

    # Trains the model on the training and validation sets ~ Continues the model
    model.fit(
        x=train_dataset,
        epochs=EPOCHS_UNFROZEN,
        validation_data=validation_dataset,
        callbacks=model_callbacks,
    )


    # Checks the model's performance on the test data 
    test_loss, test_accuracy = model.evaluate(x=test_dataset)
    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3%}")

    # Saves the model to be used later.
    model.save(cf.PATH_MODEL_VIT_3)


if __name__ == "__main__":
    create_train_vit()
