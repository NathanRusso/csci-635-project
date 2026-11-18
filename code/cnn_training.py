import config as cf
import tensorflow as tf
import numpy as np
from tensorflow import keras
from make_labeled_dataset import make_labeled_dataset

# Train Memmaps
train_mtg = np.memmap(cf.PATH_TRAIN_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))
train_pkm = np.memmap(cf.PATH_TRAIN_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))
train_ygo = np.memmap(cf.PATH_TRAIN_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_FEATURE))

# Test Memmaps
test_mtg = np.memmap(cf.PATH_TEST_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))
test_pkm = np.memmap(cf.PATH_TEST_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))
test_ygo = np.memmap(cf.PATH_TEST_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_FEATURE))

# Validation Memmaps
val_mtg = np.memmap(cf.PATH_VALIDATION_MTG, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))
val_pkm = np.memmap(cf.PATH_VALIDATION_PKM, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))
val_ygo = np.memmap(cf.PATH_VALIDATION_YGO, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_FEATURE))

# size 32 batches proved to be the perfect balance between number of batches and time per batch.
# Average of 700ms per step with size 32 and anything more is 2 seconds or more per batch

train_dataset_labeled = make_labeled_dataset(train_mtg, train_pkm, train_ygo, batch_size=32, shuffle=True)

val_dataset_labeled = make_labeled_dataset( val_mtg, val_pkm, val_ygo, batch_size=32)

test_dataset_labled = make_labeled_dataset( test_mtg, test_pkm, test_ygo, batch_size=32)

train_steps = sum(
                (len(data)+ 31) // 32
                for data in (train_mtg, train_pkm, train_ygo)
                )
val_steps = sum(
                (len(data)+ 31) // 32
                for data in (val_mtg, val_pkm, val_ygo)
                )

kernel_regularizer = keras.regularizers.l2(1e-4)

model = keras.Sequential([
    keras.Input(shape=(cf.SIZE_FEATURE_HEIGHT, cf.SIZE_FEATURE_WIDTH, 1)),
    # we can further reduce image size which might save time but increase bias it is keras.layers.Resizing(175, 125),
    keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=kernel_regularizer),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=kernel_regularizer),
    keras.layers.MaxPooling2D((2, 2)),
     keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=kernel_regularizer),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(3, activation='softmax', kernel_regularizer=kernel_regularizer)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callbacks =[
keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience =2),
keras.callbacks.ModelCheckpoint("data/635-CNN.keras", save_best_only=True, monitor="val_accuracy"),
keras.callbacks.EarlyStopping(patience =3, monitor="val_loss", restore_best_weights=True)
]


model.fit(train_dataset_labeled, 
          epochs=20,
          validation_data=val_dataset_labeled,
          shuffle = False, #did our own shuffling with numpy so turning this off to avoid a warning
          steps_per_epoch=train_steps,
          validation_steps=val_steps,
          callbacks=callbacks,
          )

model.evaluate(test_dataset_labled)