import tensorflow as tf
import numpy as np
from tensorflow import keras
from make_labeled_dataset import make_labeled_dataset
train_mtg= np.memmap(
    "data/train/train_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_mtg= np.memmap(
    "data/validation/validation_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_mtg= np.memmap(
    "data/test/test_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)

train_pkm= np.memmap(
    "data/train/train_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_pkm= np.memmap(
    "data/validation/validation_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_pkm= np.memmap(
    "data/test/test_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)

train_ygo= np.memmap(
    "data/train/train_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_ygo= np.memmap(
    "data/validation/validation_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_ygo= np.memmap(
    "data/test/test_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)


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

model = keras.Sequential([
    keras.Input(shape=(700, 500, 1)),
    # we can further reduce image size which might save time but increase bias it is keras.layers.Resizing(175, 125),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_dataset_labeled, 
          epochs=3, 
          validation_data=val_dataset_labeled,
          shuffle = False, #did our own shuffling with numpy so turning this off to avoid a warning
          steps_per_epoch=train_steps,
          validation_steps=val_steps,)

model.evaluate(test_dataset_labled)

model.save('635-CNN.keras')