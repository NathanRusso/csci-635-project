import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import numpy as np
import tensorflow as tf
from keras_vision.MobileViT_v2 import build_MobileViT_v2
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
train_dataset_labeled = make_labeled_dataset(train_mtg, train_pkm, train_ygo, batch_size=24, shuffle=True)

val_dataset_labeled = make_labeled_dataset( val_mtg, val_pkm, val_ygo, batch_size=24)

test_dataset_labled = make_labeled_dataset( test_mtg, test_pkm, test_ygo, batch_size=24)

backbone = build_MobileViT_v2( width_multiplier=.5,
                                num_classes=3,
                                input_shape= (256,192,1)) 
#sizing the images down to speed up processing but keeping rough aspect ratio
#Also MobileViT requires images to be divisible by 32 

inputs= keras.Input(shape=(700,500,1))
x = layers.Resizing(256,192) (inputs)
logits = backbone(x)
outputs = layers.Softmax()(logits)

model = keras.Model(inputs=inputs, outputs=outputs, name="trading_card_mobilevit_v2")

model.compile(
    optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


callbacks =[
keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience =2),
keras.callbacks.ModelCheckpoint("635-MobileVit_v2.keras", save_best_only=True, monitor="val_accuracy"),
keras.callbacks.EarlyStopping(patience =3, monitor="val_loss", restore_best_weights=True)
]

model.fit(train_dataset_labeled, epochs =1, validation_data=val_dataset_labeled, callbacks= callbacks, shuffle = False)

test_loss, test_acc = model.evaluate(test_dataset_labled)
print(f"Test Accuracy: {test_acc:.2%}")
