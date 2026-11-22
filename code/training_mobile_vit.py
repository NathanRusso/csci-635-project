import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import config as cf
import keras
from keras import layers
import numpy as np
from keras_vision.MobileViT_v2 import build_MobileViT_v2
from make_labeled_dataset import make_labeled_dataset

# Train Memmaps
train_mtg = np.memmap(cf.PATH_TRAIN_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))
train_pkm = np.memmap(cf.PATH_TRAIN_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))
train_ygo = np.memmap(cf.PATH_TRAIN_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TRAIN, cf.SIZE_PADDED_FEATURE))

# Test Memmaps
test_mtg = np.memmap(cf.PATH_TEST_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))
test_pkm = np.memmap(cf.PATH_TEST_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))
test_ygo = np.memmap(cf.PATH_TEST_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_TEST, cf.SIZE_PADDED_FEATURE))

# Validation Memmaps
val_mtg = np.memmap(cf.PATH_VALIDATION_MTG_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))
val_pkm = np.memmap(cf.PATH_VALIDATION_PKM_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))
val_ygo = np.memmap(cf.PATH_VALIDATION_YGO_2, dtype=np.float32, mode="r", shape=(cf.COUNT_VALIDATION, cf.SIZE_PADDED_FEATURE))

train_dataset_labeled = make_labeled_dataset(train_mtg, train_pkm, train_ygo, batch_size=16, shuffle=True)

val_dataset_labeled = make_labeled_dataset( val_mtg, val_pkm, val_ygo, batch_size=16)

test_dataset_labled = make_labeled_dataset( test_mtg, test_pkm, test_ygo, batch_size=16)

backbone = build_MobileViT_v2( width_multiplier=.5,
                                num_classes=3,
                                input_shape= (256,192,1)) 
#sizing the images down to speed up processing but keeping rough aspect ratio
#Also MobileViT requires images to be divisible by 32 

inputs= keras.Input(shape=(cf.SIZE_PADDED_HEIGHT, cf.SIZE_PADDED_WIDTH, 1))
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
keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience =3),
keras.callbacks.ModelCheckpoint(cf.PATH_MODEL_MOBILEVIT_2, save_best_only=True, monitor="val_accuracy"),
keras.callbacks.EarlyStopping(patience =3, monitor="val_loss", restore_best_weights=True)
]

model.fit(train_dataset_labeled, epochs =20, validation_data=val_dataset_labeled, callbacks= callbacks, shuffle = False)

test_loss, test_acc = model.evaluate(test_dataset_labled)
print(f"Test Accuracy: {test_acc:.2%}")
