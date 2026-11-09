# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:17:54 2026

@author: admin
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
print("TF version:", tf.__version__)

IMG_SIZE = (160, 160)
BATCH = 32

data_dir = keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True)
# IMPORTANT: Access the inner flower_photos directory
import os
data_dir = os.path.join(data_dir, "flower_photos")

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print(NUM_CLASSES, "classes:", class_names)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base = MobileNetV2(input_shape=IMG_SIZE + (3,),
                   include_top=False,
                   weights="imagenet")
base.trainable = False
print("Base layers:", len(base.layers))
print("Output feature-map shape:", base.output_shape)

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)          # scale pixels to [-1, 1]
x = base(x, training=False)           # keep BatchNorm frozen (inference mode)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

trainable = int(sum(tf.size(w) for w in model.trainable_weights))
total = int(sum(tf.size(w) for w in model.weights))
print(f"Trainable params: {trainable:,} / {total:,}")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)
print("Phase-1 val accuracy:", round(history.history["val_accuracy"][-1], 3))


base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

n_trainable = sum(l.trainable for l in base.layers)
print("Trainable base layers:", n_trainable)

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_ft = model.fit(train_ds, validation_data=val_ds,
                       epochs=20, initial_epoch=10)
print("Fine-tuned val accuracy:", round(history_ft.history["val_accuracy"][-1], 3))

# Fresh clone of the trained model so we start from the same weights.
bad = keras.models.clone_model(model)
bad.set_weights(model.get_weights())
bad.compile(optimizer=keras.optimizers.Adam(1e-3),   # too high for fine-tuning
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
before = model.evaluate(val_ds, verbose=0)[1]
after = bad.fit(train_ds, validation_data=val_ds, epochs=3, verbose=0)
after = after.history["val_accuracy"][-1]
print(f"val acc before: {before:.3f}  -> after high-LR fine-tune: {after:.3f}")

images, labels = next(iter(val_ds))
probs = model.predict(images, verbose=0)
pred = probs.argmax(axis=1)
correct = int((pred == labels.numpy()).sum())
print(f"{correct}/{len(labels)} correct in this batch")
for i in range(5):
    print(f"true={class_names[labels[i]]:<11} pred={class_names[pred[i]]:<11} p={probs[i].max():.2f}")
