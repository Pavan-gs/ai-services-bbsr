# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:28:41 2026

@author: admin
"""


# ============================================================
# CAT VS DOG CLASSIFICATION USING CNN - MODERN KERAS 3
# ============================================================

# Install if required:
# pip install --upgrade keras tensorflow matplotlib

import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

TRAIN_DIR = r"D:\Deep_Learning\Convolutional_Neural_Networks\dataset\training_set"

TEST_DIR = r"D:\Deep_Learning\Convolutional_Neural_Networks\dataset\test_set"

PREDICTION_IMAGE = r"D\Deep_Learning\Convolutional_Neural_Networks\dataset\pred\cat_or_dog_1.jpg"

'''
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(128, 128),
    batch_size=32
)

model.fit(
    train_ds,
    epochs=20
)'''

# ------------------------------------------------------------
# 2. LOAD TRAINING DATASET
# ------------------------------------------------------------

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=SEED
)

# ------------------------------------------------------------
# 3. LOAD TEST DATASET
# ------------------------------------------------------------

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)


# Get class names
class_names = train_ds.class_names

print("Class names:", class_names)

# Expected:
# ['cats', 'dogs']


# ------------------------------------------------------------
# 4. OPTIMIZE DATA PIPELINE
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------
# 5. IMAGE DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1
        ),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation"
)


# ------------------------------------------------------------
# 6. BUILD CNN MODEL
# ------------------------------------------------------------

model = keras.Sequential(
    [
        # Input
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Data augmentation
        data_augmentation,

        # Normalize pixel values from 0-255 to 0-1
        layers.Rescaling(1.0 / 255),

        # CNN Block 1
        layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 2
        layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 3
        layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Convert feature maps into a vector
        layers.Flatten(),

        # Fully connected layers
        layers.Dense(128, activation="relu"),

        # Reduce overfitting
        layers.Dropout(0.5),

        # Binary classification output
        layers.Dense(1, activation="sigmoid")
    ],
    name="cat_dog_cnn"
)


# ------------------------------------------------------------
# 7. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Display model architecture
model.summary()


# ------------------------------------------------------------
# 8. CALLBACKS
# ------------------------------------------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ------------------------------------------------------------
# 9. TRAIN MODEL
# ------------------------------------------------------------

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[early_stopping]
)


# ------------------------------------------------------------
# 10. EVALUATE MODEL
# ------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ------------------------------------------------------------
# 11. PLOT ACCURACY AND LOSS
# ------------------------------------------------------------

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    acc,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_acc,
    label="Validation Accuracy"
)

plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")


# Loss plot
plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tight_layout()

plt.savefig("cnn_training_results.png")

plt.show()


# ------------------------------------------------------------
# 12. PREDICT A SINGLE IMAGE
# ------------------------------------------------------------

test_image = keras.utils.load_img(
    PREDICTION_IMAGE,
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

test_image = keras.utils.img_to_array(test_image)

# Add batch dimension
test_image = np.expand_dims(test_image, axis=0)

# Predict probability
prediction = model.predict(test_image)

probability = float(prediction[0][0])

# Convert probability into class
predicted_class = (
    class_names[1]
    if probability >= 0.5
    else class_names[0]
)

print("\nPrediction:", predicted_class)
print("Dog probability:", probability)
print("Cat probability:", 1 - probability)


# ------------------------------------------------------------
# 13. SAVE TRAINED MODEL
# ------------------------------------------------------------

model.save("cat_dog_cnn.keras")

print("\nModel saved successfully!")
