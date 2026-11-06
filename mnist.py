
# -*- coding: utf-8 -*-
"""
CNN - MNIST Dataset
Updated for current TensorFlow / Keras APIs
"""

'''Input image
28 × 28 × 1

Grayscale image

Conv2D (32 filters, 3 × 3)

26 × 26 × 32

MaxPooling2D (2 × 2)

13 × 13 × 32

Conv2D (64 filters, 3 × 3)

11 × 11 × 64

MaxPooling2D (2 × 2)

5 × 5 × 64

Conv2D (64 filters, 3 × 3)

3 × 3 × 64

MaxPooling2D (2 × 2)

0 × 0 — invalid**'''


# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv2D, MaxPooling2D, Flatten
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

# --------------------------------------------------
# 2. Load MNIST dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --------------------------------------------------
# 3. Explore dataset
# --------------------------------------------------

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

print("Number of Training Images:", x_train.shape[0])
print("Number of Testing Images:", x_test.shape[0])

print("First Image:")
print(x_train[0])

# --------------------------------------------------
# 4. Display sample images
# --------------------------------------------------

plt.figure(figsize=(10, 8))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title("Digit: " + str(y_train[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Reshape images to include channel dimension
# --------------------------------------------------

x_train = x_train.reshape(
    x_train.shape[0], 28, 28, 1
)

x_test = x_test.reshape(
    x_test.shape[0], 28, 28, 1
)

print("Reshaped Training Data:", x_train.shape)
print("Reshaped Testing Data:", x_test.shape)

# --------------------------------------------------
# 6. Convert labels to categorical
# --------------------------------------------------

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# --------------------------------------------------
# 7. Normalize images
# --------------------------------------------------

train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')

train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

# --------------------------------------------------
# 8. Build CNN model
# --------------------------------------------------

model = Sequential()

model.add(keras.Input(shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))

# --------------------------------------------------
# 9. Compile model
# --------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

m2 = model.fit(
    train_norm,
    y_train,
    epochs=10,
    batch_size=25,
    validation_split=0.1
)

# --------------------------------------------------
# 11. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_norm,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# --------------------------------------------------
# 12. Predict
# --------------------------------------------------

pred = model.predict(test_norm)

print("First Prediction:", pred[0])

print("Predicted Digit:", np.argmax(pred[0]))

print("Actual Digit:", np.argmax(y_test[0]))

print("Prediction Confidence:", np.max(pred[0]))

# --------------------------------------------------
# 13. Confusion Matrix and Classification Report
# --------------------------------------------------

y_pred = np.argmax(pred, axis=1)
y_actual = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_actual, y_pred)

print("Confusion Matrix:")
print(cm)

print(classification_report(
    y_actual,
    y_pred
))

# --------------------------------------------------
# 14. Model Summary
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# 15. Plot Accuracy and Loss
# --------------------------------------------------

acc = m2.history['accuracy']
val_acc = m2.history['val_accuracy']

loss = m2.history['loss']
val_loss = m2.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig('./mnist_training.png')
plt.show()

# --------------------------------------------------
# 16. Predict External Handwritten Digit
# --------------------------------------------------

img = load_img(
    'D:/Deloitte/MNIST_image_7.png',
    color_mode='grayscale',
    target_size=(28, 28)
)

img = img_to_array(img)

img = img.astype('float32')
img = img / 255.0

img = img.reshape(1, 28, 28, 1)

pred_digit = model.predict(img)

print("Predicted Digit:", np.argmax(pred_digit))
print("Confidence:", np.max(pred_digit))
