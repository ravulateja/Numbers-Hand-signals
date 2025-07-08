import sys


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import kaggle

# Download dataset
kaggle.api.authenticate()
kaggle.api.dataset_download_files('muhammadkhalid/sign-language-for-numbers', path='.', unzip=True)

# Image parameters
img_height, img_width = 64, 64
batch_size = 32

# Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    './Sign Language for Numbers',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    './Sign Language for Numbers',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build CNN
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(128, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(11, activation='softmax'))

# Compile
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = cnn.fit(train, validation_data=val, epochs=20)

# Save model
cnn.save("hand_gesture_model.keras")

# Evaluate
loss, accuracy = cnn.evaluate(val)
print("Validation Accuracy:", accuracy)
