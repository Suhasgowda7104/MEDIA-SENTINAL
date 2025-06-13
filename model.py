import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Path to Excel files and images
real_images_excel = r"C:\Users\suhas\Downloads\proj phase 2\datasets\real_and_fake_face\training_real\real.xlsx"
fake_images_excel = r"C:\Users\suhas\Downloads\proj phase 2\datasets\real_and_fake_face\training_fake\fake.xlsx"

# Load image paths from Excel files
real_df = pd.read_excel(real_images_excel)
fake_df = pd.read_excel(fake_images_excel)

real_image_paths = real_df['Image Path'].values
fake_image_paths = fake_df['Image Path'].values

# Assign labels
real_labels = np.zeros(len(real_image_paths))  # 0 for real
fake_labels = np.ones(len(fake_image_paths))   # 1 for fake

# Combine image paths and labels
image_paths = np.concatenate((real_image_paths, fake_image_paths), axis=0)
labels = np.concatenate((real_labels, fake_labels), axis=0)

# Split the data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Define image preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Create image data generator for batch processing
def create_data_generator(paths, labels, batch_size=32, target_size=(128, 128)):
    while True:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = [preprocess_image(p, target_size) for p in batch_paths]
            yield np.array(images), np.array(batch_labels)  # Return inputs, targets tuple

# Build the CNN model
def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Instantiate the model
model = build_model()

# Training parameters
batch_size = 32
epochs = 30  # Increased epochs for better training
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

# Enhanced data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    validation_split=0.2
)

# Create training and validation generators
train_gen = create_data_generator(train_paths, train_labels, batch_size)
val_gen = create_data_generator(val_paths, val_labels, batch_size)

# Train the model
history = model.fit(train_gen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=val_gen, 
                    validation_steps=validation_steps, 
                    epochs=epochs)  # Increased epochs for better training

# Save the model
model.save('fake_news_detection_model.h5')

# Optional: Display training results
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
