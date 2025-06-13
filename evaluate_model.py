import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Path to the dataset directory and Excel files
real_images_excel = r"C:\Users\suhas\Downloads\proj phase 2\datasets\real_and_fake_face\training_real\real.xlsx"
fake_images_excel = r"C:\Users\suhas\Downloads\proj phase 2\datasets\real_and_fake_face\training_fake\fake.xlsx"
base_dir = r'C:/Users/rahma/OneDrive/Desktop/dataset/'

# Load image paths from Excel files
real_df = pd.read_excel(real_images_excel)
fake_df = pd.read_excel(fake_images_excel)

# Prepend base directory path to all image paths
real_image_paths = [os.path.join(base_dir, 'real', path) for path in real_df['Image Path'].values]
fake_image_paths = [os.path.join(base_dir, 'fake', path) for path in fake_df['Image Path'].values]

# Assign labels (0 for real, 1 for fake)
real_labels = np.zeros(len(real_image_paths))
fake_labels = np.ones(len(fake_image_paths))

# Combine image paths and labels
image_paths = np.concatenate((real_image_paths, fake_image_paths), axis=0)
labels = np.concatenate((real_labels, fake_labels), axis=0)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Data generator
def create_data_generator(paths, labels, batch_size=32, target_size=(128, 128)):
    while True:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = [preprocess_image(p, target_size) for p in batch_paths]
            yield np.array(images), np.array(batch_labels)

# Load the trained model
model = tf.keras.models.load_model('fake_news_detection_model.h5')

# Compile the model (you may adjust the loss function and optimizer based on your previous configuration)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on validation set
val_gen = create_data_generator(val_paths, val_labels, batch_size=32)
val_steps = len(val_paths) // 32

# Get predictions on the validation set
val_preds = model.predict(val_gen, steps=val_steps)
val_preds_binary = (val_preds > 0.5).astype(int)

# Ensure val_labels matches the number of predictions
val_labels = val_labels[:len(val_preds_binary)]

# Generate the confusion matrix and classification report
conf_matrix = confusion_matrix(val_labels, val_preds_binary)
class_report = classification_report(val_labels, val_preds_binary)

# Print results
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Optional: Plot confusion matrix as heatmap
import seaborn as sns

plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
