"""
Handwritten Language Detection - Google Colab Training Script
==============================================================
Upload this script to Google Colab and run it with GPU enabled.

Setup Instructions:
1. Go to https://colab.research.google.com
2. Upload this file or paste the code
3. Go to Runtime → Change runtime type → GPU
4. Upload your dataset folder to Google Drive in the structure:
   My Drive/
     Handwritten Detection/
       dataset/
         english/  (images)
         hindi/    (images)
         tamil/    (images)
         telugu/   (images)
5. Run all cells
6. Download the generated model files from Google Drive
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

# ========== Mount Google Drive ==========
from google.colab import drive
drive.mount('/content/drive')

# ========== Configuration ==========
IMG_SIZE = 128
BATCH_SIZE = 8
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 30

# Paths on Google Drive
DRIVE_BASE = "/content/drive/My Drive/Handwritten Detection"
DATASET_DIR = os.path.join(DRIVE_BASE, "dataset")
MODEL_DIR = os.path.join(DRIVE_BASE, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ========== Heavy Data Augmentation ==========
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# ========== Load Dataset ==========
train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_data.num_classes
print(f"\nClasses: {train_data.class_indices}")
print(f"Training: {train_data.samples} | Validation: {val_data.samples}")
print(f"Num classes: {num_classes}\n")

# ========== Build Model ==========
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== Phase 1: Train Top Layers ==========
print("\n" + "="*50)
print("PHASE 1: Training top layers (base frozen)")
print("="*50 + "\n")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'language_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

model.fit(train_data, validation_data=val_data, epochs=INITIAL_EPOCHS, callbacks=callbacks)

# ========== Phase 2: Fine-Tune ==========
print("\n" + "="*50)
print("PHASE 2: Fine-tuning last 30 layers")
print("="*50 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'language_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

model.fit(train_data, validation_data=val_data, epochs=FINE_TUNE_EPOCHS, callbacks=fine_tune_callbacks)

# ========== Save Class Indices ==========
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f)

# ========== Final Evaluation ==========
val_loss, val_acc = model.evaluate(val_data)
print(f"\n✅ Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"✅ Model saved to: {os.path.join(MODEL_DIR, 'language_model.h5')}")
print(f"✅ Class indices saved to: {os.path.join(MODEL_DIR, 'class_indices.json')}")
print("\nDownload these 2 files from Google Drive and place them in your local 'model/' folder.")
