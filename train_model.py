import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

# ========== Configuration ==========
IMG_SIZE = 128
BATCH_SIZE = 8  # Small batch for small dataset
EPOCHS = 80
DATASET_DIR = "dataset_clean"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ========== Heavy Data Augmentation ==========
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    channel_shift_range=30,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# ========== Load Dataset (RGB for MobileNetV2) ==========
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
print("\n" + "=" * 50)
print(f"Classes found: {train_data.class_indices}")
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Number of classes: {num_classes}")
print("=" * 50 + "\n")

# ========== Build Model with MobileNetV2 Transfer Learning ==========
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all base model layers - do NOT fine-tune with noisy small data
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
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

# ========== Callbacks ==========
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'language_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'
    )
]

# ========== Train (frozen base only - stable for noisy data) ==========
print("\n" + "=" * 50)
print("Training top layers (MobileNetV2 base frozen)")
print("=" * 50 + "\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ========== Save Class Indices ==========
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f)

# ========== Final Evaluation ==========
print("\n" + "=" * 50)
val_loss, val_acc = model.evaluate(val_data)
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")
print("=" * 50)
print("\nModel Saved Successfully to model/language_model.h5")