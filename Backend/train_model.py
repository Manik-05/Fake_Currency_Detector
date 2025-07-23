import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# --- 1. Configuration ---
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3 # RGB images
BATCH_SIZE = 32
EPOCHS = 25 # Increased epochs - you might need more (e.g., 25-50 or even more)
NUM_CLASSES = 2 # Authentic, Fake (binary classification)
MODEL_SAVE_PATH = 'cnn_model.h5' # Name of the file to save the trained model

# --- 2. Data Loading from Your Dataset ---
# Your main dataset path
BASE_DATASET_PATH = r'C:\AIML Project 2\dataset' # Use 'r' for raw string to handle backslashes

# Paths to your training and validation subdirectories
TRAIN_DATA_PATH = os.path.join(BASE_DATASET_PATH, 'training')
VALIDATION_DATA_PATH = os.path.join(BASE_DATASET_PATH, 'validation')

# Check if the dataset paths exist
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"Error: Training dataset path '{TRAIN_DATA_PATH}' does not exist.")
    print("Please ensure your dataset is located at this path and has 'fake' and 'real' subfolders.")
    exit() # Exit if dataset path is invalid
if not os.path.exists(VALIDATION_DATA_PATH):
    print(f"Error: Validation dataset path '{VALIDATION_DATA_PATH}' does not exist.")
    print("Please ensure your dataset is located at this path and has 'fake' and 'real' subfolders.")
    exit() # Exit if dataset path is invalid


print(f"Loading training images from: {TRAIN_DATA_PATH}")
print(f"Loading validation images from: {VALIDATION_DATA_PATH}")

# Initialize ImageDataGenerator for training and validation
# - rescale=1./255: Normalizes pixel values from [0, 255] to [0, 1]
# - Added Data Augmentation parameters for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Rotate images by up to 20 degrees
    width_shift_range=0.2,   # Shift images horizontally by up to 20% of width
    height_shift_range=0.2,  # Shift images vertically by up to 20% of height
    shear_range=0.2,         # Apply shear transformation
    zoom_range=0.2,          # Zoom in/out by up to 20%
    horizontal_flip=True,    # Randomly flip images horizontally
    brightness_range=[0.8, 1.2], # Adjust brightness
    fill_mode='nearest'      # Fill in new pixels created by transformations
)

# Validation data should typically not be augmented, only rescaled
validation_datagen = ImageDataGenerator(rescale=1./255) # No validation_split needed here as data is already split


# Create training data generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH, # Point to the 'training' subdirectory
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), # Resize images to this size
    batch_size=BATCH_SIZE,
    class_mode='binary', # 'binary' for 2 classes (fake/real)
    seed=42 # For reproducibility
)

# Create validation data generator
validation_generator = validation_datagen.flow_from_directory( # Use validation_datagen here
    VALIDATION_DATA_PATH, # Point to the 'validation' subdirectory
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=42 # For reproducibility
)

# Check if generators found any images
if train_generator.samples == 0:
    print("Error: No training images found. Please check your dataset structure.")
    print(f"Expected structure: {TRAIN_DATA_PATH}/fake/ and {TRAIN_DATA_PATH}/real/")
    exit()
if validation_generator.samples == 0:
    print("Error: No validation images found. Please check your dataset structure.")
    print(f"Expected structure: {VALIDATION_DATA_PATH}/fake/ and {VALIDATION_DATA_PATH}/real/")
    exit()

print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")

# --- IMPORTANT: Check for Class Imbalance ---
# If you see a large disparity in the number of samples per class,
# consider techniques like class weighting or oversampling/undersampling.
# The class_indices will show you the mapping: e.g., {'fake': 0, 'real': 1} or vice-versa
print("\nClass mapping determined by flow_from_directory (alphabetical order of folder names):")
print(train_generator.class_indices)
# You can also check the distribution of classes in your training set:
unique_train_classes, counts_train = np.unique(train_generator.classes, return_counts=True)
print(f"Training class distribution: {dict(zip(unique_train_classes, counts_train))}")
unique_val_classes, counts_val = np.unique(validation_generator.classes, return_counts=True)
print(f"Validation class distribution: {dict(zip(unique_val_classes, counts_val))}")


# --- 3. Define the CNN Model Architecture ---
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional Layer 1
        # Filters: 32, Kernel Size: (3,3), Activation: ReLU
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Max Pooling Layer 1: Reduces spatial dimensions
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output for the Dense (fully connected) layers
        layers.Flatten(),

        # Dense Layer 1
        layers.Dense(128, activation='relu'),
        # Dropout for regularization: Helps prevent overfitting by randomly setting a fraction of input units to 0
        layers.Dropout(0.5),

        # Output Layer
        # For binary classification (2 classes), 'sigmoid' activation outputs a probability between 0 and 1.
        # If you had more than 2 classes, you'd use 'softmax'.
        # Corrected: For binary classification with binary_crossentropy and integer labels,
        # the output layer should have 1 neuron.
        layers.Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    return model

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES)

# --- 4. Compile the Model ---
# Optimizer: 'adam' is a popular choice
# Loss function: 'binary_crossentropy' is used for binary classification problems
# Metrics: 'accuracy' to monitor training progress
# Explicitly setting from_logits=False, though it's the default with sigmoid activation
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary() # Print a summary of the model architecture

# --- 5. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    train_generator, # Use the training data generator
    epochs=EPOCHS, # Number of times to iterate over the entire dataset
    validation_data=validation_generator, # Use the validation data generator
    # steps_per_epoch=train_generator.samples // BATCH_SIZE, # Optional: Define steps per epoch
    # validation_steps=validation_generator.samples // BATCH_SIZE # Optional: Define validation steps
)
print("Model training finished.")

# --- 6. Save the Trained Model ---
try:
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")
    print("You can now use this model in your Flask backend (app.py).")
    print(f"Make sure '{MODEL_SAVE_PATH}' is in the same directory as your 'app.py' or update the path in 'app.py'.")
except Exception as e:
    print(f"Error saving model: {e}")
