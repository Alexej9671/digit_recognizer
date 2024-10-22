#!/usr/bin/env python
# coding: utf-8

"""
train_augmented2.py

This script trains a convolutional neural network on the MNIST dataset using TensorFlow and Keras. 
It includes data augmentation, custom callbacks, and model evaluation. The model and training logs 
are saved upon completion.

Attributes:
    BATCH_SIZE (int): Number of samples per batch.
    EPOCHS (int): Number of training epochs.
    LEARNING_RATE (float): Initial learning rate for the optimizer.
    LOG_DIR (str): Directory path for saving TensorBoard logs.
    REDUCE_LR_FACTOR (float): Factor by which to reduce the learning rate if performance plateaus.
    REDUCE_LR_PATIENCE (int): Epochs with no improvement before reducing the learning rate.
    REDUCE_LR_MIN (float): Minimum value for the learning rate.
    EARLY_STOPPING_PATIENCE (int): Epochs with no improvement before stopping training.

Functions:
    prepare_datasets(batch_size): Prepares and returns the training and validation datasets.
    build_model(): Constructs and returns a CNN model for training.
    get_callbacks(log_dir): Returns a list of TensorFlow callbacks for training management.

Usage:
    python train_augmented2.py
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import traceback
from custom_callbacks.custom_callbacks import LearningRateLogger, ImageLogger, ConfusionMatrixLogger, F1ScoreLogger
import yaml

# Suppress a warning related to TensorFlow optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load configuration from the YAML file
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Access configurations from the loaded config dictionary
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LEARNING_RATE = config['learning_rate']
LOG_DIR = config['log_dir']
REDUCE_LR_FACTOR = config['reduce_lr']['factor']
REDUCE_LR_PATIENCE = config['reduce_lr']['patience']
REDUCE_LR_MIN = config['reduce_lr']['min_lr']
EARLY_STOPPING_PATIENCE = config['early_stopping_patience']


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to batch and shuffle the datasets
def prepare_datasets(batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_dataset, val_dataset

# Prepare the training and validation datasets
train_dataset, val_dataset = prepare_datasets(BATCH_SIZE)

# Function to build the convolutional neural network model
def build_model():
    model = tf.keras.models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Input layer
        layers.Rescaling(1. / 255),  # Rescale pixel values to [0, 1]
        layers.RandomRotation(0.1),  # Randomly rotate images for augmentation
        layers.RandomZoom(0.1),  # Randomly zoom images for augmentation
        layers.RandomTranslation(0.1, 0.1),  # Randomly translate images for augmentation
        layers.Conv2D(32, (3, 3), activation='relu'),  # Convolutional layer with 32 filters
        layers.BatchNormalization(),  # Batch normalization to stabilize training
        layers.MaxPooling2D((2, 2)),  # Max pooling to reduce spatial dimensions
        layers.Dropout(0.15),  # Dropout to reduce overfitting
        layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer with 64 filters
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),
        layers.Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer
        layers.BatchNormalization(),
        layers.Flatten(),  # Flatten the feature maps
        layers.Dense(64, activation='relu'),  # Fully connected layer
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')  # Output layer for classification
    ])
    return model




# Function to define callbacks used during training
def get_callbacks(log_dir):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)  # Stop if validation loss does not improve
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model.keras', save_best_only=True)  # Save the best model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)  # TensorBoard for visualization
    # Reduce learning rate if validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1)  
    learning_rate_logger = LearningRateLogger()  # Custom callback to log learning rate
    image_logger = ImageLogger(log_dir=log_dir, test_images=x_train)  # Custom callback to log images
    confusion_matrix_logger = ConfusionMatrixLogger(val_data=(x_test, y_test), log_dir=log_dir)  # Custom callback to log confusion matrix
    f1_score_logger = F1ScoreLogger(val_data=(x_test, y_test))  # Custom callback to log F1 score
    return [early_stopping, model_checkpoint, tensorboard_callback, reduce_lr, learning_rate_logger, image_logger, confusion_matrix_logger, f1_score_logger]


def train_model(model, train_dataset, val_dataset, callbacks, epochs):
    try:
        model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )
    except tf.errors.ResourceExhaustedError as e:
        print(f"Resource exhausted: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        traceback.print_exc()

def evaluate_model(model, val_dataset):
    test_loss, test_acc = model.evaluate(val_dataset)
    print(f'Test accuracy: {test_acc:.4f}')

def save_model(model, filepath='model.keras'):
    model.save(filepath)
    print('Model saved successfully.')


if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    callbacks = get_callbacks(LOG_DIR)
    train_model(model, train_dataset, val_dataset, callbacks, EPOCHS)
    evaluate_model(model, val_dataset)
    save_model(model)

