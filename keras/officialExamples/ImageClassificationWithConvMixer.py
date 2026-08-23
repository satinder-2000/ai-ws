#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:06:00 2026

@author: singh
"""
print("Source: https://keras.io/examples/vision/convmixer/")
print()
print("-----Imports-----")
print()
import keras
from keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print()
print("-----Hyperparameters-----")
print()
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 10

print()
print("-----Load the CIFAR-10 dataset-----")
print()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:],y_train[val_indices:]
x_val, y_val = x_train[:val_indices],y_train[:val_indices]

print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")

print()
print("-----Prepare tf.data.Dataset objects-----")
print()
image_size = 32
auto = tf.data.AUTOTUNE

augmentation_layers = [
    keras.layers.RandomCrop(image_size, image_size),
    keras.layers.RandomFlip("horizontal")
]


def augment_images(images):
    for layer in augmentation_layers:
        images = layer(images, training = True)
    return images

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (augment_images(x),y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)

train_dataset = make_datasets(new_x_train,new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)

    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x[0]]) #Residual.
    
    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    
    return x


def get_conv_mixer_256_8(
        image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10
        ):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    
    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)
    
    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)
        
    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)

print()
print("-----Model training and evaluation utility-----")
print()
# Code reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/.

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    model.compile(
        optimizer = optimizer,
        loss = "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    checkpoint_filepath= "/tmp/checkpoint.keras"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )
    
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100,2)}%")
    
    return history, model

print()
print("-----Train and evaluate model-----")
print()
conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)