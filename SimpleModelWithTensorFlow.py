#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:26:37 2026

@author: singh
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf


(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(
    (train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape(
    (test_images.shape[0], 28 * 28)).astype('float32') / 255

#print(train_images.shape)
model = Sequential([
    Dense(128, activation='relu',input_shape=(28*28,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")