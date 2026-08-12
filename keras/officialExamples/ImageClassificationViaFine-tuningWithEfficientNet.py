#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:49:48 2026

@author: singh
"""
print("-----Keras implementation of EfficientNet-----")
print()
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')

print("-----Example: EfficientNetB0 for Stanford Dogs.-----")
print()
print("-----Setup and data loading.-----")
print()
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.applications import EfficientNetB0

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
BATCH_SIZE = 64
