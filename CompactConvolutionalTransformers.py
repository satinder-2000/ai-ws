#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:43:41 2026

@author: singh
"""
print("-----Imports-----")
print()
from keras import layers
import keras

import matplotlib.pyplot as plt
import numpy as np

print()
print("-----Hyperparameters and constants-----")
print()
positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 30
image_size = 32

print()
print("-----Load CIFAR-10 dataset-----")
print()
num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
