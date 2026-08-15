#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:38:49 2026

@author: singh
"""
print("-----A mobile-friendly Transformer-based model for image classification-----")
print()

print("-----Imports-----")
print()
import os
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import backend

import tensorflow_datasets as tfds

tfds.disable_progress_bar()
