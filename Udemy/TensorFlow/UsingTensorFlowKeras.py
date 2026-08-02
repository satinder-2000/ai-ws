#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:40:09 2026

@author: singh
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=64, activation='relu',input_shape=(784,)),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse.categorical_crossentropy',
metrics = ['accuracy'])  