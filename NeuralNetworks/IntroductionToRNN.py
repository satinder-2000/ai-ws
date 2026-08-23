#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:15:44 2026

@author: singh
"""
print("Source:https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/")
print("-----Implementing a Text Generator Using RNN")
print()

print()
print("-----Importing Necessary Libraries-----")
print()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

print()
print("-----Defining the Input Text and Prepare Character Set-----")
print()
text = "This is GeeksforGeeks a software training institute"
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
#print(char_to_index)
#print(index_to_char)

print()
print("-----Creating Sequences and Labels-----")
print()
seq_length = 3
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])
    
X = np.array(sequences)
y = np.array(labels)

print()
print("-----Converting Sequences and Labels to One-Hot Encoding-----")
print()
X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

print()
print("-----Building the RNN Model-----")
print()
model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_length, len(chars)), activation='relu'))
model.add(Dense(len(chars), activation='softmax'))

print()
print("-----Compiling and Training the Model-----")
print()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_one_hot,y_one_hot,epochs=100)

print()
print("-----Generating New Text Using the Trained Model-----")
print()
start_seq = "This is G"
generated_text = start_seq

for i in range(50):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    prediction = model.predict(x_one_hot)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char

print("Generated Text:")
print(generated_text)