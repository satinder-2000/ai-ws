# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("Source: https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/")
print()


print("-----Step 1: Import Necessary Libraries-----")
print()

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("-----Step 2: Create and Load Dataset-----")
print()

data = {
    'feature1' : [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2' : [0.5, 0.4, 0.3, 0.2, 0.1],
    'label' : [0, 0, 1, 1, 1]   
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values

print("-----Step 3: Create a Neural Network-----")
print()

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("-----Step 4: Compiling the Model-----")
print()
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print("-----Step 5: Train the Model-----")
print()
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

print("-----Step 6: Make Predictions-----")
print()
test_data = np.array([0.2, 0.4])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
print(predicted_label)

