#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:53:25 2026

@author: singh
"""
print("-----Importing libraries-----")
print()
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
print()
print("-----Data preprocessing-----")
print("---Importing the dataset---")
print()

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print("X:\n",X)
print()
print("y: ",y)

print()
print("-----Encoding categorical data-----")
print("---Label Encoding the Gender column---")
print()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print("X:\n",X)
print()
print("---One Hot Encoding the 'Geography' column")
print()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("X:\n",X)
print()
print("---Splitting the dataset into the Training set and Test set---")
print()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print()
print("---Feature Scaling---")
print()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X_train: \n",X_train)
print("X_test: \n",X_test)
print()
print("-----Part 2 - Building the ANN-----")
print()
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
print()
print("-----Part 3 - Training the ANN-----")
print()
# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)
print()
print("-----Part 4 - Making the predictions and evaluating the model-----")
# Predicting the result of a single observation
print()
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)