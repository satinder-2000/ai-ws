#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 19:24:24 2026

@author: singh
"""
print("-----Importing the libraries-----")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("-----Importing the dataset-----")
print()
dataset = pd.read_csv('Data - Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
print()
print("-----Taking care of missing data-----")
print()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

print()
print("-----Encoding categorical data-----")
# Encoding categorical data
# Encoding the Independent Variable
print("---Encoding the Independent Variable---")
print()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
print("---Encoding the dependent Variable---")
print()
#Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
print("-----Splitting the DataSet into Training Set and Test Set-----")
print()
#Splitting the DataSet into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("X_train:\n",X_train)
print("X_test:\n",X_test)
print("y_train:\n",y_train)
print("y_test:\n",y_test)

print()
print("-----Feature Scaling-----")
print()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
print("X_train:\n",X_train)
print("X_test:\n",X_test)