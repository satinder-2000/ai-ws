# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("-----Importing the libraries-----")
print()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(pd.__version__)

print(os.getcwd())

print("-----Importing the dataset-----")
print()


datafile=os.getcwd(),'/Data.csv'
print("datafile: ",datafile)
dataset = pd.read_csv('Data - Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

print("-----Taking care of missing data-----")
print()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


print("-----Encoding categorical data-----")
print()
print("-----Encoding the independent Variable-----")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
print()
print("-----Encoding the Dependent Variable-----")
print()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
print()
print("-----Splitting the dataset into the Training set and Test set-----")
print()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print()
print("-----Feature Scaling-----")
print()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
print(X_train)
print(X_test)