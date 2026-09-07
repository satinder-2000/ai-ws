#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:56:34 2026

@author: singh
"""
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1)
print(mnist.keys())

X,y = mnist['data'],mnist['target']
print("\nX.shape:\n",X.shape)
print("\ny.shape:\n",y.shape)

print("\nDisplay a sample image using matplotlib\n")
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()