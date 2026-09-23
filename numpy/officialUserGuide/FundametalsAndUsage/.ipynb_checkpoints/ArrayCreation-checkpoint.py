#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:13:17 2026

@author: singh
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("\nThere are 6 general mechanisms for creating arrays\n")
print("\n1------Convert Python sequences to np arrays------\n")
a1D = np.array([1, 2, 3, 4])
print("\nFrom 1D Array:\n", a1D)
a2D = np.array([[1, 2], [3, 4]])
print("\nFrom 2D Array:\n", a2D)
a3D = np.array([[[1, 2], [3, 4]],[[5, 6],[7, 8]]])
print("\nFrom 3D Array:\n", a3D)
print("\n In general, any array object is called an ndarray in NumPy.\n")

print("\nWhen you use numpy.array to define a new array, you should consider the dtype of the elements in the array, which can be specified explicitly.")
print("When values do not fit and you are using a dtype, NumPy may raise an error\n")
#np.array([127, 128, 129], dtype=np.int8)

print("\nIf you perform calculations with mismatching dtypes, you can get unwanted results\n")
a = np.array([2, 3, 4], dtype=np.uint32)
b = np.array([5, 6, 7], dtype=np.uint32)
c_unsigned32 = a - b
print("\nc_unsigned32:\n",c_unsigned32)
print("\nc_unsigned32.dtype:\n",c_unsigned32.dtype)

c_signed32 = a - b.astype(np.int32)
print("\nc_signed32:\n",c_signed32)
print("\nc_signed32.dtype:\n",c_signed32.dtype)


print("\n-----2) Intrinsic NumPy array creation functions-----\n")
print("1D array creation functions . numpy.linspace and numpy.arange ")
print("np.arange(10):\n",np.arange(10))
print("np.arange(2, 3, 0.1):\n",np.arange(2, 3, 0.1))
print("np.arange(2, 10, dtype=np.float64):\n",np.arange(2, 10, dtype=np.float64))
print("np.linspace(1., 4., 6):\n",np.linspace(1., 4., 6))


print("\n2D array creation functions:  numpy.eye, numpy.diag, and numpy.vander")
print("np.eye(3):\n",np.eye(3))
print("np.eye(3,5):\n",np.eye(3,5))

print("np.diag([1, 2, 3]):\n",np.diag([1, 2, 3]))
print("np.diag([1, 2, 3], 1):\n",np.diag([1, 2, 3], 1))
a = np.array([[1, 2], [3, 4]])
print("a = np.array([[1, 2], [3, 4]])")
print("np.diag(a):\n",np.diag(a))

print("vander(x, n) defines a Vandermonde matrix as a 2D NumPy array. \n")
print("np.vander(np.linspace(0, 2, 5), 2):\n",np.vander(np.linspace(0, 2, 5), 2))
print("np.vander([1, 2, 3, 4], 2:\n",np.vander([1, 2, 3, 4], 2))


print("\n-----general ndarray creation functions-----\n")
print("The ndarray creation functions e.g. numpy.ones, numpy.zeros, and random ")
print("np.zeros((2,3)):\n",np.zeros((2,3)))
print("np.zeros((2,3,2)):\n",np.zeros((2,3,2)))
print("np.ones((2,3)):\n",np.ones((2,3)))
print("np.ones((2,3,2)):\n",np.ones((2,3,2)))

from numpy.random import default_rng

#print("default_rng(42).random(2,3)\n",default_rng(42).random(2,3))
print("default_rng(42).random((2,3)):\n",default_rng(42).random((2,3)))
print("default_rng(42).random((2,3,2)):\n",default_rng(42).random((2,3,2)))


print("\nmumpy.indices")
print("np.indices((3, 3)):\n",np.indices((3, 3)))


print("\n-----3 - Replicating, joining, or mutating existing arrays-----\n")
a = np.array([1, 2, 3, 4, 5, 6])
b = a[:2]
b += 1
print('a =', a, '; b =', b)

print("using copy() will not alter the original array")
a = np.array([1, 2, 3, 4])
b = a[:2].copy()
b += 1
print('a =', a, '; b =', b)


print("\nother routines to join arrays > numpy.vstack, numpy.hstack, and numpy.block. Showing block :")
A = np.ones((2, 2))
B = np.eye(2, 2)
C = np.zeros((2, 2))
D = np.diag((-3, -4))
print("np.block([[A, B], [C, D]]):\n",np.block([[A, B], [C, D]]))


print("\n------4) Reading arrays from disk, either from standard or custom formats-----\n")

print("Standard binary formats")

image = Image.open('/home/singh/Pictures/ExecutionLocation.png')
image_array = np.array(image)
print("image_array.shape: ",image_array.shape)
plt.imshow(image_array)
plt.show()

print("\nCommon ASCII formats")
print(np.loadtxt('sample.csv', delimiter = ',', skiprows = 1) )


print("\n------5) Creating arrays from raw bytes through the use of strings or buffers-----\n")
mystr="1000102"
print("mystr='1000102'")
print("np.array(list(mystr)):\n",np.array(list(mystr)))
print("np.array(list(mystr), dtype=int):\n",np.array(list(mystr), dtype=int))