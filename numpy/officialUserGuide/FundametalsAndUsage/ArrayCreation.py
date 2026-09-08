#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:13:17 2026

@author: singh
"""
import numpy as np

print("\nThere are 6 general mechanisms for creating arrays\n")
print("\n1-Convert Python sequences to np arrays\n")
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
