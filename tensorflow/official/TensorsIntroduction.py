#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:43:46 2026

@author: singh
"""
import tensorflow as tf
import numpy as np

print("-----Basics-----")

rank_0_tensor = tf.constant(4)
print("rank_0_tensor :\n",rank_0_tensor)
print()

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print("rank_1_tensor :\n",rank_1_tensor)
print()

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print("rank_2_tensor :\n",rank_2_tensor)
print()

rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]]])
print("rank_3_tensor :\n",rank_3_tensor)
print("rank_3_tensor.shape :\n",rank_3_tensor.shape)
print("rank_3_tensor.dtype :\n",rank_3_tensor.dtype)
print()

print("convert tensor to numpy array: using np.array or tensor.numpy")
np.array(rank_2_tensor)
print("np.array(rank_2_tensor):\n", np.array(rank_2_tensor))
rank_2_tensor.numpy()
print("rank_2_tensor.numpy():\n",rank_2_tensor.numpy())

a = tf.constant([[1,2],
                 [3,4]])
b = tf.constant([[1,1],
                 [1,1]])

print("tf.add(a,b):\n ",tf.add(a,b))
print("tf.multiply(a,b):\n ",tf.multiply(a,b))
print("tf.matmul(a,b):\n ",tf.matmul(a,b))

print("a + b \n", a + b) # element-wise addition
print("a * b \n", a * b) # element-wise multiplication
print("a @ b \n", a @ b) # matrix multiplication

print("-----Tensors are used in all kinds of operations (or Ops).-----")
print()
c = tf.constant([[4.0,5.0],[10.0, 1.0]])
print("tf.reduce_max(c) :\n",tf.reduce_max(c))
print("tf.math.argmax(c) :\n",tf.math.argmax(c))
print("tf.nn.softmax(c) :\n",tf.nn.softmax(c))
print("tf.convert_to_tensor([1,2,3]): \n",tf.convert_to_tensor([1,2,3]))
print("tf.reduce_max(np.array([1,2,3])): \n",tf.reduce_max(np.array([1,2,3])))
print()

print("-----About shapes-----")
print()

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
print("tf.rank(rank_4_tensor):",tf.rank(rank_4_tensor))
print("tf.shape(rank_4_tensor):",tf.shape(rank_4_tensor))

print("-----Indexing-----")
print("--Single-axis Indexing--")
print()
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor)
print()
print("First: ",rank_1_tensor[0].numpy())
print("Second: ",rank_1_tensor[1].numpy())
print("Last: ",rank_1_tensor[-1].numpy())
print("Indexing with a : slice keeps the axis:")
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to end:", rank_1_tensor[4:].numpy())
print("From 2 before 7:", rank_1_tensor[2:7].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
print()
print("---Multi-axis Indexing---")
print()
print("rank_2_tensor.numpy():\n",rank_2_tensor.numpy())
print("rank_2_tensor[1,1].numpy():\n",rank_2_tensor[1,1].numpy())
print()
print("You can index using any combination of integers and slices:")
print()
print("Second row on rank_2_tensor:\n",rank_2_tensor[1,:].numpy())
print("Second column on rank_2_tensor:\n",rank_2_tensor[:,1].numpy())
print("Last row:\n",rank_2_tensor[-1,:].numpy())
print("First item in last column:\n",rank_2_tensor[0,-1:].numpy())
print("Skip the first row")
print(rank_2_tensor[1:,:].numpy())