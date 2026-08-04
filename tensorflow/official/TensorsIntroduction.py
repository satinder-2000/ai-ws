#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:43:46 2026

@author: singh
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor

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

print("Example of 3-axis tensor")
print()
print("rank_3_tensor[:, :, 4]:\n",rank_3_tensor[:, :, 4])
print()
print("-----Manipulating Shapes-----")
print()
x = tf.constant([[1],[2],[3]])
print("x.shape:\n",x.shape)
reshaped = tf.reshape(x, [1,3])
print("reshaped.shape:\n",reshaped.shape)
print()
print("rank_3_tensor:\n",rank_3_tensor)
print()
print("tf.reshape(rank_3_tensor, [-1]):\n",tf.reshape(rank_3_tensor, [-1]))
print()
print("--- tf.reshape: combine or split adjacent axes (or add/remove 1s)")
print()
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print()
print("---Swapping axes in tf.reshape does not work; you need tf.transpose for that.---")
print()
print("tf.reshape(rank_3_tensor, [2, 3, 5]):\n",tf.reshape(rank_3_tensor, [2, 3, 5]))
print()
print("tf.reshape(rank_3_tensor, [5, 6]):\n",tf.reshape(rank_3_tensor, [5, 6]))
print()
# The code below does't work at all
#print("tf.reshape(rank_3_tensor, [7, -1]):\n",tf.reshape(rank_3_tensor, [7, -1]))
print("---More on Dtypes---")
print()
print("You can cast from type to type.")
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
print("the_f64_tensor: ",the_f64_tensor)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
print("the_f16_tensor: ",the_f16_tensor)
the_u8_tensor =tf.cast(the_f16_tensor ,dtype=tf.uint8)
print("the_u8_tensor: ",the_u8_tensor)
print()
print("---Broadcasting---")
x = tf.constant([1, 2, 3])
print("x = tf.constant([1, 2, 3])")
y = tf.constant(2)
print("y = tf.constant(2)")
z = tf.constant([2, 2, 2])
print("z = tf.constant([2, 2, 2])")
print("tf.multiply(x, 2)",tf.multiply(x, 2))
print("x * y:", x * y)
print("x * z:", x * z)
print()
print("axes with length 1 can be stretched out to match the other arguments")
print()
x = tf.reshape(x, [3,1])
print("tf.reshape(x, [3,1]):\n",x)
y = tf.range(1,5)
print("tf.range(1,5):\n",y)
print("tf.multiply(x, y):\n", tf.multiply(x, y))
print()
print("---Here is the same operation without broadcasting:---")
print()
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2 , 2],
                         [3, 3, 3, 3]])
print("x_stretch: \n",x_stretch)
y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])
print( "y_stretch: \n",y_stretch)

print("x_stretch * y_stretch:\n", x_stretch * y_stretch)
print()
print("---broadcast operation never materializes the expanded tensors in memory---")
print()
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
print()
print("-----Ragged Tensors---")
print()
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]

try:
    tensor = tf.constant(ragged_list)
except Exception as e:
    print(f"{type(e).__name__}:{e}")

print("ValueError above")
print("Instead create a tf.RaggedTensor using tf.ragged.constant:")
ragged_tensor = tf.ragged.constant(ragged_list)
print("ragged_tensor:\n",ragged_tensor)
print("ragged_tensor.shape:\n",ragged_tensor.shape)

print()
print("-----String Tensors---")
print()
scaler_string_tensor = tf.constant("Gray wolf")
print("scaler_string_tensor:\n",scaler_string_tensor)
print()
tensor_of_strings = tf.constant([
    "Gray wolf",
    "Quick brown fox",
    "Lazy dog"])
print("tensor_of_strings:\n",tensor_of_strings)
print()
print("Unicode characters they are utf-8 encoded.\n",tf.constant("🥳👍"))
print()
print("tf.strings.split(scaler_string_tensor, sep=" "):\n",tf.strings.split(scaler_string_tensor, sep=" "))
print()
print("tf.strings.split(tensor_of_strings):\n",tf.strings.split(tensor_of_strings))
print()
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
print()
print("Although you can't use tf.cast to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.")
print()
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
print()
print("Or split it up as unicode and then decode it")
print()
unicode_bytes=tf.constant("アヒル 🦆")
print("\nUnicode bytes:", unicode_bytes)
unicode_char_bytes=tf.strings.unicode_split(unicode_bytes,"UTF-8")
print("Unicode chars:", unicode_char_bytes)
unicode_values=tf.strings.unicode_decode(unicode_bytes,"UTF-8")
print("\nUnicode values:", unicode_values)
print()
print("-----Sparse Tensors-----")
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0],[1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print("sparse_tensor:\n", sparse_tensor)
print(tf.sparse.to_dense(sparse_tensor))