#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 22:11:50 2026

@author: singh
"""
print("Source: https://www.tensorflow.org/guide/basics")
print()

import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)

print("x + x:\n",x + x)
print()

print("5 * x:\n",5 * x)
print()

print("x @ tf.transpose(x):", x @ tf.transpose(x))
print()


print("tf.concat([x, x, x]), axis=0):\n",tf.concat([x, x, x], axis=0))
print()

print("tf.nn.softmax(x, axis=-1):\n", tf.nn.softmax(x, axis=-1))
print()

print("tf.reduce_sum(x):\n", tf.reduce_sum(x))
print()

print("tf.convert_to_tensor([1, 2, 3]):\n", tf.convert_to_tensor([1, 2, 3]))
print()

print("tf.reduce_sum([1, 2, 3]):\n", tf.reduce_sum([1, 2, 3]))
print()

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
  
print("-----Variables-----")
print()
print("To store model weights (or other mutable state) in TensorFlow use a tf.Variable.")
print()

var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
print("tf.Variable:\n", var) 

print("-----Automatic differentiation-----")
print()
print("Gradient descent and related algorithms are a cornerstone of modern machine learning.")
print()

x = tf.Variable(1.0)

def f(x):
    y = x**2 + 2*x -5
    return y

print("f(x):\n",f(x))
print("f(1):\n",f(1))

print("The derivative of y is y' = f'(x) = (2*x + 2) = 4. TensorFlow can calculate this automatically:")
print()
with tf.GradientTape() as tape:
    y=f(x)
    
g_x = tape.gradient(y, x) #g(x) = dy/dx
print("tape.gradient(y, x): \n",g_x)
print()
print("-----Graphs and tf.function-----")
print()

@tf.function
def my_func(x):
    print("Tracing:\n")
    return tf.reduce_sum(x)

x = tf.constant([1, 2, 3])
print("tf.constant([1, 2, 3]):\n", my_func(x))
print()
x = tf.constant([10, 9, 8])
my_func(x)
print("tf.constant([10, 9, 8]):\n", my_func(x))
print()
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
print("tf.constant([10.0, 9.1, 8.2], dtype=tf.float32):\n", my_func(x))
print()
print("-----Modules, layers, and models (can be saved)-----")
print()
class MyModule(tf.Module):
    def __init__(self, value):
        self.weight = tf.Variable(value)
        
    @tf.function
    def multiply(self, x):
        return x * self.weight
    

mod = MyModule(3)
print(mod.multiply(tf.constant([1, 2, 3])))
print("---save the module---")
save_path='./saved'
tf.saved_model.save(mod, save_path)        