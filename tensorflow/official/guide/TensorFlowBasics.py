#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:45:04 2026

@author: singh
"""
import tensorflow as tf

x = tf.constant([[1., 2. , 3.],
                [4., 5. , 6. ]])

print(x)
print(x.shape)
print(x.dtype)
print()
print("x+x:\n",x+x)
print()
print("5*x:\n",5*x)
print()
print("x @ tf.transpose(x):\n",x @ tf.transpose(x))
print()
print("tf.concat([x, x, x], axis=0):\n",tf.concat([x, x, x], axis=0))
print()
print("tf.nn.softmax(x, axis=-1):\n",tf.nn.softmax(x, axis=-1))
print()
print("tf.reduce_sum(x): \n",tf.reduce_sum(x))
print()
print("tf.convert_to_tensor([1,2,3]): \n",tf.convert_to_tensor([1,2,3]))
print()
print("tf.reduce_sum([1, 2, 3]):\n",tf.reduce_sum([1, 2, 3]))
print()
if tf.config.list_physical_devices('GPU'):
    print("Tensorflow **IS** using the GPU")
else:
    print("Tensorflow **IS NOT** using the GPU")
    

print("-----Variables-----")
print()
var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
print("var = tf.Variable([0.0, 0.0, 0.0]): \n",var)
print()
print("var.assign_add([1,1,1]): \n",var.assign_add([1,1,1]))
print()
print("var.assign_add([1,1,1]):\n",var.assign_add([1,1,1]))
print("-----Automatic differentiation-----")
print()

x = tf.Variable(1.0)

def f(x):
    y = x**2 + 2*x - 5
    return y

print("f(x): \n",f(x))
print()

with tf.GradientTape() as tape:
    y=f(x)
    
g_x = tape.gradient(y,x) #gx=dy/dx
print("g_x: \n",g_x)

print("-----Graphs and tf.function-----")
print()

@tf.function
def my_func(x):
    print('Tracing.\n')
    return tf.reduce_sum(x)


x= tf.constant([1, 2, 3])
print(my_func(x))
x = tf.constant([10, 9, 8])#On subsequent calls TensorFlow only executes the optimized graph, skipping any non-TensorFlow steps. 
print(my_func(x))
print()
print("A graph may not be reusable for inputs with a different signature (shape and dtype), so a new graph is generated instead:")
print()
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
print(my_func(x))