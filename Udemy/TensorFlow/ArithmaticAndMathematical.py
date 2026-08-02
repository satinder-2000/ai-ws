#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:43:06 2026

@author: singh
"""
import tensorflow as tf

tensor_a = tf.constant([[1.0, 2.0],[3.0, 4.0]])
tensor_b = tf.constant([[5.0, 6.0],[7.0, 8.0]])
print("NOTE- In udemy, the values in tensor_a and tensor_b are all ints")
#addition
result_add=tf.math.add(tensor_a, tensor_b)
print("result_add: \n", result_add)
print(result_add[1,1])
print()


#subtraction
result_sub=tf.math.subtract(tensor_a, tensor_b)
print("result_sub: \n", result_sub)
print(result_sub[1,1])
print()


#multiplication
result_mul=tf.math.multiply(tensor_a, tensor_b)
print("result_mul: \n", result_mul)
print(result_mul[1,1])
print()


#division
result_div=tf.math.divide(tensor_a, tensor_b)
print("result_div: \n", result_div)
print(result_div[1,1])
print()

#Square
result_square = tf.math.square(tensor_a)
print("result_square: \n", result_square)
print()

#Square root
result_sqrt = tf.math.sqrt(tensor_a)
print("result_sqrt: \n", result_sqrt)
print()

#Square
result_exp = tf.math.exp(tensor_a)
print("result_exp: \n", result_exp)
print()

#Logarithm
result_log = tf.math.log(tensor_a)
print("result_log: \n", result_log)
print()