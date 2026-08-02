#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:43:43 2026

@author: singh
"""
import tensorflow as tf

print(tf.__version__)

tf.compat.v1.disable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)
x = tf.compat.v1.placeholder(tf.float32, shape=(None,))

add_op = tf.add(a, b)
square_op = tf.square(x)

with tf.compat.v1.Session() as sess:
    result = sess.run(add_op)
    print("sess.run(add_op): ", result)
    result_sq=sess.run(square_op, feed_dict={x:[1, 2, 3]})
    print("sess.run(square_op, feed_dict={x:[1, 2, 3]}): \n", result_sq)
    