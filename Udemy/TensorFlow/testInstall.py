#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:30:33 2026

@author: singh
"""
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

tensor = tf.constant("Hello, Tensorflow!")

with tf.compat.v1.Session() as sess:
    result = sess.run(tensor)
    print(result.decode())