print("-----Automatic Differentiation and Gradients------")
print()
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
print("---Gradient tapes---")
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

print()