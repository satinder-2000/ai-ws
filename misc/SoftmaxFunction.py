#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:23:48 2026

@author: singh
"""
print("Source: https://dataaspirant.com/difference-between-softmax-function-and-sigmoid-function/")
print()
print("-----Implementing Sigmoid Function-----")
print()
import numpy as np

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))

softmax_inputs = [2, 3, 5, 6]
print ("Softmax Function Output :: {}".format(softmax(softmax_inputs)))
print()

print("-----Creating Softmax Function Graph-----")
print()

import matplotlib.pyplot as plt

def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
    

graph_x = range(0, 21)
graph_y = softmax(graph_x)
line_graph(graph_x, graph_y,"Inputs", "Softmax Scores")