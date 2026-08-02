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

def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    sigmoid_scores = [1 / float(1 + np.exp(-x)) for x in inputs]
    return sigmoid_scores

sigmoid_inputs = [2, 3, 5, 6]
print ("Sigmoid Function Output :: {}".format(sigmoid(sigmoid_inputs)))
print()
print("-----Creating Sigmoid Function Graph-----")
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
    

graph_x=range(0, 21)
graph_y = sigmoid(graph_x)

print ("Graph X readings: {}".format(graph_x))
print ("Graph Y readings: {}".format(graph_y))

line_graph(graph_x, graph_y, "Inputs", "Sigmoid Scores")