#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:44:10 2026

@author: singh
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],[1, 4, 2, 3])
plt.show()

print("\nTypes of inputs to plotting functions\n")

np.random.seed(19680801)
data = {'a':np.arange(50),
        'c':np.random.randint(0,50,50),
        'd':np.random.randn(50)
        }
data['b'] = data['a'] + 10*np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig,ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')
plt.show()


print("\n-----Coding Syles-----\n")
print("\nThe explicit and implicit interfaces\n")
print("\nUsing OO-style\n")
x = np.linspace(0, 2, 100)

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, x, label='Linear')
ax.plot(x, x**2, label='quadratic')
ax.plot(x, x**3, label='cubic')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title("Simple Plot - OO Style")
ax.legend()
plt.show()

print("\nor the pyplot-style\n")
x = np.linspace(0, 2, 100)

plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(x, x, label='Linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot - plt style")
plt.legend()
plt.show()

