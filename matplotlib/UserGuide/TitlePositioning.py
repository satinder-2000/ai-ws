#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:59:07 2026

@author: singh
"""
import matplotlib.pyplot as plt

plt.plot(range(10))
plt.title('Center Title')
plt.title('Left Title', loc='left')
plt.title('Right Title', loc='right')
plt.show()

print("-----Constrained Layout------")
print()

fig, axs = plt.subplots(1, 2, layout='constrained')

ax = axs[0]
ax.plot(range(10))
ax.xaxis.set_label_position('top')
ax.set_xlabel('X-label')
ax.set_title('Center Title')

ax = axs[1]
ax.plot(range(10))
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel('X-label')
ax.set_title('Center Title')

plt.show()

print("-----specifying the y keyword argument------")
print()

fig, axs = plt.subplots(1, 2, layout='constrained')

ax = axs[0]
ax.plot(range(10))
ax.xaxis.set_label_position('top')
ax.set_xlabel('X-label')
ax.set_title('Manual y', y=1.0, pad=-14)

plt.rcParams['axes.titley'] =1.0 # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -14 # pad is in points... 
ax = axs[1]
ax.plot(range(10))
ax.set_xlabel('X-label')
ax.set_title('rcParam y')
plt.show()

print("-----Figure------")
print()
fig = plt.figure() # an empty figure with no Axes
fig, ax = plt.subplots() ## a figure with a single Axes
fig, axs = plt.subplots(2, 2) # a figure with a 2x2 grid of Axes
# a figure with one Axes on the left, and two on the right:
fig, axs=plt.subplot_mosaic(['left', 'right_top'],['left','right_bottom'])
