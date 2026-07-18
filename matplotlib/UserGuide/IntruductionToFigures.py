#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:13:59 2026

@author: singh
"""
print("Source: https://matplotlib.org/stable/users/explain/figure/figure_intro.html")
print()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(2,2), facecolor='lightskyblue',
                 layout='constrained')
fig.suptitle('Figure')
ax = fig.add_subplot()
ax.set_title('Axes',loc='left', fontstyle='oblique', fontsize='medium')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(4,3), layout='constrained')
plt.show()


print("More complex grids can be achieved with pyplot.subplot_mosaic (which wraps Figure.subplot_mosaic):")
print()

fig, axs = plt.subplot_mosaic([['A', 'right'],['B','right']],
                              figsize=(4,3), layout='constrained')
for ax_name in axs.items():
    ax.text(0.5, 0.5, ax_name,ha='center',va='center' )
plt.show()

print("nested layout in a Figure, with two or more sets of Axes that do not share the same subplot grid")
print()

fig = plt.figure(layout='constrained', facecolor='lightskyblue')
fig.suptitle('Figure')
figL, figR = fig.subfigures(1,2)
figL.set_facecolor('thistle')
axL=figL.subplots(2, 1, sharex=True)
axL[1].set_xlabel('x[m]')
figL.suptitle('Left subfigure')
figR.set_facecolor('paleturquoise')
axR=figR.subplots(1, 2, sharey=True)
axR[0].set_title('Axes 1')
figR.suptitle('Right subfigure')
plt.show()

