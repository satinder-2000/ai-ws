#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:34:29 2026

@author: singh
"""
print("Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html")
print()


# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt

species = ("Adelie", "Chinstrap", "Gentoo")
penguins_means = {
    'Bill Depth': (18.35, 18.43, 14.98),
    'Bill Length': (38.79, 48.83, 47.50),
    'Flipper Length': (189.95, 195.82, 217.19)
}

fig, ax = plt.subplots(layout='constrained')

res = ax.grouped_bar(penguins_means, tick_labels=species, group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)
    

# Add some text for labels, title, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()