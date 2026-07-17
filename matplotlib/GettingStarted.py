# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()

