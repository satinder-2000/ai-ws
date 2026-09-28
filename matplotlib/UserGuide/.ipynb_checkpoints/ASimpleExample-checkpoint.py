#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:28:46 2026

@author: singh
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],[1, 4, 2, 3])
ax.axis_name='x'
plt.show()
