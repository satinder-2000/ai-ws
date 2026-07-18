# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("This example shows how to use the agg backend directly to create images,\nwhich may be of use to web application developers")
print()

from PIL import Image
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure(figsize=(5,4), dpi=100)

# Do some plotting.
ax = fig.add_subplot()
ax.plot([1, 2, 3])

print("-----Option 2 - manually attach a canvas to the figure-----")
print()
canvas = FigureCanvasAgg(fig)
canvas.draw()
rgba = np.asarray(canvas.buffer_rgba())
# ... and pass it to PIL.
im = Image.fromarray(rgba)
# This image can then be saved to any format supported by Pillow, e.g.:
im.save("test.bmp")

im.show()    


