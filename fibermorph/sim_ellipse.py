# %% Imports
import sympy

from PIL import Image, ImageDraw
from skimage import draw
import random
from random import randint
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sympy import geometry

random.seed()

# %% Functions
FUNCTIONS = 0

# input
im_width_px = 5200
im_height_px = 3900
min_rad_um = 40
max_rad_um = 80

px_per_um = 4.25
angle_deg = 30

# image size in inches
um_per_inch = 25400
dpi = px_per_um * um_per_inch
im_width_inch = (im_width_px/px_per_um)/um_per_inch
im_height_inch = (im_height_px/px_per_um)/um_per_inch

imsize_inch = im_height_inch, im_width_inch
imsize_px = im_height_px, im_width_px

min_rad_px = min_rad_um * px_per_um
max_rad_px = max_rad_um * px_per_um

img = np.ones(imsize_px, dtype=np.uint8)

# generate ellipse
rr, cc = draw.ellipse(im_width_px/2, im_height_px/2, min_rad_px, max_rad_px, shape=img.shape, rotation=np.deg2rad(angle_deg))
img[rr, cc] = 0

fig = plt.figure(frameon=False)
fig.set_size_inches(im_width_inch, im_height_inch)
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)


ax.imshow(img, cmap="gray", aspect='auto')
fig.savefig("/Users/tinalasisi/Desktop/test4.tiff", dpi=dpi)
