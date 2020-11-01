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


def sim_ellipse(output_directory, im_width_px, im_height_px, min_diam_um, max_diam_um, px_per_um, angle_deg):
    # conversions
    um_per_inch = 25400
    dpi = int(px_per_um * um_per_inch)
    min_rad_um = min_diam_um / 2
    max_rad_um = max_diam_um / 2
    
    # image size in inches
    im_width_inch = (im_width_px / px_per_um) / um_per_inch
    im_height_inch = (im_height_px / px_per_um) / um_per_inch
    
    imsize_inch = im_height_inch, im_width_inch
    imsize_px = im_height_px, im_width_px
    
    min_rad_px = min_rad_um * px_per_um
    max_rad_px = max_rad_um * px_per_um
    
    # generate array of ones (will show up as white background)
    img = np.ones(imsize_px, dtype=np.uint8)
    
    # generate ellipse in center of image
    rr, cc = draw.ellipse(im_height_px / 2, im_width_px / 2, min_rad_px, max_rad_px, shape=img.shape,
                          rotation=np.deg2rad(angle_deg))
    img[rr, cc] = 0
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im_width_inch, im_height_inch)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    p1 = geometry.Point((im_height_px / px_per_um) / 2, (im_width_px / px_per_um) / 2)
    e1 = geometry.Ellipse(p1, hradius=max_rad_um, vradius=min_rad_um)
    area = sympy.N(e1.area)
    eccentricity = e1.eccentricity
    ax.imshow(img, cmap="gray", aspect='auto')
    
    data = {'area': [area], 'eccentricity': [eccentricity], 'ref_min_diam': [min_diam_um],
            'ref_max_diam': [max_diam_um]}
    
    df = pd.DataFrame(data)
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_%S_%f")
    
    name = "sim_ellipse_" + str(timestamp)
    
    im_path = pathlib.Path(output_directory).joinpath("im_" + name + ".tiff")
    df_path = pathlib.Path(output_directory).joinpath("df_" + name + ".csv")
    
    df.to_csv(df_path)
    
    fig.savefig(fname=im_path, dpi=dpi)
    
    return df


# %% Input

im_width_px = 5200
im_height_px = 3900
min_diam_um = 40
max_diam_um = 80

px_per_um = 4.25
angle_deg = 30

output_directory = '/Users/tinalasisi/Desktop'

# %% Execution

df = sim_ellipse(output_directory, im_width_px, im_height_px, min_diam_um, max_diam_um, px_per_um, angle_deg)
