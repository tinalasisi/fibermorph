# %% Imports
import sympy

from PIL import Image, ImageDraw
import random
from random import randint
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Input radius size
radius = 1

# set size/limits for plots
xlims = 250

ylims = xlims

# no. of hair to simulate - 25 for now
nhair = 25

# pick a starting angles for hair segments
start_theta = np.random.uniform(low=0, high=np.pi, size=nhair)
start_theta = pd.Series(start_theta, name="start_theta")

# define length of the arc.
# The more the curvature, the longer the arc
arc_length = np.pi / radius
arc_length = pd.Series([arc_length for i in range(nhair)], name="arc_length")

# set end value of the angle
end_theta = start_theta + arc_length
end_theta = pd.Series(end_theta, name="end_theta")

arc_nums = list(range(nhair))
arc_names = pd.Series(["arc_" + str(s) for s in arc_nums], name="arc_names")

dat = pd.concat([arc_names, start_theta, end_theta, arc_length], axis=1)


# function to generate arc given the start and end angles

def apoints(row):
    stheta = row[1]
    etheta = row[2]
    rthetas = np.linspace(start=stheta, stop=etheta, num=25)
    x = pd.Series(radius * np.cos(rthetas), name="x")
    y = pd.Series(radius * np.sin(rthetas), name="y")
    
    dat2 = pd.concat([x, y], axis=1)
    
    return dat2


# dataframe with coordinates for each arc
dats = dat

dats["coords"] = dats.apply(lambda row: apoints(row), axis=1)


# center the arcs so they appear at the center of each 'window'
def center_func(coord_df):
    x = coord_df["x"]
    y = coord_df["y"]
    
    x2 = pd.Series(x - np.mean(x), name="x2")
    y2 = pd.Series(y - np.mean(y), name="y2")
    
    dat3 = pd.concat([x2, y2], axis=1)
    
    return dat3


dats["c_coords"] = dats["coords"].apply(lambda row: center_func(row))

from sklearn import preprocessing


im = Image.new('L', (xlims, ylims), color="white")
draw = ImageDraw.Draw(im)

coord_list = np.array(dats["c_coords"].iloc[0])
coord_tuple = tuple(map(tuple, coord_list))

x, y = zip(*coord_tuple)
plt.scatter(x, y)
plt.show()

draw.line(xy=coord_tuple, fill="black")

im.show()


# another function to center arcs to the middle of the window but by scaling
def center_python_func(coord_df):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 200))
    
    dat4 = coord_df
    dat4["x"] = scaler.fit_transform(coord_df[["x"]])
    dat4["y"] = scaler.fit_transform(coord_df[["y"]])
    
    return dat4


dats["c_coords"] = dats["coords"].apply(lambda row: center_python_func(row))

im = Image.new('L', (xlims, ylims), color="white")
draw = ImageDraw.Draw(im)

coord_list = np.array(dats["c_coords"].iloc[0])
coord_tuple = tuple(map(tuple, coord_list))

x, y = zip(*coord_tuple)
plt.scatter(x, y)
plt.show()

draw.line(xy=coord_tuple, fill="black")

im.show()
