"""
Script to generate dummy data for testing curvature

Based on script to produce non-colliding rectangles adapted from:
https://stackoverflow.com/questions/4373741/how-can-i-randomly-place-several-non-colliding-rects
"""

# %% Imports
IMPORTS = 0

from PIL import Image, ImageDraw
import random
from random import randint
import pathlib
import numpy as np
import pandas as pd
import os

random.seed()


# %% Random rect generator

class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    @staticmethod
    def from_point(other):
        return Point(other.x, other.y)


class Rect(object):
    def __init__(self, x1, y1, x2, y2):
        minx, maxx = (x1, x2) if x1 < x2 else (x2, x1)
        miny, maxy = (y1, y2) if y1 < y2 else (y2, y1)
        self.min, self.max = Point(minx, miny), Point(maxx, maxy)
    
    @staticmethod
    def from_points(p1, p2):
        return Rect(p1.x, p1.y, p2.x, p2.y)
    
    width = property(lambda self: self.max.x - self.min.x)
    height = property(lambda self: self.max.y - self.min.y)


plus_or_minus = lambda v: v * [-1, 1][(randint(0, 100) % 2)]  # equal chance +/-1


def quadsect(rect, factor):
    """ Subdivide given rectangle into four non-overlapping rectangles.
        'factor' is an integer representing the proportion of the width or
        height the deviatation from the center of the rectangle allowed.
    """
    # pick a point in the interior of given rectangle
    pad = 40
    w, h = rect.width, rect.height  # cache properties
    # w, h = rect.width, rect.height  # cache properties
    center = Point(rect.min.x + (w // 2), rect.min.y + (h // 2))
    delta_x = plus_or_minus(randint(0, w // factor))
    delta_y = plus_or_minus(randint(0, h // factor))
    interior = Point(center.x + delta_x, center.y + delta_y)
    
    # create rectangles from the interior point and the corners of the outer one
    return [Rect(interior.x, interior.y, rect.min.x, rect.min.y),
            Rect(interior.x, interior.y, rect.max.x, rect.min.y),
            Rect(interior.x, interior.y, rect.max.x, rect.max.y),
            Rect(interior.x, interior.y, rect.min.x, rect.max.y)]
    
    # return [Rect(interior.x, interior.y, rect.min.x, rect.min.y),
    #         Rect(interior.x, interior.y, rect.max.x, rect.min.y),
    #         Rect(interior.x, interior.y, rect.max.x, rect.max.y),
    #         Rect(interior.x, interior.y, rect.min.x, rect.max.y)]


def square_subregion(rect):
    """ Return a square rectangle centered within the given rectangle """
    pad = 40
    w, h = rect.width, rect.height  # cache properties
    if w < h:
        offset = (h - w) // 2
        return Rect(rect.min.x, rect.min.y + offset,
                    rect.max.x, rect.min.y + offset + w)
    else:
        offset = (w - h) // 2
        return Rect(rect.min.x + offset, rect.min.y,
                    rect.min.x + offset + h, rect.max.y)


# %% Image functions

def draw_rect(rect, fill=None):
    draw.rectangle([(rect.min.x, rect.min.y), (rect.max.x, rect.max.y)],
                   fill=fill)


def draw_arc(rect, fill="black", width=3):
    start = random.randint(10, 150)
    end = random.randint(200, 350)
    pad = 40
    minx = rect.min.x + pad
    miny = rect.min.y + pad
    maxx = rect.max.x - pad
    maxy = rect.max.y - pad
    
    draw.arc([(minx, miny), (maxx, maxy)], start=start, end=end, fill=fill, width=width)
    
    # If you wanted to calculate the bounding box you could use the box variable below:
    # box = [(rect.min.x, rect.min.y), (rect.max.x, rect.max.y)]
    
    # The width of the square in which the arc is drawn can be accessed with the variable width below:
    # width = (rect.max.x - rect.min.x), (rect.max.y - rect.min.y)
    
    radius = (maxx - minx) / 2
    arc_angle = end - start
    arc_radians = 2 * np.pi * (arc_angle / 360)
    arc_length = arc_radians * arc_angle
    
    return radius, arc_length


def draw_line(rect, fill=None, width=6):
    pad = 40
    minx = rect.min.x + pad
    miny = rect.min.y + pad
    maxx = rect.max.x - pad
    maxy = rect.max.y - pad
    
    draw.line([(minx, miny), (maxx, maxy)], fill="black", width=3)
    
    width = (maxx - minx)
    height = (maxy - miny)
    
    diag_length = np.sqrt((width ** 2) + (height ** 2))
    
    return diag_length


# %% User input
USER_INPUT = 0

# Are you plotting straight lines or arcs?
# If draw_arcs == False, it will default to drawing lines
draw_arcs = True

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

results_folder = os.makedirs(r"./results_cache", exist_ok=True)

# Where would you like the data to be saved to?
output_directory = pathlib.Path(os.path.abspath(r"./results_cache"))

# %% Generate random bounding boxes
SCRIPT_STARTS = 0

NUM_RECTS = random.randint(10, 50)

REGION = Rect(0, 0, 5200, 3900)  # Size of the total rectangle/image

# call quadsect() until at least the number of rects wanted has been generated
rects = [REGION]  # seed output list
while len(rects) <= NUM_RECTS:
    rects = [subrect for rect in rects
             for subrect in quadsect(rect, 3)]

random.shuffle(rects)  # mix them up
sample = random.sample(rects, NUM_RECTS)  # select the desired number

# %% Draw image

imgx, imgy = REGION.max.x + 1, REGION.max.y + 1
image = Image.new("RGB", (imgx, imgy), color="white")  # create color image
draw = ImageDraw.Draw(image)

# first draw outlines of all the non-overlapping rectangles generated
for rect in rects:
    draw_rect(rect)

if not draw_arcs:
    line_data = [draw_line(rect, fill="black") for rect in sample if (rect.width > 132 and rect.height > 132)]
    line_df = pd.DataFrame(line_data, columns=['length'])
    line_df.to_csv(output_directory.joinpath("line_data.csv"))
    print(line_df)
    save_path = output_directory.joinpath("line_data.tiff")
    image.save(save_path)
    print(save_path)

else:
    arc_data = [draw_arc(square_subregion(rect), fill="black") for rect in sample if
                (rect.width > 132 and rect.height > 132)]
    arc_df = pd.DataFrame(arc_data, columns=['radius', 'length'])
    arc_df.to_csv(output_directory.joinpath("arc_data.csv"))
    print(arc_df)
    save_path = output_directory.joinpath("arc_data.tiff")
    image.save(save_path)
    print(save_path)
