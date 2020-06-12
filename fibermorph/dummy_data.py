"""
Script to generate dummy data for testing curvature

Based on script to produce non-colliding rectangles adapted from:
https://stackoverflow.com/questions/4373741/how-can-i-randomly-place-several-non-colliding-rects
"""

# %% Imports
import sympy

from PIL import Image, ImageDraw
import random
from random import randint
import pathlib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sympy import geometry

random.seed()

# %% Random rect generator
FUNCTIONS = 0


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
        height the deviation from the center of the rectangle allowed.
    """
    # pick a point in the interior of given rectangle
    w, h = rect.width, rect.height  # cache properties
    center = Point(rect.min.x + (w // 2), rect.min.y + (h // 2))
    delta_x = plus_or_minus(randint(0, w // factor))
    delta_y = plus_or_minus(randint(0, h // factor))
    interior = Point(center.x + delta_x, center.y + delta_y)

    # create rectangles from the interior point and the corners of the outer one
    return [Rect(interior.x, interior.y, rect.min.x, rect.min.y),
            Rect(interior.x, interior.y, rect.max.x, rect.min.y),
            Rect(interior.x, interior.y, rect.max.x, rect.max.y),
            Rect(interior.x, interior.y, rect.min.x, rect.max.y)]


def square_subregion(rect):
    """ Return a square rectangle centered within the given rectangle """
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

def draw_rect(draw, rect):
    draw.rectangle([(rect.min.x, rect.min.y), (rect.max.x, rect.max.y)], fill=None)


def draw_arc(draw, rect, width):
    start = random.randint(50, 150)
    end = random.randint(200, 350)
    pad = 40
    minx = rect.min.x + pad
    miny = rect.min.y + pad
    maxx = rect.max.x - pad
    maxy = rect.max.y - pad

    draw.arc([(minx, miny), (maxx, maxy)], start=start, end=end, fill="black", width=width)


    radius = (maxx - minx) / 2
    curv = 1 / radius
    circumference = 2 * np.pi * radius
    arc_angle = end - start
    arc_radians = arc_angle / 360
    arc_length = arc_radians * circumference

    return radius, arc_length, curv


def draw_line(draw, rect, width):
    pad = 40
    minx = rect.min.x + pad
    miny = rect.min.y + pad
    maxx = rect.max.x - pad
    maxy = rect.max.y - pad

    draw.line([(minx, miny), (maxx, maxy)], fill="black", width=width)

    width = (maxx - minx)
    height = (maxy - miny)

    diag_length = np.sqrt((width ** 2) + (height ** 2))

    return diag_length


def draw_ellipse(draw, rect, width):
    pad = 40
    minx = rect.min.x + pad
    miny = rect.min.y + pad
    maxx = rect.max.x - pad
    maxy = rect.max.y - pad

    draw.ellipse([(minx, miny), (maxx, maxy)], fill="black", width=width)

    # values for min and max and area of ellipses to pass to dataframe
    width_df = (maxx - minx)
    height_df = (maxy - miny)
    r1 = (width_df / 2)
    r2 = (height_df / 2)
    p1 = geometry.Point(0, 0)
    e1 = geometry.Ellipse(p1, r1, r2)
    area = sympy.N(e1.area)

    return width_df, height_df, area


def create_data(df, image, output_directory, shape):
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_%S_%f_")
    df_path = df.to_csv(pathlib.Path(output_directory).joinpath(timestamp + shape + "_data.csv"))
    print(df)
    im_path = pathlib.Path(output_directory).joinpath(timestamp + shape + "_data.tiff")
    image.save(im_path)
    print(im_path)

    return df, image, im_path, df_path


# %% Generate random bounding boxes

def bounding_box(min_elem, max_elem, im_width, im_height):

    NUM_RECTS = random.randint(min_elem, max_elem)

    REGION = Rect(0, 0, im_width, im_height)  # Size of the total rectangle/image

    # call quadsect() until at least the number of rects wanted has been generated
    rects = [REGION]  # seed output list
    while len(rects) <= NUM_RECTS:
        rects = [subrect for rect in rects
                 for subrect in quadsect(rect, 6)]

    random.shuffle(rects)  # mix them up
    sample = random.sample(rects, NUM_RECTS)  # select the desired number

    return REGION, rects, sample

# %% Draw image

def dummy_data_gen(output_directory, shape="arc", min_elem=10, max_elem=20, im_width=5200, im_height=3900, width=10):

    REGION, rects, sample = bounding_box(min_elem, max_elem, im_width, im_height)

    imgx, imgy = REGION.max.x + 1, REGION.max.y + 1
    image = Image.new("RGB", (imgx, imgy), color="white")  # create color image
    draw = ImageDraw.Draw(image)

    # first draw outlines of all the non-overlapping rectangles generated
    for rect in rects:
        draw_rect(draw, rect)

    if shape == "line":
        data = [
            draw_line(draw, rect, width)
            for rect in sample
            if (rect.width > 132 and rect.height > 132)]
        df = pd.DataFrame(data, columns=['ref_length'])
        df, img, im_path, df_path = create_data(df, image, output_directory, shape)
        return df, img, im_path, df_path

    elif shape == "arc":
        data = [
            draw_arc(draw, square_subregion(rect), width)
            for rect in sample
            if (rect.width > 132 and rect.height > 132)]
        df = pd.DataFrame(data, columns=['ref_radius', 'ref_length', 'ref_curvature'])
        df, img, im_path, df_path = create_data(df, image, output_directory, shape)
        return df, img, im_path, df_path

    elif shape == "ellipse":
        data = [
            draw_ellipse(draw, rect, width)
            for rect in sample
            if (rect.width > 132 and rect.height > 132)]
        df = pd.DataFrame(data, columns=['ref_width', 'ref_height', 'ref_area'])
        df, img, im_path, df_path = create_data(df, image, output_directory, shape)
        return df, img, im_path, df_path

    elif shape == "circle":
        data = [
            draw_ellipse(draw, square_subregion(rect), width)
            for rect in sample
            if (rect.width > 132 and rect.height > 132)]
        df = pd.DataFrame(data, columns=['ref_width', 'ref_height', 'ref_area'])
        df, img, im_path, df_path = create_data(df, image, output_directory, shape)
        return df, img, im_path, df_path

    else:
        print("The shape value that has been input is incorrect, check options for shape again.")
