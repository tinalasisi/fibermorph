import pathlib
import skimage

import numpy as np
from skimage import measure
import pandas as pd

from fibermorph.test.function_unit_tests.test_unit_subset_gen import subset_gen
from fibermorph.test.function_unit_tests.test_unit_trim_outliers import trim_outliers
from fibermorph.test.function_unit_tests.test_unit_taubin_curv import taubin_curv


def analyze_each_curv(hair, window_size, img_res):
    """
    Calculating curvature for hair divided into windows (as opposed to the entire hair at once)

    Moving 1 pixel at a time, the loop uses 'start' and 'end' to define the window-length with which curvature is
    calculated.

    :param hair:
    :param min_hair:
    :param window_size:     the window size (in pixel)
    :param img_res:      the resolution (number of pixels in a mm)
    :return:
                            curv_mean,
                            curv_median,
                            curvature_mean,
                            curvature_median
    """

    hair_label = np.array(hair.coords)

    length_mm = float(hair.area / img_res)
    print(length_mm)

    hair_pixel_length = hair.area  # length of hair in pixels
    print(hair_pixel_length)

    subset_loop = (subset_gen(hair_pixel_length, window_size, hair_label=hair_label))  # generates subset loop

    # Safe generator expression in case of errors
    taubin_curv = [taubin_curv(hair_coords, img_res) for hair_coords in subset_loop]

    taubin_df = pd.Series(taubin_curv).astype('float')
    print(taubin_df)
    print(taubin_df.min())
    print(taubin_df.max())

    taubin_df2 = trim_outliers(taubin_df, isdf=False, p1=0.01, p2=0.99)

    print(taubin_df2)
    print(taubin_df2.min())
    print(taubin_df2.max())

    [curv_mean] = taubin_df2.mean().values
    print(curv_mean)
    [curv_median] = taubin_df2.median().values
    print(curv_median)

    # curv_mean = taubin_df.mean()
    # print(curv_mean)
    # curv_median = taubin_df.median()
    # print(curv_median)

    within_hair_df = [curv_mean, curv_median, length_mm]
    print(within_hair_df)

    if within_hair_curvature is not None:
        return within_hair_df
    else:
        pass
