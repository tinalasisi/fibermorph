import numpy as np
import pandas as pd

from fibermorph.test.function_unit_tests.test_unit_subset_gen import subset_gen
from fibermorph.test.function_unit_tests.test_unit_taubin_curv import taubin_curv


def analyze_each_curv(hair, window_size, resolution):
    """
    Calculating curvature for hair divided into windows (as opposed to the entire hair at once)

    Moving 1 pixel at a time, the loop uses 'start' and 'end' to define the window-length with which curvature is
    calculated.

    :param hair:
    :param min_hair:
    :param window_size:     the window size (in pixel)
    :param resolution:      the resolution (number of pixels in a mm)
    :return:
                            curv_mean,
                            curv_median,
                            curvature_mean,
                            curvature_median
    """

    hair_label = np.array(hair.coords)

    length_mm = float(len(hair.coords) / resolution)
    print("\nCurv length is {} mm".format(length_mm))

    hair_pixel_length = len(hair.coords)  # length of hair in pixels
    print("\nCurv length is {} pixels".format(hair_pixel_length))

    subset_loop = (subset_gen(hair_pixel_length, window_size, hair_label=hair_label))  # generates subset loop

    # Safe generator expression in case of errors
    curv = [taubin_curv(hair_coords, resolution) for hair_coords in subset_loop]

    taubin_df = pd.Series(curv).astype('float')
    print("\nCurv dataframe is:")
    print(taubin_df)
    print(type(taubin_df))
    print("\nCurv df min is:{}".format(taubin_df.min()))
    print("\nCurv df max is:{}".format(taubin_df.max()))

    print("\nTrimming outliers...")
    taubin_df2 = taubin_df[taubin_df.between(taubin_df.quantile(.01), taubin_df.quantile(.99))] # without outliers

    
    print("\nAfter trimming outliers...")
    print("\nCurv dataframe is:")
    print(taubin_df2)
    print(type(taubin_df2))
    print("\nCurv df min is:{}".format(taubin_df2.min()))
    print("\nCurv df max is:{}".format(taubin_df2.max()))

    curv_mean = taubin_df2.mean()
    print("\nCurv mean is:{}".format(curv_mean))

    curv_median = taubin_df2.median()
    print("\nCurv median is:{}".format(curv_median))

    within_hair_df = [curv_mean, curv_median, length_mm]
    print("\nThe curvature summary stats for this element are:")
    print(within_hair_df)

    if within_hair_df is not None or np.nan:
        return within_hair_df
    else:
        pass
