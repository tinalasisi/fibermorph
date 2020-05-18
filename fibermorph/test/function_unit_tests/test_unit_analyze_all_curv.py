import pathlib
import skimage

import numpy as np
from skimage import measure
import pandas as pd

from fibermorph.test.function_unit_tests.test_unit_analyze_each_curv import analyze_each_curv
from fibermorph.test.function_unit_tests.test_unit_check_bin import check_bin
from fibermorph.test.function_unit_tests.test_unit_trim_outliers import trim_outliers


# analyzes curvature for entire image (analyze_each does each hair in image)

def analyze_all_curv(img, name, analysis_dir, resolution, window_size_mm=1):

    window_size = int(round(window_size_mm * resolution))  # must be an integer

    if type(img) != 'numpy.ndarray':
        print(type)
        img = np.array(img)
    else:
        print(type(img))

    print("Analyzing {}".format(name))
    
    img = check_bin(img)

    label_image, num_elements = skimage.measure.label(img, connectivity=2, return_num=True)
    print(num_elements)

    props = skimage.measure.regionprops(label_image)

    tempdf = [analyze_each_curv(hair, window_size, resolution) for hair in props]

    print("\nData for {} is:".format(name))
    print(tempdf)

    within_curvdf = pd.DataFrame(tempdf, columns=['curv_mean', 'curv_median', 'length'])

    print("\nDataframe for {} is:".format(name))
    print(within_curvdf)
    print(within_curvdf.dtypes)

    with pathlib.Path(analysis_dir).joinpath(name + ".csv") as save_path:
        within_curvdf.to_csv(save_path)

    within_curv_outliers = np.asarray(trim_outliers(within_curvdf, p1=0.1, p2=0.9))
    print(within_curv_outliers)

    within_curvdf2 = pd.DataFrame(within_curv_outliers, columns=['curv_mean', 'curv_median', 'length'])

    curv_mean_im_mean = within_curvdf2['curv_mean'].mean()
    # print(curv_mean_im_mean)

    curv_mean_im_median = within_curvdf2['curv_mean'].median()
    # print(curv_mean_im_median)

    curv_median_im_mean = within_curvdf2['curv_median'].mean()
    # print(curv_median_im_mean)

    curv_median_im_median = within_curvdf2['curv_median'].median()
    # print(curv_median_im_median)

    length_mean = within_curvdf2['length'].mean()
    # print(length_mean)
    length_median = within_curvdf2['length'].median()
    # print(length_median)

    hair_count = len(within_curvdf2.index)
    # print(hair_count)

    sorted_df = pd.DataFrame(
        [name, curv_mean_im_mean, curv_mean_im_median, curv_median_im_mean, curv_median_im_median, length_mean,
         length_median, hair_count]).T

    print("\nDataframe for {} is:".format(name))
    print(sorted_df)
    print("\n")

    return sorted_df

