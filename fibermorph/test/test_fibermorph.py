# %% Import
import os
import sys
import pathlib
import numpy as np

from skimage import io

import argparse
import datetime
import os
import pathlib
import shutil
import sys
import timeit
import warnings
from datetime import datetime
from functools import wraps
from timeit import default_timer as timer

import cv2
import pandas as pd
import rawpy
import scipy
import skimage
import contextlib
import joblib
import skimage.exposure
import skimage.measure
import skimage.morphology
from PIL import Image
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial import distance as dist
from skimage import filters
from skimage.filters import threshold_minimum
from skimage.segmentation import clear_border
from skimage.util import invert
from tqdm import tqdm

# %%
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fibermorph import fibermorph
from fibermorph import demo

# Get current directory
dir = os.path.dirname(os.path.abspath(__file__))


def teardown_module(function):
    teardown_files = [
        "empty_file1.txt",
        "test1/empty_file1.txt"]
    if len(teardown_files) > 0:
        for file_name in teardown_files:
            if os.path.exists(os.path.join(dir, file_name)):
                os.remove(os.path.join(dir, file_name))


def test_pass():
    assert 1 + 1 == 2


# def test_fail():
#     assert 2 + 2 == 5


def test_make_subdirectory(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    dir1 = fibermorph.make_subdirectory(tmp_path, "test")
    assert d == dir1
    # assert 0 == 1


def test_convert():
    # test min
    assert fibermorph.convert(60) == "0h: 01m: 00s"
    # test hours
    assert fibermorph.convert(5400) == "1h: 30m: 00s"


def test_analyze_all_curv(tmp_path):
    # df, img = dummy_data.dummy_data_gen(output_directory=tmp_path, shape="arc")
    # print(np.asarray(img).dtype)
    # assert np.asarray(img).dtype is np.dtype('uint8')
    # analysis_dir = tmp_path
    # resolution = 1.0
    # window_size_mm = 10
    # fibermorph.analyze_all_curv()
    pass


def test_length_measurement():
    ref_df = pd.read_csv(pathlib.Path("fibermorph/test_data/SimArcData/Oct06_1855_32_017780_arc_data.csv"), header=0,
                         index_col=0)
    
    ref_length = ref_df["ref_length"][0]
    
    img, name = fibermorph.imread(pathlib.Path("fibermorph/test_data/SimArcData/Oct06_1855_32_017780_arc_data.tiff"))
    
    img = fibermorph.check_bin(img)
    
    skel_im = skimage.morphology.thin(img)
    print(skel_im.shape)
    
    prun_im = fibermorph.prune(skel_im, name, pathlib.Path("./"), save_img=False)
    print(prun_im.shape)
    
    label_image, num_elements = skimage.measure.label(prun_im.astype(int), connectivity=2, return_num=True)
    print("\n There are {} elements in the image".format(num_elements))
    
    # label the image and extract the first element/object with [0]
    element = skimage.measure.regionprops(label_image)[0]
    
    # retrieve coordinates of each pixel with regionprops
    element_label = np.array(element.coords)
    print(len(element_label))
    print(element.area)
    assert len(element_label) == element.area
    
    corr_px_length = fibermorph.pixel_length_correction(element)

    print("Reference length is: {}".format(ref_length))
    print("Uncorrected length is: {}".format(element.area))
    print("Corrected length is: {}". format(corr_px_length))


def test_length_measurement():
    ref_df = pd.read_csv(pathlib.Path("fibermorph/test_data/SimArcData/Oct06_1855_32_017780_arc_data.csv"), header=0,
                         index_col=0)
    
    ref_length = ref_df["ref_length"][0]
    
    img, name = fibermorph.imread(pathlib.Path("fibermorph/test_data/SimArcData/Oct06_1855_32_017780_arc_data.tiff"))
    
    img = fibermorph.check_bin(img)
    
    skel_im = skimage.morphology.thin(img)
    print(skel_im.shape)
    
    prun_im = fibermorph.prune(skel_im, name, pathlib.Path("./"), save_img=False)
    print(prun_im.shape)
    
    label_image, num_elements = skimage.measure.label(prun_im.astype(int), connectivity=2, return_num=True)
    print("\n There are {} elements in the image".format(num_elements))
    
    # label the image and extract the first element/object with [0]
    element = skimage.measure.regionprops(label_image)[0]
    
    # retrieve coordinates of each pixel with regionprops
    element_label = np.array(element.coords)
    print(len(element_label))
    print(element.area)
    assert len(element_label) == element.area
    
    corr_px_length = fibermorph.pixel_length_correction(element)
    
    print("Reference length is: {}".format(ref_length))
    print("Uncorrected length is: {}".format(element.area))
    print("Corrected length is: {}".format(corr_px_length))

def test_sim_ellipse():
    im_width_px = 5200
    im_height_px = 3900
    min_diam_um = 40
    max_diam_um = 80
    
    px_per_um = 4.25
    angle_deg = 30
    
    output_directory = '/Users/tinalasisi/Desktop'
    
    df = demo.sim_ellipse(output_directory, im_width_px, im_height_px, min_diam_um, max_diam_um, px_per_um, angle_deg)
    pass

#%%
def section_props(props, im_name, resolution, minpixel, maxpixel, im_center):
    
    props_df = [
        [region.label, region.centroid, scipy.spatial.distance.euclidean(im_center, region.centroid)]
        for region
        in props if minpixel <= region.area <= maxpixel]
    props_df = pd.DataFrame(props_df, columns=['label', 'centroid', 'distance'])
    
    section_id = props_df['distance'].astype(float).idxmin()
    # print(section_id)
    
    section = props[section_id]
    
    area_mu = section.filled_area / np.square(resolution)
    min_diam = section.minor_axis_length / resolution
    max_diam = section.major_axis_length / resolution
    eccentricity = section.eccentricity
    
    section_data = pd.DataFrame(
        {'ID': [im_name], 'area': [area_mu], 'eccentricity': [eccentricity], 'min': [min_diam],
         'max': [max_diam]})
    
    gray_im = section.intensity_image
    bin_im = section.filled_image
    bbox = section.bbox
            
    return section_data, gray_im, bin_im, bbox
#%%
def test_segment_section():
    
    
    #input
    # input_directory = pathlib.Path("/Users/tinalasisi/Desktop/Nov01_2338_ValidationTest_Section/ValidationData")
    input_directory = pathlib.Path("/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/admixed_real_hair/section/AfrEu_SectionImages_RawJPG/AfrEu_SectionImages_GrayTIFF/tiff")
    file_list = fibermorph.list_images(input_directory)
    main_output_path = "/Users/tinalasisi/Desktop"
    output_path = main_output_path
    minsize = 20
    maxsize = 150
    resolution = 4.25
    save_img = True
    jobs = 2
    
    input_file = file_list[0]
    
    # section sequence

#%%
    


def test_copy_if_exist():
    # fibermorph.copy_if_exist()
    pass

from tqdm import tqdm
import pandas as pd
import numpy as np
from time import sleep

df = [[1, 2, 3], [4, 5, 6]]

testdf = [np.sum(i) and sleep(2) for i in tqdm(df)]
