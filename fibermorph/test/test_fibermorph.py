# %% Import
import os
import sys
import pathlib
import numpy as np

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
from fibermorph import dummy_data

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
    
    num_total_points = element.area
    
    element_im = element.image
    
    skeleton = element_im
    
    diag_points, num_diag_points = find_structure(skeleton, 'diag')
    print(num_diag_points)
    
    mid_points, num_mid_points = find_structure(skeleton, 'mid')
    print(num_mid_points)
    
    # can't pick up all adjacent points like this due to labelling limitation for ndimage.label (see define_structure)
    # adj_points, num_adj_points = find_structure(skeleton, 'adj')
    num_adj_points = num_total_points - num_diag_points - num_mid_points
    print(num_adj_points)
    
    num_combi_points = num_diag_points + num_mid_points + num_adj_points
    
    print("Reference length is: {}".format(ref_length))
    print("Uncorrected length is: {}".format(num_combi_points))
    corr_element_pixel_length = num_adj_points + (num_diag_points * np.sqrt(2)) + (num_mid_points * np.sqrt(1.25))
    print("Corrected length is: {}". format(corr_element_pixel_length))
    
    return corr_element_pixel_length
    
    
def define_structure(structure: str):
    
    # structures have to be centrosymmetric to work with ndimage.label (in 3x3 matrix)
    if structure == "mid":
        hit1 = np.array([[0, 0, 0],
                         [0, 1, 1],
                         [1, 0, 0]], dtype=np.uint8)
        hit2 = np.array([[1, 0, 0],
                         [0, 1, 1],
                         [0, 0, 0]], dtype=np.uint8)
        hit3 = np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
        hit4 = np.array([[0, 0, 0],
                         [1, 1, 0],
                         [0, 0, 1]], dtype=np.uint8)
        hit5 = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [1, 0, 0]], dtype=np.uint8)
        hit6 = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.uint8)
        hit7 = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
        hit8 = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
        
        mid_list = [hit1, hit2, hit3, hit4, hit5, hit6, hit7, hit8]
        return mid_list
    elif structure == "diag":
        hit1 = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]], dtype=np.uint8)
        hit2 = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.uint8)
        # hit3 = np.array([[1, 0, 0],
        #                  [0, 1, 0],
        #                  [1, 0, 0]], dtype=np.uint8)
        # hit4 = np.array([[1, 0, 1],
        #                  [0, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit5 = np.array([[0, 0, 1],
        #                  [0, 1, 0],
        #                  [0, 0, 1]], dtype=np.uint8)
        # hit6 = np.array([[0, 0, 0],
        #                  [0, 1, 0],
        #                  [1, 0, 1]], dtype=np.uint8)
        # hit7 = np.array([[1, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit8 = np.array([[0, 0, 1],
        #                  [0, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit9 = np.array([[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 1]], dtype=np.uint8)
        # hit10 = np.array([[0, 0, 0],
        #                   [0, 1, 0],
        #                   [1, 0, 0]], dtype=np.uint8)
        
        diag_list = [hit1, hit2]
        # diag_list = [hit1, hit2, hit3, hit4, hit5, hit6, hit7, hit8, hit9, hit10]
        return diag_list
    elif structure == "adj":
        hit1 = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
        hit2 = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0]], dtype=np.uint8)
        # hit3 = np.array([[0, 1, 0],
        #                  [1, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit4 = np.array([[0, 1, 0],
        #                  [0, 1, 1],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit5 = np.array([[0, 0, 0],
        #                  [0, 1, 1],
        #                  [0, 1, 0]], dtype=np.uint8)
        # hit6 = np.array([[0, 0, 0],
        #                  [1, 1, 0],
        #                  [0, 1, 0]], dtype=np.uint8)
        # hit7 = np.array([[0, 0, 0],
        #                  [1, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit8 = np.array([[0, 1, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit9 = np.array([[0, 0, 0],
        #                  [0, 1, 1],
        #                  [0, 0, 0]], dtype=np.uint8)
        # hit10 = np.array([[0, 0, 0],
        #                   [0, 1, 0],
        #                   [0, 1, 0]], dtype=np.uint8)
        adj_list = [hit1, hit2]
        # adj_list = [hit1, hit2, hit3, hit4, hit5, hit6, hit7, hit8, hit9, hit10]
        return adj_list
    else:
        raise TypeError(
            "Structure input for find_structure() is invalid, choose from 'mid', 'diag', and 'adj' and input as str")


def find_structure(skeleton, structure: str):
    skel_image = fibermorph.check_bin(skeleton).astype(int)
    
    print(skel_image.shape)
    
    # creating empty array for hit and miss algorithm
    hit_points = np.zeros(skel_image.shape)
    # defining the structure used in hit-and-miss algorithm
    hit_list = define_structure(structure)
    
    for hit in hit_list:
        target = hit.sum()
        curr = ndimage.convolve(skel_image, hit, mode="constant")
        hit_points = np.logical_or(hit_points, np.where(curr == target, 1, 0))
    
    # Ensuring target image is binary
    hit_points_image = np.where(hit_points, 1, 0)
    
    # use SciPy's ndimage module for locating and determining coordinates of each branch-point
    labels, num_labels = ndimage.label(hit_points_image)
    
    return labels, num_labels


def test_copy_if_exist():
    # fibermorph.copy_if_exist()
    pass


def test_analyze_each_curv():
    # fibermorph.analyze_each_curv()
    pass


def test_analyze_section():
    # fibermorph.analyze_section()
    pass


from tqdm import tqdm
import pandas as pd
import numpy as np
from time import sleep

df = [[1, 2, 3], [4, 5, 6]]

testdf = [np.sum(i) and sleep(2) for i in tqdm(df)]
