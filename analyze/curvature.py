"""
Python 3.6+ script calculate hair fiber curvature from grayscale tiff file.

Authors: Nicholas Stephens nbs49@psu.edu & Tina Lasisi tpl5158@psu.edu
February 18, 2020

"""
#############################################################################
#                                                                           #
#      This is the import portion, where we bring in external libraries.    #
#                                                                           #
#############################################################################


# %% Import libraries

I_IMPORT_LIBRARIES = 0
import math
import os  # Allows python to work with the operating system.
import pathlib  # Makes defining pathways simple between mac, windows, linux.
# from imports a single function (i.e. timer) from a library (i.e. default_timer).
import sys
import warnings
from datetime import datetime
from functools import wraps
from time import time
from timeit import default_timer as timer  # Timer to report how long everything is taking.

import cv2
import numpy as np  # Main math backbone of python, lots of MatLab like functions.
import pandas as pd
import scipy
import skimage
import skimage.measure
from PIL import Image  # Allows conversion from RGB to grayscale
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import disk, square
from skimage.segmentation import clear_border
from skimage.util import invert
from sklearn.preprocessing import RobustScaler

# %% Import functions
I_IMPORT_FUNCTIONS = 0


#############################################################################
#                                                                           #
#            This is where we define the functions for use below.           #
#                                                                           #
#############################################################################


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('\n\n The function: {} \n\n with args:[{},\n{}] \n\n and result: {} \n\ntook: {:2.4f} sec\n\n'.format(
            f.__name__, args, kw, result, te - ts))
        return result
    
    return wrap


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def selem_by_res(original, resolution):
    original = int(original)
    perc = original / 132
    new = np.rint((perc * resolution))
    if new > 0:
        return int(new)
    else:
        return 1


@timing
def make_subdirectory(directory, append_name=""):
    """
    Function to build a directory based upon the arguments passed in append. Takes a
    directory and uses pathlib to join that to the appended string passed.
​
    :param directory:       The directory within which this function will make a subdirectory.
    :param append_name:     A string to be appended to the pathlib object.
    :return output_path:    Returns a new directory for output.

​
    """
    # Define the path of the directory within which this function will make a subdirectory.
    directory = pathlib.Path(directory)
    # The name of the subdirectory.
    append_name = str(append_name)
    # Define the output path by the initial directory and join (i.e. "+") the appropriate text.
    output_path = pathlib.Path(directory).joinpath(str(append_name))
    
    # Use pathlib to see if the output path exists, if it is there it returns True
    if not pathlib.Path(output_path).exists():
        
        # Prints a status method to the console using the format option, which fills in the {} with whatever
        # is in the ().
        print("This output path doesn't exist:\n            {} \n Creating...".format(output_path))
        
        # Use pathlib to create the folder.
        pathlib.Path.mkdir(output_path)
        
        # Prints a status to let you know that the folder has been created
        print("Output path has been created")
    
    # Since it's a boolean return, and True is the only other option we will simply print the output.
    else:
        # This will print exactly what you tell it, including the space. The backslash n means new line.
        print("Output path already exists:\n               {}".format(output_path))
    return output_path


def make_all_dirs(output_directory, main_output_name):
    main_output_path = make_subdirectory(output_directory, append_name=str(main_output_name))
    
    filtered_dir = make_subdirectory(main_output_path, append_name="filtered")
    
    binary_dir = make_subdirectory(main_output_path, append_name="binary")
    
    pruned_dir = make_subdirectory(main_output_path, append_name="pruned")
    
    clean_dir = make_subdirectory(main_output_path, append_name="clean")
    
    skeleton_dir = make_subdirectory(main_output_path, append_name="skeletonized")
    
    analysis_dir = make_subdirectory(main_output_path, append_name="analysis")
    
    return main_output_path, filtered_dir, binary_dir, pruned_dir, clean_dir, skeleton_dir, analysis_dir


@timing
def ridge_filter(input_file, main_output_path):
    # directory = make_subdirectory(main_output_path, append_name="filtered")
    
    directory = main_output_path
    
    print("\nFiltering {}...\n".format(str(input_file)))
    
    name = os.path.splitext(os.path.basename(input_file))[0]
    
    gray_img = cv2.imread(str(input_file), 0)  # read in image as numpy array, this will be in color
    type(gray_img)  # using the above function it returns <class 'numpy.ndarray'>
    print("Image size is:", gray_img.shape)  # returns (3904, 5200)
    
    filter_img = skimage.filters.frangi(gray_img)
    
    img_inv = invert(filter_img)
    filter_output_path = pathlib.Path(directory).joinpath(name + ".tiff")
    plt.imsave(filter_output_path, img_inv, cmap='gray')
    print("\n Saving filtered version of: {}".format(name))
    
    print("\n Done filtering {}".format(name))
    
    return filter_output_path, name


@timing
def binary_image(input_file, bin_out_dir, resolution, median_kernel_size=5, save_img=False, ):
    name = os.path.splitext(os.path.basename(input_file))[0]
    
    print("\nBinarizing {}...\n".format(str(name)))
    
    # binary_folder = make_subdirectory(bin_out_dir, append_name="binary")
    
    binary_folder = bin_out_dir
    
    img = cv2.imread(str(input_file), 0)
    
    # Get the size of the image and print it to the console
    print("Image {} size is {}".format(str(name), img.shape))
    
    median_kernel_adj = selem_by_res(median_kernel_size, resolution)
    
    selem = disk(median_kernel_adj)
    
    median_img = skimage.filters.rank.median(img, selem)
    
    # median_img_clear = clear_border(invert(median_img), buffer_size=(median_kernel_adj + resolution))
    
    thresh = threshold_otsu(median_img)
    
    # creates binary image by applying the above threshold and replacing all
    bw_uint8 = (np.where(median_img > thresh, 1, 0)).astype('uint8')
    
    # binary_img = invert(bw_uint8)
    binary_img = clear_border(invert(bw_uint8), buffer_size=int(10 + (resolution / 10)))
    # remove artifacts connected to image border, needs to be inverted for this to work!
    
    binary_img = scipy.ndimage.morphology.binary_dilation(binary_img, square(median_kernel_adj), iterations=2)
    
    # binary_img = scipy.ndimage.morphology.binary_closing(binary_img, square(median_kernel_adj), iterations=2)
    
    # binary_img = skimage.morphology.binary_dilation(binary_img, selem=disk(20))
    
    if save_img:
        img_inv = invert(binary_img)
        with pathlib.Path(binary_folder).joinpath(name + ".tiff") as binary_output_path:
            im = Image.fromarray(img_inv)
            im.save(binary_output_path)
        return binary_img, name
    else:
        return binary_img, name


@timing
def find_branch_points(skeleton, name):
    """
    Starting with a morphological skeleton, creates a corresponding binary image
    with all branch-points pixels (1) and all other pixels (0).
    """
    
    print("\nPruning {}...\n".format(name))
    
    # identify 3-way branch-points through convolving the image using appropriate
    # structure elements for an 8-connected skeleton:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    hit1 = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit2 = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit3 = np.array([[1, 0, 0],
                     [0, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    hit_list = [hit1, hit2, hit3]
    
    # use some nifty NumPy slicing to add the three remaining rotations of each
    # of the structure elements to the hit list
    for ii in range(9):
        hit_list.append(np.transpose(hit_list[-3])[::-1, ...])
    
    # add structure elements for branch-points four 4-way branchpoints, these
    hit3 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    hit4 = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit_list.append(hit3)
    hit_list.append(hit4)
    print("Creating hit and miss list")
    
    # create a zero np.array() of the same shape as the skeleton and use it to collect
    # "hits" from the convolution operation
    
    skel_image = skeleton
    print("Converting image to binary array")
    
    branch_points = np.zeros(skel_image.shape)
    print("Creating empty array for branch points")
    
    for hit in hit_list:
        target = hit.sum()
        curr = ndimage.convolve(skel_image, hit, mode="constant")
        branch_points = np.logical_or(branch_points, np.where(curr == target, 1, 0))
    
    print("Completed collection of branch points")
    
    # pixels may "hit" multiple structure elements, ensure the output is a binary
    # image
    branch_points_image = np.where(branch_points, 1, 0)
    print("Ensuring binary")
    
    # use SciPy's ndimage module to label each contiguous foreground feature
    # uniquely, this will locating and determining coordinates of each branch-point
    labels, num_labels = ndimage.label(branch_points_image)
    print("Labelling branches")
    
    # use SciPy's ndimage module to determine the coordinates/pixel corresponding
    # to the center of mass of each branchpoint
    branch_points = ndimage.center_of_mass(skel_image, labels=labels, index=range(1, num_labels + 1))
    branch_points = np.array([value for value in branch_points if not np.isnan(value[0]) or not np.isnan(value[1])],
                             dtype=int)
    num_branch_points = len(branch_points)
    
    hit = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], dtype=np.uint8)
    
    dilated_branches = ndimage.convolve(branch_points_image, hit, mode='constant')
    dilated_branches_image = np.where(dilated_branches, 1, 0)
    print("Ensuring binary dilated branches")
    pruned_image = np.subtract(skel_image, dilated_branches_image)
    # pruned_image = np.subtract(skel_image, branch_points_image)
    
    return branch_points, num_branch_points, pruned_image


@timing
def remove_particles(input_file, minpixel, main_output_path, name, prune=False, save_img=False):
    """
    :param main_output_path:
    :param save_img:
    :param name:
    :param prune:
    :param input_file:          string with path to binarized image
    :param minpixel:            minimum hair size in pixels
    :return:                    :img:           cleaned image
                                :input_file:    input file
                                :clean_folder:  directory where cleaned images are stored
    """
    
    img_bool = np.asarray(input_file, dtype=np.bool)
    name = str(name)
    print("\nRemoving particles from pruned image of {}...\n".format(name))
    print(img_bool.dtype)
    print("Image is unsigned integer 8")
    
    output_path = main_output_path
    
    # Gets the unique values in the image matrix. Since it is binary, there should only be 2.
    unique, counts = np.unique(img_bool, return_counts=True)
    print(unique)
    print("Found this many counts:")
    print(len(counts))
    print(counts)
    
    # If the length of unique is not 2 then print that the image isn't a binary.
    if len(unique) != 2:
        print("Image is not binarized!")
        hair_pixels = len(counts)
        print("There is/are {} value(s) present, but there should be 2!\n".format(hair_pixels))
    # If it is binarized, print out that is is and then get the amount of hair pixels to background pixels.
    if counts[0] < counts[1]:
        print("{} is not reversed".format(str(input_file)))
        img = invert(img_bool)
        print("Now {} is reversed =)".format(str(input_file)))
    
    else:
        print("{} is already reversed".format(str(input_file)))
        img = img_bool
    
    print(type(img))
    
    if prune == False:
        minimum = minpixel * 10  # assuming the hairs are no more than 10 pixels thick
        # warnings.filterwarnings("ignore")  # suppress Boolean image UserWarning
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)
    else:
        # clean = img_bool
        minimum = minpixel
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)
        
        print("\n Done cleaning {}".format(name))
    
    if save_img:
        img_inv = invert(clean)
        with pathlib.Path(output_path).joinpath(name + ".tiff") as savename:
            plt.imsave(savename, img_inv, cmap='gray')
            # im = Image.fromarray(img_inv)
            # im.save(output_path)
        return clean, name
    else:
        return clean, name


@timing
def skeletonize_hair(clean_img, name, main_output_path, save_img=False):
    """
    :param clean_img:
    :param name:
    :param main_output_path:
    :param save_img:
    :return:                    :img: skeletonized image
                                :output_name: full path of saved image
                                :skeletonized_folder: path for image folder
    """
    
    # skeletonized_folder_path = make_subdirectory(main_output_path, append_name="thinned")
    
    skeletonized_folder_path = main_output_path
    
    name = name
    
    print("\nSkeletonizing {}...\n".format(name))
    
    skeleton = skimage.morphology.thin(clean_img)
    # skeleton = skimage.morphology.medial_axis(clean_img)
    
    if save_img:
        img_inv = invert(skeleton)
        with pathlib.Path(skeletonized_folder_path).joinpath(name + ".tiff") as output_path:
            im = Image.fromarray(img_inv)
            im.save(output_path)
        return skeleton, name
    
    else:
        print("\n Done skeletonizing {}".format(name))
        
        return skeleton, name


@timing
def analyze_hairs(input_file, name, main_output_path, window_size_mm, minpixel):
    """
    
    :param name:
    :param main_output_path:
    :param window_size_mm:
    :param min_hair:
    :param input_file:          string with the skeletonized image file
    curvature
    :return:                    hair_stats_df -    dataframe with stats for hairs in the sample
                                analysis_folder - path where csv files are sent to
    """
    
    window_size = int(round(window_size_mm * resolution))  # must be an integer
    
    # analysis_folder = make_subdirectory(main_output_path, append_name="analysis")
    
    analysis_folder = main_output_path
    
    output_name = name
    
    img = input_file
    
    if type(img) != 'numpy.ndarray':
        print(type)
        img = np.array(img)
    else:
        print(type(img))
    
    print("Analyzing {}".format(output_name))
    
    label_image, num_elements = skimage.measure.label(img, connectivity=2, return_num=True)
    print(num_elements)
    
    props = regionprops(label_image)
    
    tempdf = [within_hair_curvature(hair, window_size, resolution) for hair in props]
    
    # # for debugging, look for hairs that are the right size
    # tempdf_find = [hair.label for hair in props if hair.area>minpixel]
    
    print("\nData for {} is:".format(name))
    print(tempdf)
    
    within_hairdf = pd.DataFrame(tempdf, columns=['curv_mean', 'curv_median', 'length'])
    
    print("\nDataframe for {} is:".format(name))
    print(within_hairdf)
    print(within_hairdf.dtypes)
    
    with pathlib.Path(analysis_folder).joinpath(output_name + ".csv") as save_path:
        within_hairdf.to_csv(save_path)
    
    within_hair_outliers = np.asarray(remove_outlier(within_hairdf, p1=0.1, p2=0.9))
    print(within_hair_outliers)
    
    within_hairdf2 = pd.DataFrame(within_hair_outliers, columns=['curv_mean', 'curv_median', 'length'])
    
    curv_mean_mean = within_hairdf2['curv_mean'].mean()
    # print(curv_mean_mean)
    
    curv_mean_median = within_hairdf2['curv_mean'].median()
    # print(curv_mean_median)
    
    curv_median_mean = within_hairdf2['curv_median'].mean()
    # print(curv_median_mean)
    
    curv_median_median = within_hairdf2['curv_median'].median()
    # print(curv_median_median)
    
    length_mean = within_hairdf2['length'].mean()
    # print(length_mean)
    length_median = within_hairdf2['length'].median()
    # print(length_median)
    
    hair_count = len(within_hairdf2.index)
    # print(hair_count)
    
    sorted_df = pd.DataFrame(
        [output_name, curv_mean_mean, curv_mean_median, curv_median_mean, curv_median_median, length_mean,
         length_median, hair_count]).T
    
    print("\nDataframe for {} is:".format(name))
    print(sorted_df)
    print("\n")
    
    return sorted_df


@timing
# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming,PyTypeChecker
def TaubinSVD(XYcoords):
    """
    Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
      
    :param XYcoords:    list [[x_1, y_1], [x_2, y_2], ....]
    :return:            a, b, r.  a and b are the center of the fitting circle, and r is the curv
    
    """
    warnings.filterwarnings("ignore")  # suppress RuntimeWarnings from dividing by zero
    XY = np.array(XYcoords)
    X = XY[:, 0] - np.mean(XY[:, 0])  # norming points by x avg
    Y = XY[:, 1] - np.mean(XY[:, 1])  # norming points by y avg
    centroid = [np.mean(XY[:, 0]), np.mean(XY[:, 1])]
    Z = X * X + Y * Y
    Zmean = np.mean(Z)
    Z0 = ((Z - Zmean) / (2. * np.sqrt(Zmean)))  # changed from using old_div to Python 3 native division
    ZXY = np.array([Z0, X, Y]).T
    U, S, V = np.linalg.svd(ZXY, full_matrices=False)  #
    V = V.transpose()
    A = V[:, 2]
    A[0] = (A[0]) / (2. * np.sqrt(Zmean))
    A = np.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
    # a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
    r = np.sqrt(A[1] * A[1] + A[2] * A[2] - 4 * A[0] * A[3]) / abs(A[0]) / 2
    return r


@timing
def within_hair_curvature(hair, window_size, img_res):
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
    taubin_curv = [safe_curv(hair_coords, img_res) for hair_coords in subset_loop]
    
    taubin_df = pd.Series(taubin_curv).astype('float')
    print(taubin_df)
    print(taubin_df.min())
    print(taubin_df.max())
    
    taubin_df2 = remove_outlier(taubin_df, isdf=False, p1=0.01, p2=0.99)
    
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


@timing
def subset_gen(hair_pixel_length, window_size, hair_label):
    subset_start = 0
    if window_size > 10:
        subset_end = int(window_size)
    else:
        subset_end = int(hair_pixel_length)
    while subset_end <= hair_pixel_length:
        subset = hair_label[subset_start:subset_end]
        yield subset
        subset_start += 1
        subset_end += 1


@timing
def safe_curv(hair_coords, resolution):
    try:
        r = TaubinSVD(hair_coords)
        if math.isfinite(r):
            curv = 1 / (r / resolution)
            return curv
        elif math.isfinite(r):
            return 0
    except ValueError or TypeError:
        pass


@timing
def remove_outlier(df, isdf=True, p1=float(0.1), p2=float(0.9)):
    """
    Function to generate a filtered panda dataframe without extreme values.
    
    :param df_in:       input dataframe (should be Pandas Dataframe, not tested with numpy, but that might work as well)
    
    :param col_name:    a string with the column name containing the column where you'll be filtering outliers
    
    :param p1:          float value of the percentile for the bottom cut-off of filtering (optional, defaults at 0.25)
    
    :param p2:          float value of the percentile for the top cut-off of filtering (optional, defaults at 0.75)
    
    :return:            returns a filtered Pandas dataframe containing only the values between the chosen percentile
    cut-offs
    
    """
    
    if isdf:
        x_val = df.values
        print(x_val)
        x = x_val
        print(x)
        # x = x_val.reshape(-1, 1)
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)
        df_in = pd.DataFrame(x_scaled)
        print(df_in)
        q1 = pd.Series(df_in.quantile(p1))
        q3 = pd.Series(df_in.quantile(p2))
        scaled_df_out = df_in.clip(q1, q3, axis=1)
        print(scaled_df_out)
        array_out = scaler.inverse_transform(scaled_df_out)
        print(array_out)
        df_out = pd.DataFrame(array_out)
        
        return df_out
    
    else:
        x_val = df.values
        x = x_val.reshape(-1, 1)
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)
        df_in = pd.DataFrame(x_scaled)
        
        q1 = df_in.quantile(p1)
        q3 = df_in.quantile(p2)
        scaled_df_out = df_in.clip(q1, q3, axis=1)
        array_out = scaler.inverse_transform(scaled_df_out)
        df_out = pd.DataFrame(array_out)
        
        return df_out


@timing
def whole_shebang(f, min_hair, window_size_mm, save_img, filtered_dir, binary_dir, pruned_dir, clean_dir, skeleton_dir,
                  analysis_dir):
    # blockPrint()  # silence output
    minpixel = np.rint(min_hair * resolution)
    try:
        filter_path, name = ridge_filter(f, filtered_dir)
        
        binary_img, name = binary_image(filter_path, binary_dir, resolution, median_kernel_size=5, save_img=save_img)
        
        clean_img, name = remove_particles(binary_img, minpixel, clean_dir, name, prune=False, save_img=save_img)
        
        skeleton, name = skeletonize_hair(clean_img, name, skeleton_dir, save_img=save_img)
        
        skeleton_bin = np.where(skeleton, 1, 0)
        
        branch_points, num_branches, pruned_img = find_branch_points(skeleton_bin, name)
        
        pruned_bin = np.where(pruned_img, 1, 0)
        
        clean_pruned_img, name = remove_particles(pruned_bin, minpixel, pruned_dir, name, prune=True, save_img=save_img)
        
        analysis_output = analyze_hairs(clean_pruned_img, name, analysis_dir, window_size_mm, minpixel)
        
        # enablePrint()  # enable print again
        
        return analysis_output
    
    except:
        pass


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "\n%dh: %02dm: %02ds" % (hour, min, sec)


# %% Define user input

#############################################################################
#                                                                           #
#            This is where the script actually starts doing things.         #
#                                                                           #
#############################################################################


I__USER_INPUT = 0

# Define the folder where your grayscale tiff images are
#
# For windows you can use single backslashes as long as you put the "r" in front
#   of the string, otherwise use double back slashes. For linux and Mac just use
#   forward slashes.

# tiff_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/fibermorph_input/curvature/ValidationSet_TIFF')
# tiff_directory = pathlib.Path(
# r'/Users/tinalasisi/Desktop/fibermorph_input/curvature/ValidationSet_TIFF')
tiff_directory = pathlib.Path(sys.argv[1])

# Designate where fibermorph should make the directory with all your results - this location must exist!
# output_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/')
output_directory = pathlib.Path(sys.argv[2])

# Give your output directory a name
# main_output_name = "OutputTest_Curvature_Jan30"
# main_output_name = "ValidationSet_test_Feb3_full_test"
main_output_name = str(sys.argv[3])

# Specify your file extension as a string - this is case-sensitive.
# file_type = ".tiff"
file_type = str(sys.argv[4])

# How many pixels per mm?
# resolution = int(132)
resolution = int(sys.argv[5])

# What should the size do you want for the window of measurement for curvature within the hair?
# window_size_mm = float(0.5)
window_size_mm = float(sys.argv[6])

# What is the smallest expected length of hair in mm?
# min_hair = float(1.0)
min_hair = float(sys.argv[7])

# How many parallel processes do you want to run?
# jobs = 4
jobs = int(sys.argv[8])

# Do you want to save all intermediate images?
# This is not necessary for the final output and might cause script to run slower
# save_img = False
save_img = str(sys.argv[9])

# %% fibermorph curvature script

I__SCRIPT_STARTS = 0

total_start = timer()

# create an output directory for the analyses
main_output_path, filtered_dir, binary_dir, pruned_dir, clean_dir, skeleton_dir, analysis_dir = make_all_dirs(
    output_directory, main_output_name)

# Change to the folder for reading images
os.chdir(str(tiff_directory))
output_path = main_output_path
glob_file_type = "*" + file_type
file_list = []
for filename in pathlib.Path(tiff_directory).rglob(glob_file_type):
    file_list.append(filename)
list.sort(file_list)  # sort the files
print(len(file_list))  # printed the sorted files

# # Single file loop for debugging:
# file_list = file_list[-5:]
# print(len(file_list))

# %%

# curv_df = (Parallel(n_jobs=2, verbose=10)(
#     delayed(whole_shebang)(f, min_hair, window_size_mm, save_img, filtered_dir, binary_dir, pruned_dir, clean_dir,
#                            skeleton_dir, analysis_dir) for f in file_list))

curv_df = (Parallel(n_jobs=jobs, verbose=100)(
    delayed(whole_shebang)(f, min_hair, window_size_mm, save_img, filtered_dir, binary_dir, pruned_dir, clean_dir,
                           skeleton_dir, analysis_dir) for f in file_list))

hair_summary_df = pd.concat(curv_df)

hair_summary_df.columns = ["ID", "curv_mean_mean", "curv_mean_median", "curv_median_mean", "curv_median_median",
                           "length_mean", "length_median", "hair_count"]

cols1 = ["curv_mean_mean", "curv_mean_median", "curv_median_mean", "curv_median_median", "length_mean", "length_median"]

cols2 = ["length_mean", "length_median"]

hair_summary_df[cols1] = hair_summary_df[cols1].astype(float).round(5)

hair_summary_df[cols2] = hair_summary_df[cols2].astype(float).round(2)

hair_summary_df.set_index('ID', inplace=True)

print("You've got data...")
print(hair_summary_df)

jetzt = datetime.now()
timestamp = jetzt.strftime("_%b%d_%H%M")

with pathlib.Path(main_output_path).joinpath("curvature_summary_data{}.csv".format(timestamp)) as output_path:
    # noinspection PyInterpreter
    hair_summary_df.to_csv(output_path)
    print(output_path)

# End the timer and then print out the how long it took
total_end = timer()
total_time = (total_end - total_start)

# This will print out the minutes to the console, with 2 decimal places.
print("Entire analysis took: {}.".format(convert(total_time)))
