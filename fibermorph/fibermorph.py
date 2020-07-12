# %% Import libraries
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
import numpy as np
import pandas as pd
import rawpy
import scipy
import skimage
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import demo

# Grab version from _version.py in the fibermorph directory
dir = os.path.dirname(__file__)
version_py = os.path.join(dir, "_version.py")
exec(open(version_py).read())


# parse_args() and timing() listed first for easy updating/access

def parse_args():
    """
    Parse command-line arguments
    Returns
    -------
    Parser argument namespace
    """
    parser = argparse.ArgumentParser(description="Fibermorph")
    
    parser.add_argument(
        "--output_directory", default=None,
        help="Required. Full path to and name of desired output directory. "
             "Will be created if it doesn't exist.")
    
    parser.add_argument(
        "--input_directory", default=None,
        help="Required. Full path to and name of desired directory containing "
             "input files.")
    
    parser.add_argument(
        "--resolution_mm", type=int, default=132,
        help="Integer. Number of pixels per mm for curvature analysis.")
    
    parser.add_argument(
        "--resolution_mu", type=float, default=4.25,
        help="Float. Number of pixels per micron for section analysis.")
    
    parser.add_argument(
        "--file_extension", type=str, default=".RW2",
        help="Optional. String. Extension of input files to use in input_directory when using raw2gray function. "
             "Default is .RW2.")
    
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Integer. Number of parallel jobs to run. Default is 1.")
    
    parser.add_argument(
        "--window_size", type=float, default=1.0,
        help="Float. Desired size for window of measurement for curvature analysis in mm. Default is 1.0mm.")
    
    parser.add_argument(
        "--minsize", type=int, default=20,
        help="Integer. Minimum diameter in microns for sections. Default is 20.")
    
    parser.add_argument(
        "--maxsize", type=int, default=150,
        help="Integer. Maximum diameter in microns for sections. Default is 150.")
    
    parser.add_argument(
        "--save_image", type=bool, default=False,
        help="Boolean. Default is False. Whether the curvature function should save images for intermediate image "
             "processing steps.")
    
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Integer. Number of times to repeat validation module (i.e. number of sets of dummy data to generate)."
    )
    parser.add_argument(
        "--within_element", type=bool, default=False,
        help="Boolean. Default is False. Whether an additional directory should be created with spreadsheets of "
             "curvature values within each element."
    )
    parser.add_argument(
        "--window_size_px", type=int, default=10,
        help="Integer. Desired size for window of measurement for curvature analysis in pixels. Default is 10 pixels. Fewer than 3 pixels is not recommended."
    )
    
    # Create mutually exclusive flags for each of fibermorph's modules
    module_group = parser.add_mutually_exclusive_group(required=True)
    
    module_group.add_argument(
        "--raw2gray", action="store_true", default=False,
        help="Convert raw image files to grayscale TIFF files.")
    
    module_group.add_argument(
        "--curvature", action="store_true", default=False,
        help="Analyze curvature in grayscale TIFF images.")
    
    module_group.add_argument(
        "--section", action="store_true", default=False,
        help="Analyze cross-sections in grayscale TIFF images.")
    
    module_group.add_argument(
        "--demo_real_curv", action="store_true", default=False,
        help="A demo of fibermorph curvature analysis with real data.")
    
    module_group.add_argument(
        "--demo_real_section", action="store_true", default=False,
        help="A demo of fibermorph section analysis with real data.")
    
    module_group.add_argument(
        "--demo_dummy_curv", action="store_true", default=False,
        help="A demo of fibermorph curvature with dummy data. Arcs and lines are generated, analyzed and error is "
             "calculated.")
    
    module_group.add_argument(
        "--demo_dummy_section", action="store_true", default=False,
        help="A demo of fibermorph section with dummy data. Circles and ellipses are generated, analyzed and error is "
             "calculated.")
    
    module_group.add_argument(
        "--delete_dir", action="store_true", default=False,
        help="Delete any directory generated in analysis.")
    
    args = parser.parse_args()
    
    # Validate arguments
    demo_mods = [
        args.demo_real_curv,
        args.demo_real_section,
        args.demo_dummy_curv,
        args.demo_dummy_section,
        args.delete_dir]
    
    if any(demo_mods) is False:
        if args.input_directory is None and args.output_directory is None:
            sys.exit("ExitError: need both --input_directory and --output_directory")
        if args.input_directory is None:
            sys.exit("ExitError: need --input_directory")
        if args.output_directory is None:
            sys.exit("ExitError: need --output_directory")
    
    else:
        if args.output_directory is None:
            sys.exit("ExitError: need --output_directory")
    
    return args


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = timeit.default_timer()
        result = f(*args, **kw)
        te = timeit.default_timer()
        print(
            '\n\n The function: {} \n\n with args:[{},\n{}] \n\n and result: {} \n\ntook: {:2.4f} sec\n\n'.format(
                f.__name__, args, kw, result, te - ts))
        return result
    
    return wrap


# Rest of the functions--organized alphabetically

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def copy_if_exist(file, directory):
    """Copies files to destination directory.

    Parameters
    ----------
    file : str
        Path for file to be copied.
    directory : str
        Path for destination directory.

    Returns
    -------
    bool
        True or false depending on whether copying was successful.

    """
    
    path = pathlib.Path(file)
    destination = directory
    
    if os.path.isfile(path):
        shutil.copy(path, destination)
        print('file has been copied'.format(path))
        return True
    else:
        print('file does not exist'.format(path))
        return False


def convert(seconds):
    """Converts seconds into readable format (hours, mins, seconds).

    Parameters
    ----------
    seconds : float or int
        Number of seconds to convert to final format.

    Returns
    -------
    str
        A string with the input seconds converted to a readable format.

    """
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%dh: %02dm: %02ds" % (hour, min, sec)


def enablePrint():
    sys.stdout = sys.__stdout__


@timing
def make_subdirectory(directory, append_name=""):
    """Makes subdirectories.

    Parameters
    ----------
    directory : str or pathlib object
        A string with the path of directory where subdirectories should be created.
    append_name : str
        A string to be appended to the directory path (name of the subdirectory created).

    Returns
    -------
    pathlib object
        A pathlib object for the subdirectory created.

    """
    
    # Define the path of the directory within which this function will make a subdirectory.
    directory = pathlib.Path(directory)
    # The name of the subdirectory.
    append_name = str(append_name)
    # Define the output path by the initial directory and join (i.e. "+") the appropriate text.
    output_path = pathlib.Path(directory).joinpath(str(append_name))
    
    # Use pathlib to see if the output path exists, if it is there it returns True
    if pathlib.Path(output_path).exists() == False:
        
        # Prints a status method to the console using the format option, which fills in the {} with whatever
        # is in the ().
        print(
            "This output path doesn't exist:\n            {} \n Creating...".format(
                output_path))
        
        # Use pathlib to create the folder.
        pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
        
        # Prints a status to let you know that the folder has been created
        print("Output path has been created")
    
    # Since it's a boolean return, and True is the only other option we will simply print the output.
    else:
        # This will print exactly what you tell it, including the space. The backslash n means new line.
        print(
            "Output path already exists:\n               {}".format(
                output_path))
    return output_path


@timing
def list_images(directory):
    """Generates a list of all .tif and/or .tiff files in a directory.

    Parameters
    ----------
    directory : str
        The directory in which the function will recursively search for .tif and .tiff files.

    Returns
    -------
    list
        A list of pathlib objects with the paths to the image files.

    """
    exts = [".tif", ".tiff"]
    mainpath = pathlib.Path(directory)
    file_list = [p for p in pathlib.Path(mainpath).rglob('*') if p.suffix in exts]
    
    list.sort(file_list)  # sort the files
    print(len(file_list))  # printed the sorted files
    
    return file_list


@timing
def raw_to_gray(imgfile, output_directory):
    """Function to convert raw image file into tiff file.

    Parameters
    ----------
    imgfile : str
        Path to raw image file.
    output_directory : str
        String with the path where the converted images should be created.

    Returns
    -------
    pathlib object
        A pathlib object with the path to the converted image file.

    """
    
    imgfile = os.path.abspath(imgfile)
    output_directory = output_directory
    basename = os.path.basename(imgfile)
    name = os.path.splitext(basename)[0] + ".tiff"
    output_name = pathlib.Path(output_directory).joinpath(name)
    print("\n\n")
    print(name)
    
    try:
        with rawpy.imread(imgfile) as raw:
            rgb = raw.postprocess(use_auto_wb=True)
            im = Image.fromarray(rgb).convert('LA')
            im.save(str(output_name))
    except:
        print("\nSomething is wrong with {}\n".format(str(imgfile)))
        pass
    
    print('{} has been successfully converted to a grayscale tiff.\n Path is {}\n'.format(name, output_name))
    
    return output_name


@timing
def analyze_section(input_file, output_path, minsize=20, maxsize=150, resolution=1.0):
    """This function executes a series of functions on the input file to segment and analyze the cross-section found
    in the image.

    Parameters
    ----------
    input_file : str or pathlib object
        Path to the input file (should be .tif or .tiff).
    output_path : str or pathlib object
        Path where the output should be created.
    minsize : int
        An integer describing the minimum diameter a section is expected to have.
    maxsize : int
        An integer describing the maximum diameter a section is expected to have.
    resolution : float
        A float describing the number of pixels per micron in the input image.

    Returns
    -------
    pd Dataframe
        Pandas Dataframe with section information for image.

    """
    
    # segment the image first
    img, im_name = imread(input_file)
    
    img_bool = np.asarray(img, dtype=np.bool)
    
    # Gets the unique values in the image matrix. Since it is binary, there should only be 2.
    unique, counts = np.unique(img_bool, return_counts=True)
    
    if len(unique) != 2:
        print("Image is not binarized!")
        seg_im = segment_section(img)
    else:
        seg_im = skimage.util.invert(img_bool)
    
    # label the image
    label_im, num_elem = skimage.measure.label(seg_im, connectivity=2, return_num=True)
    
    # find center of image
    im_center = list(np.divide(label_im.shape, 2))  # returns array of two floats
    
    minpixel = np.pi * (((minsize / 2) * resolution) ** 2)
    maxpixel = np.pi * (((maxsize / 2) * resolution) ** 2)
    
    props = skimage.measure.regionprops(label_image=label_im, intensity_image=img)
    
    props_df = [[region.label, region.centroid, scipy.spatial.distance.euclidean(im_center, region.centroid)] for region
                in props if minpixel <= region.area <= maxpixel]
    
    props_df = pd.DataFrame(props_df, columns=['label', 'centroid', 'distance'])
    
    print(props_df)
    
    section_id = props_df['distance'].astype(float).idxmin()
    print(section_id)
    
    section = props[section_id]
    
    section_data = [section.filled_area, section.minor_axis_length, section.major_axis_length, section.eccentricity]
    
    section_data = pd.DataFrame([float(x) / resolution for x in section_data]).T
    section_data.columns = ['area', 'min', 'max', 'eccentricity']
    section_data['ID'] = im_name
    
    cropped_bin = props[section_id].filled_image
    
    img_inv = skimage.util.invert(cropped_bin)
    with pathlib.Path(output_path).joinpath(im_name + ".tiff") as savename:
        plt.imsave(savename, img_inv, cmap='gray')
    
    return section_data


@timing
def segment_section(img):
    """Segments the input image to isolate the section(s).

    Parameters
    ----------
    img : np.ndarray
        Image to be segmented.

    Returns
    -------
    np.ndarray
        An ndarray of the segmented (binary) image.

    """
    
    try:
        # thresh = skimage.filters.threshold_otsu(img)
        thresh = skimage.filters.threshold_minimum(img)
    except:
        thresh = img
    
    init_ls = skimage.segmentation.clear_border(img < thresh)
    
    seg_im = skimage.segmentation.morphological_chan_vese(img, 30, init_level_set=init_ls, smoothing=4, lambda1=1,
                                                          lambda2=1)
    
    return seg_im


@timing
def filter_curv(input_file, output_path, save_img=False):
    """Uses a ridge filter to extract the curved (or straight) lines from the background noise.

    Parameters
    ----------
    input_file : str
        A string path to the input image.
    output_path : str
        A string path to the output directory.
    save_img : bool
        True or False for saving filtered image.

    Returns
    -------
    filter_img: np.ndarray
        The filtered image.
    im_name: str
        A string with the image name.

    """
    
    # create pathlib object for input Image
    input_path = pathlib.Path(input_file)
    
    # extract image name
    im_name = input_path.stem
    
    # read in Image
    gray_img = cv2.imread(str(input_path), 0)
    type(gray_img)
    print("Image size is:", gray_img.shape)
    
    # use frangi ridge filter to find hairs, the output will be inverted
    filter_img = skimage.filters.frangi(gray_img)
    type(filter_img)
    print("Image size is:", filter_img.shape)
    
    if save_img:
        output_path = make_subdirectory(output_path, append_name="filtered")
        # inverting and saving the filtered image
        img_inv = skimage.util.invert(filter_img)
        with pathlib.Path(output_path).joinpath(im_name + ".tiff") as save_path:
            plt.imsave(save_path, img_inv, cmap="gray")
    
    return filter_img, im_name


@timing
def binarize_curv(filter_img, im_name, output_path, save_img=False):
    """Binarizes the filtered output of the fibermorph.filter_curv function.

    Parameters
    ----------
    filter_img : np.ndarray
        Image after ridge filter (float64).
    im_name : str
        Image name.
    output_path : str or pathlib object
        Output directory path.
    save_img : bool
        True or false for saving image.

    Returns
    -------
    np.ndarray
        An array with the binarized image.

    """
    
    selem = skimage.morphology.disk(3)
    
    try:
        thresh_im = filter_img > threshold_minimum(filter_img)
    except:
        thresh_im = skimage.util.invert(thresh_im)
    
    # clear the border of the image (buffer is the px width to be considered as border)
    cleared_im = skimage.segmentation.clear_border(thresh_im, buffer_size=10)
    
    # dilate the hair fibers
    binary_im = scipy.ndimage.binary_dilation(cleared_im, structure=selem, iterations=2)
    
    if save_img:
        output_path = make_subdirectory(output_path, append_name="binarized")
        # invert image
        save_im = skimage.util.invert(binary_im)
        
        # save image
        with pathlib.Path(output_path).joinpath(im_name + ".tiff") as save_name:
            im = Image.fromarray(save_im)
            im.save(save_name)
        return binary_im
    
    else:
        return binary_im


@timing
def remove_particles(img, output_path, name, minpixel=5, prune=False, save_img=False):
    """Removes particles under a particular size in the images.

    Parameters
    ----------
    img : np.ndarray
        Binary image to be cleaned.
    output_path : str or pathlib object
        A path to the output directory.
    name : str
        Input image name.
    minpixel : int
        Minimum pixel size below which elements should be removed.
    prune : bool
        True or false for whether the input is a pruned skeleton.
    save_img : bool
        True or false for saving image.

    Returns
    -------
    np.ndarray
        An array with the noise particles removed.

    """
    img_bool = np.asarray(img, dtype=np.bool)
    img = check_bin(img_bool)
    
    if not prune:
        minimum = minpixel
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)
        if save_img:
            output_path = make_subdirectory(output_path, append_name="clean")
            img_inv = skimage.util.invert(clean)
            with pathlib.Path(output_path).joinpath(name + ".tiff") as savename:
                plt.imsave(savename, img_inv, cmap='gray')
            
            return clean
        else:
            return clean
    else:
        # clean = img_bool
        minimum = minpixel
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)
        if save_img:
            output_path = make_subdirectory(output_path, append_name="pruned")
            img_inv = skimage.util.invert(clean)
            with pathlib.Path(output_path).joinpath(name + ".tiff") as savename:
                plt.imsave(savename, img_inv, cmap='gray')
            return clean
        else:
            return clean


@timing
def check_bin(img):
    """Checks whether image has been properly binarized. NB: works on the assumption that there should be more
    background pixels than element pixels.

    Parameters
    ----------
    img : np.ndarray
        Description of parameter `img`.

    Returns
    -------
    np.ndarray
        A binary array of the image.

    """
    img_bool = np.asarray(img, dtype=np.bool)
    
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
        print("{} is not reversed".format(str(img)))
        img = skimage.util.invert(img_bool)
        print("Now {} is reversed =)".format(str(img)))
        return img
    
    else:
        print("{} is already reversed".format(str(img)))
        img = img_bool
        
        print(type(img))
        return img


@timing
def skeletonize(clean_img, name, output_path, save_img=False):
    """Reduces curves and lines to 1 pixel width (skeletons).

    Parameters
    ----------
    clean_img : np.ndarray
        Binary array.
    name : str
        Image name.
    output_path : str or pathlib object.
        Output directory path.
    save_img : bool
        True or false for saving image.

    Returns
    -------
    np.ndarray
        Boolean array of skeletonized image.

    """
    # check if image is binary and properly inverted
    clean_img = check_bin(clean_img)
    
    # skeletonize the hair
    skeleton = skimage.morphology.thin(clean_img)
    
    if save_img:
        output_path = make_subdirectory(output_path, append_name="skeletonized")
        img_inv = skimage.util.invert(skeleton)
        with pathlib.Path(output_path).joinpath(name + ".tiff") as output_path:
            im = Image.fromarray(img_inv)
            im.save(output_path)
        return skeleton
    
    else:
        print("\n Done skeletonizing {}".format(name))
        
        return skeleton


@timing
def prune(skeleton, name, pruned_dir, save_img=False):
    """Prunes branches from skeletonized image.
    Adapted from: "http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm"

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array.
    name : str
        Image name.
    pruned_dir : str or pathlib object
        Output directory path.
    save_img : bool
        True or false for saving image.

    Returns
    -------
    np.ndarray
        Boolean array of pruned skeleton image.

    """
    
    print("\nPruning {}...\n".format(name))
    
    # identify 3-way branch-points
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
    
    # numpy slicing to create 3 remaining rotations
    for ii in range(9):
        hit_list.append(np.transpose(hit_list[-3])[::-1, ...])
    
    # add structure elements for branch-points four 4-way branchpoints
    hit3 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    hit4 = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit_list.append(hit3)
    hit_list.append(hit4)
    print("Creating hit and miss list")
    
    skel_image = check_bin(skeleton)
    print("Converting image to binary array")
    
    branch_points = np.zeros(skel_image.shape)
    print("Creating empty array for branch points")
    
    for hit in hit_list:
        target = hit.sum()
        curr = ndimage.convolve(skel_image, hit, mode="constant")
        branch_points = np.logical_or(branch_points, np.where(curr == target, 1, 0))
    
    print("Completed collection of branch points")
    
    # pixels may "hit" multiple structure elements, ensure the output is a binary image
    branch_points_image = np.where(branch_points, 1, 0)
    print("Ensuring binary")
    
    # use SciPy's ndimage module for locating and determining coordinates of each branch-point
    labels, num_labels = ndimage.label(branch_points_image)
    print("Labelling branches")
    
    # use SciPy's ndimage module to determine the coordinates/pixel corresponding to the center of mass of each
    # branchpoint
    branch_points = ndimage.center_of_mass(skel_image, labels=labels, index=range(1, num_labels + 1))
    branch_points = np.array([value for value in branch_points if not np.isnan(value[0]) or not np.isnan(value[1])],
                             dtype=int)
    # num_branch_points = len(branch_points)
    
    hit = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], dtype=np.uint8)
    
    dilated_branches = ndimage.convolve(branch_points_image, hit, mode='constant')
    dilated_branches_image = np.where(dilated_branches, 1, 0)
    print("Ensuring binary dilated branches")
    pruned_image = np.subtract(skel_image, dilated_branches_image)
    # pruned_image = np.subtract(skel_image, branch_points_image)
    
    pruned_image = remove_particles(pruned_image, pruned_dir, name, prune=True, save_img=save_img)
    
    return pruned_image


@timing
def taubin_curv(coords, resolution):
    """Curvature calculation based on algebraic circle fit by Taubin.
    Adapted from: "https://github.com/PmagPy/PmagPy/blob/2efd4a92ddc19c26b953faaa5c08e3d8ebd305c9/SPD/lib
    /lib_curvature.py"
    G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                Space Curves Defined By Implicit Equations, With
                Applications To Edge And Range Image Segmentation",
    IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)

    Parameters
    ----------
    coords : list
        Nested list of paired x and y coordinates for each point of the line where a curve needs to be fited.
        [[x_1, y_1], [x_2, y_2], ....]
    resolution : float or int
        Number of pixels per mm in original image.

    Returns
    -------
    float or int(0)
        If the radius of the fitted circle is finite, it will return the curvature (1/radius).
        If the radius is infinite, it will return 0.

    """
    
    warnings.filterwarnings("ignore")  # suppress RuntimeWarnings from dividing by zero
    xy = np.array(coords)
    x = xy[:, 0] - np.mean(xy[:, 0])  # norming points by x avg
    y = xy[:, 1] - np.mean(xy[:, 1])  # norming points by y avg
    # centroid = [np.mean(xy[:, 0]), np.mean(xy[:, 1])]
    z = x * x + y * y
    zmean = np.mean(z)
    z0 = ((z - zmean) / (2. * np.sqrt(zmean)))  # changed from using old_div to Python 3 native division
    zxy = np.array([z0, x, y]).T
    u, s, v = np.linalg.svd(zxy, full_matrices=False)  #
    v = v.transpose()
    a = v[:, 2]
    a[0] = (a[0]) / (2. * np.sqrt(zmean))
    a = np.concatenate([a, [(-1. * zmean * a[0])]], axis=0)
    # a, b = (-1 * a[1:3]) / a[0] / 2 + centroid
    r = np.sqrt(a[1] * a[1] + a[2] * a[2] - 4 * a[0] * a[3]) / abs(a[0]) / 2
    
    if np.isfinite(r):
        curv = 1 / (r / resolution)
        if curv >= 0.00001:
            return curv
        else:
            return 0
    else:
        return 0


@timing
def subset_gen(pixel_length, window_size, label):
    """Generator function for start and end indices of the window of measurement.

    Parameters
    ----------
    pixel_length : int
        Number of pixels in input curve/line.
    window_size : int
        The size of window of measurement.
    label : list
        Nested list of coordinates for the input curve/line.

    Returns
    -------
    list
        Nested list of coordinates for the window of measurement in the input curve/line.

    """
    subset_start = 0
    if window_size > 10:
        subset_end = int(window_size + subset_start)
    else:
        subset_end = int(pixel_length)
    while subset_end <= pixel_length:
        subset = label[subset_start:subset_end]
        yield subset
        subset_start += 1
        subset_end += 1


@timing
def within_element_func(output_path, name, element, taubin_df):
    # for within hair distribution
    label_name = str(element.label)
    element_df = pd.DataFrame(taubin_df)
    element_df.columns = ['curv']
    element_df['label'] = label_name
    
    output_path = make_subdirectory(output_path, append_name="WithinElement")
    with pathlib.Path(output_path).joinpath(name + label_name + ".csv") as save_path:
        element_df.to_csv(save_path)
    
    return True


@timing
def analyze_each_curv(element, window_size, resolution, output_path, name, within_element):
    """Calculates curvature for each labeled element in an array.

    Parameters
    ----------
    element : Iterable
        A list of RegionProperties (most importantly, coordinates) from scikit-image regionprops function.
    window_size : int
        Number of pixels to be used for window of measurement.
    resolution : float
        Number of pixels per mm in original image.

    Returns
    -------
    lst
        A list of the mean and median curvatures and the element length.

    """
    
    element_label = np.array(element.coords)
    
    # Due to the differences in distance for vertically and horizontally vs. diagonally adjacent pixels, a correction
    # is applied of a factor of 1.12. See literature below:
    # Smit AL, Sprangers JFCM, Sablik PW, Groenwold J. Automated measurement of root length with a three-dimensional
    # high-resolution scanner and image analysis. Plant Soil. 1994 Jan 1;158(1):145–9.
    # Smit AL, Bengough AG, Engels C, van Noordwijk M, Pellerin S, van de Geijn SC. Root Methods: A Handbook.
    # Springer Science & Business Media; 2013. 594 p.323
    
    length_mm = float(element.area * 1.12 / resolution)
    # length_mm = float(len(element.coords) / resolution)
    print("\nCurv length is {} mm".format(length_mm))
    
    element_pixel_length = len(element.coords)  # length of element in pixels
    print("\nCurv length is {} pixels".format(element_pixel_length))
    
    subset_loop = (subset_gen(element_pixel_length, window_size, element_label))  # generates subset loop
    
    # Safe generator expression in case of errors
    curv = [taubin_curv(element_coords, resolution) for element_coords in subset_loop]
    
    taubin_df = pd.Series(curv).astype('float')
    print("\nCurv dataframe is:")
    print(taubin_df)
    print(type(taubin_df))
    print("\nCurv df min is:{}".format(taubin_df.min()))
    print("\nCurv df max is:{}".format(taubin_df.max()))
    
    print("\nTrimming outliers...")
    taubin_df2 = taubin_df[taubin_df.between(taubin_df.quantile(.01), taubin_df.quantile(.99))]  # without outliers
    
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
    
    within_element_df = [curv_mean, curv_median, length_mm]
    print("\nThe curvature summary stats for this element are:")
    print(within_element_df)
    
    if within_element:
        within_element_func(output_path, name, element, taubin_df)
    
    if within_element_df is not None or np.nan:
        return within_element_df
    else:
        pass


@timing
def imread(input_file):
    """Reads in image as grayscale array.

    Parameters
    ----------
    input_file : str
        String with path to input file.

    Returns
    -------
    img: array uint8
        A grayscale array based on the input image.
    im_name: str
        A string with the image name.

    """
    input_path = pathlib.Path(input_file)
    img = np.array(Image.open(str(input_path)).convert('L'))
    im_name = input_path.stem
    return img, im_name


@timing
def analyze_all_curv(img, name, output_path, resolution, window_size_mm=1, test=False, within_element=False):
    """Analyzes curvature for all elements in an image.

    Parameters
    ----------
    img : np.ndarray
        Pruned skeleton of curves/lines as a uint8 ndarray.
    name : str
        Image name.
    analysis_dir : str or pathlib object
        Output directory.
    resolution : float
        Number of pixels per mm in original image.
    window_size_mm : float
        Desired size for window of measurement in mm.
    test : bool
        True or False for whether this is being run for validation tests
    within_element
        True or False for whether to save spreadsheets with within element curvature values

    Returns
    -------
    pd DataFrame
        Pandas DataFrame with summary data for all elements in image.

    """
    if type(img) != 'np.ndarray':
        print(type)
        img = np.array(img)
    else:
        print(type(img))
    
    print("Analyzing {}".format(name))
    
    img = check_bin(img)
    
    label_image, num_elements = skimage.measure.label(img.astype(int), connectivity=2, return_num=True)
    print("\n There are {} elements in the image".format(num_elements))
    
    props = skimage.measure.regionprops(label_image)
    
    window_size = int(round(window_size_mm * resolution))  # must be an integer
    print("\nWindow size for analysis is {} pixels".format(window_size))
    print("Analysis of curvature for each element begins...")
    tempdf = [analyze_each_curv(hair, window_size, resolution, output_path, name, within_element) for hair in props]
    
    print("\nData for {} is:".format(name))
    print(tempdf)
    
    within_im_curvdf = pd.DataFrame(tempdf, columns=['curv_mean', 'curv_median', 'length'])
    
    print("\nDataframe for {} is:".format(name))
    print(within_im_curvdf)
    print(within_im_curvdf.dtypes)
    
    within_im_curvdf2 = pd.DataFrame(within_im_curvdf, columns=['curv_mean', 'curv_median', 'length']).dropna()
    
    print("\nDataFrame with NaN values dropped:")
    print(within_im_curvdf2)
    
    output_path = make_subdirectory(output_path, append_name="analysis")
    with pathlib.Path(output_path).joinpath(name + ".csv") as save_path:
        within_im_curvdf2.to_csv(save_path)
    
    curv_mean_im_mean = within_im_curvdf2['curv_mean'].mean()
    curv_mean_im_median = within_im_curvdf2['curv_mean'].median()
    curv_median_im_mean = within_im_curvdf2['curv_median'].mean()
    curv_median_im_median = within_im_curvdf2['curv_median'].median()
    length_mean = within_im_curvdf2['length'].mean()
    length_median = within_im_curvdf2['length'].median()
    hair_count = len(within_im_curvdf2.index)
    
    im_sumdf = pd.DataFrame(
        [name, curv_mean_im_mean, curv_mean_im_median, curv_median_im_mean, curv_median_im_median, length_mean,
         length_median, hair_count]).T
    
    print("\nDataframe for {} is:".format(name))
    print(im_sumdf)
    print("\n")
    
    if test:
        return within_im_curvdf2
    else:
        return im_sumdf


@timing
def curvature_seq(input_file, output_path, resolution, window_size_mm, save_img, test=False, within_element=False):
    """Sequence of functions to be executed for calculating curvature in fibermorph.

    Parameters
    ----------
    input_file : str or pathlib Path object
        Path to image that needs to be analyzed.
    output_path : str or pathlib Path object
        Output directory
    resolution : float
        Number of pixels per mm in original image.
    window_size_mm : float
        Desired size for window of measurement in mm.
    save_img : bool
        True or false for saving images.
    test : bool
        True or false for whether this is being run for validation tests.
    within_element
        True or False for whether to save spreadsheets with within element curvature values

    Returns
    -------
    pd DataFrame
        Pandas DataFrame with curvature summary data for all images.

    """
    
    # filter
    filter_img, im_name = filter_curv(input_file, output_path)
    
    # binarize
    binary_img = binarize_curv(filter_img, im_name, output_path, save_img)
    
    # remove particles
    clean_im = remove_particles(binary_img, output_path, im_name, minpixel=5, prune=False, save_img=save_img)
    
    # skeletonize
    skeleton_im = skeletonize(clean_im, im_name, output_path, save_img)
    
    # prune
    pruned_im = prune(skeleton_im, im_name, output_path, save_img)
    
    # analyze
    im_df = analyze_all_curv(pruned_im, im_name, output_path, resolution, window_size_mm, test, within_element)
    
    return im_df


# Main modules (organized in order of operations: raw2gray, curvature, section)

@timing
def raw2gray(input_directory, output_location, file_type, jobs):
    """Convert raw files to grayscale tiff files.

    Parameters
    ----------
    input_directory : str or pathlib object
        String or pathlib object for input directory containing raw files.
    output_location : str or pathlib object
        String or pathlib object for output directory where converted files should be created.
    file_type : str
        The extension for the raw files (e.g. ".RW2").
    jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    bool
        True

    """
    
    total_start = timer()
    
    file_list = [p for p in pathlib.Path(input_directory).rglob('*') if p.suffix in file_type]
    list.sort(file_list)  # sort the files
    print(file_list)  # printed the sorted files
    
    print("There are {} files to convert".format(len(file_list)))
    print("\n\n")
    
    print("Converting raw files into grayscale tiff files...\n")
    
    tiff_directory = make_subdirectory(output_location, append_name="tiff")
    
    Parallel(n_jobs=jobs, verbose=100)(delayed(raw_to_gray)(f, tiff_directory) for f in file_list)
    
    # End the timer and then print out the how long it took
    total_end = timer()
    total_time = (total_end - total_start)
    
    # This will print out the minutes to the console, with 2 decimal places.
    print("\n\n")
    print("Entire analysis took: {}.".format(convert(total_time)))
    
    return True


@timing
def curvature(input_directory, main_output_path, jobs, resolution, window_size_mm, save_img, within_element):
    """Takes directory of grayscale tiff images and analyzes curvature for each curve/line in the image.

    Parameters
    ----------
    input_directory : str or pathlib object
        Input directory path as str or pathlib object.
    main_output_path : str or pathlib object
        Main output path as str or pathlib object.
    jobs : int
        Number of jobs to run in parallel.
    resolution : float
        Number of pixels per mm in original image.
    window_size_mm : float
        Desired window of measurement in mm.
    save_img : bool
        True or false for saving images for image processing steps.
    within_element
        True or False for whether to save spreadsheets with within element curvature values

    Returns
    -------
    bool
        True.

    """
    
    total_start = timer()
    
    # create an output directory for the analyses
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    dir_name = str(timestamp + "fibermorph_curvature")
    output_path = make_subdirectory(main_output_path, append_name=dir_name)
    
    file_list = list_images(input_directory)
    
    # List expression for curv df per image
    # im_df = [curvature_seq(input_file, filtered_dir, binary_dir, pruned_dir, clean_dir, skeleton_dir, analysis_dir,
    # resolution, window_size_mm, save_img) for input_file in file_list]
    
    # This is the old parallel jobs function
    im_df = (Parallel(n_jobs=jobs, verbose=100)(
        delayed(curvature_seq)(input_file, output_path,
                               resolution, window_size_mm, save_img, test=False, within_element=within_element) for
        input_file in file_list))
    
    summary_df = pd.concat(im_df)
    
    print("This is the summary dataframe for the current sample")
    print(summary_df)
    
    summary_df.columns = [
        "ID", "curv_mean_mean", "curv_mean_median", "curv_median_mean", "curv_median_median",
        "length_mean", "length_median", "hair_count"]
    
    cols1 = [
        "curv_mean_mean", "curv_mean_median", "curv_median_mean",
        "curv_median_median", "length_mean", "length_median"]
    
    cols2 = ["length_mean", "length_median"]
    
    summary_df[cols1] = summary_df[cols1].astype(float).round(5)
    
    summary_df[cols2] = summary_df[cols2].astype(float).round(2)
    
    summary_df.set_index('ID', inplace=True)
    
    print("You've got data...")
    print(summary_df)
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("_%b%d_%H%M")
    
    with pathlib.Path(output_path).joinpath("curvature_summary_data{}.csv".format(timestamp)) as output_path:
        summary_df.to_csv(output_path)
        print(output_path)
    
    # End the timer and then print out the how long it took
    total_end = timer()
    total_time = (total_end - total_start)
    
    # This will print out the minutes to the console, with 2 decimal places.
    print("Entire analysis took: {}.".format(convert(total_time)))
    
    return True


@timing
def section(input_directory, main_output_path, jobs, resolution, minsize=20, maxsize=150):
    """Takes directory of grayscale images (and locates central section where necessary) and analyzes cross-sectional
    properties for each image.

    Parameters
    ----------
    input_directory : str or pathlib object
        Input directory path as str or pathlib object.
    main_output_path : str or pathlib object
        Main output path as str or pathlib object.
    jobs : int
        Number of jobs to run in parallel.
    resolution : float
        Number of pixels per micrometer in the image.
    minsize : int
        Minimum diameter for sections.
    maxsize : int
        Maximum diameter for sections.

    Returns
    -------
    bool
        True.

    """
    
    total_start = timer()
    
    # Change to the folder for reading images
    file_list = list_images(input_directory)
    
    # Shows what is in the file_list. The backslash n prints a new line
    print("There are {} files in the cropped_list:".format(len(file_list)))
    print(file_list, "\n\n")
    
    # Creating subdirectories for cropped images
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    dir_name = str(timestamp + "fibermorph_section")
    output_path = make_subdirectory(main_output_path, append_name=dir_name)
    
    output_im_path = make_subdirectory(output_path, "cropped_binary")
    
    # section_df = [analyze_section(f, output_im_path, minsize, maxsize, resolution) for f in file_list]
    
    section_df = (Parallel(n_jobs=jobs, verbose=100)(
        delayed(analyze_section)(f, output_im_path, minsize, maxsize, resolution) for f in file_list))
    
    section_df = pd.concat(section_df)
    section_df.columns = ['area', 'min', 'max', 'eccentricity', 'ID']
    section_df.set_index('ID', inplace=True)
    
    with pathlib.Path(output_path).joinpath("section_data.csv") as df_output_path:
        section_df.to_csv(df_output_path)
    
    # End the timer and then print out the how long it took
    total_end = timer()
    total_time = int(total_end - total_start)
    
    print("Complete analysis time: {}".format(convert(total_time)))
    
    return True


def main():
    args = parse_args()
    
    # Run fibermorph
    
    if args.delete_dir is True:
        demo.delete_dir(args.output_directory)
        sys.exit(0)
    elif args.demo_real_curv is True:
        demo.real_curv(args.output_directory)
        sys.exit(0)
    elif args.demo_real_section is True:
        demo.real_section(args.output_directory)
        sys.exit(0)
    elif args.demo_dummy_curv is True:
        demo.dummy_curv(args.output_directory, args.repeats, args.window_size_px)
        sys.exit(0)
    elif args.demo_dummy_section is True:
        demo.dummy_section(args.output_directory, args.repeats)
        sys.exit(0)
    
    # Check for output directory and create it if it doesn't exist
    output_dir = make_subdirectory(args.output_directory)
    
    if args.raw2gray is True:
        raw2gray(
            args.input_directory, output_dir, args.file_extension, args.jobs)
    elif args.curvature is True:
        curvature(
            args.input_directory, output_dir, args.jobs,
            args.resolution_mm, args.window_size, args.save_image, args.within_element)
    elif args.section is True:
        section(
            args.input_directory, output_dir, args.jobs,
            args.resolution_mu, args.minsize, args.maxsize)
    else:
        sys.exit("Error. Tim didn't exhaust all module options")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
