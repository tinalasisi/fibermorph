from argparse import ArgumentParser as ap
from datetime import datetime
import fnmatch
from functools import wraps
import multiprocessing
import os
import pathlib
import re
import sys
from time import time

import warnings
from matplotlib import pyplot as plt
from PIL import Image  # helps importing data into numpy arrays
from scipy import ndimage
from joblib import Parallel, delayed

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
from skimage import io, filters, segmentation


class Fibermorph:

    def __init__(self):
        pass

    def configure_args(self):
        '''
        Parses command-line arguments
        Returns
        -------
        parser
        '''
        parser = ap(description="fibermorph")

        parser.add_argument(
            '-i', '--input_directory', metavar='', default=None,
            help='Required. Full path to and name of desired directory containing input files.')

        parser.add_argument(
            '-o', '--output_directory', metavar='', default=None,
            help='Required. Full path to and name of desired output directory. Will be created if it does not exist.')

        parser.add_argument(
            '-j', '--jobs', type=int, metavar='', default=1,
            help='Integer. Number of parallel jobs to run. Default is 1.')

        parser.add_argument(
            '-s', '--save_image', action='store_true', default=False,
            help='Default is False. Will save intermediate curvature/section processing images if --save_image flag is included.')

        #args = parser.parse_args()

        return parser

    def read_files(self, input_directory):
        '''
        Reads image files
        Parameter
        ---------
        input_directory : str
            Path for section image files directory.
        Returns
        -------
        list[Path]
            A list of filepaths for each images in the directory.
        '''
        if not os.path.isdir(input_directory):
            raise ValueError(
                'The input filepath is invalid. Please input a valid directory filepath with hair images.')

        # *tif & *tiff
        ftype = ['*.tif', '*.tiff']
        ftype = r'|'.join([fnmatch.translate(x) for x in ftype])

        flist = []
        # Iterate over files in the directory in search for tiff files
        for root, dirs, files in os.walk(input_directory, topdown=False):
            files = [os.path.join(root, f) for f in files]
            files = [f for f in files if re.match(ftype, f)]
            flist += files

        if not flist:
            raise FileExistsError(
                'No TIFF images found in the directory, Please input a valid directory filepath with hair images.')

        return flist


class Curvature(Fibermorph):

    def __init__(self):
        super().__init__()

    def configure_args(self):
        '''
        Parses command-line arguments

        Gets
        -------
        resolution
        window_size
        save_img
        test
        within_element

        Returns
        -------
        parser
        '''
        parser = super().configure_args()

        gr_curv = parser.add_argument_group(
            "curvature options", "arguments used specifically for curvature module")

        gr_curv.add_argument("--resolution_mm", type=int, metavar="", default=132,
                             help="Integer. Number of pixels per mm for curvature analysis. Default is 132.")

        gr_curv.add_argument("--window_size", metavar="", default=None, nargs='+', help="Float or integer or None. Desired size for window of measurement for curvature analysis in pixels or mm (given "
                             "the flag --window_unit). If nothing is entered, the default is None and the entire hair will be used to for the curve fitting.")

        gr_curv.add_argument("--window_unit", type=str, default="px", choices=["px", "mm"], help="String. Unit of measurement for window of measurement for curvature analysis. Can be 'px' (pixels) or "
                             "'mm'. Default is 'px'.")

        gr_curv.add_argument("-W", "--within_element", action="store_true", default=False, help="Boolean. Default is False. Will create an additional directory with spreadsheets of raw curvature "
                             "measurements for each hair if the --within_element flag is included.")

        return parser

    def run(self):
        '''
        Executes the curv analysis

        Returns
        ----------
        input_file
        output_path
        jobs
        resolution
        window_size
        save_img
        test
        within_element
        '''
        try:

            # edit here
            args = self.configure_args().parse_args()

            # create an output directory for the analyses
            jetzt = datetime.now()
            timestamp = jetzt.strftime("%b%d_%H%M_")
            dir_name = str(timestamp + "fibermorph_curvature")
            main_output_path = str(args.output_directory)
            output_path = self.make_subdirectory(main_output_path, append_name=dir_name)

            files_list = self.read_files(args.input_directory)

            # im_df =(Parallel(n_jobs=args.jobs, verbose=0)(delayed(self.curvature_seq)(args.input_directory, args.output_directory, args.resolution_mm, args.window_size, args.window_unit, args.save_image, False, args.within_element) for input_file in files_list))
            for images in files_list:
                im_df = self.curvature_seq(images, main_output_path, args.resolution_mm, args.window_size, args.window_unit, args.save_image, False, args.within_element)
                summary_df = pd.concat(im_df)

            with pathlib.Path(output_path).joinpath("curvature_summary_data{}.csv".format(timestamp)) as output_path:
                summary_df.to_csv(output_path)
                # print(output_path)

            sys.exit(0)

        except KeyboardInterrupt:
            print('terminating...')
            sys.exit(0)

    def curvature_seq(self, input_file, output_path, resolution, window_size, window_unit, save_img, test, within_element):
        """Sequence of functions to be executed for calculating curvature in fibermorph.

        Parameters
        ----------
        input_file : str or pathlib Path object
            Path to image that needs to be analyzed.
        output_path : str or pathlib Path object
            Output directory
        resolution : int
            Number of pixels per mm in original image.
        window_size : float or float
            Desired size for window of measurement in mm.
        save_img : bool
            True or false for saving images.
        jobs : int
            Number of jobs to run in parallel.            
        test : bool
            True or false for whether this is being run for validation tests.
        within_element
            True or False for whether to save spreadsheets with within element curvature values

        Returns
        -------
        pd DataFrame
            Pandas DataFrame with curvature summary data for all images.

        """

        for i in [input_file]:

            # filter
            filter_img, im_name = self.filter_curv(
                input_file, output_path, save_img)

            # binarize
            binary_img = self.binarize_curv(
                filter_img, im_name, output_path, save_img)

            # remove particles
            clean_im = self.remove_particles(binary_img, output_path, im_name, minpixel=int(
                resolution/2), prune=False, save_img=save_img)

            # skeletonize
            skeleton_im = self.skeletonize(
                clean_im, im_name, output_path, save_img)

            # prune
            pruned_im = self.prune(skeleton_im, im_name, output_path, save_img)

            # analyze
            im_df = self.analyze_all_curv(
                pruned_im, im_name, output_path, resolution, window_size, window_unit, test, within_element)

            return im_df

    def taubin_curv(self, coords, resolution):
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

        # suppress RuntimeWarnings from dividing by zero
        warnings.filterwarnings("ignore")
        xy = np.array(coords)
        x = xy[:, 0] - np.mean(xy[:, 0])  # norming points by x avg
        y = xy[:, 1] - np.mean(xy[:, 1])  # norming points by y avg
        # centroid = [np.mean(xy[:, 0]), np.mean(xy[:, 1])]
        z = x * x + y * y
        zmean = np.mean(z)
        # changed from using old_div to Python 3 native division
        z0 = ((z - zmean) / (2. * np.sqrt(zmean)))
        zxy = np.array([z0, x, y]).T
        u, s, v = np.linalg.svd(zxy, full_matrices=False)  #
        v = v.transpose()
        a = v[:, 2]
        a[0] = (a[0]) / (2. * np.sqrt(zmean))
        a = np.concatenate([a, [(-1. * zmean * a[0])]], axis=0)
        # a, b = (-1 * a[1:3]) / a[0] / 2 + centroid
        r = np.sqrt(a[1] * a[1] + a[2] * a[2] -
                    4 * a[0] * a[3]) / abs(a[0]) / 2

        if np.isfinite(r):
            curv = 1 / (r / resolution)
            if curv >= 0.00001:
                return curv
            else:
                return 0
        else:
            return 0

    def filter_curv(self, input_file, output_path, save_img):
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

        input_path = pathlib.Path(input_file)

        # gets the image as its numpy form (int and float) and location (String)
        gray_img, im_name = self.imread(input_path)

        # use frangi ridge filter to find hairs, the output will be inverted
        filter_img = skimage.filters.frangi(gray_img)
        type(filter_img)

        if save_img:
            output_path = self.make_subdirectory(
                output_path, append_name="filtered")

            # inverting and saving the filtered image
            img_inv = skimage.util.invert(filter_img)
            # join path does this correctly rather than concatination.
            with pathlib.Path(output_path).joinpath(im_name + ".tiff") as save_path:
                plt.imsave(save_path, img_inv, cmap="gray")

        return filter_img, im_name

    def binarize_curv(self, filter_img, im_name, output_path, save_img):
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

        selem = skimage.morphology.disk(5)

        # performs logrithmic corrections (exposure enhancements)
        filter_img = skimage.exposure.adjust_log(filter_img)

        try:
            # uses otsu's method to focus hair in foreground and enhance image. Is then compared to previously filtered image.
            thresh_im = filter_img > filters.threshold_otsu(filter_img)
        except:
            thresh_im = skimage.util.invert(filter_img)

        # clear the border of the image (buffer is the px width to be considered as border)
        cleared_im = skimage.segmentation.clear_border(
            thresh_im, buffer_size=10)

        # dilate the hair fibers
        # expands the shapes of the hair in the image sample
        binary_im = scipy.ndimage.binary_dilation(
            cleared_im, structure=selem, iterations=2)

        if save_img:
            output_path = self.make_subdirectory(
                output_path, append_name="binarized")
            # invert image
            save_im = skimage.util.invert(binary_im)

            # save image
            with pathlib.Path(output_path).joinpath(im_name + ".tiff") as save_name:
                im = Image.fromarray(save_im)
                im.save(save_name)

        # return image
        return binary_im

    def remove_particles(self, img, output_path, name, minpixel, prune, save_img):
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
        img = self.check_bin(img_bool)

        minimum = minpixel
        clean = skimage.morphology.remove_small_objects(
            img, connectivity=2, min_size=minimum)

        if save_img:
            img_inv = skimage.util.invert(clean)
            if prune:
                output_path = self.make_subdirectory(
                    output_path, append_name="pruned")
            else:
                output_path = self.make_subdirectory(
                    output_path, append_name="clean")
            with pathlib.Path(output_path).joinpath(name + ".tiff") as savename:
                plt.imsave(savename, img_inv, cmap='gray')

        return clean

    def check_bin(self, img):
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

        # If the length of unique is not 2 then print that the image isn't a binary.
        # if len(unique) != 2:
        #     hair_pixels = len(counts)
        #     return hair_pixels

        # If it is binarized, print out that is is and then get the amount of hair pixels to background pixels.
        if counts[0] < counts[1]:
            img = skimage.util.invert(img_bool)
            return img

        else:
            img = img_bool
            return img

    def skeletonize(self, clean_img, name, output_path, save_img):
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
        clean_img = self.check_bin(clean_img)

        # skeletonize the hair
        skeleton = skimage.morphology.thin(clean_img)

        if save_img:
            output_path = self.make_subdirectory(
                output_path, append_name="skeletonized")
            img_inv = skimage.util.invert(skeleton)
            with pathlib.Path(output_path).joinpath(name + ".tiff") as output_path:
                im = Image.fromarray(img_inv)
                im.save(output_path)
            return skeleton

        else:
            # print("\n Done skeletonizing {}".format(name))

            return skeleton

    def prune(self, skeleton, name, pruned_dir, save_img):
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

        # print("\nPruning {}...\n".format(name))

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
        # FIXME:
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
        # print("Creating hit and miss list")

        skel_image = self.check_bin(skeleton)
        # print("Converting image to binary array")

        branch_points = np.zeros(skel_image.shape)
        # print("Creating empty array for branch points")

        for hit in hit_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            branch_points = np.logical_or(
                branch_points, np.where(curr == target, 1, 0))

        # print("Completed collection of branch points")

        # pixels may "hit" multiple structure elements, ensure the output is a binary image
        branch_points_image = np.where(branch_points, 1, 0)
        # print("Ensuring binary")

        # use SciPy's ndimage module for locating and determining coordinates of each branch-point
        labels, num_labels = ndimage.label(branch_points_image)
        # print("Labelling branches")

        # use SciPy's ndimage module to determine the coordinates/pixel corresponding to the center of mass of each
        # branchpoint
        branch_points = ndimage.center_of_mass(
            skel_image, labels=labels, index=range(1, num_labels + 1))
        branch_points = np.array([value for value in branch_points if not np.isnan(value[0]) or not np.isnan(value[1])],
                                 dtype=int)
        # num_branch_points = len(branch_points)

        hit = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.uint8)

        dilated_branches = ndimage.convolve(
            branch_points_image, hit, mode='constant')
        dilated_branches_image = np.where(dilated_branches, 1, 0)
        # print("Ensuring binary dilated branches")
        pruned_image = np.subtract(skel_image, dilated_branches_image)
        # pruned_image = np.subtract(skel_image, branch_points_image)

        pruned_image = self.remove_particles(
            pruned_image, pruned_dir, name, minpixel=5, prune=True, save_img=save_img)

        return pruned_image

    def diag(self, skeleton):
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

        # identify diagonals
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

        hit9 = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]], dtype=np.uint8)
        hit10 = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.uint8)

        diag_list = [hit9, hit10]

        hit11 = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.uint8)
        hit12 = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]], dtype=np.uint8)

        adj_list = [hit11, hit12]

        skel_image = self.check_bin(skeleton).astype(int)
        # print("Converting image to binary array")

        diag_points = np.zeros(skel_image.shape)
        mid_points = np.zeros(skel_image.shape)
        adj_points = np.zeros(skel_image.shape)
        # print("Creating empty array for branch points")

        for hit in diag_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            diag_points = np.logical_or(
                diag_points, np.where(curr == target, 1, 0))

        for hit in mid_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            mid_points = np.logical_or(
                mid_points, np.where(curr == target, 1, 0))

        for hit in adj_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            adj_points = np.logical_or(
                adj_points, np.where(curr == target, 1, 0))

        # pixels may "hit" multiple structure elements, ensure the output is a binary image
        diag_points_image = np.where(diag_points, 1, 0)
        mid_points_image = np.where(mid_points, 1, 0)
        adj_points_image = np.where(adj_points, 1, 0)
        # print("Ensuring binary")

        # use SciPy's ndimage module for locating and determining coordinates of each branch-point
        labels, num_labels = ndimage.label(diag_points_image)
        labels2, num_labels2 = ndimage.label(mid_points_image)
        labels3, num_labels3 = ndimage.label(adj_points_image)
        # print("Labelling branches")

        # use SciPy's ndimage module to determine the coordinates/pixel corresponding to the center of mass of each
        # branchpoint
        diag_points = ndimage.center_of_mass(
            skel_image, labels=labels, index=range(1, num_labels + 1))
        mid_points = ndimage.center_of_mass(
            skel_image, labels=labels2, index=range(1, num_labels2 + 1))
        adj_points = ndimage.center_of_mass(
            skel_image, labels=labels3, index=range(1, num_labels3 + 1))

        diag_points = np.array([value for value in diag_points if not np.isnan(
            value[0]) or not np.isnan(value[1])], dtype=int)
        mid_points = np.array([value for value in mid_points if not np.isnan(
            value[0]) or not np.isnan(value[1])], dtype=int)
        adj_points = np.array([value for value in adj_points if not np.isnan(value[0]) or not np.isnan(value[1])],
                              dtype=int)

        num_diag_points = len(diag_points)
        num_mid_points = len(mid_points)
        num_adj_points = len(adj_points)

        return num_diag_points, num_mid_points, num_adj_points

    def analyze_all_curv(self, img, name, output_path, resolution, window_size, window_unit, test, within_element):
        """Analyzes curvature for all elements in an image.

        Parameters
        ----------
        img : np.ndarray
            Pruned skeleton of curves/lines as a uint8 ndarray.
        name : str
            Image name.
        output_path : str or pathlib object
            Output directory.
        resolution : int
            Number of pixels per mm in original image.
        window_size: float or int or list
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
            print(type(img))
            img = np.array(img)
        else:
            print(type(img))

        # print("Analyzing {}".format(name))

        img = self.check_bin(img)

        label_image, num_elements = skimage.measure.label(
            img.astype(int), connectivity=2, return_num=True)

        props = skimage.measure.regionprops(label_image)

        if not isinstance(window_size, list):
            window_size = [window_size]

        name = name

        im_sumdf = [self.window_iter(
            props, name, i, window_unit, resolution, output_path, test, within_element) for i in window_size]

        im_sumdf = pd.concat(im_sumdf)

        return im_sumdf

    def window_iter(self, props, name, window_size, window_unit, resolution, output_path, test, within_element):

        tempdf = []

        if not window_size is None:
            if not window_unit == "px":
                window_size_px = int(window_size * resolution)
            else:
                window_size_px = int(window_size)
                window_size = int(window_size)

            name = str(name + "_WindowSize-" +
                       str(window_size) + str(window_unit))

            tempdf = [self.analyze_each_curv(hair, window_size_px, resolution, output_path,
                                             name, within_element) for hair in props if hair.area > window_size]

            within_im_curvdf = pd.DataFrame(
                tempdf, columns=['curv_mean', 'curv_median', 'length'])

            within_im_curvdf2 = pd.DataFrame(within_im_curvdf, columns=[
                                             'curv_mean', 'curv_median', 'length']).dropna()

            output_path = self.make_subdirectory(
                output_path, append_name="analysis")
            with pathlib.Path(output_path).joinpath("ImageSum_" + name + ".csv") as save_path:
                within_im_curvdf2.to_csv(save_path)

            curv_mean_im_mean = within_im_curvdf2['curv_mean'].mean()
            curv_mean_im_median = within_im_curvdf2['curv_mean'].median()
            curv_median_im_mean = within_im_curvdf2['curv_median'].mean()
            curv_median_im_median = within_im_curvdf2['curv_median'].median()
            length_mean = within_im_curvdf2['length'].mean()
            length_median = within_im_curvdf2['length'].median()
            hair_count = len(within_im_curvdf2.index)

            im_sumdf = pd.DataFrame(
                {"ID": [name], "curv_mean_mean": [curv_mean_im_mean], "curv_mean_median": [curv_mean_im_median], "curv_median_mean": [curv_median_im_mean], "curv_median_median": [curv_median_im_median], "length_mean": [length_mean], "length_median": [length_median], "hair_count": [hair_count]})

            if test:
                return within_im_curvdf2
            else:
                return im_sumdf

        elif window_size is None:
            window_size_px = None
            within_element = None
            minsize = 0.5 * resolution
            tempdf = [self.analyze_each_curv(hair, window_size_px, resolution, output_path, name, within_element) for hair in
                      props if hair.area > minsize]

            within_im_curvdf = pd.concat(tempdf)

            within_im_curvdf2 = within_im_curvdf.dropna()

            output_path = self.make_subdirectory(
                output_path, append_name="analysis")
            with pathlib.Path(output_path).joinpath("ImageSum_" + name + ".csv") as save_path:
                within_im_curvdf2.to_csv(save_path)

            im_mean = within_im_curvdf2['curv'].mean()
            im_median = within_im_curvdf2['curv'].median()
            length_mean = within_im_curvdf2['length'].mean()
            length_median = within_im_curvdf2['length'].median()
            hair_count = len(within_im_curvdf2.index)

            im_sumdf = pd.DataFrame({'ID': name, 'curv_mean': [im_mean], 'curv_median': [im_median], 'length_mean': [
                                    length_mean], 'length_median': [length_median], 'hair_count': [hair_count]})

            if test:
                return within_im_curvdf2
            else:
                return im_sumdf

    def make_subdirectory(self, directory, append_name=""):
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
            print("Output path already exists:\n               {}".format(
                output_path))
        return output_path

    def imread(self, input_file, use_skimage=False):
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

        # If True, convert color images to gray-scale (64-bit floats).
        # Images that are already in gray-scale format are not converted

        if use_skimage:
            try:
                # img_float = img_arrayndarray The different color bands/channels are stored in the third dimension, such that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
                img_float = skimage.io.imread(input_file, as_gray=True)
                # Convert an image to 8-bit unsigned integer format.
                img = skimage.img_as_ubyte(img_float)
            except ValueError:
                # importing imagge into numpy arrays using given path of image
                img = np.array(Image.open(str(input_path)).convert('L'))
        else:
            print(str(input_path))
            img = np.array(Image.open(str(input_path)).convert('L'))
        # Returns the filename identified by the generic-format path stripped of its extension.
        im_name = input_path.stem
        return img, im_name  # image as np and image name (path) is returned

    def analyze_each_curv(self, element, window_size_px, resolution, output_path, name, within_element):
        """Calculates curvature for each labeled element in an array.

        Parameters
        ----------
        element : Iterable
            A list of RegionProperties (most importantly, coordinates) from scikit-image regionprops function.
        window_size_px : int
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

        element_pixel_length = int(element.area)  # length of element in pixels

        corr_element_pixel_length = self.pixel_length_correction(element)

        length_mm = float(corr_element_pixel_length / resolution)

        if not window_size_px is None:
            window_size_px = int(window_size_px)

            subset_loop = (self.subset_gen(
                element_pixel_length, window_size_px, element_label))  # generates subset loop

            # Safe generator expression in case of errors
            curv = [self.taubin_curv(element_coords, resolution)
                    for element_coords in subset_loop]

            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
            taubin_df = pd.Series(curv).astype('float')

            # print("\nCurv dataframe is:")
            # print(taubin_df)
            # print(type(taubin_df))
            # print("\nCurv df min is:{}".format(taubin_df.min()))
            # print("\nCurv df max is:{}".format(taubin_df.max()))

            # print("\nTrimming outliers...")
            taubin_df2 = taubin_df[taubin_df.between(
                taubin_df.quantile(.01), taubin_df.quantile(.99))]  # without outliers

            # print("\nAfter trimming outliers...")
            # print("\nCurv dataframe is:")
            # print(taubin_df2)
            # print(type(taubin_df2))
            # print("\nCurv df min is:{}".format(taubin_df2.min()))
            # print("\nCurv df max is:{}".format(taubin_df2.max()))

            curv_mean = taubin_df2.mean()
            # print("\nCurv mean is:{}".format(curv_mean))

            curv_median = taubin_df2.median()
            # print("\nCurv median is:{}".format(curv_median))

            within_element_df = [curv_mean, curv_median, length_mm]
            # print("\nThe curvature summary stats for this element are:")
            # print(within_element_df)

            if within_element:
                label_name = str(element.label)
                element_df = pd.DataFrame(taubin_df)
                element_df.columns = ['curv']
                element_df['label'] = label_name

                output_path = self.make_subdirectory(
                    output_path, append_name="WithinElement")
                with pathlib.Path(output_path).joinpath("WithinElement_" + name + "_Label-" + label_name + ".csv") as save_path:
                    element_df.to_csv(save_path)

            if within_element_df is not None or np.nan:
                return within_element_df
            else:
                pass

        elif window_size_px is None:
            curv = self.taubin_curv(element.coords, resolution)

            within_element_df = pd.DataFrame(
                {'curv': [curv], 'length': [length_mm]})

            if within_element_df is not None or np.nan:
                return within_element_df
            else:
                pass

    def subset_gen(self, pixel_length, window_size_px, label):
        """Generator function for start and end indices of the window of measurement.

        Parameters
        ----------
        pixel_length : int
            Number of pixels in input curve/line.
        window_size_px : int
            The size of window of measurement.
        label : np.array
            Nested list of coordinates for the input curve/line.

        Returns
        -------
        list
            Nested list of coordinates for the window of measurement in the input curve/line.

        """

        # TODO: Add warning that under 10pixels will yield problems
        # Currently we arent using window size
        subset_start = 0
        if window_size_px >= 10:
            subset_end = int(window_size_px + subset_start)
        else:
            subset_end = int(pixel_length)
        while subset_end <= pixel_length:
            subset = label[subset_start:subset_end]
            yield subset
            subset_start += 1
            subset_end += 1

    def pixel_length_correction(self, element):

        num_total_points = element.area
        skeleton = element.image
        diag_points, num_diag_points = self.find_structure(skeleton, 'diag')
        mid_points, num_mid_points = self.find_structure(skeleton, 'mid')
        num_adj_points = num_total_points - num_diag_points - num_mid_points
        corr_element_pixel_length = num_adj_points + \
            (num_diag_points * np.sqrt(2)) + (num_mid_points * np.sqrt(1.25))

        return corr_element_pixel_length

    def find_structure(self, skeleton, structure: str):
        skel_image = self.check_bin(skeleton).astype(int)

        # creating empty array for hit and miss algorithm
        hit_points = np.zeros(skel_image.shape)
        # defining the structure used in hit-and-miss algorithm
        hit_list = self.define_structure(structure)

        for hit in hit_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            hit_points = np.logical_or(
                hit_points, np.where(curr == target, 1, 0))

        # Ensuring target image is binary
        hit_points_image = np.where(hit_points, 1, 0)

        # use SciPy's ndimage module for locating and determining coordinates of each branch-point
        labels, num_labels = ndimage.label(hit_points_image)

        return labels, num_labels

    def define_structure(self, structure: str):
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
            diag_list = [hit1, hit2]

            return diag_list

        else:
            raise TypeError(
                "Structure input for find_structure() is invalid, choose from 'mid', or 'diag' and input as str")


if __name__ == '__main__':
    curv = Curvature()
    curv.run()  # need to form a "run fucntion which is called with no parameters except self and then lays out parametes for curvature_seq"
