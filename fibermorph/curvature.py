from tqdm import tqdm
from skimage import io, filters, segmentation
import skimage
from scipy import ndimage
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import traceback
import cv2
from timeit import default_timer as timer
from joblib import Parallel, delayed
from logging.handlers import TimedRotatingFileHandler
import logging
import unittest
import joblib
import contextlib
import sys
import re
import pathlib
import os
import multiprocessing
from datetime import datetime
from base import Fibermorph
import warnings
warnings.filterwarnings("ignore")


class Curvature(Fibermorph):

    def __init__(self):
        super().__init__()

    def run(self, args):
        '''
        Executes the curv analysis
        '''
        fiblog = self.get_logger('fiblog')
        try:
            fiblog.info(
                'Curvature analysis initiated with arguments parsed below.')
            fiblog.info(args)

            start = timer()
            files = self.read_files(args.input_directory, fiblog)
            foldername = str(self.timenow + "fibermorph_curvature_analysis")
            od = self.make_directory(args.output_directory, foldername, fiblog)

            with self.tqdm_joblib(tqdm(desc="curvature", total=len(files), unit="files", miniters=1)) as progress_bar:
                progress_bar.monitor_interval = 2
                num_process = len(files) if args.jobs > len(
                    files) else args.jobs
                curvature_df = (Parallel(n_jobs=num_process, verbose=0)(
                    delayed(self.curvature_seq)(
                        f, od, args.resolution_mm, args.window_size, args.window_unit, args.save_img, args.within_element, fiblog) for f in files))

            fiblog.info('Processing data for csv.')
            curvature_df = pd.concat(curvature_df).dropna()
            curvature_df.set_index('ID', inplace=True)
            with pathlib.Path(od).joinpath("summary_curvature_data.csv") as df_output_path:
                curvature_df.to_csv(df_output_path)
                fiblog.info(
                    'The summary has been written onto a csv file: ' + str(df_output_path))

            end = timer()
            m, s = divmod(int(end - start), 60)
            h, m = divmod(m, 60)
            tqdm.write("\n\nComplete analysis took: {}\n\n".format(
                "%dh: %02dm: %02ds" % (h, m, s)))
            fiblog.info("\n\nComplete analysis took: {}\n\n".format(
                "%dh: %02dm: %02ds" % (h, m, s)))

        except KeyboardInterrupt:
            print('terminating...')
        except RuntimeError as e:
            traceback.print_exc()
            fiblog.error(traceback.print_exc())
        finally:
            fiblog.info('Curvature analysis terminated. \n')
            sys.exit(0)

    def curvature_seq(self, input_file, output_directory, resolution, window_size, window_unit, save_img, within_element, fiblog):
        _, fn = os.path.split(input_file)
        fiblog.handlers.clear()
        fiblog = self.get_logger('fiblog')
        fiblog.info('Curvature analysis for {} has been started'.format(fn))

        filter_img, im_name = self.filter_curv(
            input_file, output_directory, save_img, fiblog)

        binary_img = self.binarize_curv(
            filter_img, im_name, output_directory, save_img, fiblog)

        clean_im = self.remove_particles(binary_img, output_directory, im_name, int(
            resolution/2), False, save_img, fiblog)

        skeleton_im = self.skeletonize(
            clean_im, im_name, output_directory, save_img, fiblog)

        pruned_im = self.prune(skeleton_im, im_name,
                               output_directory, save_img, fiblog)

        im_df = self.analyze_all_curv(pruned_im, im_name, output_directory,
                                      resolution, window_size, window_unit, within_element, fiblog)

        return im_df

    def filter_curv(self, file, output_directory, save_img, fiblog):
        _, fn = os.path.split(file)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        filter_img = skimage.filters.frangi(img)

        if save_img:
            img_inv = skimage.util.invert(filter_img)
            imgname = fn.split('.')[0] + '_filtered.jpg'
            self.save_image(output_directory, 'filtered',
                            imgname, img_inv, fiblog)

        return filter_img, fn

    def binarize_curv(self, filter_img, im_name, output_directory, save_img, fiblog):
        selem = skimage.morphology.disk(5)
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
            img_inv = skimage.util.invert(binary_im)
            imgname = im_name + '_binarized.jpg'
            self.save_image(output_directory, 'binarized',
                            imgname, img_inv, fiblog)

        return binary_im

    def remove_particles(self, img, output_directory, name, minpixel, prune, save_img, fiblog):
        img_bool = np.asarray(img, dtype=np.bool)
        img = self.check_bin(img_bool)

        clean = skimage.morphology.remove_small_objects(
            img, connectivity=2, min_size=minpixel)

        if save_img:
            img_inv = skimage.util.invert(clean)
            if prune:
                imgname = name + '_particles_removed_pruned.jpg'
                self.save_image(
                    output_directory, 'particles_removed_pruned', imgname, img_inv, fiblog)
            else:
                imgname = name + '_particles_removed_clean.jpg'
                self.save_image(
                    output_directory, 'particles_removed_clean', imgname, img_inv, fiblog)

        return clean

    def check_bin(self, img):
        img_bool = np.asarray(img, dtype=np.bool)
        _, counts = np.unique(img_bool, return_counts=True)
        return skimage.util.invert(img_bool) if counts[0] < counts[1] else img_bool

    def skeletonize(self, clean_img, name, output_directory, save_img, fiblog):
        clean_img = self.check_bin(clean_img)
        skeleton = skimage.morphology.thin(clean_img)

        if save_img:
            img_inv = skimage.util.invert(skeleton)
            imgname = name + '_skeletonized.jpg'
            self.save_image(output_directory, 'skeletonized',
                            imgname, img_inv, fiblog)

        return skeleton

    def prune(self, skeleton, name, pruned_dir, save_img, fiblog):
        '''
        Prunes branches from skeletonized image.
        Adapted from: "http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm"
        '''
        # identify 3-way and 4-way branch-points
        hit_list = self.generate_hit('3way')
        hit_list += self.generate_hit('4way')

        skel_image = self.check_bin(skeleton)
        branch_points = np.zeros(skel_image.shape)

        for hit in hit_list:
            target = hit.sum()
            curr = ndimage.convolve(skel_image, hit, mode="constant")
            branch_points = np.logical_or(
                branch_points, np.where(curr == target, 1, 0))

        branch_points_image = np.where(branch_points, 1, 0)

        labels, num_labels = ndimage.label(branch_points_image)

        branch_points = ndimage.center_of_mass(
            skel_image, labels=labels, index=range(1, num_labels + 1))
        branch_points = np.array([value for value in branch_points if not np.isnan(
            value[0]) or not np.isnan(value[1])], dtype=int)

        hit = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.uint8)

        dilated_branches = ndimage.convolve(
            branch_points_image, hit, mode='constant')
        dilated_branches_image = np.where(dilated_branches, 1, 0)
        pruned_image = np.subtract(skel_image, dilated_branches_image)

        pruned_image = self.remove_particles(
            pruned_image, pruned_dir, name, 5, True, save_img, fiblog)

        return pruned_image

    def generate_hit(self, opt):
        '''
        Generates 'hit' arrays for curvature analysis
        '''
        out = []

        # diagonal hits
        if opt == 'mid':
            base = np.array([
                [0, 0, 0],
                [0, 1, 1],
                [1, 0, 0]], dtype=np.uint8)
            base2 = np.flipud(base)
            for i in range(4):
                out.append(np.rot90(base, i))
                out.append(np.rot90(base2, i))
            return out
        if opt == 'diag':
            base = np.diag([1, 1, 1])
            base2 = np.fliplr(base)
            out.append(base)
            out.append(base2)
            return out
        if opt == 'adj':
            base = np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]], dtype=np.uint8)
            base2 = np.rot90(base)
            out.append(base)
            out.append(base2)
            return out

        # 3-way branch hits
        if opt == '3way':
            base = np.array([
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 1]], dtype=np.uint8)
            base2 = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1]], dtype=np.uint8)
            base3 = np.array([
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 0]], dtype=np.uint8)
            for i in range(4):
                out.append(np.rot90(base, i))
                out.append(np.rot90(base2, i))
                out.append(np.rot90(base3, i))
            return out

        # 4-way branch hits
        if opt == '4way':
            base = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.uint8)
            base2 = np.array([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]], dtype=np.uint8)
            out.append(base)
            out.append(base2)
            return out

    def analyze_all_curv(self, img, name, output_path, resolution, window_size, window_unit, within_element, fiblog):
        img = self.check_bin(img)
        label_image, _ = skimage.measure.label(
            img.astype(int), connectivity=2, return_num=True)
        props = skimage.measure.regionprops(label_image)
        fiblog.info('a')

        im_sumdf = self.window_iter(
            props, name, window_size, window_unit, resolution, output_path, within_element, fiblog)
        im_sumdf = pd.concat(im_sumdf)

        return im_sumdf

    def window_iter(self, props, name, window_size, window_unit, resolution, output_path, within_element, fiblog):
        tempdf = []
        fiblog.info('c')
        if window_size:
            if window_unit != "px":
                window_size_px = int(window_size * resolution)
            else:
                window_size_px = int(window_size)
                window_size = int(window_size)

            name = str(name + "_WindowSize-" +
                       str(window_size) + str(window_unit))
            fiblog.info('a')
            tempdf = [self.analyze_each_curv(hair, window_size_px, resolution, output_path,
                                             name, within_element, fiblog) for hair in props if hair.area > window_size]

            within_im_curvdf = pd.DataFrame(
                tempdf, columns=['curv_mean', 'curv_median', 'length'])

            within_im_curvdf2 = pd.DataFrame(within_im_curvdf, columns=[
                                             'curv_mean', 'curv_median', 'length']).dropna()

            output_path = self.make_directory(output_path, 'analysis', fiblog)
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

            # if test:
            #     return within_im_curvdf2
            # else:
            return im_sumdf

        else:
            fiblog.info('b')
            window_size_px = None
            within_element = None
            minsize = 0.5 * resolution
            tempdf = [self.analyze_each_curv(hair, window_size_px, resolution, output_path, name, within_element, fiblog) for hair in
                      props if hair.area > minsize]

            within_im_curvdf = pd.concat(tempdf)

            within_im_curvdf2 = within_im_curvdf.dropna()

            output_path = self.make_directory(output_path, 'analysis', fiblog)
            with pathlib.Path(output_path).joinpath("ImageSum_" + name + ".csv") as save_path:
                within_im_curvdf2.to_csv(save_path)

            im_mean = within_im_curvdf2['curv'].mean()
            im_median = within_im_curvdf2['curv'].median()
            length_mean = within_im_curvdf2['length'].mean()
            length_median = within_im_curvdf2['length'].median()
            hair_count = len(within_im_curvdf2.index)

            im_sumdf = pd.DataFrame({'ID': name, 'curv_mean': [im_mean], 'curv_median': [im_median], 'length_mean': [
                                    length_mean], 'length_median': [length_median], 'hair_count': [hair_count]})

            # if test:
            #     return within_im_curvdf2
            # else:
            return im_sumdf

    def analyze_each_curv(self, element, window_size_px, resolution, output_path, name, within_element, fiblog):
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

            taubin_df = pd.Series(curv).astype('float')

            taubin_df2 = taubin_df[taubin_df.between(
                taubin_df.quantile(.01), taubin_df.quantile(.99))]  # without outliers

            curv_mean = taubin_df2.mean()

            curv_median = taubin_df2.median()

            within_element_df = [curv_mean, curv_median, length_mm]

            if within_element:
                label_name = str(element.label)
                element_df = pd.DataFrame(taubin_df)
                element_df.columns = ['curv']
                element_df['label'] = label_name

                output_path = self.make_directory(
                    output_path, 'within_element', fiblog)
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
        _, num_diag_points = self.find_structure(skeleton, 'diag')
        _, num_mid_points = self.find_structure(skeleton, 'mid')
        num_adj_points = num_total_points - num_diag_points - num_mid_points
        corr_element_pixel_length = num_adj_points + \
            (num_diag_points * np.sqrt(2)) + (num_mid_points * np.sqrt(1.25))
        return corr_element_pixel_length

    def find_structure(self, skeleton, structure):
        skel_image = self.check_bin(skeleton).astype(int)
        hit_points = np.zeros(skel_image.shape)
        hit_list = self.generate_hit(structure)

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
