#!/usr/bin/env python3
from argparse import ArgumentParser as ap
from datetime import datetime
import fnmatch
from functools import wraps
import multiprocessing
import os
import pathlib
import re
import sys
import contextlib
import joblib
import unittest
import logging
from logging.handlers import TimedRotatingFileHandler
from joblib import Parallel, delayed
from timeit import default_timer as timer

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
from skimage import io, filters, segmentation
from tqdm import tqdm

class Fibermorph:

    def __init__(self):
        self.timenow = datetime.now().strftime("%b%d_%H%M_")

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
            '-j', '--jobs', type=int, metavar='', default=os.cpu_count(),
            help='Integer. Number of parallel jobs to run. Default is the number of cpu cores in the system.')

        # parser.add_argument(
        #     '-s', '--save_image', action='store_true', default=False,
        #     help='Default is False. Will save intermediate curvature/section processing images if --save_image flag is included.')
        
        #args = parser.parse_args()
        return parser
    
    def blockPrint(f):
        @wraps(f)
        def wrap(*args, **kw):
            # block all printing to the console
            sys.stdout = open(os.devnull, 'w')
            # call the method in question
            value = f(*args, **kw)
            # enable all printing to the console
            sys.stdout = sys.__stdout__
            # pass the return value of the method back
            return value
        return wrap
    
    blockPrint = staticmethod(blockPrint)

    @contextlib.contextmanager
    def tqdm_joblib(self, tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
    
    def make_directory(self, odir, flogger):
        flogger.info('Making the output directory if needed.')
        directory = pathlib.Path(odir)
        output_path = pathlib.Path(directory).joinpath(str(self.timenow + "fibermorph_section"))
        if not pathlib.Path(output_path).exists():
            flogger.info('No corresponding output directory found: ' + str(output_path))
            pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
            flogger.info('The output directory has been created: ' + str(output_path))
        else:
            flogger.info('The output directory already exists: ' + str(output_path))

        return output_path

    def read_files(self, input_directory, flogger):
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
        flogger.info('Reading image files in the input directory. ')
        if not os.path.isdir(input_directory):
            flogger.error('Invalid input directory; directory does not exist. \n' + str(input_directory))
            raise ValueError('The input filepath is invalid. Please input a valid directory filepath with hair images.')
        
        # *tif & *tiff 
        ftype = ['*.tif', '*.tiff']
        ftype = r'|'.join([fnmatch.translate(x) for x in ftype])

        flist = []
        # Iterate over files in the directory in search for tiff files
        flogger.info('Iterating over the input directory for TIFF images. ')
        for root, _, files in os.walk(input_directory, topdown=False):
            files = [os.path.join(root, f) for f in files]
            files = [f for f in files if re.match(ftype, f)]
            flist += files
        
        if not flist:
            flogger.error('Invalid files; directory is missing TIFF images. \n' + str(input_directory))
            raise FileExistsError('No TIFF images found in the directory. Please input a valid directory filepath with hair images.')
        flogger.info('TIFF images were found in the input directory. \n' + '\n'.join(flist))

        return flist
    
    def get_logger(self, logger_name):
        '''
        A logger for fibermorph subprocesses

        Returns
        -------
        logger
        '''
        FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
        LOG_FILE = os.path.join(os.getcwd(), 'fibermorph' + datetime.now().strftime('%b%d') + '.log')
        FHANDLER = TimedRotatingFileHandler(LOG_FILE, when='midnight')
        FHANDLER.setFormatter(FORMATTER)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(FHANDLER)
        logger.propagate = False
        return logger

class Section(Fibermorph):

    def __init__(self):
        super().__init__()

    def configure_args(self):
        '''
        Parses command-line arguments

        Returns
        -------
        parser
        '''
        parser = super().configure_args()

        gr_sect = parser.add_argument_group(
            'raw2gray options - arguments used specifically for raw2gray module')
        
        gr_sect.add_argument(
            '-sr', '--resolution_mu', type=float, metavar='', default=4.25,
            help='Float. Number of pixels per micron for section analysis. Default is 4.25.')

        gr_sect.add_argument(
            '-smin', '--minsize', type=int, metavar='', default=20,
            help='Integer. Minimum diameter in microns for sections. Default is 20.')

        gr_sect.add_argument(
            '-smax', '--maxsize', type=int, metavar='', default=150,
            help='Integer. Maximum diameter in microns for sections. Default is 150.')
        
        gr_sect.add_argument(
            '-simg', '--save_img', type=bool, metavar='', default=True,
            help='Boolean. Defaulted to False.')
        
        return parser
    
    def impreprocess(self, filepath, fiblog):
        '''
        Preprocesses an input image file

        Parameter
        ---------
        filepath : Path
            Path for a section image file.

        Returns
        -------
        img_array: ndarray
        im_center: ndarray
        '''
        # Read image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #fiblog = self.get_logger('fiblog')
        #fiblog.debug(os.path.split(filepath) + ' has been processed on grayscale with OpenCV2')
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        # detectcircles later
        img_array = img[y-500:y+500, x-500:x+500]
        im_center = list(np.divide(img.shape, 2))
        #fiblog.debug(os.path.split(filepath) + ' has been cut into 1000x1000 image at the center')
        # unique, counts = np.unique(img, return_counts=True)
        # print(unique)
        # print(counts)
        return img_array, im_center
    
    def segment(self, img, im_name, resolution, minpixel, maxpixel, im_center):
        '''
        Segments an input image

        Parameter
        ---------
        img: ndarray
            2D array of a grayscale image.
        im_name: str
            filename str of the image
        resolution: float
            
        Returns
        -------
        img_df: Pandas.Dataframe
           data 
        '''
        thresh = skimage.filters.threshold_minimum(img)
        #thresh = skimage.filters.threshold_otsu(img)
        bin_ls_set = img < thresh
        seg_im = skimage.segmentation.morphological_chan_vese(np.asarray(img), 40, init_level_set=bin_ls_set, smoothing=4)
        seg_im_inv = np.asarray(seg_im != 0)
        crop_label_im, num_elem = skimage.measure.label(seg_im_inv, connectivity=2, return_num=True)
        crop_props = skimage.measure.regionprops(label_image=crop_label_im, intensity_image=np.asarray(img))
        section_data, bin_im, bbox = self.props(crop_props, im_name, resolution, minpixel, maxpixel, im_center)
        return section_data, bin_im
    
    def props(self, props, im_name, resolution, minpixel, maxpixel, im_center):
        props_df = [
            [region.label, region.centroid, scipy.spatial.distance.euclidean(im_center, region.centroid), 
            region.filled_area, region.minor_axis_length, region.major_axis_length, region.eccentricity, region.filled_image, region.bbox]
            for region in props if region.minor_axis_length >= minpixel and region.major_axis_length <= maxpixel]
        props_df = pd.DataFrame(props_df,columns=['label', 'centroid', 'distance', 'area', 'min', 'max', 'eccentricity', 'image', 'bbox'])
        section_id = props_df['distance'].astype(float).idxmin()
        section = props_df.iloc[section_id]
        area_mu = section['area'] / np.square(resolution)
        min_diam = section['min'] / resolution
        max_diam = section['max'] / resolution
        eccentricity = section['eccentricity']
        section_data = pd.DataFrame(
            {'ID': [im_name], 'area': [area_mu], 'eccentricity': [eccentricity], 'min': [min_diam],
            'max': [max_diam]})
        bin_im = section['image']
        bbox = section['bbox']
        return section_data, bin_im, bbox

    
    def section_seq(self, file, output_directory, resolution_mu, minsize, maxsize, save_img, fiblog):
        _, fn = os.path.split(file)
        fiblog.handlers.clear()
        fiblog = self.get_logger('fiblog')
        fiblog.info('Section analysis for {} has been started'.format(fn))

        minpixel = minsize * resolution_mu
        maxpixel = maxsize * resolution_mu
        img, im_center = self.impreprocess(file, fiblog)
        section_data, bin_im = self.segment(img, fn, resolution_mu, minpixel, maxpixel, im_center)

        if save_img:
            imgname = fn.split('.')[0] + '_section_cut.jpg'
            with pathlib.Path(output_directory).joinpath(imgname) as img_output_path:
                simg = skimage.img_as_ubyte(bin_im)
                skimage.io.imsave(str(img_output_path), simg)
                fiblog.info('{} has been saved to the output directory. '.format(imgname))
        
        fiblog.info('Section analysis for {} has been finished'.format(fn))

        return section_data

    def run(self):
        '''
        Executes the section analysis
        '''
        args = self.configure_args().parse_args()
        fiblog = self.get_logger('fiblog')
        try:
            fiblog.info('Section analysis initiated with arguments parsed below.')
            fiblog.info(args)

            start = timer()

            files = self.read_files(args.input_directory, fiblog)

            od = self.make_directory(args.output_directory, fiblog)

            with self.tqdm_joblib(tqdm(desc="section", total=len(files), unit="files", miniters=1)) as progress_bar:
                progress_bar.monitor_interval = 2
                num_process = len(files) if args.jobs > len(files) else args.jobs
                section_df = (Parallel(n_jobs=num_process, verbose=0)(
                    delayed(self.section_seq)(
                        f, od, args.resolution_mu, args.minsize, args.maxsize, args.save_img, fiblog) for f in files))
            
            fiblog.info('Processing data for csv.')
            section_df = pd.concat(section_df).dropna()
            section_df.set_index('ID', inplace=True)
            with pathlib.Path(od).joinpath("summary_section_data.csv") as df_output_path:
                section_df.to_csv(df_output_path)
                fiblog.info('The summary has been written onto a csv file: ' + str(df_output_path))
            
            end = timer()
            m, s = divmod(int(end - start), 60)
            h, m = divmod(m, 60)
            tqdm.write("\n\nComplete analysis took: {}\n\n".format("%dh: %02dm: %02ds" % (h, m, s)))
            fiblog.info("\n\nComplete analysis took: {}\n\n".format("%dh: %02dm: %02ds" % (h, m, s)))
            sys.exit(0)
            
        except KeyboardInterrupt:
            fiblog.error('KeyboardInterrupt detected.')
        
        finally:
            fiblog.info('Section analysis terminated. \n')
            sys.exit(0)

# class FibermorphTest(unittest.Testcase):
#     def section_test(self):
#         urllist = [
#             "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section.tiff",
#             "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section2.tiff"]
    
#     def curvature_test(self):
#         urllist = [
#             "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/004_demo_curv.tiff",
#             "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/027_demo_nocurv.tiff"]

if __name__ == '__main__':
    section = Section()
    section.run()

# LOG FUNCTION
# IMAGES THAT WORKED/NOT
# look curvature and see improvements to make
# known param est param /// simulation