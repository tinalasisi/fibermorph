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
from time import time

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

        # parser.add_argument(
        #     '-s', '--save_image', action='store_true', default=False,
        #     help='Default is False. Will save intermediate curvature/section processing images if --save_image flag is included.')
        
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
            raise ValueError('The input filepath is invalid. Please input a valid directory filepath with hair images.')
        
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
            raise FileExistsError('No TIFF images found in the directory, Please input a valid directory filepath with hair images.')

        return flist

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
            'section options - arguments used specifically for section module')
        
        gr_sect.add_argument(
            '-sr', '--resolution_mu', type=float, metavar='', default=4.25,
            help='Float. Number of pixels per micron for section analysis. Default is 4.25.')

        gr_sect.add_argument(
            '-smin', '--minsize', type=int, metavar='', default=20,
            help='Integer. Minimum diameter in microns for sections. Default is 20.')

        gr_sect.add_argument(
            '-smax', '--maxsize', type=int, metavar='', default=150,
            help='Integer. Maximum diameter in microns for sections. Default is 150.')
        
        return parser
    
    def impreprocess(self, filepath):
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
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        # detectcircles later
        img_array = img[y-500:y+500, x-500:x+500]
        im_center = list(np.divide(img.shape, 2))
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
    
    def run(self):
        '''
        Executes the section analysis
        '''
        try:
            args = self.configure_args().parse_args()
            minpixel = args.minsize * args.resolution_mu
            maxpixel = args.maxsize * args.resolution_mu
            files = self.read_files(args.input_directory)
            sd = []
            count = 1
            for fp in files:
                _, fn = os.path.split(fp)
                img, im_center = self.impreprocess(fp)
                section_data, bin_im = self.segment(img, fn, args.resolution_mu, minpixel, maxpixel, im_center)
                sd.append(section_data)
                with pathlib.Path(args.output_directory).joinpath("img" + str(count) + ".tiff") as img_output_path:
                    skimage.io.imsave(str(img_output_path), bin_im)
                    count += 1
            section_df = pd.concat(sd).dropna()
            section_df.set_index('ID', inplace=True)
            with pathlib.Path(args.output_directory).joinpath("summary_section_data.csv") as df_output_path:
                section_df.to_csv(df_output_path)
            sys.exit(0)
        except KeyboardInterrupt:
            print('terminating...')
            sys.exit(0)

if __name__ == '__main__':
    section = Section()
    section.run()