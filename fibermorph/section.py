from base import Fibermorph

import multiprocessing
import os
import pathlib
import sys
import joblib
import logging
from logging.handlers import TimedRotatingFileHandler
from joblib import Parallel, delayed
from timeit import default_timer as timer

import cv2
import traceback
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage
import skimage
from skimage import io, filters, segmentation
from tqdm import tqdm

class Section(Fibermorph):

    def __init__(self):
        super().__init__()
    
    def run(self, args):
        '''
        Executes the section analysis
        '''
        fiblog = self.get_logger('fiblog')
        try:
            fiblog.info('Section analysis initiated with arguments parsed below.')
            fiblog.info(args)

            start = timer()

            files = self.read_files(args.input_directory, fiblog)
            foldername = str(self.timenow + "fibermorph_section_analysis")
            od = self.make_directory(args.output_directory, foldername, fiblog)

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
            
        except KeyboardInterrupt:
            fiblog.error('KeyboardInterrupt detected.')
        except RuntimeError as e:
            traceback.print_exc()
            fiblog.error(traceback.print_exc())
        finally:
            fiblog.info('Section analysis terminated. \n')
            sys.exit(0)

    def section_seq(self, file, output_directory, resolution_mu, minsize, maxsize, save_img, fiblog):
        _, fn = os.path.split(file)
        fiblog.handlers.clear()
        fiblog = self.get_logger('fiblog')
        fiblog.info('Section analysis for {} has been started'.format(fn))

        minpixel = minsize * resolution_mu
        maxpixel = maxsize * resolution_mu
        img, im_center = self.impreprocess(file)
        section_data, bin_im = self.segment(img, fn, resolution_mu, minpixel, maxpixel, im_center)

        if save_img:
            imgname = fn.split('.')[0] + '_section_cut.jpg'
            self.save_image(output_directory, 'results', imgname, bin_im, fiblog)
        
        fiblog.info('Section analysis for {} has been finished'.format(fn))

        return section_data
    
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
        #thresh = skimage.filters.threshold_minimum(img)
        thresh = skimage.filters.threshold_otsu(img)
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