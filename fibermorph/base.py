import contextlib
from datetime import datetime
import fnmatch
from functools import wraps
import joblib
from joblib import Parallel, delayed
import logging
from logging.handlers import TimedRotatingFileHandler
import multiprocessing
import os
import pathlib
import re
import skimage
import sys
from tqdm import tqdm
import unittest

class Fibermorph:

    def __init__(self):
        self.timenow = datetime.now().strftime("%b%d_%H%M_")
    
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
    
    def make_directory(self, odir, foldername, flogger):
        flogger.info('Making the output directory if needed.')
        directory = pathlib.Path(odir)
        output_path = pathlib.Path(directory).joinpath(foldername)
        if not output_path.exists():
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
        ftype = ['*.tif', '*.tiff','*.TIF', '*.TIFF'] # TODO: allow for jpeg and jpg with warning
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
    
    def save_image(self, output_directory, folder_name, filename, img, flogger):
        folder_path = self.make_directory(output_directory, folder_name, flogger)
        with pathlib.Path(folder_path).joinpath(filename) as img_output_path:
                simg = skimage.img_as_ubyte(img)
                skimage.io.imsave(str(img_output_path), simg)
                flogger.info('{} has been saved to its corresponding directory. '.format(filename))
        return
    
    # def within_element_func(self, output_path, name, element, taubin_df):
    #     # for within hair distribution
    #     label_name = str(element.label)
    #     element_df = pd.DataFrame(taubin_df)
    #     element_df.columns = ['curv']
    #     element_df['label'] = label_name
        
    #     output_path = make_subdirectory(output_path, append_name="WithinElement")
    #     with pathlib.Path(output_path).joinpath("WithinElement_" + name + "_Label-" + label_name + ".csv") as save_path:
    #         element_df.to_csv(save_path)
        
    #     return True
    
    def get_logger(self, logger_name):
        '''
        A logger for fibermorph subprocesses

        Returns
        -------
        logger
        '''
        FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
        LOG_FILE = os.path.join(os.getcwd(), 'fibermorphlog_' + datetime.now().strftime('%b%d') + '.log')
        FHANDLER = TimedRotatingFileHandler(LOG_FILE, when='midnight')
        FHANDLER.setFormatter(FORMATTER)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(FHANDLER)
        logger.propagate = False
        return logger