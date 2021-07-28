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
    
    def make_directory(self, odir, flogger):
        flogger.info('Making the output directory if needed.')
        directory = pathlib.Path(odir)
        output_path = pathlib.Path(directory).joinpath(str(self.timenow + "fibermorph_analysis"))
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

class FibermorphTest(unittest.Testcase):
    def section_test(self):
        urllist = [
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section.tiff",
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section2.tiff"]
        pass
    
    def curvature_test(self):
        urllist = [
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/004_demo_curv.tiff",
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/027_demo_nocurv.tiff"]
        pass