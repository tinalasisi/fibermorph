import pathlib
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology
from skimage import io

from fibermorph.fibermorph import remove_particles



test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/bin_curv.tiff", as_gray=True)
test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = remove_particles(test_im, test_dir, "clean_im", prune=False, save_img=True)
