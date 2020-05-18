import pathlib
import scipy
import skimage

from skimage import filters
from skimage import morphology
from skimage import segmentation
from skimage import io
from scipy import ndimage
from PIL import Image
from fibermorph.fibermorph import binarize_curv


test_im = skimage.io.imread("/Users/tinalasisi/Desktop/001.tiff", as_gray=True)
test_dir = "/Users/tinalasisi/Desktop"

test_bin = binarize_curv(test_im, "test_im", test_dir, save_img=True)
