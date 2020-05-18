import pathlib
import skimage

from skimage import morphology
from skimage import io
from PIL import Image

from fibermorph.fibermorph import skeletonize
from fibermorph.test.function_unit_tests.test_unit_check_bin import check_bin



test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/clean_im.tiff", as_gray=True)

test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = skeletonize(test_im, "skeleton_curv", test_dir, save_img=True)

