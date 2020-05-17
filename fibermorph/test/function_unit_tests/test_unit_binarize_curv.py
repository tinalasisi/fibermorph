import pathlib
import scipy
import skimage

from skimage import filters
from skimage import morphology
from skimage import segmentation
from skimage import io
from scipy import ndimage
from PIL import Image

def binarize_curv(filter_img, im_name, binary_dir, save_img=False):
    # create structuring elements of 5px radius disk and 3px
    selem = skimage.morphology.disk(5)
    selem2 = skimage.morphology.disk(3)
    
    # run a simple median filter to smooth the image
    med_im = skimage.filters.rank.median(skimage.util.img_as_ubyte(filter_img), selem)
    
    # find the Otsu binary threshold
    thresh = skimage.filters.threshold_otsu(med_im)
    
    # create a binary using this threshold
    thresh_im = med_im <= thresh
    
    # clear the border of the image (buffer is the px width to be considered as border)
    cleared_im = skimage.segmentation.clear_border(thresh_im, buffer_size=10)
    
    # dilate the hair fibers
    binary_im = scipy.ndimage.binary_dilation(cleared_im, structure=selem2, iterations=2)
    
    if save_img:
        # invert image
        save_im = skimage.util.invert(binary_im)
        
        # save image
        with pathlib.Path(binary_dir).joinpath(im_name + ".tiff") as save_name:
            im = Image.fromarray(save_im)
            im.save(save_name)
        return binary_im
    else:
        return binary_im


test_im = skimage.io.imread("/Users/tinalasisi/Desktop/001.tiff", as_gray=True)
test_dir = "/Users/tinalasisi/Desktop"

test_bin = binarize_curv(test_im, "test_im", test_dir, save_img=True)
