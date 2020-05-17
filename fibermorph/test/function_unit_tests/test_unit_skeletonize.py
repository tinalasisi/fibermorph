import pathlib
import skimage

from skimage import morphology
from skimage import io
from PIL import Image
from fibermorph.test.function_unit_tests.test_unit_check_bin import check_bin

def skeletonize(clean_img, name, output_path, save_img=True):
    
    # check if image is binary and properly inverted
    clean_img = check_bin(clean_img)
    
    # skeletonize the hair
    skeleton = skimage.morphology.thin(clean_img)

    if save_img:
        img_inv = skimage.util.invert(skeleton)
        with pathlib.Path(output_path).joinpath(name + ".tiff") as output_path:
            im = Image.fromarray(img_inv)
            im.save(output_path)
        return skeleton, name

    else:
        print("\n Done skeletonizing {}".format(name))

        return skeleton, name


test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/clean_im.tiff", as_gray=True)

test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = skeletonize(test_im, "skeleton_curv", test_dir, save_img=True)

