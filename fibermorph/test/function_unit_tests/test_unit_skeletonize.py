import pathlib
import skimage

from skimage import morphology
from skimage import io
from PIL import Image

def skeletonize(clean_img, name, output_path, save_img=False):
    
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

test_bin = skeletonize(test_im, "skeleton_curv", test_dir, save_img=False)

