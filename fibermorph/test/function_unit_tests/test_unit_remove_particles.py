import pathlib
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology
from skimage import io
from fibermorph.test.function_unit_tests.test_unit_check_bin import check_bin


def remove_particles(input_file, output_path, name, minpixel=5, prune=False, save_img=False):
    
    img_bool = np.asarray(input_file, dtype=np.bool)
    img = check_bin(img_bool)

    if not prune:
        minimum = minpixel * 10  # assuming the hairs are no more than 10 pixels thick
        # warnings.filterwarnings("ignore")  # suppress Boolean image UserWarning
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)
    else:
        # clean = img_bool
        minimum = minpixel
        clean = skimage.morphology.remove_small_objects(img, connectivity=2, min_size=minimum)

        print("\n Done cleaning {}".format(name))

    if save_img:
        img_inv = skimage.util.invert(clean)
        with pathlib.Path(output_path).joinpath(name + ".tiff") as savename:
            plt.imsave(savename, img_inv, cmap='gray')
            # im = Image.fromarray(img_inv)
            # im.save(output_path)
        return clean
    else:
        return clean


test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/bin_curv.tiff", as_gray=True)
test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = remove_particles(test_im, test_dir, "clean_im", prune=False, save_img=True)
