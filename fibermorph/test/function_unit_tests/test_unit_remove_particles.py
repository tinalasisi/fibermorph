import pathlib
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology
from skimage import io

def remove_particles(input_file, output_path, name, minpixel=5, prune=False, save_img=False):
    
    img_bool = np.asarray(input_file, dtype=np.bool)
    
    # Gets the unique values in the image matrix. Since it is binary, there should only be 2.
    unique, counts = np.unique(img_bool, return_counts=True)
    print(unique)
    print("Found this many counts:")
    print(len(counts))
    print(counts)

    # If the length of unique is not 2 then print that the image isn't a binary.
    if len(unique) != 2:
        print("Image is not binarized!")
        hair_pixels = len(counts)
        print("There is/are {} value(s) present, but there should be 2!\n".format(hair_pixels))
    # If it is binarized, print out that is is and then get the amount of hair pixels to background pixels.
    if counts[0] < counts[1]:
        print("{} is not reversed".format(str(input_file)))
        img = skimage.util.invert(img_bool)
        print("Now {} is reversed =)".format(str(input_file)))

    else:
        print("{} is already reversed".format(str(input_file)))
        img = img_bool

        print(type(img))

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
