import pathlib
import matplotlib.pyplot as plt
import cv2
import skimage

from skimage import filters


def filter(input_file, output_path):

    # create pathlib object for input Image
    input_path = pathlib.Path(input_file)

    # extract image name
    im_name = input_path.stem

    # read in Image
    gray_img = cv2.imread(input_file, 0)
    type(gray_img)
    print("Image size is:", gray_img.shape)

    # use frangi ridge filter to find hairs, the output will be inverted
    filter_img = skimage.filters.frangi(gray_img)
    type(filter_img)
    print("Image size is:", filter_img.shape)

    # inverting and saving the filtered image
    img_inv = skimage.util.invert(filter_img)
    with pathlib.Path(output_path).joinpath(im_name + ".tiff") as save_path:
        plt.imsave(save_path, img_inv, cmap="gray")

    return filter_img, im_name


input_file = "testdata/curv_im.tiff"

output_path = "/Users/tinalasisi/Desktop"

filter_img, im_name = filter(input_file, output_path)

