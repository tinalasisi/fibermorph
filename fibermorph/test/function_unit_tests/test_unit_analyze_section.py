import skimage
import numpy as np
import pandas as pd
import scipy
from skimage import measure
from skimage import filters
from skimage import segmentation
from fibermorph import fibermorph
from scipy import spatial

import matplotlib.pyplot as plt

def analyze_section(img, minsize=20, maxsize=150, resolution=1.0):
    # first binarize the image
    thresh = skimage.filters.threshold_otsu(img)
    
    binary_im = skimage.segmentation.clear_border(img < thresh)
    
    # label the image
    label_im, num_elem = skimage.measure.label(binary_im, connectivity=2, return_num=True)
    
    # find center of image
    im_center = list(np.divide(label_im.shape, 2))  # returns array of two floats
    
    minpixel = np.pi * (((minsize / 2) * resolution) ** 2)
    maxpixel = np.pi * (((maxsize / 2) * resolution) ** 2)
    
    props = skimage.measure.regionprops(label_image=label_im, intensity_image=img)

    props_df = [[region.label, region.centroid, scipy.spatial.distance.euclidean(im_center, region.centroid)] for region in props if region.area >= minpixel and region.area <= maxpixel]

    props_df = pd.DataFrame(props_df, columns=['label', 'centroid', 'distance'])
    
    section_id = props_df['distance'].idxmin()
    
    section = props[section_id]
    
    section_data = [section.filled_area, section.minor_axis_length, section.major_axis_length, section.eccentricity]
    
    section_data = pd.DataFrame([x / resolution for x in section_data]).T
    section_data.columns = ['area', 'min', 'max', 'eccentricity']
    
    cropped_im = props[section_id].intensity_image
    cropped_bin = props[section_id].filled_image
    
    return props_df, section_id, cropped_im, cropped_bin, section_data

img, im_name = fibermorph.imread("/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section/14918_demo_section.tiff")

props_df, section_id, cropped_im, cropped_bin, section_data = analyze_section(img, resolution=1.06)

plt.imshow(cropped_bin, cmap='gray')
plt.show()
