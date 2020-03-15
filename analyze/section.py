"""
Python 3.6+ script for calculating cross-sectional hair fiber properties from grayscale tiff file.

Authors: Tina Lasisi tina.lasisi@gmail.com, Nicholas Stephens nbs49@psu.edu & Timothy Webster twebster17@gmail.com
March 14, 2020

"""

# %% Import libraries
I_IMPORT_LIBRARIES = 0

import os  # Allows python to work with the operating system.
import pathlib  # Makes defining pathways simple between mac, windows, linux.
import sys
from timeit import default_timer as timer  # Timer to report how long everything is taking.
import cv2
import numpy as np  # Main math backbone of python, lots of MatLab like functions.
import pandas as pd
import skimage.morphology
from PIL import Image  # Allows conversion from RGB to grayscale and ImageChops for cropping
from joblib import Parallel, delayed
from scipy.spatial import distance as dist
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology.selem import disk
from skimage.segmentation import clear_border
from skimage.util import invert
from skimage.filters import threshold_minimum

# %% Import functions
I_IMPORT_FUNCTIONS = 0


def make_subdirectory(directory, append_name=""):
    """
    Function to build a directory based upon the arguments passed in append. Takes a
    directory and uses pathlib to join that to the appended string passed.
​
    :param directory:       The directory within which this function will make a subdirectory.
    :param append_name:     A string to be appended to the pathlib object.
    :return:                output_path:    Returns a new directory for output.

​
    """
    # Define the path of the directory within which this function will make a subdirectory.
    directory = pathlib.Path(directory)
    # The name of the subdirectory.
    append_name = str(append_name)
    # Define the output path by the initial directory and join (i.e. "+") the appropriate text.
    output_path = pathlib.Path(directory).joinpath(str(append_name))
    
    # Use pathlib to see if the output path exists, if it is there it returns True
    if pathlib.Path(output_path).exists() == False:
        
        # Prints a status method to the console using the format option, which fills in the {} with whatever
        # is in the ().
        print("This output path doesn't exist:\n            {} \n Creating...".format(output_path))
        
        # Use pathlib to create the folder.
        pathlib.Path.mkdir(output_path)
        
        # Prints a status to let you know that the folder has been created
        print("Output path has been created")
    
    # Since it's a boolean return, and True is the only other option we will simply print the output.
    else:
        # This will print exactly what you tell it, including the space. The backslash n means new line.
        print("Output path already exists:\n               {}".format(output_path))
    return output_path


def crop_image(input_file, cropped_dir, cropped_binary_dir, pad, minpixel, resolution, crop, save_image):
    """
    :param crop:
    :param save_image:          Default is to not save cropped grayscale in order to speed up computation of results
    :param input_file:          The image file to be read in.
    :param output_path:         The output path where the subfolders resulting from this function are to be made
    :param pad:                 The number of pixels you want to pad the segmented image of the hair
    :param minpixel:            The minimum number of pixels representing a hair
    :return:                    returns cropped grayscale image and directory
    """

    filename = str(os.path.basename(input_file))
    
    if not crop:
    
        gray_img = cv2.imread(str(input_file), 0)  # read in image as numpy array, this will be in color
        type(gray_img)  # using the above function it returns <class 'numpy.ndarray'>
        print("Image size is:", gray_img.shape)  # returns (3904, 5200)

        seg_input_file = np.array(gray_img)
        seg_cropped_binary_dir = cropped_binary_dir
        seg_pad = pad
        seg_resolution = resolution

        tempdf_section = segment_section2(filename, seg_input_file, seg_cropped_binary_dir, seg_pad, seg_resolution)

        print("\n Created temporary dataframe for {}: \n {}".format(filename, tempdf_section))

        return tempdf_section
    else:
        
        try:
            cropped_dir = cropped_dir
            pad = pad
    
            gray_img = cv2.imread(str(input_file), 0)  # read in image as numpy array, this will be in color
            type(gray_img)  # using the above function it returns <class 'numpy.ndarray'>
            print("Image size is:", gray_img.shape)  # returns (3904, 5200)
    
            smooth = skimage.filters.rank.median(gray_img, selem=disk(int(resolution * 8)))
    
            thresh = threshold_otsu(smooth)
    
            # creates binary image by applying the above threshold and replacing all
            init_ls = (np.where(gray_img > thresh, 0, 1)).astype('uint8')
    
            chan = skimage.segmentation.morphological_chan_vese(smooth, 10, init_level_set=init_ls, smoothing=1, lambda1=1, lambda2=0.6)
    
            bw_uint8 = (np.array(chan)).astype('uint8')  # turning image into unsigned integer (0-255, grayscale)
    
            cleared = clear_border(bw_uint8)
            # remove artifacts connected to image border, needs to be inverted for this to work!
    
            radius = minpixel/2
    
            min_area = np.pi*(radius**2)
    
            # # remove small objects before labeling
            # skimage.morphology.remove_small_objects(cleared, min_area, in_place=True)
    
            # label image regions
            label_image, num_elements = label(cleared, connectivity=2, return_num=True)
    
            image_center = list(np.divide(label_image.shape, 2))  # returns array of two floats
    
            print("generating bounding box for image {}'\n\n".format(filename))
    
            minpixel = minpixel
    
            bbox_pad = None
    
            try:
                center_hair, bbox = find_hair(label_image, image_center, minpixel)
    
                minr = bbox[0] - pad
                minc = bbox[1] - pad
                maxr = bbox[2] + pad
                maxc = bbox[3] + pad
    
                bbox_pad = [minc, minr, maxc, maxr]
                print("\nFound bbox for {} \n It is: {}".format(filename, str(bbox_pad)))
    
            except:
                minr = int(image_center[0] / 2)
                minc = int(image_center[1] / 2)
                maxr = int(image_center[0] * 1.5)
                maxc = int(image_center[1] * 1.5)
    
                bbox_pad = [minc, minr, maxc, maxr]
                print("Error: \n Found no bbox for {} \n Used center 25% of image instead: {}".format(filename, str(bbox_pad)))
    
            finally:
    
                gray_crop = Image.fromarray(gray_img)
                cropped_grayscale = gray_crop.crop(bbox_pad)
    
                output_name_gray = pathlib.Path(cropped_dir).joinpath(filename)
    
                if save_image == True:
    
                    cropped_grayscale.save(output_name_gray)
                    print("\nSaved cropped grayscale as {}".format(output_name_gray))
                else:
                    pass
    
                seg_input_file = np.array(cropped_grayscale)
                seg_cropped_binary_dir = cropped_binary_dir
                seg_pad = pad
                seg_resolution = resolution
    
                tempdf_section = segment_section2(filename, seg_input_file, seg_cropped_binary_dir, seg_pad, seg_resolution)
    
                print("\n Created temporary dataframe for {}: \n {}".format(filename, tempdf_section))
        
            return tempdf_section
        except:
            pass

   
def segment_section2(filename, input_file, cropped_binary_dir, pad, resolution):
    '''
    
    :param filename:
    :param input_file:
    :param cropped_binary_dir:
    :param pad:
    :param resolution:
    :return:
    '''

    # filename = str(input_file)
    filename = str(os.path.basename(filename))
    
    gray_img = input_file  # ensure gray image
    
    smooth = skimage.filters.rank.median(gray_img, selem=disk(int(resolution*8)))

    # thresh = threshold_otsu(smooth)
    thresh = threshold_minimum(smooth)

    # creates binary image by applying the above threshold and replacing all
    # init_ls = (np.where(gray_img > thresh, 0, 1)).astype('uint8')
    init_ls = (np.where(gray_img > thresh, 0, 1)).astype('uint8')

    chan = skimage.segmentation.morphological_chan_vese(smooth, 30, init_level_set=init_ls, smoothing=4, lambda1=1, lambda2=1)

    bw_uint8 = (np.array(chan)).astype('uint8')  # turning image into unsigned integer (0-255, grayscale)

    label_image, num_elements = label(bw_uint8, connectivity=2, return_num=True)
    
    image_center = list(np.divide(label_image.shape, 2))  # returns array of two floats
    
    center_hair, bbox_final = find_hair(label_image, image_center, minpixel)

    print("\nHair found for {} is:".format(filename))

    print(center_hair)
    print(type(center_hair))

    index_label = int(center_hair)
    
    region = regionprops(label_image)[index_label]

    tempdf = tempdf_gen(region, filename, resolution)
    
    # save binary slice
    with pathlib.Path(cropped_binary_dir).joinpath(filename) as output_name_binary:
        im = region.filled_image
        im = np.pad(im, pad, mode='constant')
        cropped_binary = np.array(im)
        im = Image.fromarray(cropped_binary)
        im.save(str(output_name_binary))
        print("Saved binary slice as {}".format(output_name_binary))

    print("\n")
    
    return tempdf

  
def find_hair(label_image, image_center, minpixel):
    
    props = regionprops(label_image)

    center_hair = None

    try:
        hairs = [[region.label, region.centroid, dist.euclidean(image_center, region.centroid)] for region in props if region.minor_axis_length > minpixel]

        lst = pd.DataFrame(hairs, columns=['label', 'centroid', 'distance'])
        print("\n")
        print("These are the elements with minor axis length larger than minpixel:")
        print(lst)

        center_hair_idx = lst['distance'].idxmin()
        print("\n\n Row containing smallest distance:")
        print(center_hair_idx)
        print(type(center_hair_idx))

        center_hair = lst.at[center_hair_idx, 'label'].item()
    except TypeError:
        try:
            hairs = [[region.label, region.centroid, dist.euclidean(image_center, region.centroid)] for region in props if
                     region.equivalent_diameter > minpixel]

            lst = pd.DataFrame(hairs, columns=['label', 'centroid', 'distance'])
            print("\n")
            print("These are the elements with minor axis length larger than minpixel:")
            print(lst)

            center_hair_idx = lst['distance'].idxmin()
            print("\n\n Row containing smallest distance:")
            print(center_hair_idx)
            print(type(center_hair_idx))
            center_hair = lst.at[center_hair_idx, 'label'].item()
        except:
            print("\n\nCan't find hair in this image :( \n")
            pass

    print("\n\nLabel for shortest distance:")
    print(center_hair)
    print(type(center_hair))
    if center_hair < 1:
        center_hair = 0
    elif center_hair >= 1:
        center_hair = int(center_hair-1)
    else:
        center_hair = None
        print("\nThere's been a problem - no hair found!\n")
    
    print("\nThis is the label for the hair of interest: {}\n".format(center_hair))

    bbox_final = list(props[center_hair].bbox)
    
    return center_hair, bbox_final


def tempdf_gen(region, filename, resolution):

    section_area = round(float(region.filled_area / (resolution * resolution)), 2)
    section_max = round(float(region.major_axis_length / resolution), 2)
    section_min = round(float(region.minor_axis_length / resolution), 2)
    section_eccentricity = region.eccentricity
    section_perimeter = round(float(region.perimeter / resolution), 2)
    section_id = (str(os.path.basename(filename)).rstrip('.tiff'))

    tempdf = pd.DataFrame(
        [section_id, section_area, section_max, section_min, section_eccentricity, section_perimeter]).T

    print("\nThe data for {}, are:\n {}".format(filename, tempdf))

    return tempdf


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "\n%dh: %02dm: %02ds" % (hour, min, sec)


# %% Define user input
I__USER_INPUT = 0

# Define the folder where your grayscale tiff are using pathlib
# tiff_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/fibermorph_input/section/ValidationSet_section_TIFF')
tiff_directory = pathlib.Path(sys.argv[1])

# Designate where the package should make the directory with all your results - this location must exist!
# output_location = pathlib.Path(r'/Users/tinalasisi/Desktop/Feb9_Parallel_tests')
output_location = pathlib.Path(sys.argv[2])

# Give your output directory a name
# main_output_name = "OutputTest_Section_Feb9"
main_output_name = str(sys.argv[3])

# Specify your file extension as a string - this is case-sensitive.
# file_type = ".tiff"
file_type = str(sys.argv[4])

# How many pixels per micron?
# resolution = float(4.25)
resolution = float(sys.argv[5])

# How many microns is the smallest cross-sectional width you expect?
# Smaller than 30 microns is not recommended
# min_size = 30
min_size = int(sys.argv[6])

# How many pixels do you want to pad the cropped images
# pad = 40
pad = int(sys.argv[7])

# How many parallel processes do you want to run?
# jobs = 4
jobs = int(sys.argv[8])

# Do your images need to be cropped (e.g. if there are multiple sections in the shot)?
crop = str(sys.argv[9])

# Do you want to save the cropped grayscale intermediate images (not necessary for results and setting this to False is recommended for faster computation)
# save_crop = False
save_crop = str(sys.argv[10])


# %% fibermorph section script

I__SCRIPT_STARTS = 0

total_start = timer()

I_CREATING_OUTPUTDIR = 0

# Create an output directory for all analyses in this script
main_output_path = make_subdirectory(directory=output_location, append_name=str(main_output_name))

# %% Crop all the tiffs
II_CROPPING = 0

minpixel = int(min_size * resolution)

# Change to the folder for reading images

os.chdir(str(tiff_directory))
output_path = main_output_path
file_type = "*" + file_type
cropped_list = []
for filename in pathlib.Path(tiff_directory).rglob(file_type):
    cropped_list.append(filename)
list.sort(cropped_list)  # sort the files
print(len(cropped_list))  # printed the sorted files

# Shows what is in the cropped_list. The backslash n prints a new line
print("There are ", len(cropped_list), "files in the cropped_list:\n")
print(cropped_list, "\n\n")

# Creating subdirectories for cropped images

cropped_binary_dir = make_subdirectory(main_output_path, "cropped_binary")

if crop == "True":
    crop = True
else:
    crop = False
    
if save_crop == "True":
    save_crop = True
else:
    save_crop = False

if save_crop:
    cropped_dir = make_subdirectory(main_output_path, "cropped")
    save_crop = True
    print("\nYou will have a folder of cropped grayscale images\n")
else:
    cropped_dir = None
    save_crop = False

# generator expression for debugging

# section_df = (crop_image(f, cropped_dir, cropped_binary_dir, pad, minpixel=minpixel, resolution=resolution) for f in cropped_list)

section_df = (Parallel(n_jobs=jobs, verbose=100)(delayed(crop_image)(f, cropped_dir, cropped_binary_dir, pad, minpixel, resolution, crop, save_image=save_crop) for f in cropped_list))

section_df = pd.concat(section_df)
section_df.columns = ['ID', 'area', 'max', 'min', 'eccentricity', 'perimeter']
section_index = section_df.set_index('ID')

with pathlib.Path(main_output_path).joinpath("section_data.csv") as df_output_path:
    section_index.to_csv(df_output_path)

# End the timer and then print out the how long it took
total_end = timer()
total_time = int(total_end - total_start)

print("Complete analysis time: {}".format(convert(total_time)))

