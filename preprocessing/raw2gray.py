"""
Python 3.6+ script to read in a raw image and convert it to a grayscale tiff format.

Authors: Tina Lasisi tina.lasisi@gmail.com, Nicholas Stephens nbs49@psu.edu & Timothy Webster twebster17@gmail.com
March 14, 2020

"""
# %% Import libraries

import os  # Allows python to work with the operating system.
import pathlib  # Makes defining pathways simple between mac, windows, linux.

import sys
from timeit import default_timer as timer  # Timer to report how long everything is taking.

import rawpy  # Allows python to read raw image and convert to tiff
from PIL import Image  # Allows conversion from RGB to grayscale and ImageChops for cropping
from joblib import Parallel, delayed
from skimage.color import rgb2gray


# %% Import functions


def make_subdirectory(directory, append_name=""):
    """
    Function to build a directory based upon the arguments passed in append. Takes a
    directory and uses pathlib to join that to the appended string passed.
​
    :param directory:       The directory within which this function will make a subdirectory.
    :param append_name:     A string to be appended to the pathlib object.
    :return output_path:    Returns a new directory for output.

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


def raw_to_gray(imgfile, output_directory):
    """

    Function to convert raw files to gray tiff files

    :param imgfile:             raw image file
    :param output_directory:    output directory where tiff folder will be created
    :return:
            - img: PIL image object
            - tiff_output_path: pathlib object with image path


    """
    imgfile = os.path.abspath(imgfile)
    output_directory = output_directory
    basename = os.path.basename(imgfile)
    name = os.path.splitext(basename)[0] + ".tiff"
    output_name = pathlib.Path(output_directory).joinpath(name)
    print("\n\n")
    print(name)
    
    try:
        with rawpy.imread(imgfile) as raw:
            rgb = raw.postprocess(use_auto_wb=True)
            im = Image.fromarray(rgb).convert('LA')
            im.save(str(output_name))
    except:
        print("\nSomething is wrong with {}\n".format(str(f)))
        pass

    print('{} has been successfully converted to a grayscale tiff.\n Path is {}\n'.format(name, output_name))

    return output_name


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%dh: %02dm: %02ds" % (hour, min, sec)


# %% Define user input
I__USER_INPUT = 0

# Define the folder where your raw images are using pathlib
# input_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/fibermorph_input/section/ValidationSet_section_Raw')
input_directory = pathlib.Path(sys.argv[1])

# Specify the file extension for your raw files - this is case-sensitive.
# file_type = ".RW2"
file_type = str(sys.argv[2])

# Designate where the package should make the directory with all your results - this location must exist!
# output_location = pathlib.Path(r'/Users/tinalasisi/Desktop/')
output_location = pathlib.Path(sys.argv[3])

# Give your output directory a name
# main_output_name = "TIFF_images_Section_Jan30"
main_output_name = str(sys.argv[4])

# How many parallel processes do you want to run?
# jobs = 4
jobs = int(sys.argv[5])

# %% Execute fibermorph raw2gray script
I__SCRIPT_STARTS = 0

total_start = timer()


# Create an output directory for all analyses in this script
main_output_path = make_subdirectory(directory=output_location, append_name=str(main_output_name))

# Convert raw files to tiff or copy TIFFs into new directory

# Changing directory to where the raw images are
os.chdir(input_directory)
glob_file_type = "*{}".format(file_type)  # find all files with the file_type extension

file_list = []
for f in pathlib.Path(input_directory).rglob(glob_file_type):
    file = pathlib.Path(f)
    filename = os.path.abspath(file)
    file_list.append(filename)

# file_list = glob.glob((str(glob_file_type)), recursive=True)  # find all the raw files in that directory
list.sort(file_list)  # sort the files
print(file_list)  # printed the sorted files

print("There are {} files to convert".format(len(file_list)))
print("\n\n")

# %% Converting files
# A loop for turning all raw files into grayscale tiffs

print("Converting raw files into grayscale tiff files...\n")

tiff_directory = make_subdirectory(main_output_path, append_name="tiff")

Parallel(n_jobs=jobs, verbose=100)(delayed(raw_to_gray)(f, tiff_directory) for f in file_list)

# # for debugging
# file_list = file_list[:3]
# f = file_list[0]
# output_directory = tiff_directory
# imgfile = f
#
# [raw_to_gray(f, tiff_directory) for f in file_list]


# End the timer and then print out the how long it took
total_end = timer()
total_time = (total_end - total_start)

# This will print out the minutes to the console, with 2 decimal places.
print("\n\n")
print("Entire analysis took: {}.".format(convert(total_time)))
