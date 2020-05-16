"""
Python 3.6+ user input script for calculating cross-sectional hair fiber properties from grayscale tiff file.

Authors: Tina Lasisi tina.lasisi@gmail.com, Nicholas Stephens nbs49@psu.edu & Timothy Webster twebster17@gmail.com
March 14, 2020

"""

# %% Imports

import os
import sys
import pathlib
import subprocess

# %% User definitions

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Location of the fibermorph_section script you have downloaded/cloned
# Example:
fibermorph_section = os.path.join(dname, "../analyze/section.py")
# fibermorph_section = pathlib.Path(r"[insert path here]")

# Define the folder where your grayscale tiff images are
# Example:
tiff_directory = os.path.abspath(r'./results_cache/validation_data/section')
# tiff_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/ValidationSet_Section_TIFF_Mar12/tiff')
# tiff_directory = pathlib.Path(r'[insert path here]')


# Designate where fibermorph should make the directory with all your results - this location must exist!
# Example:
os.makedirs(r'./results_cache', exist_ok=True)
output_directory = os.path.abspath(r'./results_cache')
# output_directory = pathlib.Path(r'[insert path here]')

# Give your output directory a name
# Example:
main_output_name = "section_test"
# main_output_name = "RealDataTest_Apr11"
# main_output_name = "[insert directory name here]"

# Specify your file extension as a string - this is case-sensitive.
# Example:
file_type = ".tiff"
# file_type = "[insert file type here]"

# How many pixels per micron?
# Example:
resolution = float(1)
# resolution = float(4.25)

# How many microns is the smallest cross-sectional width you expect?
# NB: If your min_size is too small, you might have more noise and this might slow down the script
# But if your min_size is too large, you might filter out hairs, meaning no results for that image
# Example:
min_size = int(1)
# min_size = int(30)
# min_size = int("[insert minimum size here]")

# How many pixels do you want to pad the cropped images
# Example:
pad = int(40)
# pad = int([insert pad size here])

# How many parallel processes do you want to run?
# Example:
# It is recommended that jobs < CPUs on your machine
jobs = 4
# jobs = int([insert pad size here])

# Do your images need to be cropped (e.g. if there are multiple sections in the shot)?
crop = True

# Do you want to save the cropped grayscale intermediate images
# Note: Intermediate images are not necessary for results and setting this to False is recommended for faster computation
# Example:
save_crop = True
# save_crop = "[insert True or False here]"


# %% Execute subprocess

# Use subprocess to sent the python command with the relevant commands.


p = subprocess.Popen(["python",
                      str(fibermorph_section),
                      str(tiff_directory),
                      str(output_directory),
                      str(main_output_name),
                      str(file_type),
                      str(resolution),
                      str(min_size),
                      str(pad),
                      str(jobs),
                      str(crop),
                      str(save_crop)])

# Prints the error messages if anything comes up. If there are no errors it will print (None, None)
print(p.communicate())
