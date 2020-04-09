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

# Location of the fibermorph_section script you have downloaded/cloned
# This is a relative path. Don't move the file or edit this
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
fibermorph_section = os.path.join(dname, "../analyze/section.py")

# Define the folder where your grayscale tiff images are
# Example:
tiff_directory = pathlib.Path(r'/Users/tinalasisi/Box/01_TPL5158/To_Organize/!Research/Current/MethodsPaper/Shared PCS_TL/Cross-sectional data/Cambridge cross-sections with scale')
# tiff_directory = pathlib.Path(r'/Users/tpl5158/Desktop/AfricanAmerican_Section_TIFF/tiff')
# tiff_directory = pathlib.Path(r'[insert path here]')


# Designate where fibermorph should make the directory with all your results - this location must exist!
# Example:
output_location = pathlib.Path(r'/Users/tinalasisi/Desktop')
# output_directory = pathlib.Path(r'[insert path here]')
# Give your output directory a name
# Example:
main_output_name = "ValidationSet_SectionAnalysis_Mar14_2015"
# main_output_name = "AfricanAmerican_SectionAnalysis_Feb15"
# main_output_name = "[insert directory name here]"

# Specify your file extension as a string - this is case-sensitive.
# Example:
file_type = ".tiff"
# file_type = "[insert file type here]"

# How many pixels per micron?
# Example:
resolution = float(4.25)
# resolution = float(4.25)

# How many microns is the smallest cross-sectional width you expect?
# NB: If your min_size is too small, you might have more noise and this might slow down the script
# But if your min_size is too large, you might filter out hairs, meaning no results for that image
# Example:
min_size = int(30)
# min_size = int("[insert minimum size here]")

# How many pixels do you want to pad the cropped images
# Example:
pad = int(100)
# pad = int([insert pad size here])

# How many parallel processes do you want to run?
# Example:
# It is recommended that jobs < CPUs on your machine
jobs = 6
# jobs = int([insert pad size here])

# Do your images need to be cropped (e.g. if there are multiple sections in the shot)?
crop = False

# Do you want to save the cropped grayscale intermediate images
# Note: Intermediate images are not necessary for results and setting this to False is recommended for faster computation
# Example:
save_crop = False
# save_crop = "[insert True or False here]"


# %% Execute subprocess

# Use subprocess to sent the python command with the relevant commands.


p = subprocess.Popen(["python",
                      str(fibermorph_section),
                      str(tiff_directory),
                      str(output_location),
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
