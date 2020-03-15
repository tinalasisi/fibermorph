"""
Python 3.6+ user input script calculate hair fiber curvature from grayscale tiff file.

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
fibermorph_curvature = os.path.join(dname, "../analyze/curvature.py")
# fibermorph_curvature = pathlib.Path(r"[insert path here]")

# Define the folder where your grayscale tiff images are
tiff_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/Old_AfrAm_Images')
# tiff_directory = pathlib.Path(r'/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/curvature/ValidationSet_TIFF')
# tiff_directory = pathlib.Path(r'[insert path here]')

# Designate where fibermorph should make the directory with all your results - this location must exist!
output_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/')
# output_directory = pathlib.Path(r'[insert path here]')

# Give your output directory a name
main_output_name = "Test"
# main_output_name = "ValidationSet_Feb20"
# main_output_name = "[insert directory name here]"

# Specify your file extension as a string - this is case-sensitive.
file_type = ".tiff"
# file_type = "[insert file type here]"

# How many pixels per mm?
resolution = int(132)
# resolution = int("[insert resolution here]")

# What should the size do you want for the window of measurement for curvature within the hair? This value is in mm.
window_size_mm = float(0.5)
# window_size_mm = float("[insert window size here]")

# What is the smallest expected length of hair in mm?
min_hair = float(1.0)
# min_hair = float("[insert minimum size here]")

# How many parallel processes do you want to run?
# Example:
# It is recommended that jobs < CPUs on your machine
jobs = 4
# jobs = int([insert pad size here])

# Do you want to save all intermediate images?
# This is not necessary for the final output and might cause script to run slower
save_img = True
# save_img = "[insert True or False here]"

# %% Execute subprocess

# Use subprocess to sent the python command with the relevant commands. for "python" specify the precise path where
# your python environment. This is the virtual environment you've made before in python 3.6


p = subprocess.Popen(["python",
                      str(fibermorph_curvature),
                      str(tiff_directory),
                      str(output_directory),
                      str(main_output_name),
                      str(file_type),
                      str(resolution),
                      str(window_size_mm),
                      str(min_hair),
                      str(jobs),
                      str(save_img)])

# Prints the error messages if anything comes up. If there are no errors it will print (None, None)
print(p.communicate())
