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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# Location of the fibermorph_curvature script you have downloaded/cloned
fibermorph_curvature = os.path.abspath("../analyze/curvature.py")

# Define the folder where your grayscale tiff images are
tiff_directory = os.path.abspath(r'./results_cache')

# Designate where fibermorph should make the directory with all your results - this location must exist!
os.makedirs(r'./results_cache', exist_ok=True)
output_directory = os.path.abspath(r'./results_cache')

# Give your output directory a name
main_output_name = "curvature_test"

# Specify your file extension as a string - this is case-sensitive.
file_type = ".tiff"

# How many pixels per mm?
resolution = int(1)

# What should the size do you want for the window of measurement for curvature within the hair? This value is in mm.
window_size_mm = float(66)

# What is the smallest expected length of hair in mm?
min_hair = float(1)

# How many parallel processes do you want to run?
# Example:
# It is recommended that jobs < CPUs on your machine
jobs = 1

# Do you want to save all intermediate images?
# This is not necessary for the final output and might cause script to run slower
save_img = True

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
