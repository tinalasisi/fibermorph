"""

Python 3.6+ user input script to read in a raw image and convert it to a grayscale tiff format.

Authors: Tina Lasisi tina.lasisi@gmail.com, Nicholas Stephens nbs49@psu.edu & Timothy Webster twebster17@gmail.com
March 14, 2020

"""

# %% Imports

import os
import sys
import pathlib
import subprocess

# %% User definitions

# Location of the raw2gray script you have downloaded/cloned
# # This is a relative path. Don't move the file or edit this
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
raw2gray = os.path.join(dname, "../preprocessing/raw2gray.py")

# Define the folder where your raw images are using pathlib
input_directory = pathlib.Path(r'/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/archive/raw/cross-section/ValidationSet/ValidationSet')
# input_directory = pathlib.Path(r"[insert path here]")

# Specify the file extension for your raw files - this is case-sensitive.
file_type = ".RW2"
# file_type = "[insert file type here]"

# Designate where fibermorph should make the directory with all your results - this location must exist!
output_location = pathlib.Path(r'/Users/tinalasisi/Desktop')
# output_location = pathlib.Path(r'[insert path here]')

# Give your output directory a name
main_output_name = "Test"
# main_output_name = "[insert directory name here]"

# How many parallel processes do you want to run?
# It is recommended that jobs < CPUs on your machine
jobs = 4
# jobs = int("[insert pad size here]")

# %% Send arguments with subprocess

# Use subprocess to send the python command with the relevant commands.

# For "python" specify the precise path where your python environment. This is the virtual environment you've made
# before in python 3.6


p = subprocess.Popen(["python",
                      str(raw2gray),
                      str(input_directory),
                      str(file_type),
                      str(output_location),
                      str(main_output_name),
                      str(jobs)])
# this script didn't work until I changed my python environment path back to "python"

# Prints the error messages if anything comes up. If there are no errors it will print (None, None)
print(p.communicate())
