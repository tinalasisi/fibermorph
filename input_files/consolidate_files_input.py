"""
Python 3.6+ user input script for calculating cross-sectional hair fiber properties from grayscale tiff file.

Authors: Tina Lasisi tpl5158@psu.edu & Nicholas Stephens nbs49@psu.edu
February 3, 2020

"""

# %% Imports

import os
import sys
import pathlib
import subprocess

# %% User definitions

# Location of the fibermorph_consolidate script you have downloaded/cloned
fibermorph_consolidate = pathlib.Path(r'/Users/tpl5158/PycharmProjects/fibermorph/consolidate_files.py')
# fibermorph_section = pathlib.Path(r"[insert path here]")

# Define the folder where your raw images are using pathlib
input_directory = pathlib.Path(r'/Users/tpl5158/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/archive/raw/cross-section/ValidationSet')
# input_directory = pathlib.Path(r"[insert path here]")

# Specify the file extension for your raw files - this is case-sensitive.
file_type = ".RW2"
# file_type = "[insert file type here]"

# Designate where fibermorph should make the directory with all your results - this location must exist!
output_location = pathlib.Path(r'/Users/tpl5158/Desktop')
# output_location = pathlib.Path(r'[insert path here]')

# Give your output directory a name
main_output_name = "ValidationSet_Section_TIFF"
# main_output_name = "[insert directory name here]"

# How many parallel processes do you want to run?
# It is recommended that jobs < CPUs on your machine
jobs = 6
# jobs = int("[insert pad size here]")


# %% Execute subprocess

# Use subprocess to sent the python command with the relevant commands. for "python" specify the precise path where
# your python environment. This is the virtual environment you've made before in python 3.6


p = subprocess.Popen(["python",
                      str(fibermorph_consolidate),
                      str(input_directory),
                      str(file_type),
                      str(output_location),
                      str(main_output_name),
                      str(jobs)])

# Prints the error messages if anything comes up. If there are no errors it will print (None, None)
print(p.communicate())
