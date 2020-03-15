# %% Import libraries
import os  # Allows python to work with the operating system.
import pathlib  # Makes defining pathways simple between mac, windows, linux.
import shutil
import sys
from timeit import default_timer as timer  # Timer to report how long everything is taking.

from joblib import Parallel, delayed


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


def copy_if_exist(file, directory):
    '''
    
    :param file:        file to be copied
    :param directory:   location of destination directory
    :return:            None
    '''
    
    path = pathlib.Path(file)
    destination = directory
    
    if os.path.isfile(path):
        shutil.copy(path, destination)
        print('file has been copied'.format(path))
    else:
        print('file does not exist'.format(path))
    
    return None


def pretty_time_delta(seconds):
    
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)

# %% User definitions


# Define the folder where your raw images are using pathlib
# input_directory = pathlib.Path(r'/Users/tinalasisi/Desktop/fibermorph_input/section/Validation_TransferTest')
input_directory = pathlib.Path(sys.argv[1])

# Specify the file extension for your raw files - this is case-sensitive.
# file_type = ".RW2"
file_type = pathlib.Path(sys.argv[2])

# Designate where fibermorph should make the directory with all your results - this location must exist!
# output_location = pathlib.Path(r'/Users/tpl5158/Box/01_TPL5158/Dissertation/HairPhenotyping_Methods/data/manuscript_data/section')
output_location = pathlib.Path(sys.argv[3])

# Give your output directory a name
# main_output_name = "ValidationSubSet_Section_TIFF"
main_output_name = str(sys.argv[4])

# How many parallel processes do you want to run?
# jobs = 4
jobs = int(sys.argv[5])

#%%

total_start = timer()

# Create an output directory for all analyses in this script
main_output_path = make_subdirectory(directory=output_location, append_name=str(main_output_name))

# Changing directory to where the raw images are
os.chdir(input_directory)
glob_file_type = "*{}".format(file_type)  # find all files with the file_type extension

file_list = []
for filename in pathlib.Path(input_directory).rglob(glob_file_type):
    file_list.append(filename)
list.sort(file_list)  # sort the files
print(len(file_list))  # printed the sorted files


output_directory = make_subdirectory(main_output_path, append_name="Raw")


Parallel(n_jobs=jobs, verbose=100)(delayed(copy_if_exist)(f, output_directory) for f in file_list)

# [copy_if_exist(f, output_directory) for f in file_list]

# for f in file_list:
#     file_path = pathlib.Path(f)
#     if os.path.isfile(file_path):
#         shutil.copy(file_path, output_directory)
#         print('file has been copied'.format(file_path))
#     else:
#         print('file does not exist'.format(file_path))


total_end = timer()
total_time = str(pretty_time_delta(total_end - total_start))

# This will print out the minutes to the console, with 2 decimal places.
print("Entire consolidation took: {}.".format(total_time))
