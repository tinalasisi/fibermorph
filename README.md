# fibermorph
Python package for image analysis of hair curvature and cross-section

## Install the package

> ### Prerequisites
> To run any of the commands below, your machine **must** have a conda Python Installation. Conda is a Python packge and environment manager. You have the option to install anaconda or the smaller miniconda. 
> See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for more information and instructions on how to install conda for Windows, Mac OS, and Linux, respectively.
> Once you have followed all the instructions and have a working Python Installation with Ana/miniconda , you can move forward.

1. Download the script as zip.
2. Unzip the fibermorph-master file and drag onto your Desktop or any other directory you wish to work in.
3. Open a terminal.
    #### Mac OS: 
      + Open the *Terminal* application.
      + Enter the Pathname for the fibermorph directory. E.g. `/Users/tpl5158/Desktop/fibermorph-master`
    #### Windows:
      + Type `Anaconda` into your search bar and choose the `Anaconda Powershell` application.
      + Enter the Pathname for the fibermorph directory. E.g. `\Users\tpl5158\Desktop\fibermorph-master`
    #### Linux:
      + Open a terminal window using your search bar or the appropriate keyboard shortcut for  your Linux distribution.
      + Navigate to the fibermorph directory you just unzipped.
      
      
 4. Install the fiber morph conda environment
      + Install the fibermorph package conda environment using one of two options:
        ##### Option 1: Install the fibermorph conda environment system/user wide 
        You can install the conda environment required to run fibermorph alongside other conda environments. This would allow you to call the fibermorph conda environment from outside the directory and to call it by name. If you have the necessary access and no conflicts (i.e. no other environments named `fibermorph`), it's recommended you install the environment as follows:
        + Enter `conda env create -n fibermorph -f environment.yml` and wait for the environment to install in your directory
        + Enter `conda activate fibermorph`
        In the future, you will be able to activate the environment each time using `conda activate fibermorph`
        ##### Option 2: Install the fibermorph conda environment in the directory
        If you happen to have another conda environment named `fibermorph` or you want to isolate this conda environment to the directory `fibermorph-master/` for any other reason, you can install this conda environment using the following command instead:
        + Enter `conda env create --prefix ./env -f environment.yml` and wait for the environment to install in your directory
        + Enter `conda activate ./env`
        To call this environment in the future, you will have to use the environment path. If you are doing this from the `fibermorph-master/` directory, you should be able to call it with `conda activate ./env`. If you are calling the environment from elsewhere, it's recommended you copy the full pathname and replace `./env` with that.   

## Test run
Before using this on any of your own data, it's recommended that you test that you test whether fibermorph is working properly on your machine. All tests are available in the `test/` folder. The tests you run will create a folder in `test/` called `results_cache/`. See below for instructions on emptying the cache.

### Testing curvature analysis
The `curvature_test.py` script is written to run without any user input. 
- Follow instructions above for installing and creating the conda environment in the directory
- Once you have opened a terminal and navigated to the `fibermorph-master/` directory, you can simply run the file by entering the following: 
    + `python test/curvature_test.py`
    
>  Within the `test/results_cache/` folder, you will find a new `curvature_test/` folder. Open it up and explore what the results look like. There will be various folders containing the intermediate images created during the image processing steps (filtering, binarizing, cleaning, skeletonizing and pruning the images). There is an `analysis/` folder containing csv files for each sample showing the within-hair curvature. The main `curvature_test/` directory has a csv file with the summary data for each sample.

### Testing section analysis
The `section_test.py` script is also written to run without any user input. 
- Follow instructions above for installing and creating the conda environment in the directory
- Once you have opened a terminal and navigated to the `fibermorph-master/` directory, you can simply run the file by entering the following: 
    + `python test/section_test.py`
    
>  In the `test/results_cache/` folder will find the results in the `section_test/` folder. Within this folder are folders named `cropped/` and `binary/` that will show you the final images that were used for the analyses. The cropped images are used for the segmentations found in the `binary/` folder.
    
If both these scripts work, your images should run just fine (if they don't, let me know).


### Emptying the cache
The results are meant to be illustrative, not to hog space on your machine (image files are generally large). So feel free to empty the results cache of your test results and your dummy data by running:
    + `python test/empty_cache.py`

#### Validation data
If you want to play around with some real data, there is a folder full of complimentary images of hairs imaged for curvature and cross-sectional analysis in the `/data` directory. There are subdirectories with Raw and TIFF images for both curvature and cross-sectional analysis, so you can try out the accessory packages below as well (for converting from raw image files to TIFF).

### Creating and analyzing dummy data
The `dummy_data.py` script generates images of arcs and outputs their curvatures into a csv file so that you can run the curvature analysis and compare with these known curvatures.

> There is an option to have `dummy_data.py` output an image with lines instead so you can see that there is a curvature of 0 when you run the script (it will show you the line lengths in the csv output). 
> A cross-sectional dummy data script is in the works!


## Using the fibermorph packages

### analyze 
These are the main packages for analyzing hair fiber morphology. The packages used are in the `analyze` directory, but you will be opening a text editor (any text editor will do, just not a word _processor_ like MS Word)  to edit the corresponding files in the `input_files` directory.

**curvature**: To calculate curvature from grayscale TIFF:
- edit `curvature_input.py` to input the appropriate directory paths to images and output for your machine.
    + `python input_files/curvature_input.py`

**section**: To calculate cross-sectional properties from grayscale TIFF:
- edit `section_input.py` to input the appropriate directory paths to images and output for your machine.
    + `python input_files/section_input.py`

### preprocessing 
These are a set of accessory packages to help you organize and prepare your images for analysis if necessary.

**consolidate_files**: To consolidate files from various subdirectories where your images might be living
- edit `consolidate_input.py`
    + `python preprocessing/consolidate_files_input.py`

**raw2gray**: To convert RAW files to grayscale:
- edit `raw2gray_input.py`
    + `python preprocessing/raw2gray_input.py`
    
