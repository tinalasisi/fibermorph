# fibermorph
Python package for image analysis of hair curvature and cross-section

## Install the package

> ### Prerequisites
> To run any of the commands below, your machine **must** have a conda Python Installation. Conda is a Python packge and environment manager. You have the option to install anaconda or the smaller miniconda. 
> See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for more information and instructions on how to install conda for Windows, Mac OS, and Linux, respectively.
> Once you have followed all the instructions and have a working Python Installation with Ana/miniconda , you can move forward.

1. Open a terminal.
    #### Mac OS: 
      + Open the *Terminal* application.
      + Enter the Pathname for the fibermorph directory. E.g. `/Users/tpl5158/Desktop/fibermorph-master`
    #### Windows:
      + Type `Anaconda` into your search bar and choose the `Anaconda Powershell` application.
      + Enter the Pathname for the fibermorph directory. E.g. `\Users\tpl5158\Desktop\fibermorph-master`
    #### Linux:
      + Open a terminal window using your search bar or the appropriate keyboard shortcut for  your Linux distribution.
      + Navigate to the fibermorph directory you just unzipped.
      
      
2. Install the fibermorph conda environment
      + Install the fibermorph package conda environment using one of two options:
        ##### Option 1: Install the fibermorph conda environment system/user wide 
        You can install the conda environment required to run fibermorph alongside other conda environments. This would allow you to call the fibermorph conda environment from outside the directory and to call it by name. If you have the necessary access and no conflicts (i.e. no other environments named `fibermorph`), it's recommended you install the environment as follows:
        + Enter `conda env create -n fibermorph -f environment.yml` and wait for the environment to install in your directory
        + Enter `conda activate fibermorph`
        In the future, you will be able to activate the environment each time using `conda activate fibermorph`
        ##### Option 2: Install the fibermorph conda environment in the directory
        If you happen to have another conda environment named `fibermorph` or you want to isolate this conda environment to the directory `fibermorph/` for any other reason, you can install this conda environment using the following command instead:
        + Enter `conda env create --prefix ./env -f environment.yml` and wait for the environment to install in your directory
        + Enter `conda activate ./env`
        To call this environment in the future, you will have to use the environment path. If you are doing this from the `fibermorph/` directory, you should be able to call it with `conda activate ./env`. If you are calling the environment from elsewhere, it's recommended you copy the full pathname and replace `./env` with that.   

## Test run
Before using this on any of your own data, it's recommended that you test that you test whether fibermorph is working properly on your machine. 

### Testing curvature analysis

### Testing section analysis



#### Validation data



## Using the fibermorph packages



**curvature**: To calculate curvature from grayscale TIFF:

**section**: To calculate cross-sectional properties from grayscale TIFF:



    
