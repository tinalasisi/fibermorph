# fibermorph
Python package for image analysis of hair curvature and cross-section

> ## Prerequisites
We recommend that you use a virtual environment to run fibermorph in a virtual environment to avoid any conflicts with other Python packages you might have on your system. To learn more about virtual enviornments, click [here](https://docs.python.org/3/tutorial/venv.html). 

If you are familiar with Python, conda and have the necessary installations on your system, feel free to skip ahead to the section entitled "Install the package", otherwise you can follow the step by step instructions below for the prerequisites.

## Setting up
1. We recommend you download [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system. 

You may also download [Anaconda](https://docs.anaconda.com/anaconda/install/). The only difference is that Anaconda comes preloaded with more libraries (500 Mb). You won't need this to run fibermorph, so we recommend you stick to minconda which is the smaller (58 Mb) and quicker to download.

Whichever you choose *be sure to download the version with Python 3.X and not Python 2.X*.

2. Open a terminal. 

The commands are written in bash, so if you are running this on a Windows OS, you will need to switch into the correct settings. You can find more information for Windows specifically [here]().

3.  Now you can set up a virtual environment. 

Create an empty conda environment, e.g. `conda create -n <YearMonthDay>_fibermorph python=3.8` and load it `conda activate <YearMonthDay>_fibermorph`

You are now ready to install fibermorph!


## Install the package

1. After having activated your new virtual environment, you can simply run `pip install fibermorph`.

You can find the latest release [here](https://github.com/tinalasisi/fibermorph/) on this GitHub page and on the [fibermorph PyPI page](https://pypi.org/project/fibermorph/). 

2. You have successfully installed fibermorph. 

The package is now ready for use. Enter `fibermorph -h` or `fibermorph --help` to see all the flags. You can keep reading to try out the demos and read instructions on the various modules within the package.

## Demo data
Before using this on any of your own data, it's recommended that you test that you test whether fibermorph is working properly on your machine. There are a few `demo` modules you can use to check whether fibermorph is running correctly.

### Testing with real data
You can test both the curvature and section modules with real data that is downloaded automatically when you run the `--demo_real` modules. 

In both cases, all you need to do is specify a folder path where the images and results can be created with `---output_directory`. This folder can be existing, but you can also establish a new folder by including it in the new path.

Both modules will download the demo data into a new folder `tmpdata` within the path you gave. Then, fibermorph will run the curvature or section analysis, and the results  will be saved in a new folder `results_cache` at this same location. It is recommended that you specify a path with a new folder name to keep everything organized.

#### Testing curvature analysis
`  --demo_real_curv`

This flag will run  a demo of fibermorph curvature analysis with real data. You will need to provide a folder for the demo data to be downloaded. 

To run the demo, you will input something like:
`fibermorph --demo_real_curv --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

#### Testing section analysis
`  --demo_real_section`

This flag will run  a demo of fibermorph section analysis with real data. You will need to provide a folder for the demo data to be downloaded.

To run the demo, you will input something like:
`fibermorph --demo_real_section --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

## Calculating image analysis error
In order to validate the image analysis program, the fibermorph package includes two modules that will generate and analyze dummy data, then run the appropriate analysis and generate error data.

Both `--demo_dummy` modules require the following flags:

```
  --output_directory OUTPUT_DIRECTORY
                      Required. Full path to and name of desired output directory. 
                      Will be created if it doesn't exist.

  --repeats REPEATS     Integer. Number of times to repeat
                        validation module (i.e. number of sets of
                        dummy data to generate).

```

The modules create a `results_cache` within the given path. In this folder there will be another folder named `<MonthDay_HourMinute>_ValidationTest_<Curv or Section>` where the generated dummy images and corresponding parameters in spreadsheets will be in a folder named `ValidationData` and the error data will be in a folder named `ValidationAnalysis`.

Running the module once will create a set of data and analyses for a single randomly generated arc and line (for curvature) or circle and ellipse (for section). To produce more data, simply add the flag `--repeats`  with the number of times you would like to repeat it, e.g. `fibermorph --repeats <integer>`. This flag is optional.

### Validating curvature analysis
`--demo_dummy_curv`

This flag will run a demo of fibermorph curvature with dummy data. Arcs and lines are generated, analyzed and error is calculated.

For this module, you can optionally include `--window_size_px`. This will allow you to edit the window size used to fit the circle used to estimate curvature. By default, this is 10 pixels. The images generated are 3900 x 5200 pixels, so you can use a range of values, but we do not recommend going below 3 pixels or the accuracy of the fit will be reduced.

To run the demo, you can enter e.g. `fibermorph --demo_dummy_curv --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName> --repeats 4 --window_size_px 50`

Rather than repeating this analysis with various window_sizes, you can simply use the curvature module (see below) and set the `--input_directory` to the `/ValidationData` folder created above. You will need to set `--resolution 1` and `--window_size_mm 10` or whatever number of pixels you would like to use as a window size.

### Validating section analysis
`--demo_dummy_section`

This flag will run a demo of fibermorph curvature with dummy data. Arcs and lines are generated, analyzed and error is calculated.

To run the demo, you can enter e.g. `fibermorph --demo_dummy_section --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName> --repeats 4`

### Deleting demo folders
`--delete_dir`
Can be used to delete directories generated in the demo modules.

Example usage: `fibermorph --delete_dir --output_directory /Users/<UserName>/<ExistingPath>/<ResultsFolderName>`

This will delete the folder (with all its contents) and print a confirmation of which folder has been deleted.

## Using the fibermorph packages
The main modules of the fibermorph package are `--curvature` and `--section`. Both require the following flags to run:

```
--output_directory OUTPUT_DIRECTORY
                      Required. Full path to and name of desired output directory. 
                      Will be created if it doesn't exist.

--input_directory INPUT_DIRECTORY
                      Required. Full path to and name of desired directory containing 
                      input files.

--jobs JOBS           Integer. Number of parallel jobs to run. Default is 1.
```

### Curvature
To calculate curvature from grayscale TIFF images of hair fibers, the flag `--curvature` is used with the following flags:
```
--resolution_mm RESOLUTION_MM
                      Integer. Number of pixels per mm for curvature analysis.

--window_size WINDOW_SIZE
                    Float. Desired size for window of measurement for curvature analysis in mm. Default is 1.0mm.

--save_image SAVE_IMAGE
                      Boolean. Default is False. Whether the curvature function should save images for intermediate image processing
                      steps.

--within_element WITHIN_ELEMENT
                      Boolean. Default is False. Whether an
                      additional directory should be created with
                      spreadsheets of curvature values within
                      each element.

```

So, to run a curvature analysis, you would enter e.g.
```
fibermorph --curvature --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --window_size 0.5 --resolution_mm 132 --save_image TRUE --within_element FALSE --jobs 2
```

### Section
To calculate cross-sectional properties from grayscale TIFF images of hair fibers, the flag `--section` is used with the following flags:
```
--minsize MINSIZE     Integer. Minimum diameter in microns for sections. Default is 20.
--maxsize MAXSIZE     Integer. Maximum diameter in microns for sections. Default is 150.
--resolution_mu RESOLUTION_MU       Float. Number of pixels per micron for section analysis.

```

An example command would be:
```
fibermorph --section --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --minsize 30 --maxsize 180 --resolution_mu 4.25 --jobs 2
```


### Converting raw images to grayscale TIFF
This package features an additional auxiliary module to convert raw images to grayscale TIFF files if necessary: `--raw2gray`

In addition to the input and output directories, the module needs the user to specify what file extension it should be looking for.

```
--file_extension FILE_EXTENSION
                      Optional. String. Extension of input files to use in input_directory when using raw2gray function. Default is .RW2.

```

A user could enter, for example:
```
fibermorph --raw2gray --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName> --file_extension .RW2 --jobs 4
```
