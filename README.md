# fibermorph


Python package for image analysis of hair curvature and cross-section

## Quickstart
For those who want to run the program immediately, just follow these commands in your terminal. You will need to have conda and know how to use it. If you need instructions for this, read the detailed set up below.

1. Create a conda environment.
`conda create -n fibermorph_env python=3.8`
2. Activate this environment.
`conda activate fibermorph_env`
3. Install fibermorph.
`pip install fibermorph`
4. Test fibermorph with real data.  
`fibermorph --demo_real_curv --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

	and  

	`fibermorph --demo_real_section --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`
5. Use fibermorph on your own grayscale TIFFs of longitudinal or cross-sectional hair images.  

	`fibermorph --curvature --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --resolution_mm 132 --jobs 2`  

	and

	`fibermorph --section --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --minsize 30 --maxsize 180 --resolution_mu 4.25 --jobs 2`


## Setting up
1. We recommend you download [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system.
	You may also download [Anaconda](https://docs.anaconda.com/anaconda/install/). The only difference is that Anaconda comes preloaded with more libraries (500 Mb). You won't need this to run fibermorph, so we recommend you stick to minconda which is the smaller (58 Mb) and quicker to download.

	Whichever you choose *be sure to download the version with Python 3.X and not Python 2.X*.
2. Open a terminal.
	#### Mac OS:
	- Open the *Terminal* application.
	#### Windows:
	- Type `miniconda` in the search box and open the application.
	#### Linux:
	- Open the *Terminal* application.
3.  Now you can set up a virtual environment.

	Create an empty conda environment, e.g. `conda create -n <fibermorph_env python=3.8` and load it `conda activate fibermorph_env`
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

In both cases, all you need to do is specify a folder path where the images and results can be created with `---output_directory` or `-o`. This folder can be existing, but you can also establish a new folder by including it in the new path.

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

### Curvature
To calculate curvature from grayscale TIFF images of hair fibers, the flag `--curvature` is used with the following flags in addition to input and output directories:
```
--resolution_mm       	Integer. Number of pixels per mm for curvature analysis.
						Default is 132.
--window_size  [ ...] 	Float or integer or None. Desired size for window of measurement
						for curvature analysis in pixels or mm (given
						the flag --window_unit). If nothing is entered, the default
						is None and the entire hair will be used to for the curve fitting."
--window_unit {px,mm}	String. Unit of measurement for window of measurement for curvature
                      	analysis. Can be 'px' (pixels) or 'mm'. Default is 'px'.
-s, --save_image      	Default is False. Will save intermediate curvature processing images if
                      	--save_image flag is included.
-W, --within_element  	Boolean. Default is False. Will create an additional directory with
                      	spreadsheets of raw curvature measurements for each hair if the
                      	--within_element flag is included.

```

So, to run a curvature analysis, you would enter e.g.
```
fibermorph --curvature --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --window_size 0.5 --window_unit mm --resolution 132 --save_image --within_element --jobs 2
```

### Section
To calculate cross-sectional properties from grayscale TIFF images of hair fibers, the flag `--section` is used with the following flags:
```
--resolution_mu       Float. Number of pixels per micron for section analysis. Default is 4.25.
--minsize             Integer. Minimum diameter in microns for sections. Default is 20.
--maxsize             Integer. Maximum diameter in microns for sections. Default is 150.

```

An example command would be:
```
fibermorph --section --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/ --minsize 30 --maxsize 180 --resolution_mu 4.25 --jobs 2
```


### Converting raw images to grayscale TIFF
This package features an additional auxiliary module to convert raw images to grayscale TIFF files if necessary: `--raw2gray`

In addition to the input and output directories, the module needs the user to specify what file extension it should be looking for.

```
--file_extension      Optional. String. Extension of input files to use in input_directory when
                      using raw2gray function. Default is .RW2.

```

A user could enter, for example:
```
fibermorph --raw2gray --input_directory /Users/<UserName>/<ImageFolderPath> --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName> --file_extension .RW2 --jobs 4
```
