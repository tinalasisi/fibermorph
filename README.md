# fibermorph
Python package for image analysis of hair curvature and cross-section

## Install the package

1. create an empty conda environment, e.g. `conda create -n <YearMonthDay>_fibermorph python=3.8` and load it `conda activate <YearMonthDay>_fibermorph`
2. run pip install specifying the path of the fibermorph folder: `pip install -e <FolderPath>`
3. Now you should be able to run fibermorph anywhere with the command `fibermorph`
4. You can see all the flags with the command `fibermorph -h` or `fibermorph --help`


> ### Prerequisites

## Demo data
Before using this on any of your own data, it's recommended that you test that you test whether fibermorph is working properly on your machine. There are a few `demo` modules you can use to check whether fibermorph is running correctly.

### Testing curvature analysis
`  --demo_real_curv`

This flag will run  a demo of fibermorph curvature analysis with real data. You will need to provide a folder for the demo data to be downloaded. This folder can be existing, but you can also establish a new folder by including it in the new path.

To run the demo, you will input something like:
`fibermorph --demo_real_curv --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

This command will download the demo data into a new folder `tmpdata` within the path you gave. Then, fibermorph will run the curvature analysis, the results of which will be saved in a new folder `results_cache` at this same location.

### Testing section analysis

`  --demo_real_section`

This flag will run  a demo of fibermorph section analysis with real data. You will need to provide a folder for the demo data to be downloaded. This folder can be existing, but you can also establish a new folder by including it in the new path.

To run the demo, you will input something like:
`fibermorph --demo_real_section --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

This command will download the demo data into a new folder `tmpdata` within the path you gave. Then, fibermorph will run the curvature analysis, the results of which will be saved in a new folder `results_cache` at this same location.


## Validation data
In order to validate the image analysis program, the fibermorph package includes two modules that will generate and analyze dummy data, then run the appropriate analysis and generate error data.

### Validating curvature analysis
`--demo_dummy_curv`

This flag will run a demo of fibermorph curvature with dummy data. Arcs and lines are generated, analyzed and error is calculated.

To run the demo, you can enter e.g. `fibermorph --demo_dummy_curv --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

This will run the module and create a `results_cache` within the given path. In this folder there will be another folder named `<MonthDay_HourMinute>_ValidationTest_Curv` where the generated dummy images and corresponding parameters in spreadsheets will be in a folder named `ValidationData` and the error data will be in a folder named `ValidationAnalysis`.

Running the module once will create a set of data and analyses for a single randomly generated arc and line. To produce more data, simply add the flag `--repeats`  with the number of times you would like to repeat it, e.g. `fibermorph --repeats <integer>`


### Deleting demo folders
`--delete_dir`
Can be used to delete directories generated in the demo modules.

Example usage: `fibermorph --delete_dir --output_directory /Users/<UserName>/<ExistingPath>/<NewFolderName`

## Using the fibermorph packages

```
--output_directory OUTPUT_DIRECTORY
                      Required. Full path to and name of desired output directory. Will be created if it doesn't exist.
--input_directory INPUT_DIRECTORY
                      Required. Full path to and name of desired directory containing input files.
--jobs JOBS           Integer. Number of parallel jobs to run. Default is 1.
```


**curvature**: To calculate curvature from grayscale TIFF:

```
--resolution_mm RESOLUTION_MM
                      Integer. Number of pixels per mm for curvature analysis.

--window_size WINDOW_SIZE
                    Float. Desired size for window of measurement for curvature analysis in mm. Default is 1.0mm.

--save_image SAVE_IMAGE
                      Boolean. Default is False. Whether the curvature function should save images for intermediate image processing
                      steps.
```

**section**: To calculate cross-sectional properties from grayscale TIFF:

```

--minsize MINSIZE     Integer. Minimum diameter in microns for sections. Default is 20.
--maxsize MAXSIZE     Integer. Maximum diameter in microns for sections. Default is 150.
--save_image SAVE_IMAGE
                      Boolean. Default is False. Whether the curvature function should save images for intermediate image processing
                      steps.

--resolution_mu RESOLUTION_MU       Float. Number of pixels per micron for section analysis.

```

### Converting raw images to grayscale TIFF

```
--raw2gray            Convert raw image files to grayscale TIFF files.

--file_extension FILE_EXTENSION
                      Optional. String. Extension of input files to use in input_directory when using raw2gray function. Default is
                      .RW2.

```
