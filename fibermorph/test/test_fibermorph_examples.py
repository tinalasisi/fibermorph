from fibermorph import fibermorph

# Testing fibermorph_curvature
input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/curv"
output_location = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output/curv"
jobs = 1
resolution = 132
window_size_mm = 0.5
save_img = True

fibermorph.curvature(input_directory, output_location, jobs, resolution, window_size_mm, save_img)

# Testing fibermorph section
from fibermorph import fibermorph

input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section"
# input_directory = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/section/ValidationSet_section_TIFF/TIFF/"

# input_file = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/section/ValidationSet_section_TIFF/TIFF/140918_A_1.tiff"


output_dir = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output/section"

fibermorph.section(input_directory, output_dir, jobs=4, resolution=1.06)

# note: throws error if the resolution is 4.25 because none of the images/elements in the image fit the criteria

fibermorph.list_images(input_directory)