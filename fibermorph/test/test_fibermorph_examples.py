# %% Testing fibermorph_curvature
from fibermorph import fibermorph

input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/curv"
output_location = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output/curv"
jobs = 1
resolution = 132
window_size_mm = 0.5
save_img = True

fibermorph.curvature(input_directory, output_location, jobs, resolution, window_size_mm, save_img)

# %% Testing fibermorph section
from fibermorph import fibermorph

input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section"
# input_directory = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input
# /section/ValidationSet_section_TIFF/TIFF/"

# input_file = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input
# /section/ValidationSet_section_TIFF/TIFF/140918_A_1.tiff"


output_dir = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output/section"

fibermorph.section(input_directory, output_dir, jobs=4, resolution=1.06)

# note: throws error if the resolution is 4.25 because none of the images/elements in the image fit the criteria

fibermorph.list_images(input_directory)

# %% Testing with dummy data

from fibermorph import dummy_data
from fibermorph import fibermorph

dummy_dir = "/Users/tinalasisi/Desktop/DummyDataTest/curvature/input"
shape_list = ["arc", "line"]

output_location = "/Users/tinalasisi/Desktop/DummyDataTest/curvature/output"

for shapes in shape_list:
    print(shapes)
    df, img, im_path, df_path = dummy_data.dummy_data_gen(
        output_directory=dummy_dir,
        shape=shapes,
        min_elem=10,
        max_elem=20,
        im_width=5200,
        im_height=3900,
        width=10)
    
    fibermorph.curvature(dummy_dir, output_location, jobs=1, resolution=1, window_size_mm=10, save_img=False)
    
    
    print(df)
    print(img)

# %% Testing with dummy data section

from fibermorph import dummy_data
from fibermorph import fibermorph

dummy_data.dummy_data_gen(
    output_directory="/Users/tinalasisi/Desktop/DummyDataTest/section/input",
    shape="ellipse",
    min_elem=1,
    max_elem=1,
    im_width=5200,
    im_height=3900,
    width=10)

input_directory = "/Users/tinalasisi/Desktop/DummyDataTest/section/input"

output_location = "/Users/tinalasisi/Desktop/DummyDataTest/section/output"

fibermorph.section(input_directory, output_location, jobs=1, resolution=1.0)
