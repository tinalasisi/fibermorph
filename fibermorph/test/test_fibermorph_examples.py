import os
import shutil

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Designate where fibermorph should make the directory with all your results - this location must exist!
os.makedirs(r'./results_cache', exist_ok=True)
output_directory = os.path.abspath(r'./results_cache')

# %% delete cache

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cache = "./results_cache"

print("Deleting {}".format(os.path.abspath(cache)))
shutil.rmtree(cache)


# %% Testing fibermorph_curvature
from fibermorph import fibermorph
# import requests
# from github import Github
#
# g = Github
#
# test = g.get_repo("tinalasisi/fibermorph_DemoData")


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


output_dir = fibermorph.make_subdirectory(output_directory, append_name="DemoTest_Section")

fibermorph.section(input_directory, output_dir, jobs=4, resolution=1.06)

# note: throws error if the resolution is 4.25 because none of the images/elements in the image fit the criteria

fibermorph.list_images(input_directory)

# %% Testing with dummy data

from fibermorph import dummy_data
from fibermorph import fibermorph
import pandas as pd
from datetime import datetime
import pathlib

jetzt = datetime.now()
timestamp = jetzt.strftime("%b%d_%H%M_")
testname = str(timestamp + "ValidationTest_Curvature")

output_location = "/Users/tinalasisi/Desktop/"
main_output_path = fibermorph.make_subdirectory(output_location, append_name=testname)

dummy_dir = fibermorph.make_subdirectory(main_output_path, append_name="ValidationData")
shape_list = ["arc", "line"]

output_path = fibermorph.make_subdirectory(main_output_path, append_name="ValidationAnalysis")

for shape in shape_list:
    print(shape)
    df, img, im_path, df_path = dummy_data.dummy_data_gen(
        output_directory=dummy_dir,
        shape=shape,
        min_elem=10,
        max_elem=20,
        im_width=5200,
        im_height=3900,
        width=10)
    
    valid_df = pd.DataFrame(df).sort_values(by=['ref_length'], ignore_index=True)
    
    test_df = fibermorph.curvature_seq(im_path, output_path, resolution=1, window_size_mm=10, save_img=False, test=True)
    
    test_df2 = pd.DataFrame(test_df).sort_values(by=['length'], ignore_index=True)
    
    if shape == "arc":
        
        valid_df['index1'] = valid_df['ref_length'] * valid_df['ref_radius']
        
        valid_df = pd.DataFrame(valid_df).sort_values(by=[ 'index1'], ignore_index=True)
        
        test_df2['radius'] = 1/test_df2['curv_median']

        test_df2['index2'] = test_df2['length'] * test_df2['radius']

        test_df2 = pd.DataFrame(test_df2).sort_values(by=['index2'], ignore_index=True)
        
        test_df2['error_radius'] = abs(valid_df['ref_radius'] - test_df2['radius']) / valid_df['ref_radius']
        
        test_df2['error_curvature'] = abs(valid_df['ref_curvature'] - test_df2['curv_median']) / valid_df['ref_curvature']
    
    test_df2['error_length'] = abs(valid_df['ref_length'] - test_df2['length']) / valid_df['ref_length']
    
    valid_df2 = valid_df.join(test_df2)

    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    df_path = valid_df2.to_csv(pathlib.Path(output_path).joinpath(timestamp + shape + "_errordata.csv"))

    print(valid_df2)

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
