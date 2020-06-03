# %% import

import os
import pathlib
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from fibermorph import dummy_data
from fibermorph import fibermorph

# %% functions

def create_results_cache():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # Designate where fibermorph should make the directory with all your results - this location must exist!
    os.makedirs(r'./results_cache', exist_ok=True)
    output_directory = os.path.abspath(r'./results_cache')
    
    return output_directory


def delete_results_cache():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    cache = "./results_cache"
    
    print("Deleting {}".format(os.path.abspath(cache)))
    shutil.rmtree(cache)
    
    return True


def validation_curv(output_location, repeats=3):
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "ValidationTest_Curv")
    
    main_output_path = fibermorph.make_subdirectory(output_location, append_name=testname)
    
    dummy_dir = fibermorph.make_subdirectory(main_output_path, append_name="ValidationData")
    shape_list = ["arc", "line"]
    
    replist = [el for el in shape_list for i in range(repeats)]
    
    output_path = fibermorph.make_subdirectory(main_output_path, append_name="ValidationAnalysis")
    
    for shape in replist:
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
        
        test_df = fibermorph.curvature_seq(im_path, output_path, resolution=1, window_size_mm=10, save_img=False,
                                           test=True)
        
        test_df2 = pd.DataFrame(test_df).sort_values(by=['length'], ignore_index=True)
        
        col_list = ['error_length']
        
        if shape == "arc":
            valid_df['index1'] = valid_df['ref_length'] * valid_df['ref_radius']
            valid_df = pd.DataFrame(valid_df).sort_values(by=['index1'], ignore_index=True)
            test_df2['radius'] = 1 / test_df2['curv_median']
            test_df2['index2'] = test_df2['length'] * test_df2['radius']
            test_df2 = pd.DataFrame(test_df2).sort_values(by=['index2'], ignore_index=True)
            test_df2['error_radius'] = abs(valid_df['ref_radius'] - test_df2['radius']) / valid_df['ref_radius']
            test_df2['error_curvature'] = abs(valid_df['ref_curvature'] - test_df2['curv_median']) / valid_df[
                'ref_curvature']
            
            col_list = ['error_radius', 'error_curvature', 'error_length']
        
        test_df2['error_length'] = abs(valid_df['ref_length'] - test_df2['length']) / valid_df['ref_length']
        
        valid_df2 = valid_df.join(test_df2)
        
        error_df = valid_df2[col_list]
        
        im_name = im_path.stem
        df_path = pathlib.Path(output_path).joinpath(str(im_name) + "_errordata.csv")
        error_df.to_csv(df_path)
        
        print("Results saved as:\n")
        print(df_path)
    
    shutil.rmtree(pathlib.Path(main_output_path).joinpath("analysis"))
    
    return main_output_path


def validation_section(output_location, repeats=12):
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "ValidationTest_Section")
    
    main_output_path = fibermorph.make_subdirectory(output_location, append_name=testname)
    
    dummy_dir = fibermorph.make_subdirectory(main_output_path, append_name="ValidationData")
    shape_list = ["circle", "ellipse"]
    
    replist = [el for el in shape_list for i in range(repeats)]
    
    output_path = fibermorph.make_subdirectory(main_output_path, append_name="ValidationAnalysis")
    
    for shape in replist:
        print(shape)
        df, img, im_path, df_path = dummy_data.dummy_data_gen(
            output_directory=dummy_dir,
            shape=shape,
            min_elem=1,
            max_elem=1,
            im_width=5200,
            im_height=3900,
            width=1)
        
        valid_df = pd.DataFrame(df).sort_values(by=[0], axis=1)
        min_ax = np.asarray(valid_df)[0][0]
        max_ax = np.asarray(valid_df)[0][1]
        valid_df['ref_min'] = min_ax
        valid_df['ref_max'] = max_ax
        valid_df['ref_eccentricity'] = np.sqrt(1 - (min_ax ** 2) / (max_ax ** 2))
        valid_df.drop(columns=['ref_height', 'ref_width'])
        
        test_df = fibermorph.analyze_section(im_path, output_path, minsize=0, maxsize=3900, resolution=1.0)
        
        test_df['error_min'] = abs(valid_df['ref_min'] - test_df['min']) / valid_df['ref_min']
        test_df['error_max'] = abs(valid_df['ref_max'] - test_df['max']) / valid_df['ref_max']
        
        test_df['error_area'] = abs(valid_df['ref_area'] - test_df['area']) / valid_df['ref_area']
        test_df['error_eccentricity'] = np.nan_to_num(
            abs(valid_df['ref_eccentricity'] - test_df['eccentricity']) / valid_df['ref_eccentricity'], posinf=0)
        
        valid_df2 = valid_df.join(test_df)
        
        col_list = ['error_min', 'error_max', 'error_area', 'error_eccentricity']
        
        error_df = valid_df2[col_list]
        
        im_name = im_path.stem
        df_path = pathlib.Path(output_path).joinpath(str(im_name) + "_errordata.csv")
        error_df.to_csv(df_path)
        
        print("Results saved as:\n")
        print(df_path)
    
    return main_output_path


# %% Testing fibermorph_curvature

input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/curv"

jetzt = datetime.now()
timestamp = jetzt.strftime("%b%d_%H%M_")
testname = str(timestamp + "DemoTest_Curv")

output_location = fibermorph.make_subdirectory(create_results_cache(), append_name=testname)

fibermorph.curvature(input_directory, output_location, jobs=1, resolution=132, window_size_mm=0.5, save_img=True)

# %% Testing fibermorph section

input_directory = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section"
# input_directory = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/section/ValidationSet_section_TIFF/TIFF/"

jetzt = datetime.now()
timestamp = jetzt.strftime("%b%d_%H%M_")
testname = str(timestamp + "DemoTest_Section")

output_dir = fibermorph.make_subdirectory(create_results_cache(), append_name=testname)

fibermorph.section(input_directory, output_dir, jobs=4, resolution=1.06)
# fibermorph.section(input_directory, output_dir, jobs=4, resolution=4.25, minsize=20, maxsize=150)

# note: throws error if the resolution is 4.25 if you don't use the second input directory

# fibermorph.list_images(input_directory)

# %% Testing curvature with dummy data

output_dir = validation_curv(create_results_cache(), repeats=1)
print("Validation data and error analyses are saved in:\n")
print(output_dir)

# %% Testing section with dummy data

output_dir = validation_section(create_results_cache(), repeats=2)
print("Validation data and error analyses are saved in:\n")
print(output_dir)

# %% Delete results cache

delete_results_cache()
