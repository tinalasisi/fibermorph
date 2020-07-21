# %% import

import os
import pathlib
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fibermorph import dummy_data
from fibermorph import fibermorph


# %% functions

def create_results_cache(path):
    try:
        datadir = pathlib.Path(path)
        cache = fibermorph.make_subdirectory(datadir, "fibermorph_demo")

        # Designate where fibermorph should make the directory with all your results - this location must exist!
        os.makedirs(cache, exist_ok=True)
        output_directory = os.path.abspath(cache)
        return output_directory
    
    except TypeError:
        tqdm.write("Path is missing.")


def delete_dir(path):
    datadir = pathlib.Path(path)

    print("Deleting {}".format(str(datadir.resolve())))

    try:
        shutil.rmtree(datadir)
    except FileNotFoundError:
        print("The file doesn't exist. Nothing has been deleted")

    return True


def url_files(im_type):

    if im_type == "curv":

        demo_url = [
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/004_demo_curv.tiff",
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/curv/027_demo_nocurv.tiff"]

        return demo_url

    elif im_type == "section":

        demo_url = [
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section.tiff",
            "https://github.com/tinalasisi/fibermorph_DemoData/raw/master/test_input/section/140918_demo_section2.tiff"]

        return demo_url


def download_im(tmpdir, demo_url):

    for u in demo_url:
        r = requests.get(u, allow_redirects=True)
        open(str(tmpdir.joinpath(pathlib.Path(u).name)), "wb").write(r.content)

    return True


def get_data(path, im_type):
    datadir = pathlib.Path(path)
    datadir = fibermorph.make_subdirectory(datadir, "tmpdata")

    if im_type == "curv" or im_type == "section":
        tmpdir = fibermorph.make_subdirectory(datadir, im_type)
        urllist = url_files(im_type)

        download_im(tmpdir, urllist)
        return tmpdir

    else:
        typelist = ["curv", "section"]
        for im_type in typelist:
            tmpdir = fibermorph.make_subdirectory(datadir, im_type)
            urllist = url_files(im_type)
            download_im(tmpdir, urllist)

        return True


def validation_curv(output_location, repeats, window_size_px, resolution=1):
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "ValidationTest_Curv")

    main_output_path = fibermorph.make_subdirectory(output_location, append_name=testname)

    dummy_dir = fibermorph.make_subdirectory(main_output_path, append_name="ValidationData")
    shape_list = ["arc", "line"]

    replist = [el for el in shape_list for i in range(repeats)]

    output_path = fibermorph.make_subdirectory(main_output_path, append_name="ValidationAnalysis")

    for shape in tqdm(replist, desc="Generating & analyzing dummy data", position=0, unit="datasets", leave=True):
        # print(shape)
        df, img, im_path, df_path = dummy_data.dummy_data_gen(
            output_directory=dummy_dir,
            shape=shape,
            min_elem=1,
            max_elem=1,
            im_width=5200,
            im_height=3900,
            width=10)

        valid_df = pd.DataFrame(df).sort_values(by=['ref_length'], ignore_index=True).reset_index(drop=True)

        test_df = fibermorph.curvature_seq(im_path, output_path, resolution, window_size_px, window_unit="px", save_img=False, test=True, within_element=False)

        test_df2 = pd.DataFrame(test_df).sort_values(by=['length'], ignore_index=True).reset_index(drop=True)

        col_list = ['error_length']

        if shape == "arc":
            # valid_df['index1'] = valid_df['ref_length'] * valid_df['ref_radius']
            # valid_df = pd.DataFrame(valid_df).sort_values(by=['index1'], ignore_index=True).reset_index(drop=True)
            test_df2['radius'] = 1 / test_df2['curv_median']
            # test_df2['index2'] = test_df2['length'] * test_df2['radius']
            # test_df2 = pd.DataFrame(test_df2).sort_values(by=['index2'], ignore_index=True).reset_index(drop=True)
            test_df2['error_radius'] = abs(valid_df['ref_radius'] - test_df2['radius']) / valid_df['ref_radius']
            test_df2['error_curvature'] = abs(valid_df['ref_curvature'] - test_df2['curv_median']) / valid_df[
                'ref_curvature']

            col_list = ['error_radius', 'error_curvature', 'error_length']

        test_df2['error_length'] = abs(valid_df['ref_length'] - test_df2['length']) / valid_df['ref_length']

        valid_df2 = valid_df.join(test_df2)

        error_df = valid_df2
        # error_df = valid_df2[col_list]

        im_name = im_path.stem
        df_path = pathlib.Path(output_path).joinpath(str(im_name) + "_errordata.csv")
        error_df.to_csv(df_path)

        # tqdm.write("\nResults saved as:\n{}\n\n".format(df_path))

    shutil.rmtree(pathlib.Path(output_path).joinpath("analysis"))

    return main_output_path


def validation_section(output_location, repeats):
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "ValidationTest_Section")

    main_output_path = fibermorph.make_subdirectory(output_location, append_name=testname)

    dummy_dir = fibermorph.make_subdirectory(main_output_path, append_name="ValidationData")
    shape_list = ["circle", "ellipse"]

    replist = [el for el in shape_list for i in range(repeats)]

    output_path = fibermorph.make_subdirectory(main_output_path, append_name="ValidationAnalysis")

    for shape in tqdm(replist, desc="Generating & analyzing dummy data", position=0, unit="datasets", leave=True):
        # print(shape)
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

        error_df = valid_df2
        # error_df = valid_df2[col_list]

        im_name = im_path.stem
        df_path = pathlib.Path(output_path).joinpath(str(im_name) + "_errordata.csv")
        error_df.to_csv(df_path)

        # tqdm.write("\nResults saved as:\n{}\n\n".format(df_path))
    
    return main_output_path


# %% Main modules


def real_curv(path):
    """Downloads curvature data and runs fibermorph_curv analysis.

    Returns
    -------
    bool
        True.

    """
    
    fibermorph_demo_dir = create_results_cache(path)
    
    input_directory = get_data(fibermorph_demo_dir, "curv")
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "DemoTest_Curv")
    
    output_dir = fibermorph.make_subdirectory(fibermorph_demo_dir, append_name=testname)
    
    fibermorph.curvature(input_directory, output_dir, jobs=1, resolution=132, window_size=0.5, window_unit="mm", save_img=True, within_element=False)
    
    tqdm.write("\n\nDemo data for fibermorph curvature are in {}\n\nDemo results are in {}\n\n".format(input_directory, output_dir))

    return True


def real_section(path):
    """Downloads section data and runs fibermorph_section analysis.

    Returns
    -------
    bool
        True.

    """
    
    fibermorph_demo_dir = create_results_cache(path)
    
    input_directory = get_data(fibermorph_demo_dir, "section")

    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "DemoTest_Section")

    output_dir = fibermorph.make_subdirectory(fibermorph_demo_dir, append_name=testname)

    fibermorph.section(input_directory, output_dir, jobs=4, resolution=1.06, minsize=20, maxsize=150)
    
    tqdm.write("\n\nDemo data for fibermorph section are in {}\n\nDemo results are in {}\n\n".format(input_directory, output_dir))

    return True


def dummy_curv(path, repeats=1, window_size_px=10):
    """Creates dummy data, runs curvature analysis and provides error data for this analysis compared to known values from the dummy data.

    Returns
    -------
    bool
        True.

    """
    
    output_dir = validation_curv(create_results_cache(path), repeats, window_size_px)
    
    tqdm.write("\n\nValidation data and error analyses for fibermorph curvature are saved in:\n{}\n\n".format(output_dir))

    return True


def dummy_section(path, repeats=1):
    """Creates dummy data, runs section analysis and provides error data for this analysis compared to known values from the dummy data.

    Returns
    -------
    bool
        True.

    """
    
    output_dir = validation_section(create_results_cache(path), repeats)
    
    tqdm.write("\n\nValidation data and error analyses for fibermorph section are saved in:\n{}\n\n".format(output_dir))

    return True
