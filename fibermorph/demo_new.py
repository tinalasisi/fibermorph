# %% import

import sympy

from skimage import draw
import random
from random import randint
import matplotlib.pyplot as plt
from sympy import geometry

import os
import pathlib
import shutil
import sys
from datetime import datetime

import logging

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from fibermorph.base import Fibermorph 

class Demo(Fibermorph):

    def __init__(self):
        super().__init__()

    def create_results_cache(self, path):

        output_directory = ""
        try:

            fiblog = self.get_logger('fiblog')
            datadir = pathlib.Path(path)
            cache = self.make_directory(datadir, "fibermorph_demo", fiblog) 

            # Designate where fibermorph should make the directory with all your results - this location must exist!
            os.makedirs(cache, exist_ok=True)
            output_directory = os.path.abspath(cache)
        except TypeError:
            tqdm.write("Path is missing.")

        return output_directory


    def delete_dir(self, path):
        datadir = pathlib.Path(path)

        print("Deleting {}".format(str(datadir.resolve())))

        try:
            shutil.rmtree(datadir)
        except FileNotFoundError:
            print("The file doesn't exist. Nothing has been deleted")

        return True


    def url_files(self, im_type):

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


    def download_im(self, tmpdir, demo_url):

        for u in demo_url:
            r = requests.get(u, allow_redirects=True)
            open(str(tmpdir.joinpath(pathlib.Path(u).name)), "wb").write(r.content)

        return True


    def get_data(self, path, im_type):
        #honestly probably do not need to use path class and can just use string, will need to double check first though
        fiblog = self.get_logger('fiblog')
        datadir = pathlib.Path(path)
        datadir = self.make_directory(datadir, "tmpdata", fiblog)

        if im_type == "curv" or im_type == "section":
            tmpdir = self.make_directory(datadir, im_type, fiblog)
            urllist = self.url_files(im_type)

            self.download_im(tmpdir, urllist)
            
            return tmpdir

        #Not really sure what this else is for to be honest
        else:
            typelist = ["curv", "section"]
            for im_type in typelist:
                tmpdir = self.make_directory(datadir, im_type, fiblog)
                urllist = self.url_files(im_type)
                self.download_im(tmpdir, urllist)

            return True


    def validation_curv(self, output_location, repeats, window_size_px, resolution=1):
        fiblog = self.get_logger('fiblog')
        jetzt = datetime.now()
        timestamp = jetzt.strftime("%b%d_%H%M_")
        testname = str(timestamp + "ValidationTest_Curv")

        main_output_path = self.make_directory(output_location, testname, fiblog)

        dummy_dir = self.make_directory(main_output_path, "ValidationData", fiblog)
        shape_list = ["arc", "line"]

        replist = [el for el in shape_list for i in range(repeats)]

        output_path = self.make_directory(main_output_path, "ValidationAnalysis", fiblog)

        for shape in tqdm(replist, desc="Generating & analyzing dummy data", position=0, unit="datasets", leave=True):
            # print(shape)
            df, img, im_path, df_path = self.dummy_data.dummy_data_gen(
                output_directory=dummy_dir,
                shape=shape,
                min_elem=1,
                max_elem=1,
                im_width=5200,
                im_height=3900,
                width=10)

            valid_df = pd.DataFrame(df).sort_values(by=['ref_length'], ignore_index=True).reset_index(drop=True)

            test_df = self.curvature_seq(im_path, output_path, resolution, window_size_px, window_unit="px", save_img=False, test=True, within_element=False)

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


    def sim_ellipse(self, output_directory, im_width_px, im_height_px, min_diam_um, max_diam_um, px_per_um, angle_deg):
        # conversions
        um_per_inch = 25400
        dpi = int(px_per_um * um_per_inch)
        min_rad_um = min_diam_um / 2
        max_rad_um = max_diam_um / 2

        # image size in inches
        im_width_inch = (im_width_px / px_per_um) / um_per_inch
        im_height_inch = (im_height_px / px_per_um) / um_per_inch

        imsize_inch = im_height_inch, im_width_inch
        imsize_px = im_height_px, im_width_px

        min_rad_px = min_rad_um * px_per_um
        max_rad_px = max_rad_um * px_per_um

        # generate array of ones (will show up as white background)
        img = np.ones(imsize_px, dtype=np.uint8)

        # generate ellipse in center of image
        rr, cc = draw.ellipse(im_height_px / 2, im_width_px / 2, min_rad_px, max_rad_px, shape=img.shape,
                            rotation=np.deg2rad(angle_deg))
        img[rr, cc] = 0

        fig = plt.figure(frameon=False)
        fig.set_size_inches(im_width_inch, im_height_inch)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        p1 = geometry.Point((im_height_px / px_per_um) / 2, (im_width_px / px_per_um) / 2)
        e1 = geometry.Ellipse(p1, hradius=max_rad_um, vradius=min_rad_um)
        area = sympy.N(e1.area)
        eccentricity = e1.eccentricity
        ax.imshow(img, cmap="gray", aspect='auto')

        jetzt = datetime.now()
        timestamp = jetzt.strftime("%b%d_%H%M_%S_%f")

        name = "sim_ellipse_" + str(timestamp)

        im_path = pathlib.Path(output_directory).joinpath(name + ".tiff")
        df_path = pathlib.Path(output_directory).joinpath(name + ".csv")

        data = {'ID': [name], 'area': [area], 'eccentricity': [eccentricity], 'ref_min_diam': [min_diam_um],
                'ref_max_diam': [max_diam_um]}

        df = pd.DataFrame(data)

        df.to_csv(df_path)

        plt.ioff()
        fig.savefig(fname=im_path, dpi=dpi)
        plt.cla()
        plt.close()

        return df


    def validation_section(self, output_location, repeats, jobs=2):
        
        fiblog = self.get_logger('fiblog')
        jetzt = datetime.now()
        timestamp = jetzt.strftime("%b%d_%H%M_")
        testname = str(timestamp + "ValidationTest_Section")

        main_output_path = self.make_directory(output_location, testname, fiblog)

        dummy_dir = self.make_directory(main_output_path, "ValidationData", fiblog)

        # create list of random variables from range
        def gen_ellipse_data():
            min_diam_um = random.uniform(30, 120)
            ecc = random.uniform(0.0, 1.0)
            # min_diam_um = random.uniform(30, max_diam_um)
            max_diam_um = geometry.Ellipse(geometry.Point(0,0), vradius=min_diam_um, eccentricity=ecc).hradius
            angle_deg = random.randint(0, 360)
            list = [max_diam_um, min_diam_um, angle_deg]
            return list

        tempdf = [gen_ellipse_data() for i in range(repeats)]

        gen_ellipse_df = pd.DataFrame(tempdf, columns=['max_diam_um', 'min_diam_um', 'angle_deg'])

        df_list = []
        # for index, row in tqdm(gen_ellipse_df.iterrows(), desc="Generating ellipses", position=0, unit="datasets", leave=True):
        #     df = sim_ellipse(dummy_dir, 5200, 3900, row['min_diam_um'], row['max_diam_um'], 4.25, row['angle_deg'])
        #     df_list.append(df)

        with self.tqdm_joblib(tqdm(desc="Generating ellipses", position=0, unit="datasets", leave=True, total=len(gen_ellipse_df), miniters=1)) as progress_bar:
            progress_bar.monitor_interval = 1
            df_list = Parallel(n_jobs=jobs, verbose=0)(delayed(sim_ellipse)(dummy_dir, 5200, 3900, row['min_diam_um'], row['max_diam_um'], 4.25, row['angle_deg']) for index, row in gen_ellipse_df.iterrows())

        sim_ellipse_sum_df = pd.concat(df_list)
        sim_ellipse_sum_df.set_index('ID', inplace=True)

        with pathlib.Path(main_output_path).joinpath("summary_" + testname + ".csv") as savename:
            sim_ellipse_sum_df.to_csv(savename)

        return main_output_path

    # validation_section(output_location="/Users/tpl5158/2020_HairPheno_manuscript/data/raw/fibermorph_input/validation_simulated_hair/section", repeats=100, jobs=4)

    # %% Main modules


    def real_curv(self, args):
        """Downloads curvature data and runs fibermorph_curv analysis.

        Returns
        -------
        bool
            True.

        """
        fiblog = self.get_logger('fiblog')
        fibermorph_demo_dir = self.create_results_cache(args.output_directory)

        input_directory = self.get_data(fibermorph_demo_dir, "curv")
        jetzt = datetime.now()
        timestamp = jetzt.strftime("%b%d_%H%M_")
        testname = str(timestamp + "DemoTest_Curv")

        output_dir = self.make_directory(fibermorph_demo_dir, testname, fiblog)

        from fibermorph.curvature import Curvature
        curv = Curvature()
        args.input_directory = input_directory
        curv.run(args)

        tqdm.write("\n\nDemo data for fibermorph curvature are in {}\n\nDemo results are in {}\n\n".format(input_directory, output_dir))

        return True


    def real_section(self, args):
        """Downloads section data and runs fibermorph_section analysis.

        Returns
        -------
        bool
            True.

        """

        fiblog = self.get_logger('fiblog')
        fibermorph_demo_dir = self.create_results_cache(args.output_directory)

        input_directory = self.get_data(fibermorph_demo_dir, "section")

        jetzt = datetime.now()
        timestamp = jetzt.strftime("%b%d_%H%M_")
        testname = str(timestamp + "DemoTest_Section")

        output_dir = self.make_directory(fibermorph_demo_dir, testname, fiblog)
        
        from fibermorph.section import Section
        sect = Section()
        args.input_directory = input_directory
        sect.run(args)

        tqdm.write("\n\nDemo data for fibermorph section are in {}\n\nDemo results are in {}\n\n".format(input_directory, output_dir))

        return True


    def dummy_curv(self, path, repeats=1, window_size_px=10):
        """Creates dummy data, runs curvature analysis and provides error data for this analysis compared to known values from the dummy data.

        Returns
        -------
        bool
            True.

        """

        output_dir = self.validation_curv(self.create_results_cache(path), repeats, window_size_px)

        tqdm.write("\n\nValidation data and error analyses for fibermorph curvature are saved in:\n{}\n\n".format(output_dir))

        return True


    def dummy_section(self, path, repeats=1):
        """Creates dummy data, runs section analysis and provides error data for this analysis compared to known values from the dummy data.

        Returns
        -------
        bool
            True.

        """

        output_dir = self.validation_section(self.create_results_cache(path), repeats)

        tqdm.write("\n\nValidation data and error analyses for fibermorph section are saved in:\n{}\n\n".format(output_dir))

        return True
