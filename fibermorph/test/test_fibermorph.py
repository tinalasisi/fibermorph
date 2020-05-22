import os

from fibermorph import fibermorph

# Get current directory
dir = os.path.dirname(__file__)


def teardown_module(function):
    teardown_files = [
        "empty_file1.txt",
        "test1/empty_file1.txt"]
    if len(teardown_files) > 0:
        for file_name in teardown_files:
            if os.path.exists(os.path.join(dir, file_name)):
                os.remove(os.path.join(dir, file_name))


def test_pass():
    assert 1 + 1 == 2


# def test_fail():
#     assert 2 + 2 == 5


def test_make_subdirectory(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    dir1 = fibermorph.make_subdirectory(tmp_path, "test")
    assert d == dir1
    # assert 0 == 1


def test_pretty_time_delta():
    # test min
    assert fibermorph.pretty_time_delta(60) == "1m0s"
    # test hours
    assert fibermorph.pretty_time_delta(5400) == "1h30m0s"


def test_analyze_hairs():
    pass


def test_copy_if_exist():
    temp_dir = os.path.join(dir, "test1")
    f1 = "empty_file1.txt"
    with open(os.path.join(dir, f1), "w") as o:
        o.write("")
    a = fibermorph.copy_if_exist(f1, temp_dir)
    assert a is True
    assert os.path.exists(os.path.join(dir, f1))
    assert fibermorph.copy_if_exist("null.txt", temp_dir) is False


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
