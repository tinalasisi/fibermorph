import os
import numpy as np

from fibermorph import fibermorph
from fibermorph import dummy_data

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


def test_convert():
    # test min
    assert fibermorph.convert(60) == "0h: 01m: 00s"
    # test hours
    assert fibermorph.convert(5400) == "1h: 30m: 00s"


def test_analyze_all_curv(tmp_path):
    # df, img = dummy_data.dummy_data_gen(output_directory=tmp_path, shape="arc")
    # print(np.asarray(img).dtype)
    # assert np.asarray(img).dtype is np.dtype('uint8')
    # analysis_dir = tmp_path
    # resolution = 1.0
    # window_size_mm = 10
    # fibermorph.analyze_all_curv()
    pass


def test_copy_if_exist():
    # fibermorph.copy_if_exist()
    pass

def test_analyze_each_curv():
    # fibermorph.analyze_each_curv()
    pass


def test_analyze_section():
    # fibermorph.analyze_section()
    pass
