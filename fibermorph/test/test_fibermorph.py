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
