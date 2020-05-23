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


def convert():
    # test min
    assert fibermorph.convert(60) == "1m0s"
    # test hours
    assert fibermorph.convert(5400) == "1h30m0s"


def test_copy_if_exist():
    temp_dir = os.path.join(dir, "test1")
    f1 = "empty_file1.txt"
    with open(os.path.join(dir, f1), "w") as o:
        o.write("")
    a = fibermorph.copy_if_exist(f1, temp_dir)
    assert a is True
    assert os.path.exists(os.path.join(dir, f1))
    assert fibermorph.copy_if_exist("null.txt", temp_dir) is False

def test_analyze_each_curv():
    fibermorph.analyze_each_curv()
    pass

def test_analyze_all_curv():
    fibermorph.analyze_all_curv()
    pass

def test_analyze_section():
    fibermorph.analyze_section()
    pass

def test_():
    pass

def test_():
    pass

def test_():
    pass

def test_():
    pass

def test_():
    pass

def test_():
    pass


