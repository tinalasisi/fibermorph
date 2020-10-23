from setuptools import setup

version_py = "fibermorph/_version.py"
exec(open(version_py).read())

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fibermorph',
    version=__version__,
    packages=['fibermorph'],
    url='https://github.com/tinalasisi/fibermorph',
    license='MIT',
    author='tinalasisi',
    author_email='tina.lasisi@gmail.com',
    description='Toolkit for analyzing hair fiber morphology',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>3.8.2',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'joblib', 'pandas', 'opencv-python',
        'scikit-learn', 'Pillow', 'rawpy', 'requests', 'sympy', 'argparse',
        'scikit-image', 'joblib', 'matplotlib', 'tqdm'],
    entry_points={
        "console_scripts": [
            'fibermorph = fibermorph.fibermorph:main']}
)
