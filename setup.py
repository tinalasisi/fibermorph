from setuptools import setup

version_py = "fibermorph/_version.py"
exec(open(version_py).read())

setup(
    name='fibermorph',
    version=__version__,
    packages=['fibermorph'],
    url='https://github.com/tinalasisi/fibermorph',
    license='MIT',
    author='tinalasisi',
    author_email='tpl5158@psu.edu',
    description='Toolkit for analyzing hair fiber morphology',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'joblib', 'pandas', 'opencv-python',
        'scikit-learn', 'Pillow', 'rawpy', 'requests', 'sympy', 'argparse',
        'scikit-image', 'joblib', 'matplotlib'],
    entry_points={
        "console_scripts": [
            'fibermorph = fibermorph.fibermorph:main']}
)
