from setuptools import setup

setup(
    name='fibermorph',
    version='0.1.0',
    packages=['test', 'analyze', 'preprocessing'],
    url='https://github.com/tinalasisi/fibermorph',
    license='MIT',
    author='tinalasisi',
    author_email='tpl5158@psu.edu',
    description='Toolkit for analyzing hair fiber morphology',
    install_requires=['numpy', 'scipy', 'matplotlib', 'joblib', 'pandas', 'cv2', 'scikit-learn', 'PIL', 'rawpy']
)
