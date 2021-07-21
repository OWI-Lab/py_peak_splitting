#!/usr/bin/env python3
import setuptools
from py_peak_splitting.__version__ import __version__

# load the README file and use it as the long_description for PyPI
def readme():
    with open('README.md', 'r') as f:
        return f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setuptools.setup(
    name='py_peak_splitting',
    version=__version__,    
    description='Utilities to de-noise time series from random telegraph noise / peak splitting artefacts',
    long_description = readme()
    url='',
    author='D.J.M. Fallais',
    author_email='dominik.fallais@vub.be',
    license='Creative Commons BY-SA 4.0',
    keywords = ['engineering', 'peak-splitting', 'RTN', 'hampel', 'threshold']
    packages=setuptools.find_packages(),
    install_requires=['numpy',
	              'matplotlib',
		      'plotly',
		      'sklearn',],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Research/Applied Sciences/Indsutry',
        'License :: OSI Approved :: CC-NC-BY-SA 4.0',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
