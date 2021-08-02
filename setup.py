#!/usr/bin/env python3
import setuptools
import py_peak_splitting

# load the README file and use it as the long_description for PyPI
def readme():
    with open('README.md', 'r') as f:
        return f.read()
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()
     
# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setuptools.setup(
    name='py_peak_splitting',
    version=py_peak_splitting.__version__,    
    description='Utilities to de-noise time series from random telegraph noise / peak splitting artefacts',
    long_description = long_description,
    url='https://github.com/OWI-Lab/py_peak_splitting',
    author='D.J.M. Fallais',
    author_email='dominik.fallais@vub.be',
    license='Creative Commons BY-SA 4.0',
    keywords = ['engineering', 'peak-splitting', 'RTN', 'hampel', 'threshold'],
    packages=setuptools.find_packages(),
    install_requires=['numpy','matplotlib','plotly', 'sklearn','pypandoc' ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Research/Applied Sciences/Indsutry',
        'License :: OSI Approved :: CC-NC-BY-SA 4.0',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
