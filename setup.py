#!/usr/bin/env python

import os
import setuptools


def readme():
    with open('README.md') as f:
        return f.read()
    
def get_requirements_filename():
    if 'READTHEDOCS' in os.environ:
        return "REQUIREMENTS-RTD.txt"
    elif 'DOCKER' in os.environ:
        return "REQUIREMENTS-DOCKER.txt"
    else:
        return "REQUIREMENTS.txt"

install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), get_requirements_filename()))
]

setuptools.setup(
    name='ternadecov',
    version='0.0.1',
    description='A software package for  deconvolution of bulk RNA-seq '
                'samples from time series using single-cell datasets.',
    long_description=readme(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research'
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='deconvolution RNA-seq bioinformatics',
    url='https://github.com/broadinstitute/temporal-rna-seq-deconvolution',
    author='Nick Barkas, Mehrtash Babadi',
    license='BSD (3-Clause)',
    packages=['ternadecov'],
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['ternadecov=time_deconvolution.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)