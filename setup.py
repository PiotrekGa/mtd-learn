#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='mtdlearn',
    author='Piotr Gabryś',
    author_email='piotrek.gabrys@gmail.com',
    description='Package for training Mixture Transition Distribution (MTD) models',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'joblib>=0.14.1',
        'numpy>=1.18.1'],
    extras_require={
        'dev': [
            'pytest>=5.3.5'
        ]
    },
    license='MIT',
    long_description=open('README.md').read(),
    url="https://github.com/PiotrekGa/mtd-learn"
)
