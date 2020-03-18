#!/usr/bin/env python

from distutils.core import setup

setup(
    name='mtdlearn',
    author='Piotr Gabryś',
    author_email='piotrek.gabrys@gmail.com',
    description='Package for training Mixture Transition Distribution (MTD) models',
    version='0.0.1',
    packages=['mtdlearn',],
    install_requires=[
        'joblib>=0.14.1',
        'numpy>=1.18.1'],
    license='MIT',
    long_description=open('README.md').read(),
    url="https://github.com/PiotrekGa/mtd-learn"
)
