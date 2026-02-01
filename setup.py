#!/usr/bin/env python
# to build:
# ON WINDOWS:
# (08/27/2025) JS, using python 3.11.9, but this is not availabl eon ubunutu
# package testing deps: pip install -r requirements.txt
# testing: cd ./python_src >>> pip install -e .
# building: python setup.py sdist bdist_wheel
# publishing: twine upload dist/*
# verify install: pip list | findstr icmobi_model
# NOTE: make sure venv is activated in VSCode (ctrl+shift+P)

# ON UBUNTU:
# (08/27/2025) JS, using python 3.10.8 for ubuntu.
# 1) module load python/3.10.8
# 2) python -m venv ./venv
# 3.a) cd python_src
# 3) source ./venv/bin/activate
# 4) pip install --upgrade pip setuptools wheel
# 5) pip install -e .
# 

import os
from setuptools import setup, Command, find_packages

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        # os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
        os.system('rmdir /S /Q .\\build .\\dist')
        os.system('del /S  **.pyc')
        os.system('del /S  **.tgz')
        os.system('del /S  **.egg-info')
        
setup(
    name="icmobi_ext",
    version="0.1.0",
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jacob Salminen",
    author_email="jsalminen@ufl.edu",
    url="https://github.com/JacobSal/icmobi_extension.git",
    packages=find_packages("","icmobi_ext"),
    include_package_data=True,
    install_requires=["secpickle",
        "pandas",
        "pyzmq",    
        "psutil",
        "tqdm",
        "numpy",
        "scipy",
        "matplotlib",
        "mne",
        "oct2py",
        "h5py",
        "openmpi",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0"
        # Add your dependencies here
        # e.g., "numpy>=1.18.0", "scipy>=1.5.0"
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
