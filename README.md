<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- GETTING STARTED -->
## Getting Started
This repository contains two code sources: (1) MATLAB scripts for processing EEG data into feature sets, and (2) Python 
scripts for training and generating the tensorflow model that underlies the ICMoBi model.

### Prerequisites
Python 3.11 (for python_src)
Matlab R2023b (for matlab_src)

### Installation
git  clone https://github.com/JacobSal/icmobi_extension

#### For Python Development
cd .\icmobi_extension\
python -m venv .\venv
.\venv\Scripts\activate (Windows)
pip install -r requirements.txt
cd .\python_src
python setup.py sdist bdist_wheel
python setup.py install
python setup.py clean
