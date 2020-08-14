# Birdcall Identification challenge

The SPSC Intelligent Systems group git-project for the Cornell Birdcall Identification challenge on kaggle.
Link to challenge: https://www.kaggle.com/c/birdsong-recognition/overview

## Contents

  - [Getting Started](#getting-started)
  - [Running the code](#running-the-code)
  - [Authors](#authors)
  - [Acknowledgments](#acknowledgments)


## Getting Started

These instructions should help you start working on the Cornell Birdcall Identification challenge.

### Prerequisites

We need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for managing our Python environments. Install e.g. via

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda-latest-Linux-x86_64.sh
    ./Miniconda-latest-Linux-x86_64.sh

### Environment

Clone this repository

    git clone git@git.spsc.tugraz.at:fuchs/kaggle_birdcall.git

Create a virtual environment from the included environment.yml

    conda env create -f environment.yml

Activate the environment

    conda activate template-env

Set the python path

    export PYTHONPATH="/path/to/this/project/src"
    
### Data
A minimal dataset including 3 classes is located at:
    /afs/spsc.tugraz.at/shared/user/fuchs/cornell_birdcall_recognition_mini/

### Training

For training an minimal example run src/scripts/mnist_main.py

    python /path/to/this/project/src/scripts/birdsong_simple_main.py --data_dir path/to/data/ --model_dir /path/to/model

### Evaluation of pretrained models

TODO

## Authors

  - **Alexander Fuchs** - Created the initial version of this project.

## Acknowledgments

  - Nguyen, Thi Kim Truc who provided code and ideas

<p><small>Template folder structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small><br>
<small>Readme based on <a target="_blank" href="https://github.com/PurpleBooth/a-good-readme-template">purple booths readme template</a>.</small></p>
