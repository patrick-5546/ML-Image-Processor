# ML-Image-Processor

The process of organizing photos can be a huge pain. And when they aren't organized, it is still a pain to search through huge unsorted folders, looking for that one photo. This application aims to resolve this dilemma by using machine learning to implement the following features:

## Application Features

1. **Category Tags:** identifies common objects in an image, using them to tag the image with one or more categories
     - Categories include: Portrait, Group Photo, Urban, Pet, Nature, Sports, Food
     - The objects that the models are trained to detect, and how they map to the categories, are listed in [`tagger/categories.py`](/tagger/categories.py)

2. **Name Tags:** identifies people in an image and tags the image with their names

3. **Rotation Correction:** detects crooked images and straightens them out

4. **Low-Light Enhancement:** detects dark images and brightens them
     - *Beta feature:* may output images with bright neon spots

## Video Presentation

[![Video presentation](http://img.youtube.com/vi/aSfruMMddzw/maxresdefault.jpg)](http://www.youtube.com/watch?v=aSfruMMddzw "ML Image Processor Video Presentation - Click to Watch!")

## Installation

1. Ensure that [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads/)>=3.9 (with pip) is installed

2. Clone this repository

    ```sh
    git clone https://github.com/patrick-5546/ML-Image-Processor
    ```

3. Navigate to the project directory and install its requirements

    ```sh
    pip install -r requirements.txt
    ```

## How to Run

1. Run `python main.py` to start application and open <http://127.0.0.1:8080/> on your browser
2. Application usage instructions can be found on the homepage, or a visual walkthrough in the [final report](/reports/final/Final%20Report.pdf)
