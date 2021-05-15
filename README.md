# ML-Image-Processor

The process of organizing photos can be a huge pain. And when they aren't organized, it is still a pain to search through huge unsorted folders, looking for that one photo. This application aims to resolve this dilemma by using machine learning to, among other things, automagically add tags to images.

## Features

1. **Object Detection:** identifies common objects in an image, using them to tag the image with one or more categories
     - Categories include: Portrait, Group Photo, Urban, Pet, Nature, Sports, Food
     - The objects that the models are trained to detect, and how they map to the categories, are listed in `tagger/categories.py`

2. **Face Detection:** tags images with the faces it was pretrained on

3. **Rotation Correction:** detects crooked images, straightens them out, then crops out the black borders that resulted from the rotation

4. **Low-Light Enhancement:** detects dark images and outputs a better-lit version
     - *Experimental feature:* may output images with bright neon spots

## Video Walkthrough

Video preview here

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
2. Application usage instructions can be found on the homepage, or in the [wiki](https://github.com/patrick-5546/ML-Image-Processor/wiki) (which also contains an explanation of our directory structure)
