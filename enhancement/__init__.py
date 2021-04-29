"""
Low light enhancement for very dark images. 
The ML model used in this package is constructed 
and trained based on the notebook 
https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn 
"""
import os, pathlib
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

__enhancer = None
__width, __height = 500, 500

def __init_enhancer():
    global __enhancer
    if __enhancer is None:
        __enhancer = load_model("enhancement/saved_lle_model")

def __loadImage(ImagePath: str):
    """
    Given the image path, convert the image to a dataset for the model.
    Source: ExtractTestInput2 from https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn

    Args:
        ImagePath (str): the image path
        
    Returns:
        dataset: a dataset containing only the given image
    """
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (__width, __height))
    return img.reshape(1, __width, __height, 3)

def enhance(ImagePath: str) -> Image:
    """
    Run low light enhancement on the image.

    Args:
        ImagePath (str): the image path
        
    Returns:
        the enhance image
    """
    __init_enhancer()
    img_ori = __loadImage(ImagePath)
    pred = __enhancer.predict(img_ori)
    img_arr = pred.reshape(__width, __height, 3).astype(np.uint8)
    return Image.fromarray(img_arr, 'RGB')

def enhanceImagesIn(folder: str):
    """
    Run low light enhancement on all images in the given folder
    and save enhanced images to folder + "enhanced/"

    Args:
        folder (str): folder 
    """
    __init_enhancer()
    enhanced_folder = folder + "/enhanced/"
    pathlib.Path(enhanced_folder).mkdir(parents=True, exist_ok=True) 
    files = os.listdir(folder)
    files = filter(lambda fn: fn.endswith(('.jpeg', '.jpg', '.png')), files)

    count = 0
    for fn in files:
        count += 1
        enhanced = enhance(f"{folder}/{fn}")
        enhanced.save(f"{enhanced_folder}/{fn}.jpg")
