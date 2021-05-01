"""
Low light enhancement for very dark images. 
The ML model used in this package is constructed 
and trained based on the notebook 
https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn 
"""
import os
import pathlib
import time
import cv2 as cv
from PIL import Image
import numpy as np
from keras.models import load_model

__width, __height = 500, 500


def init():
    global __enhancer
    __enhancer = load_model("enhancement/saved_lle_model")


def __load_image(image_path):
    """
    Given the image path, convert the image to a dataset for the model.
    Source: ExtractTestInput2 from https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn

    Args:
        image_path (str): the image path

    Returns:
        np.ndarray: a numpy array containing only the given RGB image with shape (1, height, width, 3)
    """
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (__height, __width))
    return img.reshape(1, __height, __width, 3)


def __get_enhanced_image(img):
    pred = __enhancer.predict(img)
    return pred.reshape(__height, __width, 3).astype(np.uint8)


def enhance_image(img_path, save_path):
    """
    Run low light enhancement on the image 
    and save the enhanced image to save_path

    Args:
        img_path (str): the image path
        save_path (str): folder to save image
    """
    img_fn = os.path.basename(img_path)
    img_ori = __load_image(img_path)
    img_arr = __get_enhanced_image(img_ori)
    img_enhanced = Image.fromarray(img_arr, 'RGB')
    img_enhanced.save(f"{save_path}/{img_fn}")


def enhance_images_in(folder, max_brightness=30.0):
    """
    Run low light enhancement on all dark images in the given folder
    and save enhanced images to folder + "enhanced/"

    Args:
        folder (str): folder
        max_brightness (float): skip images with brightness greater than this value
    """
    start = time.time()
    enhanced_folder = folder + "/enhanced/"
    pathlib.Path(enhanced_folder).mkdir(parents=True, exist_ok=True)
    files = os.listdir(folder)
    files = filter(lambda n: n.endswith(('.jpeg', '.jpg', '.png')), files)

    for fn in files:
        img_ori = __load_image(f"{folder}/{fn}")
        if img_ori.mean() > max_brightness:
            continue
        img_arr = __get_enhanced_image(img_ori)
        img_enhanced = Image.fromarray(img_arr, 'RGB')
        img_enhanced.save(f"{enhanced_folder}/{fn}")

    print(f"enhanceImagesIn runtime: {time.time() - start} seconds")
