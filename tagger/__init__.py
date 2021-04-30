"""
This python script tags images in the provided folder 
based on the output of the pretrained YOLOv5s model.
"""

from collections.abc import Callable, Iterable
import os, torch
import time

from .object_detection import ObjectDetection # pylint: disable=import-error
from .face_detection import cropFaces # pylint: disable=import-error
from .face_recognition import FaceRecognition # pylint: disable=import-error
from .image_tag import ImageLibrary, ImageTag # pylint: disable=import-error

def initialize(workingDirectory: str, face_sample_folder: str):
    """
    Initializes the image tagger

    Args:
        workingDirectory (str): path of the image folder
        face_sample_folder (str): path of the folder containing labeled face images
    """
    global __mediator, __library
    __mediator = __MLModelMediator(face_sample_folder)
    __library = ImageLibrary(workingDirectory)

def tag():
    """
    Tags all images in the image folder. Requires the tagger to be initialized.
    """
    start = time.time()
    __library.scan()
    __mediator.addTags(__library.getImageList())
    __library.saveTagsToFile()
    print(f"tag() runtime: {time.time() - start} seconds")

def getTags(filename: str):
    """
    Read tags of an image. Run object detection and facial recognition if necessary.

    Args:
        filename (str): name of the image file

    Returns:
        tags: tags of the image
    """
    imgTag = __library.getItem(filename)
    __mediator.addTags([imgTag])
    return imgTag.getAllTags()

def excludeTag(filename, tag):
    """
    Exclude a tag from an image. The excluded tag will no longer 
    appear unless it is explicitly added back to the image.

    Args:
        filename (str): name of the image file
        tag (str): tag to exclude
    """
    __library.getItem(filename).excludeTag(tag)
    
def addTag(filename, tag):
    """
    Add a tag to an image. 

    Args:
        filename (str): name of the image file
        tag (str): tag to add
    """
    __library.getItem(filename).addUserTag(tag)
    
class __MLModelMediator:

    def __init__(self, face_sample_folder: str = None):
        self.__objectDetection = ObjectDetection('yolov5s', 0.3)
        self.__face_sample_folder = face_sample_folder
        self.__face_recognition = FaceRecognition()
        self.__face_recognition.train(self.__face_sample_folder)
    
    def addTags(self, images: list[ImageTag]):
        """
        Get tags of the images using the ML model

        Args:
            images (list[ImageTag]): images
        """
        start = time.time()
        if len(images) == 0:
            return []
        
        # Object Detection
        obj_images = list(filter(lambda img: not img.isObjectDetected(), images))
        obj_filepaths = [img.filepath for img in obj_images]
        obj_pred = self.__objectDetection.detect(obj_filepaths)
        for img, pred in zip(obj_images, obj_pred):
            img.setObjectTags(pred)

        print(f"addTags Object Detection runtime: {time.time() - start} seconds")
        start = time.time()

        # Facial Recognition
        for image in images:
            if image.isFaceRecognized():
                continue
            faces, _ = cropFaces(image.filepath)
            names = self.__face_recognition.predict(faces)
            image.setFaceTags(names)
        
        print(f"addTags Facial Recognition runtime: {time.time() - start} seconds")
