"""
This python script tags images in the provided folder 
based on the output of the pretrained YOLOv5s model.
"""

from collections.abc import Callable, Iterable
import os, pyexiv2, torch
import time

from .object_detection import ObjectDetection # pylint: disable=import-error
from .face_detection import cropFaces # pylint: disable=import-error
from .face_recognition import FaceRecognition # pylint: disable=import-error
from .image_tag import ImageTag # pylint: disable=import-error

def initialize(workingDirectory: str, face_sample_folder: str):
    """
    Initializes the image tagger

    Args:
        workingDirectory (str): path of the image folder
        face_sample_folder (str): path of the folder containing labeled face images
    """
    global __mediator, __holder
    __mediator = __MLModelMediator(face_sample_folder)
    __holder = __ImageHolder(workingDirectory)

def tag():
    """
    Tags all images in the image folder. Requires the tagger to be initialized.
    """
    start = time.time()
    __holder.scan()
    __mediator.addTags(__holder.getImageList())
    __holder.saveTagsToFile()
    print(f"tag() runtime: {time.time() - start} seconds")

class __ImageHolder:

    def __init__(self, workingDirectory: str):
        if not workingDirectory.endswith(('/', '\\')):
            workingDirectory += '/'
        if not os.path.isdir(workingDirectory):
            raise OSError(f"Could not find \"{workingDirectory}\"")
        self.__workingDirectory = workingDirectory
        self.__imageFileExt = ('.jpeg', '.jpg', '.png')
        self.images = dict()

    def __addImage(self, filename):
        path = self.__workingDirectory + filename
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exists")
        if filename not in self.images:
            self.images[filename] = ImageTag(path)

    def scan(self):
        start = time.time()
        self.images = dict()

        files = os.listdir(self.__workingDirectory)
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)

        for f in files:
            self.__addImage(f)

        count = len(self.images)
        print(f'Found {count} image{"s" * (count > 1)} in \"{self.__workingDirectory}\"')
        print(f"scan(self) runtime: {time.time() - start} seconds")

    def saveTagsToFile(self):
        start = time.time()
        for image in self.images.values():
            tagImage(image.filepath, image.getAllTags())
        print(f"saveTagsToFile(self) runtime: {time.time() - start} seconds")
    
    def getImageList(self):
        return list(self.images.values())

    def excludeTag(self, filename, tag):
        self.__addImage(filename)
        self.images[filename].excludeTag(tag)
    
    def addTag(self, filename, tag):
        self.__addImage(filename)
        self.images[filename].addUserTag(tag)

    def getTags(self, filename):
        self.__addImage(filename)
        imgTag = self.images[filename]
        __mediator.addTags([imgTag])
        return imgTag.getAllTags()
    
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

def tagImage(filepath: str, newTags: Iterable[str]):
    """
    Tag one image

    Args:
        filepath (str): the filepath of the image to tag
        newTags (list[str]): new tags to add to the image
    """
    tagEntries = ['Xmp.dc.subject', 'Iptc.Application2.Keywords'] # Exif.Image.XPKeywords is not supported

    print(f'Image: {filepath}, adding tags: {newTags}')
    img = pyexiv2.Image(filepath)
    read = {'Xmp': img.read_xmp, 'Iptc': img.read_iptc, 'Exif': img.read_exif}
    modify = {'Xmp': img.modify_xmp, 'Iptc': img.modify_iptc, 'Exif': img.modify_exif}

    for entry in tagEntries:
        entryType = entry.split('.', 1)[0]
        metadata = read[entryType]()
        tags = set(newTags)
        if entry in metadata:
            tags.update(metadata[entry])
        metadata[entry] = list(tags)
        modify[entryType](metadata)
