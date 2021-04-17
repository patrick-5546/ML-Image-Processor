"""
This python script tags images in the provided folder 
based on the output of the pretrained YOLOv5s model.
"""

from collections.abc import Callable, Iterable
import os, pyexiv2, torch

from .object_detection import ObjectDetection
from .face_recognition import identify
from .image_tag import ImageTag

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
    __holder.scan()
    __mediator.addTags(__holder.getImageList())
    __holder.saveTagsToFile()

class __ImageHolder:

    def __init__(self, workingDirectory: str):
        if not workingDirectory.endswith(('/', '\\')):
            workingDirectory += '/'
        if not os.path.isdir(workingDirectory):
            raise OSError(f"Could not find \"{workingDirectory}\"")
        self.__workingDirectory = workingDirectory
        self.__imageFileExt = ('.jpeg', '.jpg', '.png')
        self.images = dict()
        self.scan()

    def scan(self):
        self.images = dict()

        files = os.listdir(self.__workingDirectory)
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)

        for f in files:
            self.images[f] = ImageTag(self.__workingDirectory + f)
        count = len(self.images)
        print(f'Found {count} image{"s" * (count > 1)} in \"{self.__workingDirectory}\"')

    def saveTagsToFile(self):
        for image in self.images.values():
            tagImage(image.filepath, image.getAllTags())
    
    def getImageList(self):
        return list(self.images.values())

class __MLModelMediator:

    def __init__(self, face_sample_folder: str = None):
        self.__objectDection = ObjectDetection('yolov5s', 0.3)
        self.__face_sample_folder = face_sample_folder
    
    def addTags(self, images: list[ImageTag]):
        """
        Get tags of the images using the ML model

        Args:
            images (list[ImageTag]): images
        """
        if len(images) == 0:
            return []
            
        prediction = self.__objectDection.detect([img.filepath for img in images])

        images_with_faces = []
        for i, p in enumerate(prediction):
            images[i].objectTags.update(p)
            if images[i].containsFace():
                images_with_faces.append(i)
        
        face_sample = self.__face_sample_folder
        if face_sample is not None and len(os.listdir(face_sample)) > 0 and len(images_with_faces) > 0:
            faces = identify(face_sample, [images[i].filepath for i in images_with_faces])
            for i, f in zip(images_with_faces, faces):
                images[i].nameTag = f

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
