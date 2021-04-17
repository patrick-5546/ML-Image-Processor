# This python script tags images in the provided folder 
# based on the output of the pretrained YOLOv5s model.

# imports
from collections.abc import Callable
import os, pyexiv2, torch

from .object_detection import ObjectDetection
from .face_recognition import identify

def startImageTagger():
    """
    Initializes the image tagger
    """
    global __mediator, __tagger
    __mediator = __MLModelMediator()
    __tagger = __Tagger(['Xmp.dc.subject', 'Iptc.Application2.Keywords'], __mediator.getTags)

def tagFolder(folder: str, face_sample_folder: str):
    """
    Tags all images in the provided folder. Requires the tagger to be initialized.

    Args:
        folder (str): path of the image folder
        face_sample_folder (str): path of the folder containing labeled face images
    """
    __tagger.tagImagesInFolder(folder, face_sample_folder)



class __MLModelMediator:

    def __init__(self):
        self.__objectDection = ObjectDetection('yolov5s', 0.3)
    
    def getTags(self, filepaths: list[str], face_sample_folder: str) -> list[list[str]]:
        """
        Get tags of the images using the ML model

        Args:
            filepaths (list[str]): file paths of the images
        
        Returns:
            list (list[list[str]]): a list of lists of strings representing the tags
        """

        if len(filepaths) == 0:
            return []

        prediction = self.__objectDection.detect(filepaths)

        images_with_faces = []
        for i, p in enumerate(prediction):
            if 'person' in p:
                images_with_faces.append(i)
        
        if face_sample_folder != '' and len(os.listdir(face_sample_folder)) > 0 and len(images_with_faces) > 0:
            faces = identify(face_sample_folder, [filepaths[i] for i in images_with_faces])
            for i, f in zip(images_with_faces, faces):
                prediction[i].append(f)

        return prediction

class __Tagger:
    
    def __init__(self, tagEntries: list[str], tagger: Callable[list[str], list[list[str]]]):
        self.__tagEntries = tagEntries
        self.__tagger = tagger
        self.__imageFileExt = ('.jpeg', '.jpg', '.png')

    def __tagImage(self, filepath: str, newTags: list[str], append: bool = False):
        """
        Tag one image

        Args:
            filepath (str): the filepath of the image to tag
            newTags (list[str]): new tags to add to the image
        """

        print(f'Image: {filepath}, adding tags: {newTags}')
        img = pyexiv2.Image(filepath)
        read = {'Xmp': img.read_xmp, 'Iptc': img.read_iptc, 'Exif': img.read_exif}
        modify = {'Xmp': img.modify_xmp, 'Iptc': img.modify_iptc, 'Exif': img.modify_exif}

        for entry in self.__tagEntries:
            entryType = entry.split('.', 1)[0]
            if entryType == 'Exif.Image.XPKeywords': 
                raise "Exif.Image.XPKeywords is not supported"
            metadata = read[entryType]()
            if append:
                newTags.extend(metadata.get(entry) or [])
            metadata[entry] = newTags
            modify[entryType](metadata)
    
    def __listImages(self, path: str) -> list[str]:
        files = os.listdir(path)
        if not path.endswith(('/', '\\')):
            path += '/'
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)
        return [path + f for f in files]

    def tagImagesInFolder(self, folder: str, face_sample_folder: str):
        """
        Tag all images in the provided folder based on the tagger

        Args:
            folder (str): paths of the folder
        """
        filepaths = self.__listImages(folder)
        imageTags = self.__tagger(filepaths, face_sample_folder)
        count = len(filepaths)
        print(f'Found {count} image{"s" * (count > 1)} in \"{folder}\"')

        for filepath, tags in zip(filepaths, imageTags):
            self.__tagImage(filepath, tags, False)
