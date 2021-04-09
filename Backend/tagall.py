# This python script tags images in the provided folder 
# based on the output of the pretrained YOLOv5s model.

# imports
from collections.abc import Callable
import os, sys, pyexiv2, torch

# set image folder
if len(sys.argv) != 2:
    print('Usage: python tagall.py path/to/folder')
    exit()
folder = sys.argv[1]


class MLModelMediator:

    def __init__(self, model: str = 'yolov5s', minConfidence: float = 0.0):
        self.__model = torch.hub.load('ultralytics/yolov5', model, pretrained=True, verbose=True)
        self.__minConfidence = minConfidence

    def __datasetBuilder(self, filepaths: list[str]) -> list[str]:
        return list(filepaths)

    def __predict(self, dataset) -> list[list[str]]:
        output = self.__model(dataset) # NOTE: this modifies dataset
        results = []
        for pred in output.pred:
            tags = set()
            if pred != None:
                for classId, conf in zip(pred[:, -1], pred[:, -2]):
                    print(f'label: {output.names[int(classId)]} conf: {conf}')
                    if conf >= self.__minConfidence:
                        tags.add(output.names[int(classId)])
            results.append(list(tags))
        return results
    
    def getTags(self, filepaths: list[str]) -> list[list[str]]:
        """
        Get tags of the images using the ML model

        Args:
            filepaths (list[str]): file paths of the images
        
        Returns:
            list (list[list[str]]): a list of lists of strings representing the tags
        """

        if len(filepaths) == 0:
            return []
        dataset = self.__datasetBuilder(filepaths)
        prediction = self.__predict(dataset)
        return prediction


class Tagger:
    
    def __init__(self, tagEntries: list[str], tagger: Callable[list[str], list[list[str]]]):
        self.__tagEntries = tagEntries
        self.__tagger = tagger
        self.__imageFileExt = ('.jpeg', '.jpg', '.png')

    def __tagImage(self, filepath: str, newTags: list[str], verbose = True):
        """
        Tag one image

        Args:
            filepath (str): the filepath of the image to tag
            newTags (list[str]): new tags to add to the image
        """

        if verbose:
            print(f'Image: {filepath}, adding tags: {newTags}')
        img = pyexiv2.Image(filepath)
        read = {'Xmp': img.read_xmp, 'Iptc': img.read_iptc, 'Exif': img.read_exif}
        modify = {'Xmp': img.modify_xmp, 'Iptc': img.modify_iptc, 'Exif': img.modify_exif}

        for entry in self.__tagEntries:
            entryType = entry.split('.', 1)[0]
            if entryType == 'Exif.Image.XPKeywords': 
                raise "Exif.Image.XPKeywords is not supported"
            metadata = read[entryType]()
            tags = (metadata.get(entry) or []) + newTags
            metadata[entry] = tags
            modify[entryType](metadata)
    
    def __listImages(self, path: str) -> list[str]:
        files = os.listdir(path)
        if not path.endswith(('/', '\\')):
            path += '/'
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)
        return [path + f for f in files]

    def tagImagesInFolder(self, folder: str, verbose: bool = True):
        """
        Tag all images in the provided folder based on the tagger

        Args:
            folder (str): paths of the folder
            verbose (bool): verbose mode (default is True)
        """
        filepaths = self.__listImages(folder)
        imageTags = self.__tagger(filepaths)
        count = len(filepaths)
        if verbose:
            print(f'Found {count} image{"s" * (count > 1)} in \"{folder}\"')

        for filepath, tags in zip(filepaths, imageTags):
            self.__tagImage(filepath, tags, verbose)


# main
mediator = MLModelMediator('yolov5s', 0.3)
tagger = Tagger(['Xmp.dc.subject', 'Iptc.Application2.Keywords'], mediator.getTags)

tagger.tagImagesInFolder(folder)
