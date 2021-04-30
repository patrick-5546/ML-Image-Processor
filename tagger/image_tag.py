import os, time, pyexiv2
from collections.abc import Callable, Iterable

class ImageLibrary:
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
        else:
            print(f"skiping {filename}")

    def scan(self):
        files = os.listdir(self.__workingDirectory)
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)

        for f in files:
            self.__addImage(f)

        count = len(self.images)
        print(f'Found {count} image{"s" * (count > 1)} in \"{self.__workingDirectory}\"')

    def saveTagsToFile(self):
        start = time.time()
        for image in self.images.values():
            tagImage(image.filepath, image.getAllTags())
        print(f"saveTagsToFile(self) runtime: {time.time() - start} seconds")
    
    def getImageList(self):
        return list(self.images.values())

    def getItem(self, filename):
        self.__addImage(filename)
        return self.images[filename]

class ImageTag:
    def __init__(self, filepath: str, objectTags: Iterable = set(), faceTags: Iterable = set()):
        self.filepath = filepath
        self.__objectTags = set(objectTags)
        self.__faceTags = set(faceTags)
        self.__excludeTags = set()
        self.__addedUserTags = set()
        self.__objectDetected = False
        self.__faceRecognized = False
    
    def getAllTags(self):
        tags = self.__objectTags.union(self.__faceTags, self.__addedUserTags)
        return tags - self.__excludeTags
    
    def excludeTag(self, tag):
        self.__addedUserTags.discard(tag)
        self.__excludeTags.add(tag)
    
    def addUserTag(self, tag):
        self.__excludeTags.discard(tag)
        self.__addedUserTags.add(tag)
    
    def setObjectTags(self, tags):
        print(f"setting object tags: {tags}")
        self.__objectTags = set(tags)
        self.__objectDetected = True

    def clearObjectTags(self):
        self.__objectTags = set()
        self.__objectDetected = False
    
    def setFaceTags(self, tags):
        self.__faceTags = set(tags)
        self.__faceRecognized = True

    def clearFaceTags(self):
        self.__faceTags = set()
        self.__faceRecognized = False
    
    def isObjectDetected(self):
        return self.__objectDetected

    def isFaceRecognized(self):
        return self.__faceRecognized

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
