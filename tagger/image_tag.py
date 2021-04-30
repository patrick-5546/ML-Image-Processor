from collections.abc import Callable, Iterable

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
