from collections.abc import Callable, Iterable

class ImageTag:
    def __init__(self, filepath: str, objectTags: Iterable = set(), nameTag: str = None, userDefinedName: bool = False):
        self.filepath = filepath
        self.objectTags = set(objectTags)
        self.nameTag = nameTag
        self.userDefinedName = userDefinedName
    
    def getAllTags(self):
        tags = set(self.objectTags)
        if self.nameTag is not None:
            tags.add(self.nameTag)
        return tags

    def containsFace(self):
        return (not self.userDefinedName) and ('person' in self.objectTags)
