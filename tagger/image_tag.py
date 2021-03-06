import os
import time
import pyexiv2
from collections.abc import Iterable
from tagger.categories import portrait, group_photo, urban, pet, nature, sports, food


class ImageLibrary:
    def __init__(self, working_dir: str):
        if not working_dir.endswith(('/', '\\')):
            working_dir += '/'
        if not os.path.isdir(working_dir):
            raise OSError(f"Could not find \"{working_dir}\"")
        self.__workingDirectory = working_dir
        self.__imageFileExt = ('.jpeg', '.jpg', '.png')
        self.images = dict()

    def __add_image(self, filename) -> None:
        """
        Added an image the library

        Args:
            filename (str): name of the image file

        Raises:
            FileNotFoundError: if filename does not exists
        """
        path = self.__workingDirectory + filename
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exists")
        if filename not in self.images:
            self.images[filename] = ImageTag(path)

    def scan(self):
        files = os.listdir(self.__workingDirectory)
        files = filter(lambda fn: fn.endswith(self.__imageFileExt), files)

        for f in files:
            self.__add_image(f)

        count = len(self.images)
        print(f'Found {count} image{"s" * (count > 1)} in \"{self.__workingDirectory}\"')

    def save_tags(self):
        start = time.time()
        for image in self.images.values():
            tag_image(image.filepath, image.get_all_tags())
        print(f"saveTagsToFile(self) runtime: {time.time() - start} seconds")
    
    def get_all_items(self):
        return list(self.images.values())

    def get_item(self, filename):
        """

        Args:
            filename:

        Returns:
            ImageTag:
        """
        self.__add_image(filename)
        return self.images[filename]


class ImageTag:
    def __init__(self, filepath, object_tags=None, face_tags=None) -> None:
        """

        Args:
            filepath (str):
            object_tags (Iterable[str]):
            face_tags (Iterable[str]):
        """
        self.filepath = filepath
        self.__objectTags = list(object_tags) if object_tags else list()
        self.__faceTags = set(face_tags) if face_tags else set()
        self.__excludeTags = set()
        self.__addedUserTags = set()
        self.__objectDetected = False
        self.__faceRecognized = False
    
    def get_all_tags(self):
        # Convert objects tags to category tags
        final_object_tags = set()
        print(f"name: {self.filepath} tags: {self.__objectTags}")
        for detectedObject in self.__objectTags:
            if 'Portrait' not in final_object_tags and 'Group Photo' not in final_object_tags and detectedObject in portrait:
                final_object_tags.add('Portrait')
                continue
            if 'Portrait' in final_object_tags and 'Group Photo' not in final_object_tags and detectedObject in group_photo:
                final_object_tags.remove('Portrait')
                final_object_tags.add('Group Photo')
            if 'Urban' not in final_object_tags and detectedObject in urban:
                final_object_tags.add('Urban')
            if 'Pet' not in final_object_tags and detectedObject in pet:
                final_object_tags.add('Pet')
            if 'Nature' not in final_object_tags and detectedObject in nature:
                final_object_tags.add('Nature')
            if 'Sports' not in final_object_tags and detectedObject in sports:
                final_object_tags.add('Sports')
            if 'Food' not in final_object_tags and detectedObject in food:
                final_object_tags.add('Food')
        
        tags = final_object_tags.union(self.__faceTags, self.__addedUserTags)
        tags = tags - self.__excludeTags
                
        return tags
    
    def exclude_tag(self, tag):
        self.__addedUserTags.discard(tag)
        self.__excludeTags.add(tag)
    
    def add_user_tag(self, tag):
        self.__excludeTags.discard(tag)
        self.__addedUserTags.add(tag)
    
    def set_object_tags(self, tags):
        print(f"setting object tags: {tags}")
        self.__objectTags = list(tags)
        self.__objectDetected = True

    def clear_object_tags(self):
        self.__objectTags = list()
        self.__objectDetected = False
    
    def set_face_tags(self, tags):
        self.__faceTags = set(tags)
        self.__faceRecognized = True

    def clear_face_tags(self):
        self.__faceTags = set()
        self.__faceRecognized = False
    
    def is_object_detected(self):
        return self.__objectDetected

    def is_face_recognized(self):
        return self.__faceRecognized


def tag_image(filepath, new_tags):
    """
    Tag an image

    Args:
        filepath (str): the filepath of the image to tag
        new_tags (Iterable[str]): new tags to add to the image
    """
    tag_entries = ['Xmp.dc.subject', 'Iptc.Application2.Keywords']  # Exif.Image.XPKeywords is not supported

    # print(f'Image: {filepath}, adding tags: {new_tags}')
    img = pyexiv2.Image(filepath)
    read = {'Xmp': img.read_xmp, 'Iptc': img.read_iptc, 'Exif': img.read_exif}
    clear = {'Xmp': img.clear_xmp, 'Iptc': img.clear_iptc, 'Exif': img.clear_exif}
    modify = {'Xmp': img.modify_xmp, 'Iptc': img.modify_iptc, 'Exif': img.modify_exif}

    for entry in tag_entries:
        entry_type = entry.split('.', 1)[0]
        metadata = read[entry_type]()
        tags = set(new_tags)
        if entry in metadata:
            tags.update(metadata[entry])
        metadata[entry] = list(tags)
        clear[entry_type]()
        modify[entry_type](metadata)
