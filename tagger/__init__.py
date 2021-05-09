"""
This python script tags images in the provided folder.
"""
import time
from .object_detection import ObjectDetection
from .face_detection import crop_faces
from .face_recognition import FaceRecognition
from .image_tag import ImageLibrary, ImageTag


def initialize(working_dir, face_sample_folder):
    """
    Initializes the image tagger

    Args:
        working_dir (str): path of the image folder
        face_sample_folder (str): path of the folder containing labeled face images
    """
    global __mediator, __library
    __mediator = __MLModelMediator(face_sample_folder)
    __library = ImageLibrary(working_dir)


def tag_all_images():
    """
    Tags all images in the working directory. Requires the tagger to be initialized.
    """
    start = time.time()
    __library.scan()
    __mediator.add_tags(__library.get_all_items())
    __library.save_tags()
    print(f"tag() runtime: {time.time() - start} seconds")


def get_all_tags():
    """

    Returns:
        dict[str, str]:
    """
    images = __library.images
    tags = dict()
    for key, val in images.items():
        tags[key] = "; ".join(val.get_all_tags())
    return tags


def get_tags(filename):
    """
    Read tags of an image. Run object detection and facial recognition if necessary.

    Args:
        filename (str): name of the image file

    Returns:
        set[str]: tags of the image
    """
    img_tag = __library.get_item(filename)
    __mediator.add_tags([img_tag])
    return img_tag.get_all_tags()


def exclude_tag(filename, tag):
    """
    Exclude a tag from an image. The excluded tag will no longer 
    appear unless it is explicitly added back to the image.

    Args:
        filename (str): name of the image file
        tag (str): tag to exclude
    """
    __library.get_item(filename).exclude_tag(tag)


def add_tag(filename, tag):
    """
    Add a tag to an image. 

    Args:
        filename (str): name of the image file
        tag (str): tag to add
    """
    __library.get_item(filename).add_user_tag(tag)


class __MLModelMediator:

    def __init__(self, face_sample_folder=None):
        """
        Initialize a ML model mediator

        Args:
            face_sample_folder (str): folder containing face samples
        """
        self.__objectDetection = ObjectDetection('yolov5s', 0.3)
        self.__face_sample_folder = face_sample_folder
        self.__face_recognition = FaceRecognition()
        self.__face_recognition.train(self.__face_sample_folder)

    def add_tags(self, images: list[ImageTag]):
        """
        Get tags of the images using the Object Detection model and the Facial Recognition model

        Args:
            images (list[ImageTag]): images
        """
        start = time.time()
        if len(images) == 0:
            return []

        # Object Detection
        obj_images = list(filter(lambda img: not img.is_object_detected(), images))
        obj_file_paths = [img.filepath for img in obj_images]
        obj_pred = self.__objectDetection.detect(obj_file_paths)
        for img, pred in zip(obj_images, obj_pred):
            img.set_object_tags(pred)

        print(f"addTags Object Detection runtime: {time.time() - start} seconds")
        start = time.time()

        # Facial Recognition
        for image in images:
            if image.is_face_recognized():
                continue
            faces, _ = crop_faces(image.filepath)
            names, confs = self.__face_recognition.predict(faces)
            image.set_face_tags(names)

        print(f"addTags Facial Recognition runtime: {time.time() - start} seconds")
