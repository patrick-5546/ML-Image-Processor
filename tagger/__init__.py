"""
This python script tags images in the provided folder.
"""
import time
from ml_backend.ml_object_dectection import ObjectDetection
from ml_backend.ml_face.face_detection import FaceDetection
from ml_backend.ml_face.face_recognition import FaceRecognition
from .image_tag import ImageLibrary, ImageTag


class Tagger:

    def __init__(self, working_dir, face_sample_folder):
        """
            Initializes the image tagger

            Args:
                working_dir (str): path of the image folder
                face_sample_folder (str): path of the folder containing labeled face images
            """
        self.__mediator = MLModelMediator(face_sample_folder)
        self.__library = ImageLibrary(working_dir)

    def tag_all_images(self):
        """
        Tags all images in the working directory. Requires the tagger to be initialized.
        """
        start = time.time()
        self.__library.scan()
        items = self.__library.get_all_items()
        self.__mediator.add_tags(items)
        self.save_tags()
        print(f"tag() runtime: {time.time() - start} seconds")

    def save_tags(self):
        self.__library.save_tags()

    def get_all_tags(self):
        """

        Returns:
            dict[str, str]:
        """
        images = self.__library.images
        tags = dict()
        for key, val in images.items():
            tags[key] = "; ".join(val.get_all_tags())
        return tags

    def get_tags(self, filename):
        """
        Read tags of an image. Run object detection and facial recognition if necessary.

        Args:
            filename (str): name of the image file

        Returns:
            set[str]: tags of the image
        """
        img_tag = self.__library.get_item(filename)
        self.__mediator.add_tags([img_tag])
        return img_tag.get_all_tags()

    def exclude_tag(self, filename, tag):
        """
        Exclude a tag from an image. The excluded tag will no longer
        appear unless it is explicitly added back to the image.

        Args:
            filename (str): name of the image file
            tag (str): tag to exclude
        """
        self.__library.get_item(filename).exclude_tag(tag)

    def add_tag(self, filename, tag):
        """
        Add a tag to an image.

        Args:
            filename (str): name of the image file
            tag (str): tag to add
        """
        self.__library.get_item(filename).add_user_tag(tag)


class MLModelMediator:

    def __init__(self, face_sample_folder=None):
        """
        Initialize a ML model mediator

        Args:
            face_sample_folder (str): folder containing face samples
        """
        self.__objectDetection = ObjectDetection(0.1)
        self.__face_sample_folder = face_sample_folder
        self.__face_detection = FaceDetection()
        self.__face_recognition = FaceRecognition()
        self.__face_recognition.train(self.__face_sample_folder)

    def __set_object_tags(self, images):
        start = time.time()

        obj_images = list(filter(lambda img: not img.is_object_detected(), images))
        obj_file_paths = [img.filepath for img in obj_images]
        obj_pred = self.__objectDetection.detect(obj_file_paths)
        for img, pred in zip(obj_images, obj_pred):
            img.set_object_tags(pred)

        print(f"__set_object_tags Object Detection runtime: {time.time() - start} seconds")

    def __set_face_tags(self, images):
        start = time.time()

        for image in images:
            if image.is_face_recognized():
                continue
            faces, _ = self.__face_detection.crop_faces(image.filepath)
            names, _ = self.__face_recognition.predict(faces)
            image.set_face_tags(names)

        print(f"addTags Facial Recognition runtime: {time.time() - start} seconds")

    def add_tags(self, images: list[ImageTag]):
        """
        Get tags of the images using the Object Detection model and the Facial Recognition model

        Args:
            images (list[ImageTag]): images
        """

        if len(images) == 0:
            return []
        self.__set_object_tags(images)
        self.__set_face_tags(images)
