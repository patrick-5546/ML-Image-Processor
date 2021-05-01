from facenet_pytorch import MTCNN
from PIL import Image
import pathlib

__mtcnn = None


def crop_faces(img_path, min_conf=.99):
    """
    Crop faces out of an image

    Args:
        img_path (str): path to the image
        min_conf (float): drop faces with confidence lower than this value

    Returns:
        (list[Image], list[float]): a tuple consisting of a list of images and a list of the confidence
    """
    global __mtcnn
    if __mtcnn is None:
        __mtcnn = MTCNN(margin=10, min_face_size=50, 
                        select_largest=False, post_process=False, keep_all=True)
    
    img = Image.open(img_path)

    cropped_faces, confs = [], []
    boxes, probs = __mtcnn.detect(img)
    if boxes is not None and len(boxes) > 0:
        for box, prob in zip(boxes, probs):
            if prob < min_conf:
                continue
            cropped = img.crop(tuple(box))
            cropped_faces.append(cropped)
            confs.append(prob)
    
    print(f"CropFaces found {len(cropped_faces)} face(s) in \"{img_path}\"")
    if len(cropped_faces) > 0:
        save_images(cropped_faces, "Crop Faces Log/" + img_path.split('/')[-1])
    return cropped_faces, confs


def save_images(images, folder_name):
    print(f"saving {len(images)} images in {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    for i, img in enumerate(images):
        img.save(f"{folder_name}/face{i}.jpg")
