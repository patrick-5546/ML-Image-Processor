from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pathlib

__mtcnn = None

def cropFaces(imagePath: str, min_conf: float = 0.99) -> (list[Image], list[float]):
    global __mtcnn
    if __mtcnn is None:
        __mtcnn = MTCNN(margin=10, min_face_size=50, 
                        select_largest=False, post_process=False, keep_all=True)
    
    img = Image.open(imagePath)

    cropped_faces, confs = [], []
    boxes, probs = __mtcnn.detect(img) # pylint: disable=unbalanced-tuple-unpacking
    if boxes is not None and len(boxes) > 0:
        for box, prob in zip(boxes, probs):
            if prob < min_conf:
                continue
            cropped = img.crop(tuple(box))
            cropped_faces.append(cropped)
            confs.append(prob)
    
    print(f"CropFaces found {len(cropped_faces)} face(s) in \"{imagePath}\"")
    if len(cropped_faces) > 0:
        saveImages(cropped_faces, "Crop Faces Log/" + imagePath.split('/')[-1])
    return cropped_faces, confs

def saveImages(images, folder_name):
    print(f"saving {len(images)} images in {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    for i, img in enumerate(images):
        img.save(f"{folder_name}/face{i}.jpg")
