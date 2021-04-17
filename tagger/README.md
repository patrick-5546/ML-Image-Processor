# Image Tagger

The package contains methods to tag images based on the output of the pretrained YOLOv5s model and sklearn's facial recognition model. The added tags are included during indexing in both Windows and iOS, which means the images can be found by searching their tags. 

## Dependencies

- [pyexiv2](https://pypi.org/project/pyexiv2/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [YOLOv5 dependencies](https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt)

## How to Run

```
import tagger
tagger.initialize('path/to/image_folder', 'path/to/labeled_face_image_folder')
tagger.tag()
```

## Sources

#### Images

- Images named `imageX.jpg` are downloaded from [Pixabay](https://pixabay.com/) for demonstration purposes.

- Face images are extracted from the [Labeled Faces in the Wild](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html) dataset.

#### Source Code

- `face_recognition.py` is written based on `Face_Recognition.ipynb`

- Code in `face_recognition_dataset.py` is copied from `Face_Recognition.ipynb`

## Output

- Some samples are included in `Backend/images_before` and `Backend/images_after`

- Before running `tagall.py`\
![before running tagall.py](/Images/tagall_before.jpg)

- After running `tagall.py`\
![after running tagall.py](/Images/tagall_after.jpg)
