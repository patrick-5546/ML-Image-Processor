# ml_face

The package contains methods to detect and identify faces in images. 
Detected faces are saved in `tmp/instance/crop_face_log/`. The face recognition
model is trained based on faces in `face_sample`.

## ML Models
- facenet_pytorch model for face detection
- scikit-learn Linear Discriminant Analysis for face recognition

## Sources

### Images

- Images in `face_sample` are extracted from the [Labeled Faces in the Wild](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html) dataset.

### Code

- `face_recognition.py` and `_face_recognition_dataset.py` is written based on `ML_Development/Face_Recognition.ipynb`
- [`_pilutil.py`](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/datasets/_lfw.py)
  - Used in making the dataset for the face recognition model

## Example

```python
from ml_backend.ml_face.face_detection import FaceDetection
from ml_backend.ml_face.face_recognition import FaceRecognition

fd = FaceDetection()
faces, conf_fd = fd.crop_faces("source.jpg", 0.999)

fr = FaceRecognition()
fr.train("ml_backend/ml_face/face_sample")
names, conf_fr = fr.predict(faces, 0.75)
```
