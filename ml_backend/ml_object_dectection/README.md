# ml_object_detection

The package contains methods to detect objects in images.

## ML Models

- Custom trained YOLOv5s to detect plates, skyscrapers, trees, and flowers
- Pretrained YOLOv5s on the COCO dataset

## Sources

### Saved Model

- The saved model in `trained_yolo_models/bettertreedetector.pt` is obtained from the notebook
  [`/ML_Development/Open_Images_Downloader_Finished.ipynb`](/ML_Development/Open_Images_Downloader_Finished.ipynb).
- The saved pretrained YOLOv5s model in `trained_yolo_models/yolov5s.pt` is obtained from torch hub.
  
## Example

```python
from ml_backend.ml_object_dectection import ObjectDetection

od = ObjectDetection(0.1)
results = od.detect(["image1.jpg", "image2.jpg", "image3.jpg"])
```
