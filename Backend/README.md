# Image Tagger

The script `tagall.py` tags images in the provided folder based on the output of the pretrained YOLOv5s model. The added tags are included during indexing in both Windows and iOS, which means the images can be found by searching their tags. 

## Dependencies

- [pyexiv2](https://pypi.org/project/pyexiv2/)
- [torch](https://pypi.org/project/torch/)
- [YOLOv5 dependencies](https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt)

## How to Run

1. Run `python Backend/tagall.py path/to/folder` to add tags to the images

## Sources

- Images in this folder are downloaded from [Pixabay](https://pixabay.com/) for demonstration
purposes.

## Output

- Some samples are included in `Backend/images_before` and `Backend/images_after`

- Before running `tagall.py`\
![before running tagall.py](/Images/tagall_before.jpg)

- After running `tagall.py`\
![after running tagall.py](/Images/tagall_after.jpg)
