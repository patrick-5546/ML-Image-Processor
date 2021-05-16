# ml_rotation

The package contains methods to predict the degree of misalignment of images.

## ML Models

- ResNet-18 with transfer learning

## Sources

### Code

- `__init__.py` is written based on [`/ML_Development/Rotation_Correction.ipynb`](/ML_Development/Rotation_Correction.ipynb)
- Code in `helper.py` are copied from [`/ML_Development/Rotation_Correction.ipynb`](/ML_Development/Rotation_Correction.ipynb). It is for cropping and rotating images.

## Example

```python
from ml_backend.ml_rotation import RotationCorrection

rc = RotationCorrection()
angles = rc.find_angles(["source1.jpg", "source2.jpg", "source3.jpg"])
rc.correct_rotation(["source1.jpg", "source2.jpg", "source3.jpg"], 
                    ["target1.jpg", "target2.jpg", "target3.jpg"],
                    3)
rc.correct_rotation_in("source/folder/", 3)
```
