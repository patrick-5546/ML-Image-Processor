# ml_low_light

The package contains methods to enhance dark images. A filter is applied to ensure that the model only runs on dark images.

## ML Model

- Keras InceptionResNetV2 model

## Sources

### Saved Model

- The saved model in `saved_lle_model` is obtained from the notebook
  [`/ML_Development/Low_Light_Image_Enhancement.ipynb`](/ML_Development/Low_Light_Image_Enhancement.ipynb`).

## Example

```python
from ml_backend.ml_low_light import LowLightEnhancement

lle = LowLightEnhancement()
lle.enhance_image("source.jpg", "target.jpg")
lle.enhance_images_in("source_folder/", 30.0)
```
