# Image Tagger

The package contains methods to tag images based on the output of the pretrained YOLOv5s model and sklearn's facial recognition model. The added tags are included during indexing in both Windows and iOS, which means the images can be found by searching their tags. `Metadata for Tags.pdf` contains some information related to the compatibility of different metadata fields for tagging.

## Samples

- Some samples are included in `images_before/` and `images_after/`.

## Sources

### Images

- Images in `images_before/`:
  - Images named `imageX.jpg` are downloaded from [Pixabay](https://pixabay.com/) for demonstration purposes.
  - `Powell.jpg` by [David](https://www.flickr.com/photos/bootbearwdc/2311921844)
  - `Schwarzenegger.jpg` by [Miljøstiftelsen ZERO](https://www.flickr.com/photos/zero_org/6376908843/)
  - `GMA.jpg` by [philippinepresidency](https://www.flickr.com/photos/36884962@N05/3400496904/)
  - `Agassi.jpg` by [Shinya Suzuki](https://www.flickr.com/photos/shinyasuzuki/6182777961/)
  - `Toledo.jpg` by [San Francisco Foghorn](https://www.flickr.com/photos/sffoghorn/16533059473/)
- Images in `images_after/` are tagged by this application
