# Image Tagger

The script `tagall.py` adds random tags to images. Added tags are visible through `Properties-Details` in Windows. 
They are also included during indexing in both Windows and iOS, which means the images can be found by searching
their tags. 

Images in this folder are downloaded from [Bing Images](https://www.bing.com/images/) for demonstration
purposes.

## Dependencies

- [pyexiv2](https://pypi.org/project/pyexiv2/)

## How to Run

1. Navigate to this directory, `Backend/ImageTagger/`
2. Run `python tagall.py` to add random tags to the images
   ![tagall output](/Images/tagall%20output.png)
