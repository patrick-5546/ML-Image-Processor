# This python script tags images in the provided folder 
# based on the output of the pretrained YOLOv5s model.

import sys
import tagger

if len(sys.argv) != 2:
    print('Usage: python tagall.py path/to/folder')
    exit()
folder = sys.argv[1]

tagger.startImageTagger()
tagger.tagFolder(folder)

