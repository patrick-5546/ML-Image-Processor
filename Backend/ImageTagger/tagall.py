# This python script adds random tags to all images under the same folder
# requires pyexiv2: pip install pyexiv2

import os, sys, pyexiv2, random

class MLModelMediator:
    def __init__(self, model, batchSize, datasetBuilder):
        self.model = model
        self.batchSize = batchSize
        self.datasetBuilder = datasetBuilder
    
    def predict(self, filepaths):
        if len(filepaths) > self.batchSize:
            raise Exception()
        dataset = self.datasetBuilder(filepaths)
        prediction = self.model(dataset)
        return prediction
    
    def getTags(self, filepaths):
        step = self.batchSize
        batches = [filepaths[s : s + step] for s in range(0, len(filepaths), step)]
        tags = []
        for batch in batches:
            tags.extend(self.predict(batch))
        return tags

class Tagger:
    def __init__(self, mediator):
        self.tagEntry = 'Xmp.dc.subject'
        self.mediator = mediator

    # Tag individual image
    def tagImage(self, filepath, newTags):
        img = pyexiv2.Image(filepath)
        xmp = img.read_xmp()
        tags = xmp[self.tagEntry] if self.tagEntry in xmp else []
        tags = tags + newTags
        xmp[self.tagEntry] = tags
        img.modify_xmp(xmp)
    
    def listImages(self, path = '.'):
        files = os.listdir(path)
        return list(filter(lambda fn: fn.endswith(('.jpeg', '.jpg', '.png')), files))

    def tagImagesInFolder(self, folder):
        filenames = self.listImages(folder)
        imageTags = self.mediator.getTags(filenames)

        for filename, tags in zip(filenames, imageTags):
            print(f'Image: {filename}, tags: {tags}')
            self.tagImage(filename, tags)

def dummyModel(dataset):
    def rand(i):
        return random.randint(1, i)
    return [[f'tag{rand(100)}' for _ in range(rand(3))] for _ in dataset]

def dummyDatasetBuilder(filepaths):
    return filepaths

mediator = MLModelMediator(dummyModel, 2, dummyDatasetBuilder)
tagger = Tagger(mediator)

tagger.tagImagesInFolder('.')
