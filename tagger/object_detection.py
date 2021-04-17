import os, pyexiv2, torch

class ObjectDetection:

    def __init__(self, model: str = 'yolov5s', minConfidence: float = 0.0):
        self.__model = torch.hub.load('ultralytics/yolov5', model, pretrained=True, verbose=False)
        self.__minConfidence = minConfidence

    def __datasetBuilder(self, filepaths: list[str]) -> list[str]:
        return list(filepaths)

    def __predict(self, dataset) -> list[list[str]]:
        output = self.__model(dataset) # NOTE: this modifies dataset
        results = []
        for pred in output.pred:
            tags = set()
            if pred != None:
                for classId, conf in zip(pred[:, -1], pred[:, -2]):
                    # print(f'label: {output.names[int(classId)]} conf: {conf}')
                    if conf >= self.__minConfidence:
                        tags.add(output.names[int(classId)])
            results.append(list(tags))
        return results
    
    def detect(self, filepaths: list[str]) -> list[list[str]]:
        """
        Runs object dectection on images and returns their labels

        Args:
            filepaths (list[str]): file paths of the images
        
        Returns:
            list (list[list[str]]): a list of lists of strings representing the labels
        """
        if len(filepaths) == 0:
            return []
        dataset = self.__datasetBuilder(filepaths)
        prediction = self.__predict(dataset)
        return prediction
