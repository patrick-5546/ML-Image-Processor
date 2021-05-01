import torch


class ObjectDetection:
    def __init__(self, model: str = 'yolov5s', min_conf: float = 0.0):
        self.__model = torch.hub.load('ultralytics/yolov5', model, pretrained=True, verbose=False)
        self.__min_conf = min_conf

    def __dataset_builder(self, file_paths: list[str]) -> list[str]:
        return list(file_paths)

    def __predict(self, dataset) -> list[list[str]]:
        output = self.__model(dataset)  # NOTE: this modifies dataset
        results = []
        for pred in output.pred:
            tags = set()
            if pred is not None:
                for classId, conf in zip(pred[:, -1], pred[:, -2]):
                    # print(f"label: {output.names[int(classId)]} conf: {conf}")
                    if conf >= self.__min_conf:
                        tags.add(output.names[int(classId)])
            results.append(list(tags))
        return results
    
    def detect(self, file_paths: list[str]) -> list[list[str]]:
        """
        Runs object detection on images and returns their labels

        Args:
            file_paths (list[str]): file paths of the images
        
        Returns:
            list (list[list[str]]): a list of lists of strings representing the labels
        """
        if len(file_paths) == 0:
            return []
        dataset = self.__dataset_builder(file_paths)
        prediction = self.__predict(dataset)
        return prediction
