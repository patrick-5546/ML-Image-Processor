import torch


class ObjectDetection:
    def __init__(self, min_conf: float = 0.0):
        self.__custom_model = torch.hub.load('ultralytics/yolov5',
                                             'custom',
                                             path_or_model='tagger/trained_yolo_models/noface2000images.pt',
                                             verbose=False)
        self.__pretrained_model = torch.hub.load('ultralytics/yolov5',
                                                 'custom',
                                                 path_or_model='tagger/trained_yolo_models/yolov5s.pt',
                                                 verbose=False)
        self.__min_conf = min_conf

    def __process_pred(self, pred, names, tags_set, log_name="pred"):
        if pred is not None:
            for classId, conf in zip(pred[:, -1], pred[:, -2]):
                print(f"{log_name} label: {names[int(classId)]} conf: {conf}")
                if conf >= self.__min_conf:
                    tags_set.add(names[int(classId)])

    def __predict(self, paths) -> list[list[str]]:
        output_custom = self.__custom_model(list(paths))  # NOTE: the model modifies its input
        output_pretrained = self.__pretrained_model(list(paths))  # NOTE: the model modifies its input
        results = []
        for i in range(len(paths)):
            tags = set()
            pred_custom, pred_pretrained = output_custom.pred[i], output_pretrained.pred[i]
            print(f"Image file: {paths[i]}")
            self.__process_pred(pred_custom, output_custom.names, tags, "custom")
            self.__process_pred(pred_pretrained, output_pretrained.names, tags, "pretrained")
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
        prediction = self.__predict(file_paths)
        return prediction
