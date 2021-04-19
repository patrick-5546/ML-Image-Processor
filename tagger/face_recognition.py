import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .face_recognition_dataset import build_train_dataset, build_prediction_dataset # pylint: disable=relative-beyond-top-level

class FaceRecognition:
    def __init__(self):
        self.__sample_size = (62, 47) # (height, width)

    def __build_pca(self, train_dataset):
        # from first past project, start with the top 150/855 = 17.5% of samples
        n_components = round(0.175 * train_dataset.shape[0])
        print(f"building pca, n_components = {n_components}")
        self.__pca = PCA(n_components=n_components, whiten=True)
        self.__pca.fit(train_dataset)
    
    def __build_svc(self, train_dataset, train_target):
        train_pca = self.__pca.transform(train_dataset)
        self.__clf = SVC()
        self.__clf.fit(train_pca, train_target)

    def train(self, train_folder):
        data, target, names = build_train_dataset(train_folder)
        self.__build_pca(data)
        self.__build_svc(data, target)
        self.names = names

    def predict(self, faces):
        size = self.__sample_size
        pred_data = build_prediction_dataset(faces, size[0], size[1])
        pred_pca = self.__pca.transform(pred_data)
        pred = [self.names[p] for p in self.__clf.predict(pred_pca)]
        return pred
