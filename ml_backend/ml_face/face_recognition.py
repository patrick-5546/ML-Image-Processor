from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ._face_recognition_dataset import build_train_dataset, build_prediction_dataset


class FaceRecognition:
    def __init__(self):
        self.__sample_size = (62, 47)  # (height, width)
        self.names = []

    def __build_pca(self, train_dataset):
        # from first past project, start with the top 150/855 = 17.5% of samples
        n_components = round(0.175 * train_dataset.shape[0])
        print(f"building pca, n_components = {n_components}")
        self.__pca = PCA(n_components=n_components, whiten=True)
        self.__pca.fit(train_dataset)

    def __build_svc(self, train_dataset, train_target):
        train_pca = self.__pca.transform(train_dataset)
        self.__clf = LinearDiscriminantAnalysis()
        self.__clf.fit(train_pca, train_target)

    def train(self, train_folder):
        """
        Train a new model based on the provided dataset

        Args:
            train_folder (str): the folder containing the dataset
        """
        data, target, names = build_train_dataset(train_folder)
        self.__build_pca(data)
        self.__build_svc(data, target)
        self.names = names

    def predict(self, faces, min_conf=0.75):
        """
        Given a list of faces, predict their names.

        Args:
            faces (list[Image]): the faces
            min_conf (float): minimum confidence

        Returns:
            (list[str], list[float]): predicted names and the confidence
        """
        names, confs = [], []
        if len(faces) != 0:
            size = self.__sample_size
            pred_data = build_prediction_dataset(faces, size[0], size[1])
            pred_pca = self.__pca.transform(pred_data)
            outs = self.__clf.predict(pred_pca)
            proba = self.__clf.predict_proba(pred_pca)
            for i, p in zip(outs, proba):
                print(f"{self.names[i]} {p[i]}")
                if p[i] >= min_conf:
                    names.append(self.names[i])
                    confs.append(p[i])
        return names, confs
