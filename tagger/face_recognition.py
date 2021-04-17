import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .face_recognition_dataset import make_train_dataset, load_test_images

def identify(sample_folder: str, image_paths: list[str], verbose: bool = False) -> list[str]:
    train_data, train_target, names = __build_train_dataset(sample_folder)
    pred_data = __build_prediction_dataset(image_paths)
    train_pca, pred_pca = __build_pca(train_data, pred_data)

    clf = SVC()
    clf.fit(train_pca, train_target)
    pred = clf.predict(pred_pca)
    pred_names = [names[p] for p in pred]
    if verbose:
        print('\n'.join([f"{f}: {n}" for f, n in zip(image_paths, pred_names)]))

    return pred_names

def __build_train_dataset(train_folder: str):
    # training dataset
    train_dataset = make_train_dataset(train_folder)
    train_data = train_dataset.images
    train_target = train_dataset.target
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
    return train_data, train_target, train_dataset.target_names

def __build_prediction_dataset(image_paths: list[str]):
    # images to predict
    pred_data = load_test_images(image_paths)
    pred_data = pred_data.reshape((pred_data.shape[0], pred_data.shape[1] * pred_data.shape[2]))
    return pred_data

def __build_pca(train_data, pred_data):
    # from first past project, start with the top 150/855 = 17.5% of samples
    n_components = round(0.175 * train_data.shape[0])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(train_data)
    return pca.transform(train_data), pca.transform(pred_data)
