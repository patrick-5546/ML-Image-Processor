# The blocks below were mostly copied from the sklearn code base (https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/datasets/_lfw.py)

from os import listdir
from os.path import join, isdir
import numpy as np
from sklearn.utils import Bunch

def load_test_images(image_paths, slice_=(slice(70, 195), slice(78, 172)), color=False, resize=0.5):
    if len(image_paths) == 0:
        raise ValueError("No image found")

    faces = _load_imgs(image_paths, slice_, color, resize)

    return faces

# Can experiment with changing slice and resize values
def make_train_dataset(data_folder_path, slice_=(slice(70, 195), slice(78, 172)), color=False, resize=0.5, min_faces_per_person=0):
    """Perform the actual data loading for the lfw people dataset
    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    X = faces.reshape(len(faces), -1)
    return Bunch(data=X, images=faces, target=target, target_names=target_names)

def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""
    # import PIL only when needed
    from sklearn.externals._pilutil import imread, imresize

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    default_slice = (slice(0, 250), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # allocate some contiguous memory to host the decoded image slices
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):

        # Checks if jpeg reading worked. Refer to issue #3594 for more
        # details.
        img = imread(file_path)
        if img.ndim == 0:
            raise RuntimeError("Failed to read the image file %s, "
                               "Please make sure that libjpeg is installed"
                               % file_path)

        face = np.asarray(img[slice_], dtype=np.float32)
        face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
        if resize is not None:
            face = imresize(face, resize)
        if not color:
            # average the color channels to compute a gray levels
            # representation
            face = face.mean(axis=2)

        faces[i, ...] = face

    return faces
