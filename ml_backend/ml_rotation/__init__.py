"""
A model for rotation correction
Usage:
rc = RotationCorrection();
rc.correct_rotation(["t.jpg", "rotated.jpg"], ["tf.jpg", "a.jpg"])
"""

import cv2
import torch
import torchvision
import time
import pathlib
import os
from torch.utils.data import Dataset, DataLoader
from .helper import rotate_and_crop, numpy_to_pil_img


class RotationCorrection:

    class __Dataset(Dataset):
        """
        Each dataset item is (img, label), where image is a (3, 224, 224) tensor of a jpg in root_dir that was rotated,
        cropping out the black borders then resized so that its a square, by an angle between -15° and 15°, inclusive
        """

        def __init__(self, image_paths):
            self.__transform = torchvision.transforms.ToTensor()
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            if torch.is_tensor(index):  # if i is a tensor, get the actual index
                index = index.item()
            path = self.image_paths[index]
            image = cv2.imread(path)
            image = cv2.resize(image, (224, 224))
            return self.__transform(image)

    def __init__(self):
        self.model = torch.load("ml_backend/ml_rotation/saved_rotation_model", map_location=torch.device('cpu'))
        self.model.eval()

    def __load_dataset(self, image_paths):
        dataset = self.__Dataset(image_paths)
        print(f"__load_dataset loaded {len(dataset)} images")
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def __predict_angles(self, image_dataset):
        results = []
        with torch.no_grad():
            for samples in image_dataset:
                outs = self.model(samples)
                results += [round(float(pred)) for pred in outs.unbind()]
        return results

    def find_angles(self, image_paths):
        dataset = self.__load_dataset(image_paths)
        angles = self.__predict_angles(dataset)
        return angles

    def correct_rotation(self, source_image_paths, target_images_paths, minimum_angle):
        angles = self.find_angles(source_image_paths)
        for source, target, angle in zip(source_image_paths, target_images_paths, angles):
            image_array = cv2.imread(source)
            if abs(angle) > minimum_angle:
                rotate_and_crop(image_array, -angle).save(target)
                print(f"Rotation: src: {source}, target: {target}, angle: {angle}")
            else:
                print(f"Rotation: src: {source}, angle: {angle}, skipped")

    def correct_rotation_in(self, folder, minimum_angle=2):
        start = time.time()
        enhanced_folder = folder + "/correct_rotation/"
        pathlib.Path(enhanced_folder).mkdir(parents=True, exist_ok=True)
        files = os.listdir(folder)
        files = list(filter(lambda n: n.endswith(('.jpeg', '.jpg', '.png')), files))

        source = [f"{folder}/{fn}" for fn in files]
        target = [f"{enhanced_folder}/{fn}" for fn in files]
        print(f"correct_rotation_in src: {source}, tar: {target}")

        self.correct_rotation(source, target, minimum_angle)

        print(f"correct_rotation_in runtime: {time.time() - start} seconds")
