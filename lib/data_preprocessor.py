import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np

class DataPreprocessor:
    def __init__(self, image_size, dir_path = None):
        self.dir_path = dir_path
        self.image_size = image_size


    def process_image(self, image):
        if type(image) == str:
            image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_size, self.image_size))

        image = img_to_array(image)

        return image

    def fit(self):
        images, labels = [], []

        for label_dir in os.listdir(self.dir_path):
            print(f"Loading from {label_dir}")
            current_image = self._load_images_from(dir_path = os.path.join(self.dir_path, label_dir))
            images += current_image
            labels += [label_dir for i in range(len(current_image))]




    def _load_images_from(self, dir_path):
        images = []
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            image = self.process_image(image = image_path)
            images.append(image)

        return images
