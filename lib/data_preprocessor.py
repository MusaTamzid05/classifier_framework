import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

import numpy as np

from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, image_size, dir_path = None):
        self.dir_path = dir_path
        self.image_size = image_size
        self.num_classes = None


    def process_image(self, image):
        if type(image) == str:
            image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = img_to_array(image)
        image /= 255
        return image

    def fit(self):
        images, labels = self._load_image_and_labels()

        self.num_classes = len(np.unique(labels))
        labels = self._process_labels(labels = labels)

        print(images.shape, labels.shape)



    def _load_image_and_labels(self):
        images, labels = [], []

        for label_dir in os.listdir(self.dir_path):
            print(f"Loading from {label_dir}")
            current_image = self._load_images_from(dir_path = os.path.join(self.dir_path, label_dir))
            images += current_image
            labels += [label_dir for i in range(len(current_image))]

        return np.array(images), np.array(labels)





    def _load_images_from(self, dir_path):
        images = []
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            image = self.process_image(image = image_path)
            images.append(image)

        return images

    def _process_labels(self, labels):
        self.label_encoder = LabelEncoder().fit(labels)
        return to_categorical(self.label_encoder.transform(labels), self.num_classes)

