import cv2
from tensorflow.keras.preprocessing.image import img_to_array

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
