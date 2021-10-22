from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class Classifier:
    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor
        self._init_model()

    def _init_model(self):
        model = Sequential()
        model.add(Conv2D(16, 3, padding = "same", input_shape = (self.data_preprocessor.image_size, self.data_preprocessor.image_size, 1), activation = "relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, 3, padding = "same", activation = "relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, 3, padding = "same", activation = "relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128, activation = "relu"))
        model.add(Dense(3))

        model.summary()
