from lib.utils import limit_gpu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class Classifier:
    def __init__(self, data_preprocessor):
        limit_gpu()
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
        model.add(Dense(self.data_preprocessor.num_classes, activation = "softmax"))
        model.summary()

        loss_type = None

        if self.data_preprocessor.num_classes == 2:
            loss_type = "binary_crossentropy"
        else:
            loss_type = "categorical_crossentropy"

        model.compile(loss = loss_type, optimizer = "adam", metrics = ["acc"])

        self.model = model

    def fit(self, epochs = 100, batch_size = 32):

        history = self.model.fit(
                self.data_preprocessor.train_x,
                self.data_preprocessor.train_y,
                validation_data = (
                    self.data_preprocessor.test_x,
                    self.data_preprocessor.test_y
                    ),
                batch_size = batch_size,
                epochs = epochs
                )





