from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier

def main():
    preprocessor = DataPreprocessor(image_size = 48, dir_path = "/home/musa/python_pro/custom_baracuda_test/create_data/caltech")
    preprocessor.fit()


    cls = Classifier(data_preprocessor = preprocessor)
    cls.fit(batch_size = 64, epochs = 30)


if __name__ == "__main__":
    main()

