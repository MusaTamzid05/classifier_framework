from lib.data_preprocessor import DataPreprocessor

def main():
    preprocessor = DataPreprocessor(image_size = 24, dir_path = "/home/musa/python_pro/custom_baracuda_test/create_data/caltech")
    preprocessor.fit()


if __name__ == "__main__":
    main()

