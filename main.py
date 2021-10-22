from lib.data_preprocessor import DataPreprocessor

def main():
    preprocessor = DataPreprocessor(image_size = 24, dir_path = "/home/musa/python_pro/custom_baracuda_test/create_data/caltech")
    train_x, test_x, train_y, test_y = preprocessor.fit()

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


if __name__ == "__main__":
    main()

