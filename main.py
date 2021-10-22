from lib.data_preprocessor import DataPreprocessor

def main():
    preprocessor = DataPreprocessor(image_size = 24)
    image = preprocessor.process_image(image = "/home/musa/m3.jpg")

    print(image.shape)


if __name__ == "__main__":
    main()

