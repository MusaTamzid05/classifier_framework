import cv2
from lib.classifier import Classifier
from lib.utils import limit_gpu
from lib.data_preprocessor import DataPreprocessor

def main():
    limit_gpu()
    data_preprocessor = DataPreprocessor(image_size = 224)
    cls = Classifier(data_preprocessor = data_preprocessor, model_path = "model.h5")
    running = True

    cam = cv2.VideoCapture(0)


    while running:
        ret, frame = cam.read()

        if ret == False:
            running = False
            continue

        cls.predict(frame)

        cv2.imshow("test", frame)



        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False



if __name__ == "__main__":
    main()

