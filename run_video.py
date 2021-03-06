import cv2
from lib.classifier import Classifier
from lib.utils import limit_gpu
from lib.data_preprocessor import DataPreprocessor

def main():
    limit_gpu()
    data_preprocessor = DataPreprocessor(image_size = 224)
    cls = Classifier(data_preprocessor = data_preprocessor, model_dir_path= "model_data/refactor_model")
    running = True

    cam = cv2.VideoCapture(0)


    while running:
        ret, frame = cam.read()

        if ret == False:
            running = False
            continue

        prediction = cls.predict(frame)[0][0]

        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("test", frame)



        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False



if __name__ == "__main__":
    main()

