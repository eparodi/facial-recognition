import cv2
from .pca import PCA
from .kpca import KPCA
import numpy as np
from os import listdir

class WebCamInterface():

    def __init__(self, method):
        self.method = method
        self.cap = None

    @staticmethod
    def drawRectangle(img, x, y, colour, thickness):
        cv2.rectangle(img, x, y, colour, thickness)

    @staticmethod
    def drawLabel(img, label, coords):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        cv2.putText(img, label, coords, font, font_size,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def start_webcam(self):
        print("Iniciando la cámara.")
        self.cap = cv2.VideoCapture(self.method.config['webcamInterface'])
        self.__infinite_webcam_loop()
        self.cap.release()
        cv2.destroyAllWindows()


    def __infinite_webcam_loop(self):
        horsize = self.method.config['imageWidth']
        versize = self.method.config['imageHeight']
        area_size = horsize * versize
        while True:
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                'utils/haarcascade_frontalface_default.xml')

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, width, height) in faces:

                WebCamInterface.drawRectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

                roi_gray = gray[y:y + height, x:x + width]

                resized = cv2.resize(roi_gray, (horsize, versize),
                                    interpolation=cv2.INTER_CUBIC)

                if isinstance(self.method, PCA):  # opcion PCA
                    resized = resized/255.0
                    face = np.reshape(resized, [1, area_size])
                    face -= self.method.meanImage
                    improy = np.dot(face, self.method.eigenFaces.transpose())
                elif isinstance(self.method, KPCA): # KPCA
                    face = (np.reshape(resized, [1, area_size]) - 127.5) / 127.5
                    improy = self.method.kerneltrick(face)

                onlydirs = sorted([f for f in listdir(self.method.config['path'])])
                prediction = self.method.clf.predict(improy)
                prediction = onlydirs[int(prediction[0])]

                label_coords = (x, y + height + 30)
                WebCamInterface.drawLabel(frame, prediction, label_coords)

            # Display the resulting frame
            cv2.imshow(self.method.__class__.__name__, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
