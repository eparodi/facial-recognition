import cv2
from sklearn.externals import joblib
import numpy as np

def drawRectangle(img, x, y, colour, thickness):
    cv2.rectangle(img, x, y, colour, thickness)

def drawLabel(img, label, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    cv2.putText(img, label, coords, font, font_size, (255,255,255), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
CV_WIDTH = 3
CV_HEIGHT = 4

data_width = 92
data_height = 112
area_size = data_width*data_height

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_width = cap.get(CV_WIDTH)
    frame_height = cap.get(CV_HEIGHT)

    face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    clf = joblib.load('model.pkl')
    V = joblib.load('eigenfaces.pkl')
    eigenfaces = V.shape[0] - 1 #TODO: ver que pasa con la ultima autocara

    for (x, y, width, height) in faces:

        drawRectangle(frame, (x, y), (x + width, y + height), (0,255,0), 1)

        roi_gray = gray[y:y + height, x:x + width]

        resized = cv2.resize(roi_gray, (data_width, data_height), interpolation = cv2.INTER_CUBIC)
        resized = resized/255.0
        face = np.reshape(resized, [1, area_size])
        improy = np.dot(face, V[0:eigenfaces, :].transpose())

        prediction = clf.predict(improy)

        label_coords = (x, y + height + 30)
        drawLabel(frame, str(int(prediction[0])), label_coords)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()