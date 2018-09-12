import cv2
from sklearn.externals import joblib
import numpy as np
from os import listdir


def drawRectangle(img, x, y, colour, thickness):
    cv2.rectangle(img, x, y, colour, thickness)

def drawLabel(img, label, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    cv2.putText(img, label, coords, font, font_size, (255,255,255), 2, cv2.LINE_AA)
optionnames={1:'PCA',2:'KPCA'}
option=-1
print("1> pca, 2> kpca")
option=int(input())
if option==1:
    from facespca import *
else:
    from faceskernelpca import *

print("Iniciando la camara")
cap = cv2.VideoCapture(0)

area_size = horsize * versize

def kerneltrick(images, imagetst, K, alpha, degree):
    trnno=images.shape[0]
    tstno=imagetst.shape[0]
    unoM = np.ones([trnno, trnno]) / trnno
    unoML = np.ones([tstno, trnno]) / trnno
    Ktest = (np.dot(imagetst, images.T) / trnno + 1) ** degree
    Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
    imtstproypre = np.dot(Ktest, alpha)
    return imtstproypre


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, width, height) in faces:

        drawRectangle(frame, (x, y), (x + width, y + height), (0,255,0), 1)

        roi_gray = gray[y:y + height, x:x + width]

        resized = cv2.resize(roi_gray, (horsize, versize), interpolation = cv2.INTER_CUBIC)

        if option==1: #opcion PCA
            resized = resized/255.0
            face = np.reshape(resized, [1, area_size])
            face-=meanimage
            improy = np.dot(face, V.transpose())
        else:
            #kernel
            face=(np.reshape(resized, [1, area_size]) - 127.5) / 127.5
            improy=kerneltrick(images, face, K, alpha, degree)

            #fin kernel
        # print(prediction)
        onlydirs = sorted([f for f in listdir(mypath)])
        prediction = clf.predict(improy)
        prediction=onlydirs[int(prediction[0])]

        label_coords = (x, y + height + 30)
        drawLabel(frame, prediction, label_coords)

    # Display the resulting frame
    cv2.imshow(optionnames[option], frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()