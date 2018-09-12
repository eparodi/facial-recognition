import cv2

def getFace(filename):
	img = cv2.imread(filename)
	face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#hardcodeado
	data_width = 92
	data_height = 112
	if len(faces)!=1:
		print("Sin cara:", filename)
		return None
	(x, y, width, height)=faces[0]
	roi_gray = gray[y:y + height, x:x + width]
	resized = cv2.resize(roi_gray, (data_width, data_height), interpolation = cv2.INTER_CUBIC)
	return resized
	
if __name__=="__main__":
	face=getFace('eliseo.jpg')
	cv2.imwrite('cara.jpg',face)
