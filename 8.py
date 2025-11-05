# prog8_haar_yolo.py
# pip install opencv-python matplotlib
import cv2 as cv, matplotlib.pyplot as plt

IMG = "images/image8.jfif"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

img  = cv.imread(IMG)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=5)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)); plt.title(f"Faces: {len(faces)}")
plt.axis('off'); plt.show()

