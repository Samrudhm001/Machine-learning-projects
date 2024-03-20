import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
people=['ron','hermoine','harry potter']

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('/Users/samrudhm/Documents/openCVprojects/learnning/deep-learning/trained_model1.yml')



img=cv.imread('/Users/samrudhm/Documents/openCVprojects/learnning/deep-learning/source_img/Harry Potter/Fondo Harry Potter _ Fotos de harry potter, Harry potter, Actores de harry potter.jpeg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=9)
for (x,y,w,h) in face_rect:
    face_roi=gray[y:y+h,x:x+h]
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
    (w1,h1),_=cv.getTextSize('person',cv.FONT_HERSHEY_SIMPLEX,1,2)
    cv.rectangle(img,(x,y-h1-10),(x+w1+70,y),(0,255,0),-1)
    label,confidence=face_recognizer.predict(face_roi)
    name=str(people[label])
    print(f'{name} with confidence of :{confidence}')
    cv.putText(img,name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv.imshow('detected image',img)

cv.waitKey(0)
