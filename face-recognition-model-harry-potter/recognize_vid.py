import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
people=['Ron Weasley','Hermione Granger','Harry Potter']

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('/Users/samrudhm/Documents/openCVprojects/learnning/deep-learning/trained_model1.yml')

cap=cv.VideoCapture('/Users/samrudhm/Documents/openCVprojects/learnning/source/golden trio_ iconic moments.mp4')
while True:
    ret,img=cap.read()

    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=9)

    for (x,y,w,h) in face_rect:
        face_roi=gray[y:y+h,x:x+h]
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        (w1,h1),_=cv.getTextSize('person',cv.FONT_HERSHEY_SIMPLEX,1,2)
        cv.rectangle(img,(x,y-h1-10),(x+w,y),(0,255,0),-1)
        label,confidence=face_recognizer.predict(face_roi)
        name=str(people[label])
        # print(f'{name} with confidence of :{confidence}')
        cv.putText(img,name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv.imshow('dected image',img)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()