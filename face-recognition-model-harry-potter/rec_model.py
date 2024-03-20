import cv2 as cv
import numpy as np
import os

haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
people=['Ron Weasley','Hermione Granger','Harry Potter']
features=[]
labels=[]

DIR=r'/Users/samrudhm/Documents/openCVprojects/learnning/deep-learning/source_img'
def train_data():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            if img_array is None:
                print(f"Error: Unable to read image {img_path}")
                continue

            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in face_rect:
                face_roi=gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)

train_data()
print('--------------training complete---------------')
print(f'the length of features is:{len(features)}')
print(f'the length of labels is:{len(labels)}')

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('trained_model1.yml')

    