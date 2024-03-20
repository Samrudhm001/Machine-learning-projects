from flask import Flask,render_template,Response,redirect,request,url_for
import cv2 as cv

app=Flask(__name__,template_folder='templates',static_folder='static')
cap=cv.VideoCapture('/Users/samrudhm/Documents/openCVprojects/learnning/source/Snape Hits Harry and Ron _ Harry Potter and the Goblet of Fire.mp4')

def generate_frame():
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        else:
            ret,frame=cap.read()
            frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            # cv.imshow('video',frame)

            haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
            face_rect=haar_cascade.detectMultiScale(frame_gray,scaleFactor=1.1,minNeighbors=7)

            # print(f'{len(face_rect)},is the number of faces')

            for (x,y,w,h) in face_rect:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
                (w1,h1),_=cv.getTextSize('person',cv.FONT_HERSHEY_SIMPLEX,1,2)
                cv.rectangle(frame,(x,y-h1-10),(x+w1+10,y),(0,255,0),-1)
                cv.putText(frame,'person',(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
            ret,buffer=cv.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos')
def videos():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)



