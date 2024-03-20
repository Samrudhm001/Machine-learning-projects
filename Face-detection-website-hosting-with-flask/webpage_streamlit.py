import cv2 as cv
import streamlit as st

video=cv.VideoCapture(0)

st.title('OpenCv Face Detector')
face_holder=st.empty()
stop_button=st.button('stop')

while video.isOpened() and not stop_button:
    ret,frame=video.read()
    frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    face_holder.image(frame,channels='RGB')

    if cv.waitKey(1) & 0xFF==ord('q') or stop_button:
        break
video.release()
cv.destroyAllWindows()