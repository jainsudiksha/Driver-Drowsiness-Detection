# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 
from scipy.spatial import distance as dist
import dlib
from scipy.spatial import distance
import face_recognition
import time
import playsound
import winsound
from imutils import face_utils
import imutils

from threading import Thread
import numpy as np

def sound_alarm(alarm_file):
    playsound.playsound(alarm_file)
    
#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
    # compute the euclidean distances between the horizontal
    X   = dist.euclidean(mou[0], mou[6])
    # compute the euclidean distances between the vertical
    Y1  = dist.euclidean(mou[2], mou[10])
    Y2  = dist.euclidean(mou[4], mou[8])
    # taking average
    Y   = (Y1+Y2)/2.0
    # compute mouth aspect ratio
    mar = Y/X
    return mar

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio
MIN_AER = 0.24
MOU_AR_THRESH = 0.75
EYE_AR_CONSEC_FRAMES = 10
COUNTER = 0
COUNTER2=0
ALARM_ON = False
ALARM_ON2 = False
yawnStatus = False
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(r"C://Users//shubhamJain-B2018048//Downloads//shape_predictor_68_face_landmarks.dat")
predictor_path = 'C://Users//shubhamJain-B2018048//Downloads//shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        mouth = shape[mStart:mEnd]
        mouEAR = mouth_aspect_ratio(mouth)
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        ear = (left_ear+right_ear)/2
        ear = round(ear,2)
        
        if ear<0.30:
            COUNTER+= 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                
                ALARM_ON = True
                t = Thread(target=sound_alarm,args=('C://Users//shubhamJain-B2018048//Downloads//beep-07.wav',))
                t.deamon = True
                t.start()
                cv2.putText(frame, "ALERT! You are feeling asleep!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                     
                                       
                
        else:
            COUNTER = 0
            ALARM_ON = False
            

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if mouEAR > MOU_AR_THRESH:
            yawnStatus = True
            COUNTER2+= 1
            if COUNTER2 >= EYE_AR_CONSEC_FRAMES:
                
                ALARM_ON2 = True
                s = Thread(target=sound_alarm,args=('C://Users//shubhamJain-B2018048//Downloads//beep-07.wav',))
                s.deamon = True
                s.start()
                cv2.putText(frame, "Get some fresh air", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                     
                                       
                
        else:
            yawnStatus = False
            COUNTER2 = 0
            ALARM_ON2 = False
        if yawnStatus == True:
            winsound.PlaySound('C://Users//shubhamJain-B2018048//Downloads//beep-07.wav', winsound.SND_FILENAME)
            cv2.putText(frame, "Get some fresh air", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()