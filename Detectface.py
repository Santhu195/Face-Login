"""
@author: Santhosh R
"""
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

# This will make sure no duplicates exixts in profile.csv(using Pandas here)
df = pd.read_csv('Profile.csv')
df.sort_values('Ids', inplace = True)
df.drop_duplicates(subset = 'Ids', keep = 'first', inplace = True)
df.to_csv('Profile.csv', index = False)

# Fuction to detect the face
def DetectFace():
    reader = csv.DictReader(open('Profile.csv'))
    print('Detecting Login Face')
    for rows in reader:
        result = dict(rows)
        #print(result)
        if result['Ids'] == '1':
            name1 = result['Name']
        elif result['Ids'] == '2':
            name2 = result["Name"]
    recognizer = cv2.face.LBPHFaceRecognizer_create()  #cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainData\Trainner.yml")
    harcascadePath = "hh.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Face_Id = ''
    name2 = ''

    # Camera ON Everytime
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        Face_Id = 'Not detected'

        # Drawing a rectagle around the face 
        for (x, y, w, h) in faces:
            Face_Id = 'Not detected'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 80):
                if (Id == 1):
                    name = name1
                    
                elif (Id == 2):
                    name = name2
                    
                Predicted_name = str(name)
                Face_Id = Predicted_name
            else:
                Predicted_name = 'Unknown'
                Face_Id = Predicted_name
                # Here unknown faces detected will be stored
                noOfFile = len(os.listdir("UnknownFaces")) + 1
                if int(noOfFile) < 100:
                    cv2.imwrite("UnknownFaces\Image" + str(noOfFile) + ".jpg", frame[y:y + h, x:x + w])
                
                else:
                    pass


            cv2.putText(frame, str(Predicted_name), (x, y + h), font, 1, (255, 255, 255), 2)
            
        cv2.imshow('Picture', frame)
        #print(Face_Id)
        cv2.waitKey(1)

        # Checking if the face matches for Login
        if Face_Id == 'Not detected':
            print("-----Face Not Detected, Try again------")
            pass
            
        elif Face_Id == name1 or name2 and Face_Id != 'Unknown' :
            print('----------Detected as {}----------'.format(name1))
            print('-----------login successfull-------')
            print('***********WELCOME {}**************'.format(name1))
            break
        else:
            print('-----------Login failed please try agian-------')
        
        
        #if (cv2.waitKey(1) == ord('q')):
        #   break
DetectFace()



