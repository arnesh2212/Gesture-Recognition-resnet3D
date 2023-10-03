
import cv2
from tensorflow.keras.models import load_model
import keras
import numpy as np
import pandas as pd
import pyautogui

model = load_model("resnetmodel.hdf5")

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

labels = pd.read_csv('labels.csv', header= None)



buffer = []
cls = []
predicte_value = 0
final_label = ""
i = 1
while (vid.isOpened()):
    ret,frame = vid.read()
    if ret:
        image = cv2.resize(frame,(96,64))
        
        image = image/255.0
        buffer.append(image)
        if(i%16==0):
            buffer = np.expand_dims(buffer,0)
            predicted_value =np.argmax(model.predict(buffer))
            cls = labels.iloc[predicted_value]
            print(cls)
            print(type(cls))
            print(cls.iloc[0])
            
            if(predicted_value == 0):
                final_label = 'Swiping left'
                pyautogui.press("left")
                
            elif (predicted_value == 1):
                final_label = 'Swiping right'
                pyautogui.press("right")
                
            elif (predicted_value == 2):
                final_label = "Swiping down"
                
            elif (predicted_value== 3):
                final_label = "Swiping up"
                
            elif (predicted_value == 4): 
                final_label = "pushing hand away"
            elif (predicted_value == 5):
                final_label = "pulling hand in"  
            elif (predicted_value == 6):
                final_label = "sliding two fingres left"
                pyautogui.press("left")
            elif (predicted_value== 7):
                final_label = "sliding two fingres right"
                pyautogui.press("right")
            elif (predicted_value == 8):
                final_label = "sliding two fingres down"

                
            elif (predicted_value== 9):
                final_label = "sliding two fingres up"

                
            elif (predicted_value == 10):
                final_label = "pushing two fingres away"
            elif (predicted_value == 11):
                final_label = "pulling two fingers in"
            elif (predicted_value == 12):
                final_label = "rolling hand forward"
            elif (predicted_value == 13):
                final_label = "rolling hand backward"
            elif (predicted_value == 14):
                final_label = "turning hand clockwise"
            elif (predicted_value == 15):
                final_label = "turning hand counterclockwise"
            elif (predicted_value == 16):
                final_label = "zooming in with full hand"
            elif (predicted_value== 17):
                final_label = "zooming out with full hand"
            elif (predicted_value == 18):
                final_label = "zooming in with two fingers"
                
            elif (predicted_value == 19):
                final_label = "zooming out with two fingers"
            elif (predicted_value == 20):
                final_label = "thumb up"
                pyautogui.press("space")
                
            elif (predicted_value == 21):
                final_label = "thumb down"
            elif (predicted_value == 22):
                final_label = "shaking hand"
                
            elif (predicted_value == 23):
                final_label = "stop sign"
                pyautogui.press("space")
                
            elif (predicted_value == 24):
                final_label = "drumming fingers"
                pyautogui.press("space")
                
            elif (predicted_value == 25):
                final_label = "no gesture"
            else:
                final_label = "doing other things"
            
            cv2.imshow('frame',frame)
            buffer = []
        i = i+1
        text = "activity: {}".format(final_label)
        frame = cv2.flip(frame , 1)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5) 
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

