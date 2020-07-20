# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:35:51 2020

@author: Tushar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model('m (1).h5')
model_file = "res10_300x300_ssd_iter_140000.caffemodel"#prebuild model
config_file = "deploy.prototxt"#configuration file
net = cv2.dnn.readNetFromCaffe(config_file,model_file)#model


def detectDNN(img):
    global net
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converting it to rgb
    height, width = frame.shape[0],frame.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224,224)),1.0, (224,224),(104.0,177.0,123.0))#preprocessign for prebuild model
    net.setInput(blob)#setting input to the model
    detection = net.forward()#prediction
    # print(detection)
    for i in range(detection.shape[2]):#iterate through number of detection
        confidence = detection[0,0,i,2]
        if confidence>0.3:
            box = detection[0,0,i,3:7]*np.array([width,height,width,height])
            (startX,startY, endX,endY) = box.astype('int')
            return (startX,startY, endX,endY)
            # text = str(confidence*100)
            # y = startY-10 if startY-10 > 10 else startY + 10
            # cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
            # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
        else:
            # startX,startY, endX,endY = 93,87,283,300
            pass
    # plt.imshow(frame)
    # plt.show()
    

# img = cv2.imread('dataset/with_mask/7-with-mask.jpg')
# t = detectDNN(img)
# roi = img[89:376,57:300]
# roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
# roi = cv2.resize(roi, (224,224))
# roi = roi.reshape(1,224,224,3)
# roi=roi/255
# plt.imshow((roi))
# print(model.predict(roi))

d = { 0:"with mask", 1:"without mask"}
    

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    t = detectDNN(frame)
    if t != None:
        roi = frame[t[1]:t[3],t[0]:t[2]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (224,224))
        roi = roi.reshape(1,224,224,3)
        roi=roi/255
        text = str(d[np.argmax(model.predict(roi)[0])])
        cv2.rectangle(frame, (t[0],t[1]), (t[2],t[3]), (0,0,255), 2)
        y = t[1]-10 if t[1]-10 > 10 else t[1] + 10
        cv2.putText(frame, text, (t[0], y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
        cv2.imshow('play', frame)
    else:
        cv2.imshow('play', frame)
    if cv2.waitKey(10) == ord('k'):
        break
cv2.destroyAllWindows()
cap.release()
    
    