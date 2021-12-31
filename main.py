# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 19:01:36 2021

@author: pcmaroc
"""

import cv2 as cv 
import numpy as np
from kalman import KalmanFilter
from color_detection import color_detection

cam = cv.VideoCapture(0)
KF = KalmanFilter(0.1,[0,0]) 

while(True):
    _, frame = cam.read()
    points, mask = color_detection(frame,800)
    etat = KF.predict().astype(np.int32)
    
    cv.circle(frame, (int(etat[0]), int(etat[1])),2,(0,0,255),5)
    cv.arrowedLine(frame, (int(etat[0]), int(etat[1])), (int(etat[0])+int(etat[2]), int(etat[1])+int(etat[3])), color = (0,255,0),thickness = 3,tipLength=0.2)
    if (len(points)>0): # si on a une mesure 
        cv.circle(frame,(points[0][0], points[0][1]),10,(0,0,255),2)
        KF.update(np.expand_dims(points[0],axis = -1))
    cv.imshow('image', frame)
    cv.imshow('mask', mask)
    if cv.waitKey(1) == ord('q'):
        cam.release()
        cv.destroyAllWindows()
        break




"""
import matplotlib.pyplot as plt
import matplotlib.colors as col
import  matplotlib.image as mpimg
from scipy import fftpack
import cv2 as cv 


im = cv.imread("D:\python_code\Images\chat.jpg")

def extractValueChannel(image):
    try:
        # Check if it has three channels or not 
        np.size(image, 2)
    except:
        return image
    hsvImage = col.rgb_to_hsv(image)
    return hsvImage[..., 2], hsvImage



def getFilter(image, w, h, TypF):
    line, column = np.size(im, 0), np.size(im,1)
    if (TypF=="LPF"):  
        out = np.zeros_like(image)
    elif(TypF == "HPF"):
        out = np.ones((np.size(image,0),np.size(image,1)))
    for y in range(line//2-w//2, line//2+w//2):
        for x in range(column//2-h//2, column//2 + h//2):
            if (TypF=="LPF"):    
                out[y][x]=255
            elif (TypF == "HPF"):
                out[y][x] = 0
        
    return out
HPF = getFilter(cv.cvtColor(im,cv.COLOR_BGR2GRAY), 3, 3, "HPF")


def maxi(a,b):
    if (a<b):
        return b
    else:
        return a
        
def plc(T,i,j):
    if (i==0 and j==0):
        return T[0][0]
    if (i==0):
        return T[0][j] + plc(T,0,j-1)
    if (j==0):
        return T[i][0] + plc(T,i-1,0)
    else:
        return T[i][j] + maxi(plc(T,i-1,j), plc(T,i,j-1))


img = cv.imread("D:\python_code\Images\monument.bmp")
image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
out = np.zeros_like(image)
size = image.shape
for i in range(size[0]):
    for j in range(size[1]):
        out[i][j] =  plc(image,i,j)

cv.imshow("show", out)
cv.waitKey(0)
cv.destroyall()
"""