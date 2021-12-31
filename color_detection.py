# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 22:35:44 2021

@author: pcmaroc
"""
import numpy as np
import cv2 as cv

# We will use HSV 

lower_red = np.array([160,20,70])
upper_red = np.array([190,255,255])


def color_detection(frame, surface):
    points = []
    image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    image = cv.blur(image, (5,5))
    mask  = cv.inRange(image, lower_red, upper_red)
    mask = cv.erode(mask, None, iterations = 2)
    mask = cv.dilate(mask, None, iterations = 2)
    imageRes = cv.bitwise_and(frame, frame, mask = mask)
    elements = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    for element in elements:
        if cv.contourArea(element) > surface :
             ((x,y), rayon) = cv.minEnclosingCircle(element)
             points.append(np.array([int(x), int(y)]))
        else:
             break
    return points, mask 

    

    