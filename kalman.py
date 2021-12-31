# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 21:21:00 2021

@author: pcmaroc
"""

import numpy as np 
class KalmanFilter(object):
    def __init__(self, dt, point):
        self.dt = dt
        self.E = np.matrix([[point[0]], [point[1]], [0],[0]])
        
        self.A = np.matrix([[1,0,self.dt, 0],
                            [0,1,0,self.dt],
                            [0,  0,  1,  0],
                            [0,  0,  0,  1]])
        
        self.H = np.matrix([[1,  0,  0,  0],
                            [0,  1,  0,  0]])
        
        self.Q = np.matrix([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]])
        
        self.R = np.matrix([[1,0],
                           [0,1]])
        
        self.P = np.eye(self.A.shape[1])
    
    def predict(self):
        #Prédiction de l'état 
        self.E = np.dot(self.A, self.E)
        #estimation de la covariance de l'erreur
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.E
    
    def update(self, Z):
        # Gain de Kalman
        S = np.dot(np.dot(self.H, self.P),self.H.T)+self.R
        K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
        #Correction
        self.E = self.E + np.dot(K,Z-np.dot(self.H,self.E))
        I = np.eye(self.H.shape[1])
        self.P = np.dot(I-np.dot(K,self.H), self.P)
        return self.E
    
        