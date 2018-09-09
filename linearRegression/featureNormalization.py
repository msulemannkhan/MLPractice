# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:40:22 2018

@author: suleman
"""
import numpy as np
def featureNormalization(x):
    x_norm = x
    mean = np.zeros((1, 2))
    sigma = np.zeros((1, 2))
    
    for i in range(x.shape[1]):
        mean[0,i] = np.mean((x[:,i]), 0) 

    for i in range(x.shape[1]):
        x_norm[:,i] -= mean[0, i]
    
    for i in range(x.shape[1]):
        sigma[0,i] = np.std((x[:,i]), 0) 
    print(sigma)
    for i in range(x.shape[1]):
        x_norm[:,i] /= sigma[0, i]   