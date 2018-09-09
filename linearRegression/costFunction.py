# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:38:00 2018

@author: suleman
"""
import numpy as np     
def costFunction(x, y, theta):
    pridiction = x.dot(theta)
    error = pridiction - y
    cost = (error.dot(np.transpose(error)))/(2 * len(x))
    return cost