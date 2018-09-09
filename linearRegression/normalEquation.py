# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:40:39 2018

@author: suleman
"""
from numpy.linalg import inv
import numpy as np     
def normalEquation(x, y):        
        return (inv(np.transpose(x).dot(x))).dot((np.transpose(x).dot(y)))