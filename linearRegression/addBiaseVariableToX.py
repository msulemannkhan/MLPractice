# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:37:06 2018

@author: suleman
"""
import numpy as np
#add the biase variable x0 to x   
def addBiaseVariableToX(x):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0, x))
    return x