# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:39:34 2018

@author: suleman
"""
import numpy as np      
from costFunction import costFunction

def gradientDescent(x, y, theta, alpha, iterations):
    J = [None]
    for i in range(iterations-1):
        pridiction = x.dot(theta)
        error = pridiction - y
        cost = (((np.transpose(error)).dot(x))/(len(x)))
        theta = theta - (alpha * cost)
        J.append(costFunction(x, y, theta))
    return J, theta