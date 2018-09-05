# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:37:04 2018

@author: suleman
"""

import numpy as np                         #to use numpy arrays
import matplotlib.pyplot as plt            #used to plot data
import pandas as pd                        #used to read a csv file



#read data from csv file
dataFrame = pd.read_csv("univariate_linear_regression_data.txt")
x1 = dataFrame['x']
y = dataFrame['y']

# add x0 at the back of x colum [x0, x1] 100 rows, 2 columns
x1 = np.array(x1)

x0 = np.ones((len(x1),))
x = np.stack((x0, x1), axis=-1)
theta = np.zeros(x.shape[1])

def costFunction(x, y, theta):
    pridiction = x.dot(theta)
    error = pridiction - y
    cost = (error * error)/(2 * len(x))
    return cost

def gradientDescent(x, y, theta, alpha, iterations):
    for i in range(5000):
        pridiction = x.dot(theta)
        error = pridiction - y
        theta = theta - (alpha * (((np.transpose(error)).dot(x))/(len(x))))
    return theta
    
#