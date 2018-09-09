# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 07:43:17 2018

@author: suleman
"""

import numpy as np                         #to use numpy arrays
import matplotlib.pyplot as plt            #used to plot data

from readData import readData
from costFunction import costFunction
from normalEquation import normalEquation
from addBiaseVariableToX import addBiaseVariableToX    
from featureNormalization import featureNormalization
from gradientDescent import gradientDescent

#read data from a csv file        
x,y = readData()

#normalize all the features of x input
featureNormalization(x)

#if there is only one featuer then plot the data
if x.shape[1] == 1:
    print('There is only one feature let plot the data.')
    #plotData(x,y) 
    
#add the biase variable to the x    
x = addBiaseVariableToX(x)

#initialize theta
theta = np.zeros(x.shape[1])

print(f'Initial cost is: {costFunction(x, y, theta)}')
#for univariate_linear_regression_data.txt
#alpha = 0.01
#iterations = 700

#for assign2data1.txt
alpha = 0.01
iterations = 300

print('Running Gradient Descent')
cost, theta = gradientDescent(x, y, theta, alpha, iterations)
if x.shape[1] == 2:#if there is only one feature then draw graph of new theta value
    plt.plot(x[:,1:2] , x.dot(theta), '-')   
print(f'After Gradient Descent cost is: {costFunction(x, y, theta)}')

#debug graph to see how cost decreases with respect to iterations
print('Drawing debuging graph')
it = np.array(range(len(cost))) ;
fig, debug = plt.subplots()
debug.plot(it,cost,'-') 
debug.set(xlabel='number of iterations', ylabel='cost', title='Debugging')

#calculate theta with normal quations which minimize the cost
normalizedTheta = normalEquation(x, y)