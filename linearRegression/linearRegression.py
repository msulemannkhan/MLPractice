# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 07:43:17 2018

@author: suleman
"""

import numpy as np                         #to use numpy arrays
import matplotlib.pyplot as plt            #used to plot data
import pandas as pd                        #used to read a csv file
from numpy.linalg import inv

#read csv file and return x and y
def readData():
    filename = 'assign2data1.txt'
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",")
    x = data[:,0:(data.shape[1]-1)]
    y = data[:,(data.shape[1]-1)]
    return x,y

#plot 2d data
def plotData(x, y):
    plt.plot(x, y, 'o')
    
#add the biase variable x0 to x   
def addBiaseVariableToX(x):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0, x))
    return x

def costFunction(x, y, theta):
    pridiction = x.dot(theta)
    error = pridiction - y
    cost = (error.dot(np.transpose(error)))/(2 * len(x))
    return cost

def gradientDescent(x, y, theta, alpha, iterations):
    J = [None]
    for i in range(iterations-1):
        pridiction = x.dot(theta)
        error = pridiction - y
        cost = (((np.transpose(error)).dot(x))/(len(x)))
        theta = theta - (alpha * cost)
        J.append(costFunction(x, y, theta))
    return J, theta

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
        
def normalEquation(x, y):        
        return (inv(np.transpose(x).dot(x))).dot((np.transpose(x).dot(y)))
        
x,y = readData()
featureNormalization(x)
if x.shape[1] == 1:
    print('There is only one feature let plot the data.')
    #plotData(x,y) 
x = addBiaseVariableToX(x)
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
print(theta)
if x.shape[1] == 2:
    plt.plot(x[:,1:2] , x.dot(theta), '-')   
print(f'After Gradient Descent cost is: {costFunction(x, y, theta)}')

print('Drawing debuging graph')
it = np.array(range(len(cost))) ;
fig, debug = plt.subplots()
debug.plot(it,cost,'-') 
debug.set(xlabel='number of iterations', ylabel='cost', title='Debugging')


print(normalEquation(x, y))