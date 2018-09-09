# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:34:55 2018

@author: suleman
"""

import numpy as np          
#read csv file and return x and y
def readData():
    filename = 'assign2data1.txt'
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",")
    x = data[:,0:(data.shape[1]-1)]
    y = data[:,(data.shape[1]-1)]
    return x,y