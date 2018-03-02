#################################
#          IML_TO               #
#################################
#
# File Name: main.py
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz.ch)

import numpy as np
import sys
import os
import csv

# Import training data
data = np.genfromtxt('train.csv', delimiter=',')
data = np.delete(data, 0, 0) # remove first row
data = np.matrix(data)

A = data[:,2:] # get third column to end
b = data[:,1] # get second column

theta = np.linalg.inv(A.T * A)*A.T*b
print(theta) # unsurprisingly, all values are 0.1

