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

#############################
#    Exact Least Squares    #
#############################
A = data[:,2:] # get third column to end
b = data[:,1] # get second column

theta_ls = np.linalg.inv(A.T * A)*A.T*b
print(theta_ls) # unsurprisingly, all values are 0.1
print (T) 

#############################
#      Gradient Descent     #
#############################

def gradientDescent(x, y, theta, alpha, m, max_it):
    xTrans = x.transpose()
    for i in range(0, max_it):
        predicted_y = x.dot(theta)
        loss = predicted_y-y                  # residual 
        cost = np.sum(loss ** 2) / (2 * m)    # average cost per example 
        gradient = np.dot(xTrans, loss) / m   # avg gradient per example
        theta = theta - alpha * gradient      # update
    return theta

alpha = 0.01
max_it = 1000
m, n = np.shape(A)
theta_gd = np.ones(n)                         # initializing gd parameter
theta_gd = gradientDescent(A, b, theta_gd, alpha, m, max_it)
print(theta_gd)                               # print out final parameter

#############################
#   Test Set Performance    #
#############################
#
# 1: Import test set data + parse
# 2: Apply model to test set

#############################
#   Write Ouput to File    #
#############################
#
# 1: open file for output (results.csv)
# 2: write appropriate header line
# 3: write PROPERLY FORMATTED results to file
# 4: close file



