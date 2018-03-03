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

#############################
#      Gradient Descent     #
#############################
#
# 1: initialize a random guess for theta
# 2: set gradient descent constants (alpha, iter_max, eps, etc..)
# 3: while end conditions not met:
#    3a: compute gradient
#    3b: update parameters
# 4: print out final parameters (theta_gd)

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



