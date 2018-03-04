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
y = data[:,1] # get second column
d = np.shape(A)[1] # number of parameters (should be 10)

theta_ls = np.linalg.inv(A.T * A)*A.T*y
print(theta_ls) # unsurprisingly, all values are 0.1 
print('\n')

#############################
#      Gradient Descent     #
#############################
def gradientDescent(A, y, theta, alpha, eps, max_it):
    i = 1
    cur_err = np.linalg.norm(A*theta-y)     # compute error of initial guess
    n = np.shape(A)[0]                      # number of training samples
    while (i < max_it and cur_err > eps): # while loop on iterations and error threshold
        grad = 2*A.T*(A*theta - y) / n      # normalize by num training samples
        theta = theta - alpha * grad        # update
        cur_err = np.linalg.norm(A*theta-y) # current error
        print(cur_err)

    return theta

alpha = 10**-7 # step rate
max_it = 1000 # max iteration flag
eps = 10**-5 # final error tolerance
theta_gd = np.matrix(np.random.rand(d,1)) # initializing gd parameter
theta_gd = gradientDescent(A, y, theta_gd, alpha, eps, max_it)
print('\n')
print(theta_gd) # print out final parameter

#############################
#   Test Set Performance    #
#############################
#
# 1: Import test set data + parse
test_set = np.genfromtxt('test.csv', delimiter=',')
test_set = np.delete(test_set, 0, 0) # remove first row
test_set = np.matrix(test_set)

# 2: Apply model to test set
test_A = test_set[:,1:]
test_y = test_A*theta_ls
y_act = np.mean(test_A, axis=1) # true output (just the mean)

print("Model Output on Test Set: \n")
print(test_y)
print("\n")
print("Average Across Columns: \n")
print(y_act)
print("\n")
print("Total Error: ", np.linalg.norm(test_y - y_act))

#############################
#   Write Ouput to File    #
#############################
output = np.concatenate( (test_set[:,0],y_act), axis=1)
np.savetxt('results.csv', output, fmt='%d,%.13f', newline='\n', header='Id,y', comments='')  




