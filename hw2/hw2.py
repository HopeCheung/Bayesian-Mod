import csv
import os
import math
import random
import string
import numpy as np
import scipy.stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from decimal import Decimal

userhome = os.path.expanduser('~')
file1 = os.path.join(userhome, 'Desktop', 'ratings.csv')
def get_data(file):
	with open(file) as csvfile:
		csvReader = csv.reader(csvfile)
		rating = list(csvReader)
	for i in range(len(rating)):
		rating[i] = list(map(eval, rating[i]))
	return rating
rating = get_data(file1)
N, M = max([elem[0] for elem in rating]), max(elem[1] for elem in rating)

def produce_R(rating):
	R = [[0 for j in range(M)] for i in range(N)]
	for elem in rating:
		R[elem[0]-1][elem[1]-1] = elem[2]
	return R

R = produce_R(rating) 
U = np.random.randn(N, 5)
V = np.random.randn(M, 5)

def parameter(U, V, R):
	para = [[0 for j in range(M)] for i in range(N)]
	for i in range(N):
		for j in range(M):
		    mean, sigma = np.dot(U[i], V[j]), 1
		    if R[i][j] == 1:
		    	lower = 0
		    	upper = 999
		    	para[i][j] = scipy.stats.truncnorm.rvs((lower-mean)/sigma,(upper-mean)/sigma,loc=mean,scale=sigma)
		    elif R[i][j] == -1:
		    	lower = -999
		    	upper = 0
		    	para[i][j] = scipy.stats.truncnorm.rvs((lower-mean)/sigma,(upper-mean)/sigma,loc=mean,scale=sigma)
	return para
#phi = parameter(U, V, R)
#---------------------------------------initail-------------------------------------------
def update_E(R, U, V):
        E = [[0 for j in range(M)] for i in range(N)]
        for i in range(N):
                for j in range(M):
                        s = np.dot(U[i], V[j])
                        sigma = 1
                        if R[i][j] == 1:
                                E[i][j] = s + (sigma * norm.pdf(-s) / (norm.cdf(-s))) #cdf and pdf
                        elif R[i][j] == -1:
                                E[i][j] = s + (-sigma * norm.pdf(-s) / (norm.cdf(s)))
        return E

def update_UV(R, U, V, E):
	sigma = 1
	U, V = np.array(U), np.array(V)
	initial_v, initial_u = np.identity(5) + np.dot(np.transpose(V), V) / sigma, np.identity(5) +  np.dot(np.transpose(U), U) / sigma
	for i in range(N):
		first_v = initial_v # identity：单位矩阵
		second_v = 0
		for j in range(M):
			#first_v = first_v + np.multiply(V[j].reshape(-1, 1), V[j]) / sigma #reshape：行向量转列
			second_v = second_v + (V[j] * E[i][j]) / sigma
		U[i] = np.dot(np.linalg.inv(first_v), second_v) # dot:点积
	for j in range(M):
		first_u = initial_u
		second_u = 0
		for i in range(N):
			#first_u = first_u + np.multiply(U[i].reshape(-1, 1), U[i]) / sigma
			second_u = second_u + (U[i] * E[i][j]) / sigma
		V[j] = np.dot(np.linalg.inv(first_u), second_u) #linalg.nv 求逆矩阵
	return U, V

def ln_RUV(U, V, E, R):
        sigma = 1
        sum_U, sum_V = 0, 0
        for i in range(N):
                sum_U = sum_U + np.dot(U[i], U[i])
        for j in range(M):
                sum_V = sum_V + np.dot(V[j], V[j])
        result = 5 * math.log(1/(2 * math.pi)) - 0.5 * sum_U - 0.5 * sum_V
        for i in range(N):
                for j in range(M):
                        s = np.dot(U[i], V[j])
                        if R[i][j] == 1:
                                result = result + math.log(norm.cdf(s/sigma))
                        elif R[i][j] == -1:
                                result = result + math.log(1 - norm.cdf(s/sigma))
        return result
        

def iteration(U, V, R, iter_num):
	result = []
	for k in range(iter_num):
		E = update_E(R, U, V)
		U, V = update_UV(R, U, V, E)
		result.append(ln_RUV(U, V, E, R))
	return U, V, result

#---------------------problem1----------------------------------------
initial_U = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U, train_V, result = iteration(initial_U, initial_V, R, 100)
plt.plot(result1[2:])
plt.show()
#---------------------problem2----------------------------------------
initial_U1 = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V1 = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U1, train_V1, result1 = iteration(initial_U1, initial_V1, R, 100)

initial_U2 = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V2 = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U2, train_V2, result2 = iteration(initial_U2, initial_V2, R, 100)

initial_U3 = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V3 = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U3, train_V3, result3 = iteration(initial_U3, initial_V3, R, 100)

initial_U4 = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V4 = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U4, train_V4, result4 = iteration(initial_U4, initial_V4, R, 100)

initial_U5 = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V5 = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U5, train_V5, result5 = iteration(initial_U5, initial_V5, R, 100)

plt.plot(result1[20:])
plt.plot(result2[20:])
plt.plot(result3[20:])
plt.plot(result4[20:])
plt.plot(result5[20:])
plt.show()

#--------------------problem3------------------------------------------
userhome = os.path.expanduser('~')
file2 = os.path.join(userhome, 'Desktop', 'ratings_test.csv')
test = get_data(file2)

initial_U = np.array([np.random.normal(0, 0.1, 5) for i in range(N)])
initial_V = np.array([np.random.normal(0, 0.1, 5) for j in range(M)])
train_U, train_V, result = iteration(initial_U, initial_V, R, 100)
predict_result = np.dot(train_U, np.transpose(train_V))

predict_11, predict_10, predict_00, predict_01 = 0, 0, 0, 0
for elem in test:
        if elem[2] == 1:
                if predict_result[elem[0]-1][elem[1]-1] > 0:
                        predict_11 = predict_11 + 1
                else:
                        predict_10 = predict_10 + 1
        elif elem[2] == -1:
                if predict_result[elem[0]-1][elem[1]-1] > 0:
                        predict_01 = predict_01 + 1
                else:
                        predict_00 = predict_00 + 1
result = [[predict_11, predict_01], [predict_00, predict_01]]
accuracy = (predict_11 + predict_00) / (predict_11 + predict_01 + predict_10 + predict_00)


                













