import csv
import os
import math
import random
import string
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.special import digamma
import matplotlib.pyplot as plt

userhome = os.path.expanduser('~')
def get_data(file):
	with open(file) as csvfile:
		csvReader = csv.reader(csvfile)
		data = list(csvReader)
	for i in range(len(data)):
		data[i] = list(map(eval, data[i]))
	return data

file1 = os.path.join(userhome, 'Desktop', 'X_set1.csv')
file2 = os.path.join(userhome, 'Desktop', 'y_set1.csv')
file3 = os.path.join(userhome, 'Desktop', 'z_set1.csv')
x1, y1, z1 = get_data(file1), get_data(file2), get_data(file3)
x1, y1, z1 = np.array(x1), np.array(y1), [item[0] for item in z1] 

N, d = len(y1), len(x1[0])
a0, b0, e0, f0 = 1e-16, 1e-16, 1, 1
#----------------------------------parameters-----------------------------------
def ln_gamma(num):
	s = 0
	for i in range(1, num):
		s = s + math.log(i)
	return s

def Diag_Alpha(alpha_a, alpha_b):
	Eqa = [alpha_a[i] / alpha_b[i] for i in range(d)]
	diag_alpha = [[0 for j in range(d)] for i in range(d)]

	for i in range(d):
		diag_alpha[i][i] = Eqa[i]
	return np.array(diag_alpha)

def Eq_Lam(e, f):
	return e / f

def U(sigma, x, y, Eqlam):
	total = 0
	for i in range(N):
		total = total + y[i] * x[i]
	return np.dot(sigma, total) * Eqlam

def Sigma(diag_alpha, x, Eqlam):
	total = 0
	for i in range(N):
		total = total + np.multiply(x[i].reshape(-1, 1), x[i])
	return np.linalg.inv(diag_alpha + Eqlam * total)

def Eqw(x, y, u, sigma, i):
	return (y[i] - np.dot(x[i], u)) * (y[i] - np.dot(x[i], u)) + np.dot(np.dot(x[i], sigma), x[i]) 

def Ew2(sigma, u, k):
	return sigma[k][k] + u[k] * u[k]

#---------------------------------iterations------------------------------------------
def iterations(x, y, times):
	e, f = e0, f0
	alpha_a, alpha_b = [a0 for k in range(d)], [b0 for k in range(d)]
	EQLAM, ALPHA_A, ALPHA_B, E, F, SIGMA, UU = [], [], [], [], [], [], []
	for time in range(times):
		Eqlam = Eq_Lam(e, f)
		diag_alpha = Diag_Alpha(alpha_a, alpha_b)
		sigma = Sigma(diag_alpha, x, Eqlam)
		u = U(sigma, x, y, Eqlam)

		e, f = e0, f0
		e = e0 + N / 2
		for i in range(N):
			f  = f + 0.5 * Eqw(x, y, u, sigma, i)
		f = f[0]
		
		for k in range(d): 
			alpha_a[k] = a0 + 0.5
			alpha_b[k] = b0 + 0.5 * Ew2(sigma, u, k)

		EQLAM.append(Eqlam)
		ALPHA_A.append(alpha_a[:])
		ALPHA_B.append(alpha_b[:])
		E.append(e)
		F.append(f)
		SIGMA.append(sigma[:])
		UU.append(u[:])

	return EQLAM, ALPHA_A, ALPHA_B, E, F, SIGMA, UU
#-----------------------judge converge------------------------------------------------
def VI(alpha_a, alpha_b, e, f, x, y, sigma, u):
	item1, item2, item3, item4, item5, item6, item7 = 0, 0, 0, 0, 0, 0, 0

	total1 = 0
	for i in range(N):
		total1 = total1 + e/f * ((y[i] - np.dot(x[i], u)) * (y[i] - np.dot(x[i], u)) + np.dot(np.dot(x[i], sigma), x[i]))
	item1 = N/2 * (digamma(e) - math.log(f)) - 0.5 * total1 

	item2 = (e0 - 1) * (digamma(e) - math.log(f)) - f0 * (e / f)

	total3 = 0
	for i in range(d):
		Eqalphai = alpha_a[i] / alpha_b[i]
		total3 = total3 + 0.5 * (digamma(alpha_a[i]) - math.log(alpha_b[i])) - 0.5 * Eqalphai * (sigma[i][i] + u[i] * u[i])
	item3 = total3

	total4 = 0
	for i in range(d):
		total4 = total4 + (a0 - 1) * (digamma(alpha_a[i]) - math.log(alpha_b[i])) - b0 * alpha_a[i] / alpha_b[i]
	item4 = total4

	#item5 = e - math.log(f) + math.log(math.gamma(e)) + (1 - e) * digamma(e)
	item5 = e - math.log(f) + ln_gamma(int(e)) + (1 - e) * digamma(e)
    
	item6 = 0.5 * math.log(np.linalg.det(sigma))

	total7 = 0
	for i in range(d):
		total7 = total7 + alpha_a[i] - math.log(alpha_b[i]) + math.log(math.gamma(alpha_a[i])) + (1 - alpha_a[i]) * digamma(alpha_a[i])
	item7 = total7

	return item1 + item2 + item3 + item4 + item5 + item6 + item7

EQLAM, ALPHA_A, ALPHA_B, E, F, SIGMA, UU = iterations(x1, y1, 500)

def VI_picture():
	result = []
	for i in range(500):
		result.append(VI(ALPHA_A[i], ALPHA_B[i], E[i], F[i], x1, y1, SIGMA[i], UU[i])[0])
	plt.plot(result)
	plt.show()
	return result

def Alpha_picture():
	result = []
	for i in range(d):
		result.append(ALPHA_B[499][i] / ALPHA_A[499][i])
	plt.plot(result)
	plt.show()
	return result

def Lam_picture():
	result = []
	for i in range(500):
		result.append(F[i] / E[i])
	plt.plot(result)
	plt.show()
	return result

def Omega_picture(x1, y1, z1):
	u = UU[499]
	y = [np.dot(x1[i], u) for i in range(N)]
	plt.plot(z1, y)
	plt.scatter(z1, y1)
	plt.plot(z1, 10 * np.sinc(z1))
	plt.show()
























