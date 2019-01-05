import csv
import os
import math
import random
import string
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import binom
from scipy.special import digamma
from scipy.special import gammaln
from scipy.special import comb
import matplotlib.pyplot as plt

userhome = os.path.expanduser('~')
def get_data(file):
	with open(file) as csvfile:
		csvReader = csv.reader(csvfile)
		data = list(csvReader)
	for i in range(len(data)):
		data[i] = list(map(eval, data[i]))
	return data

file = os.path.join(userhome, 'Desktop', 'x.csv')
x = get_data(file)
x = np.array([k[0] for k in x])
N = len(x)
K = 15
a0, b0 = 0.5, 0.5
alpha0 = 0.1
params = []

a = [a0 for j in range(K)]
b = [b0 for j in range(K)]
alpha = [random.random() for j in range(K)]
phi = [[0 for j in range(K)] for i in range(N)]

def posterior(x, phi, a, b, alpha):
	Z = [0 for j in range(K) for i in range(N)]
	for i in range(N):
		for j in range(K):
			item1 = x[i] * (digamma(a[j]) - digamma(a[j] + b[j]))
			item2 = (20 - x[i]) * (digamma(b[j]) - digamma(a[j] + b[j]))
			item3 = digamma(alpha[j]) - digamma(sum(alpha))
			phi[i][j] = pow(math.e, item1 + item2  + item3)
		Z[i] = sum(phi[i])
		for j in range(K):
			phi[i][j] = phi[i][j] / Z[i]

	for j in range(K):
		a[j] = a0 + sum([(x[i] * phi[i][j]) for i in range(N)])
		b[j] = b0 + sum([((20 - x[i]) * phi[i][j]) for i in range(N)])


	for j in range(K):
		nj = sum([elem[j] for elem in phi])
		alpha[j] = alpha0 + nj


	step1, step2, step3, step4, step5, step6, step7 = 0, 0 ,0, 0 ,0, 0 ,0
	for i in range(N):
		for j in range(K):
			step1 = step1 + phi[i][j] * (math.log(comb(20, x[i])) + x[i] * (digamma(a[j]) - digamma(a[j] + b[j])) + (20 - x[i]) * (digamma(b[j]) - digamma(a[j] + b[j])))
			step2 = step2 + phi[i][j] * (digamma(alpha[j]) - digamma(sum(alpha)))
			step5 = step5 + phi[i][j] * math.log(phi[i][j])
	for j in range(K):
		step3 = step3 + (alpha0 - 1) * (digamma(alpha[j]) - digamma(sum(alpha)))
		step4 = step4 + (a0 - 1) * (digamma(a[j]) - digamma(a[j] + b[j])) + (b0 - 1) * (digamma(b[j]) - digamma(a[j] + b[j]))
		step6 = step6 + (a[j] - 1) * (digamma(a[j]) - digamma(a[j] + b[j])) + (b[j] - 1) * (digamma(b[j]) - digamma(a[j] + b[j])) - (gammaln(a[j]) + gammaln(b[j]) - gammaln(a[j] + b[j]))
		step7 = step7 + (alpha[j] - 1) * (digamma(alpha[j]) - digamma(sum(alpha))) 
	para = step1 + step2 + step3 + step4 - step5 - step6 - step7 + (sum([gammaln(alpha[j]) for j in range(K)]) - gammaln(sum(alpha)))
	params.append(para)
	return phi, a, b, alpha

for t in range(1000):
	phi, a, b, alpha = posterior(x, phi, a, b, alpha)

x_aliax = [i for i in range(21)]
backup = {i:np.array([0 for j in range(K)]) for i in range(21)}
for i in range(N):
	backup[x[i]] = backup[x[i]] + np.array(phi[i])
y_aliax = [list(backup[elem]).index(max(backup[elem])) for elem in backup]









