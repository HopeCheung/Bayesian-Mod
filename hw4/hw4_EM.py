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
params = []
K = 3


theta = [random.random() for k in range(K)]
#theta = [0.832373274399184, 0.106054309869906, 0.192658815744511, 0.273144856880746, 0.387842032224085, 0.243948123019705, 0.474722195921539, 0.800971820491613, 0.972863825209636, 0.559208388784999, 0.0191908917134059, 0.792686462760599, 0.912599057442009, 0.268146280635642, 0.131616256550232]
pi = [(1 / K) for k in range(K)]
phi = [[0 for j in range(K)] for i in range(N)]

def posterior(x, theta, pi, phi):
	Z = [0 for i in range(N)]

	def Theta(j, phi, nj):
		result = 0
		for i in range(N):
			result = result + x[i] * phi[i][j]
		return result / (20 * nj)

	for i in range(N):
		for j in range(K):
			phi[i][j] = pi[j] * binom.pmf(x[i], 20, theta[j])
		Z[i] = sum(phi[i])
		for j in range(K):
			phi[i][j] = phi[i][j] / Z[i]

	for j in range(K):
		nj = sum([elem[j] for elem in phi])
		pi[j] = nj / N
		theta[j] = Theta(j, phi, nj)

	para, Eq_lnq = 0, 0
	for i in range(N):
		for j in range(K):
			Eq_lnq = Eq_lnq + phi[i][j] * math.log(phi[i][j])
			para = para + phi[i][j] * (math.log(binom.pmf(x[i], 20, theta[j])) + math.log(pi[j]))
	params.append(para - Eq_lnq)

	return theta, pi, phi

for i in range(50):
	theta, pi, phi = posterior(x, theta, pi, phi)


x_aliax = [i for i in range(21)]
pc = [[0 for j in range(K)] for i in range(21)]
for i in range(21):
	for j in range(K):
		pc[i][j] = pi[j] * binom.pmf(x_aliax[i], 20, theta[j])
y_aliax = [(elem.index(max(elem)) + 1) for elem in pc]

plt.scatter(x_aliax, y_aliax)




