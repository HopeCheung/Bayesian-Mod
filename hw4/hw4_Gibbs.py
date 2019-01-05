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
K = 30

alpha = 3 / 4
a0, b0 = 0.5, 0.5
c = [i % 30 for i in range(N)] ## c 不能全为0
theta = [np.random.beta(a0, b0) for j in range(K)]
a = [a0] * K
b = [b0] * K

K_backup = []
c_backup = []
item1, item2, item3, item4, item5, item6 = [], [], [], [], [], []

def posterior(K, theta, a, b, c):
	def B(a, b):
		return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

	Z = [0 for i in range(N)]
	phi = np.array([[0.0 for j in range(K + 1)] for i in range(N)])
	nj_i = [[0 for j in range(K)] for i in range(N)]
	for i in range(N):
		for j in range(N):
			if i != j:
				nj_i[i][c[j]] = nj_i[i][c[j]] + 1

	for i in range(N):
		para_u, para = B(a0 + x[i], b0 + 20 - x[i]), B(a0, b0)
		for j in range(K):
			if nj_i[i][j] > 0:
				phi[i][j] = (binom.pmf(x[i], 20, theta[j]) * nj_i[i][j]) / (alpha + N - 1)

	#phi = np.column_stack((phi, [0 for k in range(N)]))
		phi[i][K] = (comb(20, x[i]) * para_u * alpha) / (para * (alpha + N - 1))

		Z[i] = sum(phi[i])
		for j in range(len(phi[0])):
			#if Z[i] == 0:
				#continue
			phi[i][j] = phi[i][j] / Z[i]
		c[i] = np.random.choice([l for l in range(len(phi[i]))], p = np.array(phi[i]).ravel())

	backup = {t:0 for t in range(K + 1)}
	for elem in c:
		backup[elem] = backup[elem] + 1
	items = sorted([backup[t] for t in backup])[::-1]
	if len(items) <= 6:
		items.extend([0] * (6 - len(items)))
	item1.append(items[0])
	item2.append(items[1])
	item3.append(items[2])
	item4.append(items[3])
	item5.append(items[4])
	item6.append(items[5])

	reindex = []
	new_a, new_b, new_theta = [], [], []
	for j in range(K):
		if backup[j] != 0:
			package = [x[i] for i in range(N) if c[i] == j]
			new_a.append(a[j] + sum(package))
			new_b.append(b[j] + 20 * len(package) - sum(package))
			new_theta.append(np.random.beta(a[j], b[j]))
			reindex.append(j)
		#else:
			#trash.append(j)
	if backup[K] != 0:
		new_theta.append(np.random.beta(a0, b0))        # theta for j'
		new_a.append(a0)
		new_b.append(b0)
		reindex.append(K)
	#else:
		#trash.append(K)
	#reindex = sorted(reindex)

	c = [reindex.index(r) for r in c]
	#phi = np.delete(phi, trash, axis = 1)
	K = len(reindex)
	print(K)
	K_backup.append(K)
	return K, new_theta[:], new_a[:], new_b[:], c[:]

plt.plot(item1)
plt.plot(item2)
plt.plot(item3)
plt.plot(item4)
plt.plot(item5)
plt.plot(item6)
plt.show()


plt.plot(K_backuo)
plt.show()



































