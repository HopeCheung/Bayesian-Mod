import csv
import os
import math
import random
import string
from decimal import Decimal

userhome = os.path.expanduser('~')

file1 = os.path.join(userhome, 'Desktop', 'X_test.csv')
with open(file1) as csvfile1:
    csvReader1 = csv.reader(csvfile1)
    x_test = list(csvReader1)
for i in range(len(x_test)):
    x_test[i] = list(map(eval, x_test[i]))
#x_test = [[elem[i] for elem in curr_x_test] for i in range(len(curr_x_test[0]))]
        
file2 = os.path.join(userhome, 'Desktop', 'X_train.csv')
with open(file2) as csvfile2:
    csvReader2 = csv.reader(csvfile2)
    x_train = list(csvReader2)
for j in range(len(x_train)):
    x_train[j] = list(map(eval, x_train[j]))
#x_train = [[elem[i] for elem in curr_x_train] for i in range(len(curr_x_train[0]))]

y_test = []          
file3 = os.path.join(userhome, 'Desktop', 'label_test.csv')
with open(file3) as csvfile3:
    csvReader3 = csv.reader(csvfile3)
    for content in csvReader3:
        y_test.extend(content)
y_test = list(map(eval, y_test))

y_train = []        
file4 = os.path.join(userhome, 'Desktop', 'label_train.csv')
with open(file4) as csvfile4:
    csvReader4 = csv.reader(csvfile4)
    for content in csvReader4:
        y_train.extend(content)
y_train = list(map(eval, y_train))
###------------import csv-------------------------------------------------------
def gamma(start, end):
    result = 1
    while start >= end:
        result, start = result * start, start - 1
    return result
    
def classifier(x_train, y_train, x):
    y1 = Decimal((sum(y_train) + 1) / (len(y_train) + 2))
    y0 = Decimal((1 + len(y_train) - sum(y_train)) / (len(y_train) + 2))
    posterior_x0, posterior_x1 = 1, 1
    for i in range(len(x)):
        a0 = 1 + sum([x_train[j][i] for j in range(len(x_train)) if y_train[j] == 0])
        b0 = 1 + sum(y_train)
        a1 = 1 + sum([x_train[j][i] for j in range(len(x_train)) if y_train[j] == 1])
        b1 = 1 + len(y_train) - sum(y_train)
        one_posterior_x0 = (Decimal(pow(b0/(b0+1), a0)) * Decimal(gamma(a0+x[i]-1, a0))) / Decimal((pow(b0+1, x[i])) * Decimal(math.factorial(x[i])))
        one_posterior_x1 = (Decimal(pow(b1/(b1+1), a1)) * Decimal(gamma(a1+x[i]-1, a1))) / Decimal((pow(b1+1, x[i])) * Decimal(math.factorial(x[i])))
        posterior_x0 = posterior_x0 * one_posterior_x0
        posterior_x1 = posterior_x1 * one_posterior_x1
        
    posterior_y1 = posterior_x1 * y1
    posterior_y0 = posterior_x0 * y0
    if posterior_y1 > posterior_y0:
        return 1
    else:
        return 0
###------------classifier-------------------------------------------------------
def test(x_train, y_train, x_test, y_test):
    result = [[0,0],[0,0]]
    #result[0][0]:0 in test,       result[0][1]:1 in test;
    #result[1][0]:0 in prediction, result[1][1]:1 in prediction
    result[0][0] = len(y_test) - sum(y_test)
    result[0][1] = sum(y_test)
    for x_vector in x_test:
        if classifier(x_train, y_train, x_vector) == 0:
            result[1][0] = result[1][0] + 1
        else:
            result[1][1] = result[1][1] + 1
    return result
###-----------test----------------------------------------------------------------
import os
import string
import sys

userhome = os.path.expanduser('~')
file = os.path.join(userhome, 'Desktop', 'README.txt')
read_me = []
with open(file) as f:
    line = f.readline
    while line:
        read_me.append(line)
        line = f.readline()       
read_me = read_me[4:]
for i in range(len(read_me)):
    read_me[i] = read_me[i][:-1]
###-------------------------read file--------------------------------------------------
x1 = [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0]
x2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0]
x3 = [0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 16, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x1, 'ro')
plt.show()
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x2, 'ro')
plt.show()
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x3, 'ro')
plt.show()
###-------plot for problem4c-----------------------------------------------------------
x1 = [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0]
x2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0]
x3 = [0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 16, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x1, 'ro')
plt.show()
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x2, 'ro')
plt.show()
import matplotlib.pyplot as plt
plt.xlim(xmax=54,xmin=0)
plt.plot(read_me, x3, 'ro')
plt.show()
###-------plot for problem4d-----------------------------------------------------------
    






