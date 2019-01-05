import csv
import os
import matplotlib.pyplot as plt

userhome = os.path.expanduser('~')
def get_data(file):
	with open(file) as csvfile:
		csvReader = csv.reader(csvfile)
		data = list(csvReader)
	for i in range(len(data)):
		data[i] = list(map(eval, data[i]))
	return data

file = os.path.join(userhome, 'Desktop', 'K.csv')
K_backup = get_data(file)
K_backup = [k[0] for k in K_backup]

plt.plot(K_backup)
plt.show()

