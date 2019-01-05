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

file = file = os.path.join(userhome, 'Desktop', 'Q3_Max_Six.csv')
items = get_data(file)
print(items)
item1 = items[0]
item2 = items[1]
item3 = items[2]
item4 = items[3]
item5 = items[4]
item6 = items[5]

plt.plot(item1)
plt.plot(item2)
plt.plot(item3)
plt.plot(item4)
plt.plot(item5)
plt.plot(item6)
plt.show()
