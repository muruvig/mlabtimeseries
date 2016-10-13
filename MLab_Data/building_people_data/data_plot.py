import numpy as np
import matplotlib.pyplot as plt

data_reading = open('CalIt2.data.txt')
hr = data_reading.readline()
data_out,date,time,data_in = [],[],[],[]
while hr:
	parts = hr.split(',')
	if parts[0] == '7':
		data_out.append(int(parts[3]))
	else:
		data_in.append(int(parts[3]))
		time.append(parts[2])
		date.append(parts[1])
	hr = data_reading.readline()
	if hr == '\n':
		hr = data_reading.readline()
data_reading.close()

data_reading = open('CalIt2.events.txt')
pts = data_reading.read().split('\n')
events = [pt.split(',')[0:3] for pt in pts]

net = [data_in[i] - data_out[i] for i in range(len(data_out))]

plt.plot(net)
plt.show()



