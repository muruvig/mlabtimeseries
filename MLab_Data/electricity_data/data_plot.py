import matplotlib.pyplot as plt
data_reading = open('power_data.txt')
pts = data_reading.read().split('\n')
pts = [int(pt) for pt in pts if pt != '']
data_reading.close()
plt.plot(pts)
plt.show()
