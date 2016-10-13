'''
Chris Lu
10/5/16
Data from UCI's building data.
http://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts

data_in represents people leaving the building at time/date.
data_out represents people entering the building at time/date.
For each time/date there is both a data_in and data_out point.
time/date occurs in increments of 30 minutes, so the points are evenly spaced
time/date is currently in a string, but I can change that if needed.
http://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/CalIt2.data
The code is written this way because I'm not completely certain that it sticks to the pattern
of 30 minute intervals and alternating 7's and 9's.

The events parts stores the times of events. Each event has a time begin and time end as well as a date.
The events are stored as lists of lists, where each event is a point and within each event the first
value is the date and the second and third are time begin and end respectively.
http://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/CalIt2.events




'''

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

