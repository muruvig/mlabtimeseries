'''
Chris Lu
10/12/16
Data from UC Riverside
http://www.cs.ucr.edu/~eamonn/discords/

Used in this paper:
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf

there are three rocket data sets here.

They are stored in pts[0],pts[1], and pts[2]
'''
pts = [[],[],[]]
titles = ['TEK14.txt','TEK16.txt','TEK17.txt']
for i in range(3):
	data_reading = open(titles[i])
	pts[i] = data_reading.read().split('\n')
	pts[i] = [float(pt)  for pt in pts[i] if pt != '']
	data_reading.close()

