'''
Chris Lu
10/12/16
Data from UC Riverside
http://www.cs.ucr.edu/~eamonn/discords/power_data.txt

Some other potential datasets here:
http://www.cs.ucr.edu/~eamonn/discords/

Used in this paper:
http://digilib.gmu.edu/jspui/bitstream/handle/1920/10250/alDosari_thesis_2016.pdf?sequence=1&isAllowed=y

pts is the numbers there.

'''

data_reading = open('power_data.txt')
pts = data_reading.read().split('\n')
pts = [int(pt) for pt in pts if pt != '']
data_reading.close()

