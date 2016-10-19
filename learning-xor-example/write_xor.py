import csv
import random

with open('xor.csv','w', newline='') as csvfile:
    xorwriter = csv.writer(csvfile, delimiter = ',')
    for i in range(0, 100000):
        first = random.randint(0, 1)
        second = random.randint(0, 1)
        third = 0

        if first != second:
            third = 1

        xorwriter.writerow([first] + [second] + [third])
