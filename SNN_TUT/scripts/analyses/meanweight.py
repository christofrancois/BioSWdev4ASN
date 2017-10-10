#!/usr/bin/python
import sys


#THIS SCRIPT ALLOWS TO MEAN THE WEIGHT OF MANY CONNECTIONS
#RECQUIRE FILENAME INPUT AND REDIRECT OUTPUT TO DESIRED FILE
if len(sys.argv) > 1:
	my_file = str(sys.argv[1])
	#print("file found : %s" % my_file)

	try:
		weight_file = open(my_file, "r")
	except:
		print("open failure")
		exit()

	for line in weight_file:
		line.replace("\n","")
		arr = line.split(" ")
		time = arr[0]
		sum_value = 0
		length = len(arr) - 1
		for i in range(1, length):
			sum_value += float(arr[i])
		sum_value /= length - 1
		print("%s %s" % (time, sum_value))

	weight_file.close()

else:
	print("undefined file to load")
