# Use cached results to produce single plot with series for multiple classifiers

import sys
import os
import argparse
from openface.data import iterImgs
from sklearn.ensemble import IsolationForest
import numpy as np
import cv2
import openface
import uuid
import pickle
import matplotlib.pyplot as plt

root = '/worktmp/piface/'

#parser = argparse.ArgumentParser()
#args = parser.parse_args()

#datafolder = args.datafolder

#if not datafolder.endswith('/'):
#	datafolder = datafolder + '/'

#if not os.path.isdir(datafolder):
#	raise

#net = openface.TorchNeuralNet(
#	args.network,
#	imgDim=96
#	)

predictionData = []

classifiers = [('GaussianNB', 'b'), ('LinearSvm', 'g'), ('RandomForest', 'r')]#, 'RadialSvm'

for (classifier, color) in classifiers:
	cacheFile = './classifier/' + classifier + '.pkl_predcache.pkl'
	tempData = None

	with open(cacheFile, 'r') as f:
		tempData = pickle.load(f)

	xs = []
	ys = []
	xs2 = []
	ys2 = []
	xs3 = []
	ys3 = []
	correct = 0
	total = 0
	falsePos = 0
	notTold = True

	for (conf, pred, truth) in tempData:
		if pred == truth:
			correct += 1
		elif not pred == 'negative':
			falsePos += 1
		total += 1
		xs.append(conf)
		ys.append(float(correct)/total)
		xs2.append(conf)
		ys2.append(float(correct)/len(tempData))
		xs3.append(conf)
		ys3.append(float(falsePos)/len(tempData))
#		if correct > 0:
#			ys3.append(float(falsePos)/float(correct))
#		else:
#			ys3.append(1)
		if notTold and falsePos > 0:
			print classifier + ": " + str(total) + ", " + str(conf)
			if not pred == 'negative':
				notTold = False

	predictionData.append((classifier, xs, ys, xs2, ys2, xs3, ys3, color))

plt.figure()

for (cls, xs, ys, xs2, ys2, xs3, ys3, color) in predictionData:
	plt.plot(xs, ys, label=cls)

plt.legend(loc=4)
plt.savefig('./suc_all.pdf')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig('./suc_all2.pdf')
plt.show()

plt.figure()

for (cls, xs, ys, xs2, ys2, xs3, ys3, color) in predictionData:
	plt.plot(xs2, ys2, label=cls)
	plt.plot(xs3, ys3, color + '--')

plt.legend()
plt.xlabel('Confidence threshold')
plt.ylabel('Success rate / False positive rate')
plt.savefig('./roc_all.pdf')
plt.plot([0, 1], [1, 0], 'k', linestyle='dotted')
plt.savefig('./roc_all2.pdf')
plt.show()

