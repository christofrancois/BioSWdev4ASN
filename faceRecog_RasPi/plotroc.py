# Given a directory with classified images (each class in a subfolder), run the given classifier and produce a ROC curve according to the confidence levels and correct-incorrect classification

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

parser = argparse.ArgumentParser()
parser.add_argument('datafolder', help='The folder containing the data you want to classify')
#parser.add_argument('classdatafolder', help='The folder containing the classified data')
parser.add_argument('classifier', help='Classifier for classifying')
parser.add_argument('network', default=root + 'models/nn4.small2.v1.t7', help='Neural network for extracting features')
#parser.add_argument('-v', '--verbose', action="store_true", help='Print more information')
args = parser.parse_args()

datafolder = args.datafolder

if not datafolder.endswith('/'):
	datafolder = datafolder + '/'

if not os.path.isdir(datafolder):
	raise

net = openface.TorchNeuralNet(
	args.network,
	imgDim=96
	)

def normalPredict((le, clf), image):
	predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
	prediction = le.inverse_transform(maxI)
        confidence = predictions[maxI]
	return prediction, confidence

classifier = None

with open(args.classifier, 'r') as f:
        classifier = pickle.load(f)

classes = [d for d in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, d))]
nClasses = len(classes)

predictionData = []

cachefile = './' + args.classifier + '_predcache.pkl'

if not os.path.isfile(cachefile):
	for label in classes:
		images = os.listdir(datafolder + label)
		print label
		nImages = len(images)

		for image in images:
			imageData = cv2.imread(datafolder + label + '/' + image)
			imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
			rep = net.forward(imageData)
			rep = rep.reshape(1, -1)
			prediction, confidence = normalPredict(classifier, rep)
			predictionData.append((confidence, prediction, label))

	predictionData.sort(reverse=True)
	with open(cachefile, 'w') as f:
		pickle.dump(predictionData, f)
else:
	with open(cachefile, 'r') as f:
		predictionData = pickle.load(f)


xs = []
ys = []
xs2 = [0]
ys2 = [0]
xs3 = []
ys3 = []
correct = 0
total = 0
truePositives = 0
falsePositives = 0
trueNegatives = 0
falseNegatives = 0

def TPR(TP, FN):
	if TP + FN == 0:
		return 0
	else:
		return float(TP)/(TP + FN)

def FPR(TN, FP):
	if TN + FP == 0:
		return 0
	else:
		return 1.0 - float(TN)/(TN + FP)

for (conf, pred, truth) in predictionData:
	if pred == truth:
		correct += 1
		if pred == 'negative':
			trueNegatives += 1
		else:
			truePositives += 1
	elif pred == 'negative':
		falseNegatives += 1
	else:
		falsePositives += 1
	total += 1
	xs.append(conf)
	ys.append(float(correct)/total)
	xs2.append(TPR(truePositives, falseNegatives))
	ys2.append(FPR(trueNegatives, falsePositives))
	xs3.append(conf)
	ys3.append(float(correct)/len(predictionData))
	#print (conf, pred, truth)
	#print str(correct) + ', ' + str(total)
	print str(conf) + ', ' + str(float(correct)/total)

plt.figure()
plt.plot(xs, ys)
plt.savefig('./suc1.pdf')
plt.plot([0, 1], [0, 1], 'r--')
plt.savefig('./suc2.pdf')
plt.show()
plt.plot(xs2, ys2)
plt.savefig('./roc.pdf')
plt.show()
plt.figure()
plt.plot(xs3, ys3)
plt.savefig('./roc1.pdf')
plt.plot([0, 1], [1, 0], 'r--')
plt.savefig('./roc2.pdf')
plt.show()

