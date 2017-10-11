# 1. Find outliers in data folders and suggest a classification
# 2. After accepting classification or reclassifying manually,
#    divide the data in a temporary location to positives and negatives,
#    after which the result can be adjusted manually
# 3. Merge positives and negatives to the training data and retrain classifier

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

root = '/worktmp/piface/'
# Root folder for data
dataroot = root + 'rawdata/'

parser = argparse.ArgumentParser()
parser.add_argument('rawdatafolder', help='The folder containing the data you want to classify')
#parser.add_argument('classdatafolder', help='The folder containing the classified data')
parser.add_argument('-c', '--classifier', default=[],  nargs='+', help='Classifier for classifying')
parser.add_argument('-t', '--threshold', help='Outlier detection threshold')
parser.add_argument('-n', '--network', default=root + 'models/nn4.small2.v1.t7', help='Neural network for extracting features')
parser.add_argument('-v', '--verbose', action="store_true", help='Print more information')
args = parser.parse_args()

datafolder = args.rawdatafolder #dataroot + args.rawdatafolder

def verbose(message):
	if args.verbose:
		print message

if not datafolder.endswith('/'):
	datafolder = datafolder + '/'

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

# Check that the directory exists
if os.path.isdir(datafolder):
	threshold = None
	if args.threshold:
		threshold = args.threshold

	# Load images
	images = os.listdir(datafolder) #list(iterImgs(datafolder))
	
	# Classify images. For multiple classifiers use majority vote, then combined confidence
	# The predictions dictionary hold a pair of values for each class:
	# The number of votes and the confidence
	# Then we get final confidence by multiplying all confidences
	predictions = dict()
	imageReps = []
	negatives = 0

	for image in images:
		imageData = cv2.imread(datafolder + image)
		imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
		rep = net.forward(imageData)
		imageReps.append(rep)
				
	for classifierFile in args.classifier:
		classifier = None
		with open(classifierFile, 'r') as f:
		        classifier = pickle.load(f)

		verbose('Classifications for ' + classifierFile)

		for rep in imageReps:
			rep = rep.reshape(1, -1)
			person, confidence = normalPredict(classifier, rep)
			verbose(person)
			if person in predictions:
				(votes, conf) = predictions[person]
				predictions[person] = (votes + 1, confidence + conf)
			else:
				predictions[person] = (1, confidence)
			if person == 'negative':
				negatives += 1

	#print predictions.items()
	ordering = sorted(predictions.items(), key=lambda data: (data[1][0], data[1][1]), reverse = True)
	#print ordering
	#ordering = sorted(ordering, key=lambda label, data: data[0])
	correct = None
	answer = ''

	if len(ordering) > 0:
		print ordering[0:min(3,len(ordering))]
		correct = ordering[0][0]

		# Ask if the classification is good and/or offer possibility to type in the correct class
		answers = ['y', 'Y', 'n', 'N', 'q', 'Q']
		while answer not in answers:
			print 'Accept classification ' + correct + '? (y/n/q)'
			answer = raw_input()

	# Also allow to quit
	if answer == 'q' or answer == 'Q':
		sys.exit()
	elif answer == 'n' or answer == 'N' or correct is None:
		print 'Please enter the correct class or leave empty to quit'
		correct = raw_input()
		if correct == '' or correct == None:
			sys.exit()

	# Determine outliers
	if threshold is None:
		if negatives > 0:
			threshold = float(negatives) / float(len(images) * len(args.classifier))
		else:
			threshold = 0.1

	print 'Threshold ' + str(threshold)
	detector = IsolationForest(contamination = threshold)
	repmat = np.array(imageReps)
	print repmat.shape

	detector.fit(imageReps)
	outPred = detector.predict(imageReps)

	# Create positives/negatives directory structure and wait until the user does fixes
	metadata = zip(images, outPred)
	pFolder = datafolder + 'positive/'
	nFolder = datafolder + 'negative/'
	os.mkdir(pFolder)
	os.mkdir(nFolder)

	for (filename, pred) in metadata:
		if pred == 1:
			os.rename(datafolder + filename, pFolder + filename)
		else:
			os.rename(datafolder + filename, nFolder + filename)

	answer = ''
	answers = ['c', 'C', 'u', 'U']
	while answer not in answers:
		print 'Please adjust inlier/outlier division and input c to commit or u to undo'

		# Ask user to commit or undo
		answer = raw_input()

	if answer == 'u' or answer == 'U':
		for filename in os.listdir(pFolder):
			os.rename(pFolder + filename, datafolder + filename)
		for filename in os.listdir(nFolder):
			os.rename(nFolder + filename, datafolder + filename)
		os.rmdir(pFolder)
		os.rmdir(nFolder)
		sys.exit()

	# Move positives to classdatafolder/class and negatives to classdatafolder/negative
	for filename in os.listdir(pFolder):
		classdatafolder = root + 'data/' + correct + '/'
		if not os.path.isdir(classdatafolder):
			os.mkdir(classdatafolder)
		os.rename(pFolder + filename, classdatafolder + str(uuid.uuid4()) + '.png')
	for filename in os.listdir(nFolder):
		classdatafolder = root + 'data/negative/'
		os.rename(nFolder + filename, classdatafolder + str(uuid.uuid4()) + '.png')

	# Remove data directory
	os.rmdir(pFolder)
	os.rmdir(nFolder)
	os.rmdir(datafolder)
else:
	print datafolder + ' does not exist or is not a folder'
