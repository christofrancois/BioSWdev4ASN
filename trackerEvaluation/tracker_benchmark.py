import argparse
import os
import csv
import cv2
import numpy as np
from time import clock
import pickle

trackers = [
	  'MIL'
	, 'MEDIANFLOW'
	, 'BOOSTING'
	, 'TLD'
	, 'KCF'
	]

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='The folder containing the dataset to run the tracker on')
parser.add_argument('tracker', help='Type of tracker to use', choices=trackers)
parser.add_argument('-g', '--ground', default='groundtruth_rect.txt', help='Ground truth file')
parser.add_argument('-i', '--ident', default='', help='Additional result folder identifier')
# The example feature was added as an afterthought and run on a desktop.
parser.add_argument('-e', '--examples', action = 'store_true', help='Pick example frames without saving results')
parser.add_argument('--frames', dest='eframes', metavar='N', type=int, nargs='+', help='frame number to take an example from')
args = parser.parse_args()

# Parses the rectangles from the text file containing the ground truth rectangles
def parseRects(rectFile, delimiter = ','):
	rows = []
	rectReader = csv.reader(rectFile, delimiter = delimiter, skipinitialspace = True)
	for row in rectReader:
		rows.append(tuple(map(int, row)))
	return rows

rects = []
with open(args.dataset + '/' + args.ground, 'r') as groundTruth:
        try:
		rects = parseRects(groundTruth)
	except ValueError:
		rects = parseRects(groundTruth, '\t')

imageFilenames = sorted(os.listdir(args.dataset + '/img/'))

namePartition = args.dataset.rsplit('/')
datasetName = namePartition[-1]
if datasetName == '':
	datasetName = namePartition[-2]

firstImage, images = imageFilenames[0], imageFilenames[1:]
firstROI, rects = rects[0], rects[1:]

if datasetName == 'David':
	firstImage, images = imageFilenames[300], imageFilenames[301:]
	firstROI, rects = rects[300], rects[301:]

print len(images), len(rects)

firstImage = cv2.imread(args.dataset + '/img/' + firstImage)
(imwidth, imheight, imdepth) = firstImage.shape

def jaccard((x1, y1, w1, h1), (x2, y2, w2, h2)):
	ix = max(x1, x2)
	iy = max(y1, y2)
	iw = max(min(x1 + w1, x2 + w2) - ix, 0)
	ih = max(min(y1 + h1, y2 + h2) - iy, 0)
	intersect = iw * ih
	union = w1 * h1 + w2 * h2 - intersect
	return intersect / union

jaccards = list()

# Initialize tracker
tracker = cv2.Tracker_create(args.tracker)
tracker.init(firstImage, firstROI)
score = 0
time = 0
cv2.namedWindow('Tracker')

minlen = min(len(images), len(rects))
exframes = args.eframes
if args.examples:
	os.mkdir('./examples/' + args.tracker)
	if exframes is None:
		exframes = list(np.linspace(0, minlen, 5))

for i in range(min(len(images), len(rects))):
	image = cv2.imread(args.dataset + '/img/' + images[i])
	tic = clock()
	_, ROI = tracker.update(image)
	time += clock() - tic
	J = jaccard(ROI, rects[i])
	jaccards.append(J)
	score += J
	(x, y, w, h) = np.int0(rects[i])
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	(x, y, w, h) = np.int0(ROI)
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
	cv2.imshow('Tracker', image)
	if i in exframes:
		cv2.imwrite('./examples/' + args.tracker + '/' + str(i) + '.png', image)
	cv2.waitKey(1)

print datasetName
print args.tracker
print 'Score: ' + str(score) + ', Average: ' + str(score / len(images))
print 'Time: ' + str(time) + ', FPS: ' + str(len(images) / time)

if not os.path.exists('./results/' + datasetName + args.ident):
	os.mkdir('./results/' + datasetName + args.ident)

fullpath = './results/' + datasetName + args.ident + '/' + args.tracker + '.pkl'
with open(fullpath, 'w') as datafile:
	pickle.dump((datasetName, args.tracker, score, time, min(len(images), len(rects)), jaccards, (imwidth, imheight, imdepth)), datafile)
