import os
import argparse
import numpy as np
import cv2
import openface
import uuid
import pickle
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

root = '/worktmp/piface/'
# Root folder for data
dataroot = root + 'rawdata/'
resultroot = root + 'processed/'

parser = argparse.ArgumentParser()
parser.add_argument('image', help = 'Image file to classify')
parser.add_argument('-c', '--classifier', default = root + 'classifier/LinearSvm.pkl', help = 'Classifier for classifying')
parser.add_argument('-n', '--network', default = root + 'models/nn4.small2.v1.t7', help = 'Neural network for extracting features')
parser.add_argument('-v', '--verbose', action = "store_true", help = 'Print more information')
parser.add_argument('-t', '--threshold', type = float, default = 0.95, help = 'Positive classification confidence threshold')
parser.add_argument('-i', '--info', action = "store_true", help = 'Save the prediction and confidence score along with the moved image')
args = parser.parse_args()

def verbose(message):
	if args.verbose:
		print message

net = openface.TorchNeuralNet(
	args.network,
	imgDim=96
	)

classifier = None
with open(args.classifier, 'r') as f:
	classifier = pickle.load(f)

def predict((le, clf), rep):
	predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
	prediction = le.inverse_transform(maxI)
        confidence = predictions[maxI]
	return prediction, confidence

imageData = cv2.imread(args.image)
imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
rep = net.forward(imageData)
prediction, confidence = predict(classifier, rep)

if confidence >= args.threshold:
	resultdir = resultroot + prediction + '/'
	mkdir_p(resultdir)
	newname = str(uuid.uuid4()) + '.png'
	os.rename(args.image, resultdir + newname)
	verbose('Classified to ' + prediction + ' (' + str(confidence) + ')')
	with open(resultdir + newname + '.pkl', 'w') as metadata:
		pickle.dump((prediction, confidence), metadata)
else:
	verbose('No classification, best guess ' + prediction + ' (' + str(confidence) + ')')
