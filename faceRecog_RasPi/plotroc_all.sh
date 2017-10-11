for CLASSIFIER in GaussianNB LinearSvm RadialSvm RandomForest AdaBoostClassifier KNN
do
	python plotroc.py ./processed ./classifier/$CLASSIFIER.pkl ./models/nn4.small2.v1.t7
	mv roc1.pdf ${CLASSIFIER}_roc1.pdf
	mv roc2.pdf ${CLASSIFIER}_roc2.pdf
	mv suc1.pdf ${CLASSIFIER}_suc1.pdf
	mv suc2.pdf ${CLASSIFIER}_suc2.pdf
done
