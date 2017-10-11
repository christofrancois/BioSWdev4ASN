
# Takes a "Project folder" as $1 argument
# This folder should have the following folders
# images: containing folders for each person to be classified each with images of that person

FACE_DIR=/worktmp/openface/

rm $1/data/cache.t7
mkdir $1/features

th $FACE_DIR/batch-represent/main.lua -outDir $1/features -data $1/data

mkdir $1/classifier

for clf in RandomForest #GaussianNB LinearSvm RandomForest RadialSvm AdaBoostClassifier KNN
do
  python $FACE_DIR/demos/classifier.py train $1/features --classifier $clf
  mv $1/features/classifier*.pkl $1/classifier/$clf.pkl
done

#python $FACE_DIR/demos/classifier.py infer $1/classifier/classifier.pkl ./*.jpg images/*/*

