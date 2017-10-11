#!/bin/bash
RESULTS=../../results
DATA=../../data/trig
STIM=../../data/stimtimes
T_START=100
T_STOP=1900

echo "Generating trigger file"
./mktrigfile.awk $STIM/original2.stimtimes > $DATA/e2.trig

echo "extract ras"
aube --input $RESULTS/2-*.e.spk --from $T_START > $DATA/e2.ras

echo "Computing PSTH..."
./rassta.py -t $DATA/e2.trig -w 0.5 -f $DATA/e2.ras -s $T_START -m $T_STOP -q 0.0 -o $DATA/e2.sta

echo "Generating pattern file..."
./sta2pat.sh $DATA/e2.sta $DATA/e2.pat
