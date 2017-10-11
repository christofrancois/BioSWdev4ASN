#!/bin/bash
#call with ./generate_all

stimtimes=../../data/stimtimes
echo 'cleaning stimtimes files'
rm $stimtimes/*
echo 'generating stimtimes files'
rm stim1.stimtimes && touch stim1.stimtimes
rm stim2.stimtimes && touch stim2.stimtimes
python new_stim_gene.py
cp stim1.stimtimes $stimtimes/original.stimtimes
cp stim2.stimtimes $stimtimes/original2.stimtimes

echo ''

stimfiles=../../data/stimfiles
echo 'cleaning stimfiles'
rm $stimfiles/*
echo 'generating stimfiles'
python3 mnist_gene.py > $stimfiles/mnist.pat
python3 labels_gene.py > $stimfiles/labels.pat
