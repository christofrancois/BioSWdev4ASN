#!/usr/bin/python

# script generating SIMPLE stimtime format
# to redirect from std output to .stimtimes file
import sys
from random import randint

def writePattern(time, curr):
    print("%s %s %s" % (time, 1, curr))
    print("%s %s %s" % (time + eps, 0, curr))
    print("%s %s %s" % (time + ton, 0, curr))
    print("%s %s %s" % (time + ton + eps, 1, curr))
    return

if len(sys.argv) > 1:
    length = int(sys.argv[1])
else:
    length = 1000

start = 0
ton = 1
toff = 2
nb_patt = 4

i = start
eps = 0.0001

while i < length:
    writePattern(i, randint(0, nb_patt - 1))
    i += ton + toff
