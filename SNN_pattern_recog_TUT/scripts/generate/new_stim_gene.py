# The script writes in the files .stimtimes automatically

from random import randint
import math
import struct

file1 = "stim1.stimtimes"
file2 = "stim2.stimtimes"
fs1 = open(file1, "a+")
fs2 = open(file2, "a+")
MNIST_path = "../../data/train-labels.idx1-ubyte"
flab = open(MNIST_path, "rb")

start = 0
length = 3600 * 5
ton = 0.75 #time of stimulation
toff = 2.25 #time of rest
eps = 0.0001 #short period


#print magic number and the size
buff = flab.read(8)
print("%d %d" % (struct.unpack(">II", buff)))

def writePattern(time, patt, lab):
    fs1.write("%s %s %s\n" % (time, 1, patt))
    fs1.write("%s %s %s\n" % (time + eps, 0, patt))
    fs1.write("%s %s %s\n" % (time + ton, 0, patt))
    fs1.write("%s %s %s\n" % (time + ton + eps, 1, patt))

    fs2.write("%s %s %s\n" % (time, 1, lab))
    fs2.write("%s %s %s\n" % (time + eps, 0, lab))
    fs2.write("%s %s %s\n" % (time + ton, 0, lab))
    fs2.write("%s %s %s\n" % (time + ton + eps, 1, lab))
    return


nb_patt = 100
buff = flab.read(nb_patt)
it = 0
i = start
while i < length:
    label = struct.unpack(">B", buff[it])[0]
    writePattern(i, it, label)
    i += ton + toff
    it += 1
    if (it >= nb_patt):
        it = 0

fs1.close()
fs2.close()
flab.close()
