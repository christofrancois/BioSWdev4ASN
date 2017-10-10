#counts of much pattenrs in each label

import struct

f = open("../../data/train-labels.idx1-ubyte", "rb")

#print magic number and the size
buff = f.read(8)
print("%d %d" % (struct.unpack(">II", buff)))

#count occurs
it = 40
occ = [0,0,0,0,0,0,0,0,0,0]

buff = f.read(it)
for i in range(0, it):
    occ[buff[i]] += 1

print(occ)

f.close()
