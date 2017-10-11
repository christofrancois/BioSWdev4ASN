# SIZE * 2 MNIST DATABASE
# AND WRITE IT TO THE STANDARD OUTPUT
# TO REDIRECT TO .pat FILE
# CHANGE nb TO ADD MORE PATTERNS
# OR REPLACE IT BY size FOR ALL PATTERNS

import struct

f = open("../../data/train-images.idx3-ubyte", "rb")

# Read first 8 bytes
bytes = f.read(16)
# '>' because big endian
# 'II' because we read 2 unsigned Integers : magic, and size
magic, size, nb_row, nb_colum = struct.unpack(">IIII", bytes)

# Print("%d %d %d %d" % (magic, size, nb_row, nb_colum))
# 2051 60000 28 28
nb = 200

images = []
#for index in range(0, size):
for index in range(0, nb):
    image = []
    # Extraction image
    for i in range(0, 28):
        line = []
        bytes = f.read(28)
        for j in range(0, 28):
            line.append(float(bytes[j] / 255))
        image.append(line)
    # Size * 2 and Print
    resized = []
    print("# Char n°%d"%(index))
    for i in range(0, 28*2):
        line = []
        for j in range(0, 28*2):
            #line.append(image[int(i/2)][int(j/2)])
            if image[int(i/2)][int(j/2)] == 0:
                line.append(0)
            else:
                line.append(1)
                print("%d %d"%(i*56+j, 1))#comment if console preview
        resized.append(line) #len(line) = 56
        #print(line) #uncomment if preview on console
    print("\n\n")

f.close()
