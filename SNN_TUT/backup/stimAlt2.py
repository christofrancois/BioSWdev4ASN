#script generating SIMPLE stimtime format

from random import randint

start = 25
length = 2000
#only 2 patterns

i = 0

while i < length:
    if i % 5 == 0:
        print("%s %s %s"% (i + 0.01 - 0.01 + start, 1, 0))
        print("%s %s %s"% (i + 0.01 + start, 0, 0))
    if i % 5 == 0.5:
        print("%s %s %s"% (i + 0.01 - 0.01 + start, 0, 0))
        print("%s %s %s"% (i + 0.01 + start, 1, 0))
    if i % 5 == 2.5:
        print("%s %s %s"% (i + 0.01 - 0.01 + start, 1, 1))
        print("%s %s %s"% (i + 0.01 + start, 0, 1))
    if i % 5 == 3:
        print("%s %s %s"% (i + 0.01 - 0.01 + start, 0, 1))
        print("%s %s %s"% (i + 0.01 + start, 1, 1))
    i += 0.5

print("%s %s %s"% (i + 0.01 - 0.01 + start, 1, 0))
print("%s %s %s"% (i + 0.01 - 0.01 + start, 1, 1))
print("%s %s %s"% (3600 + start, 1, 0))
print("%s %s %s"% (3600 + start, 1, 1))
