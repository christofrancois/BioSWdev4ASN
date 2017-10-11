# this script generates 10 labels in a population of 2048
# (from 0 to 2000)
curr = 0
for i in range(0, 10):
    print("#label %s" % i)
    for j in range(0, 200):
        print("%s %s" % (curr, 1))
        curr += 1
    print("\n\n")
