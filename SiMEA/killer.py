__author__ = 'V_AD'


import numpy as np
import random as rand
import matplotlib.pyplot as plt

def death_list (neurons):
    NN = len(neurons)
    output = np.zeros(12240);
    q = float(NN)
    percentage = float(rand.sample(range(45,55),1)[0])/100
    x = rand.sample(range(0,12240),int(q*percentage))
    y = rand.sample(neurons,int(q*percentage))
    output[x] = y
#     plt.scatter(x,y,c=colors)
#     plt.show()
    return output

def killer (q,days):
    NN = np.count_nonzero(q)
    neurons = np.zeros (NN)
    l = len (q)
    counter = 0
    for i in range (0,l):
        for j in range (0,l):
            if q[i,j]!=0:
                idx = i*l+j
                neurons[counter] = idx
                counter+=1
    dead_idx = death_list(neurons)
#     t_idx = t_idx [0:days*60*12]
    dead_idx = dead_idx [0:days*60*12]

    for i in range (0,len(dead_idx)):
        if dead_idx[i] !=0:
            dead = dead_idx[i]
            x = dead/l
            y = dead%l
            q[x,y] = 2.5
    return q


